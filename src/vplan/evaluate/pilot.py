"""Feedback pilot artifacts: offline cohort preparation, validated live runs and the paired report.

Layout: DIR/{tasks,oracle,exclusions}.jsonl + manifest.json (written last, so a crashed
preparation never validates) and DIR/runs/<run_id>/{run,summary}.json, events.jsonl,
results.jsonl and report.md. Saved evidence is refused against a changed task, config
or source snapshot so results can never mix snapshots.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import statistics
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urlparse

from vplan.config import CONFIG_DIR, config_hash, config_snapshot
from vplan.planners.nemotron import NemotronPlanner
from vplan.records import GeneratedTask, PairedResult, RoundRecord, TaskMetadata, TaskOracle, TaskSpec
from vplan.repair import _terminal, evaluate_pair
from vplan.tasks import HARD_STRATA, STRATA, CohortPreparationError, prepare_cohort, prepare_hard_cohort

ROOT = CONFIG_DIR.parent
SCHEMA_VERSION = 1
HEADER = "Development pilot (feasible tasks only) — not a held-out result"
ARMS = ("repair", "resample")
COHORT_FILES = ("tasks.jsonl", "oracle.jsonl", "exclusions.jsonl")
MANIFEST_KEYS = ("schema_version", "n", "files", "config_hashes", "source_hashes")
# Cohort generator -> its strata. Manifests from before the "generator" field are pilot cohorts.
GENERATORS = {"pilot": STRATA, "hard": HARD_STRATA}
# Auth, billing and wrong-path errors will not recover by retrying the next task.
FATAL_STATUS = frozenset({401, 402, 403, 404})
# Consecutive failed requests across tasks usually mean a quota or rate cap, not a planning outcome.
MAX_CONSECUTIVE_SERVICE_FAILURES = 3
_WINDOWS = os.name == "nt"


class PilotError(RuntimeError):
    """A refused preparation or run; the message is safe to print (never contains credentials)."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, data: dict) -> None:
    # Replace, never truncate in place: a crash mid-write must not leave a half-written status file.
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(row + "\n" for row in rows), encoding="utf-8")


def source_hashes() -> dict[str, str]:
    """sha256 of every package source and config file, keyed by repo-relative POSIX path."""
    paths = sorted((ROOT / "src" / "vplan").rglob("*.py")) + sorted(CONFIG_DIR.glob("*.json"))
    return {p.relative_to(ROOT).as_posix(): _sha256(p) for p in paths}


def source_digest(hashes: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode("utf-8")).hexdigest()


def _manipulapy() -> dict:
    # find_spec locates the package without importing it (import is slow and prints a banner).
    try:
        version = importlib.metadata.version("ManipulaPy")
    except importlib.metadata.PackageNotFoundError:
        version = None
    spec = importlib.util.find_spec("ManipulaPy")
    return {"version": version, "location": None if spec is None else spec.origin}


def _versions() -> dict:
    out = {}
    for name in ("openai", "pybullet", "ManipulaPy"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def check_path_arg(value: str) -> None:
    """The project runs Windows Python from WSL, which would resolve /mnt/c/... against the current drive."""
    if _WINDOWS and str(value).replace("\\", "/").startswith("/mnt/"):
        raise PilotError(f"{value} is a WSL path; pass a relative or Windows path (the interpreter is Windows Python)")


def _git() -> dict | None:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              timeout=10, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True,
                                timeout=10, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return {"head": head, "dirty": bool(status.strip())}


def prepare(output: Path, *, n: int = 20, seed: int = 0, max_candidates: int = 100, progress=None,
            generator: str = "pilot") -> dict:
    """Freeze a witness-validated cohort (no model calls) into separate public/private files."""
    if generator not in GENERATORS:
        raise PilotError(f"unknown generator {generator!r}; expected one of {', '.join(GENERATORS)}")
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise PilotError(f"output directory {output} is not empty; refusing to overwrite")
    hashes = source_hashes()
    try:
        build = prepare_hard_cohort if generator == "hard" else prepare_cohort
        cohort = build(n=n, seed=seed, max_candidates=max_candidates, progress=progress)
    except CohortPreparationError as exc:
        # Keep the rejection log for diagnosis; without a manifest the directory can never be run.
        output.mkdir(parents=True, exist_ok=True)
        _write_jsonl(output / "exclusions.jsonl", (e.model_dump_json() for e in exc.cohort.exclusions))
        raise
    if source_hashes() != hashes:
        raise PilotError("source or config files changed during preparation; discard and prepare again")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise PilotError(f"output directory {output} became non-empty during preparation")
    _write_jsonl(output / "tasks.jsonl", (json.dumps({
        "task_id": t.metadata.task_id, "seed": t.metadata.seed, "stratum": t.metadata.stratum,
        "task": t.task.model_dump(mode="json")}) for t in cohort.tasks))
    _write_jsonl(output / "oracle.jsonl", (json.dumps({
        "task_id": t.metadata.task_id, "oracle": t.oracle.model_dump(mode="json")}) for t in cohort.tasks))
    _write_jsonl(output / "exclusions.jsonl", (e.model_dump_json() for e in cohort.exclusions))
    manifest = {
        "schema_version": SCHEMA_VERSION, "created_utc": _utc_now(), "scope": HEADER,
        "generator": generator, "n": n, "seed": seed, "max_candidates": max_candidates, "attempted": cohort.attempted,
        "excluded": len(cohort.exclusions),
        "strata": dict(Counter(t.metadata.stratum for t in cohort.tasks)),
        "config_hashes": cohort.config_hashes,
        "files": {name: _sha256(output / name) for name in COHORT_FILES},
        "source_hashes": hashes, "source_digest": source_digest(hashes),
        "python": platform.python_version(), "manipulapy": _manipulapy(), "git": _git()}
    _write_json(output / "manifest.json", manifest)
    return manifest


class Prepared(NamedTuple):
    tasks: list[GeneratedTask]
    manifest: dict
    manifest_sha256: str
    config: dict[str, dict]
    source_digest: str
    drift: list[str]
    generator: str


def load_prepared(input_dir: Path, *, allow_source_drift: bool = False) -> Prepared:
    """Validate a prepared cohort against the current checkout before any model call."""
    input_dir = Path(input_dir)
    manifest_path = input_dir / "manifest.json"
    if not manifest_path.is_file():
        raise PilotError(f"{manifest_path} is missing: not a prepared cohort, or preparation did not finish")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except ValueError:
        raise PilotError(f"{manifest_path} is not valid JSON") from None
    missing = [k for k in MANIFEST_KEYS if not isinstance(manifest, dict) or k not in manifest]
    if missing:
        raise PilotError(f"manifest.json lacks required keys: {', '.join(missing)}")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise PilotError(f"unsupported manifest.json schema_version {manifest.get('schema_version')!r}")
    generator = manifest.get("generator", "pilot")
    if generator not in GENERATORS:
        raise PilotError(f"manifest.json names unknown generator {generator!r}")
    for name in COHORT_FILES:
        path = input_dir / name
        if not path.is_file() or _sha256(path) != manifest["files"].get(name):
            raise PilotError(f"{name} does not match manifest.json (tampered or partial file)")
    cfgs = config_snapshot()
    changed = sorted(k for k in set(cfgs) | set(manifest["config_hashes"])
                     if k not in cfgs or config_hash(cfgs[k]) != manifest["config_hashes"].get(k))
    if changed:
        raise PilotError(f"configuration changed since preparation: {', '.join(changed)}")
    current, prepared = source_hashes(), manifest["source_hashes"]
    drift = sorted(p for p in set(current) | set(prepared) if current.get(p) != prepared.get(p))
    if drift and not allow_source_drift:
        raise PilotError("source changed since preparation (pass --allow-source-drift to record and proceed): "
                         + ", ".join(drift))
    public = [json.loads(line) for line in (input_dir / "tasks.jsonl").read_text(encoding="utf-8").splitlines()]
    private = [json.loads(line) for line in (input_dir / "oracle.jsonl").read_text(encoding="utf-8").splitlines()]
    if [r["task_id"] for r in public] != [r["task_id"] for r in private] or len(public) != manifest["n"]:
        raise PilotError("tasks.jsonl and oracle.jsonl disagree on task ids or count")
    tasks = [GeneratedTask(metadata=TaskMetadata(task_id=pub["task_id"], seed=pub["seed"], stratum=pub["stratum"]),
                           task=TaskSpec.model_validate(pub["task"]), oracle=TaskOracle.model_validate(priv["oracle"]))
             for pub, priv in zip(public, private)]
    foreign = sorted({t.metadata.stratum for t in tasks} - set(GENERATORS[generator]))
    if foreign:
        raise PilotError(f"tasks.jsonl has stratum {', '.join(foreign)}, which the {generator} generator never makes")
    return Prepared(tasks, manifest, _sha256(manifest_path), cfgs, source_digest(current), drift, generator)


def read_env_file(path: Path) -> dict[str, str]:
    """Minimal KEY=VALUE parser for an explicitly named file; errors never echo line content."""
    try:
        # utf-8-sig: Windows editors often save a BOM, which would otherwise corrupt the first key.
        text = Path(path).read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise PilotError(f"cannot read env file {path}: {type(exc).__name__}") from None
    values = {}
    for number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, sep, value = line.partition("=")
        if not sep or not key.strip():
            raise PilotError(f"{path}: line {number} is not KEY=VALUE")
        value = value.strip()
        end = value.find(value[0], 1) if value[:1] in ("\"", "'") else -1
        if end > 0:
            value = value[1:end]  # quoted: keep '#' inside, drop anything after the closing quote
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0]
        values[key.strip()] = value
    return values


def run_live(input_dir: Path, *, env_file: Path, model: str | None = None, temperature: float = .7,
             limit: int | None = None, allow_source_drift: bool = False, allow_paid: bool = False,
             run_id: str | None = None) -> Path:
    """Credentials come only from env_file. Without allow_paid only free OpenRouter models run."""
    env = read_env_file(env_file)
    base_url, api_key = env.get("LLM_BASE_URL", ""), env.get("LLM_API_KEY", "")
    model = model or env.get("LLM_MODEL", "")
    if not (base_url and api_key and model):
        raise PilotError(f"{env_file} must set LLM_BASE_URL, LLM_API_KEY and LLM_MODEL (or pass --model)")
    parsed = urlparse(base_url)
    host = parsed.hostname or ""
    if not allow_paid and not (parsed.scheme == "https" and host == "openrouter.ai"
                               and model.startswith("nvidia/") and model.endswith(":free")):
        raise PilotError(f"refusing {host or base_url!r} / {model!r}: only https://openrouter.ai 'nvidia/...:free' "
                         "models run without --allow-paid (paid calls need explicit approval)")
    provider = "openrouter" if host == "openrouter.ai" else host
    planner = NemotronPlanner(model=model, provider=provider, base_url=base_url, api_key=api_key,
                              temperature=temperature)
    return run_pilot(input_dir, planner, base_url_host=host, limit=limit,
                     allow_source_drift=allow_source_drift, run_id=run_id)


def run_pilot(input_dir: Path, planner, *, base_url_host: str, limit: int | None = None,
              allow_source_drift: bool = False, run_id: str | None = None) -> Path:
    """Evaluate tasks in order, appending and flushing evidence after every event and task."""
    input_dir = Path(input_dir)
    prepared = load_prepared(input_dir, allow_source_drift=allow_source_drift)
    tasks, cfgs = prepared.tasks, prepared.config
    if limit is not None and limit < 1:
        raise PilotError("--limit must be >= 1")
    selected = tasks[:limit]
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = input_dir / "runs" / run_id
    try:
        run_dir.mkdir(parents=True)
    except FileExistsError:
        raise PilotError(f"run directory {run_dir} exists; refusing to overwrite evidence") from None
    meta = {"schema_version": SCHEMA_VERSION, "run_id": run_id, "status": "running", "started_utc": _utc_now(),
            "model": planner.model, "provider": planner.provider, "base_url_host": base_url_host,
            "temperature": planner.temperature, "manifest_sha256": prepared.manifest_sha256,
            "generator": prepared.generator,
            "config_hashes": {k: config_hash(v) for k, v in cfgs.items()},
            "n_prepared": len(tasks), "n_tasks": len(selected), "limit": limit, "source_drift": prepared.drift,
            "source_digest": prepared.source_digest, "git": _git(), "python": platform.python_version(),
            "versions": _versions(),
            "manifest_manipulapy_version": (prepared.manifest.get("manipulapy") or {}).get("version")}
    _write_json(run_dir / "run.json", meta)
    provider_state = {"consecutive": 0, "fatal": None}
    stop_reason = None
    try:
        with open(run_dir / "events.jsonl", "a", encoding="utf-8") as events, \
                open(run_dir / "results.jsonl", "a", encoding="utf-8") as results:
            def sink(event: dict) -> None:
                events.write(json.dumps(event) + "\n")
                events.flush()
                if event.get("event") == "request_completed":
                    call = event["call"]
                    if call["outcome"] != "service_failure":
                        provider_state["consecutive"] = 0
                    else:
                        provider_state["consecutive"] += 1
                        if call.get("status_code") in FATAL_STATUS:
                            provider_state["fatal"] = call["status_code"]

            for index, task in enumerate(selected, 1):
                # Checked between tasks so every recorded task is whole; at most one task's requests are spent.
                if provider_state["fatal"] is not None:
                    stop_reason = f"provider status {provider_state['fatal']} (auth, billing or endpoint path)"
                elif provider_state["consecutive"] >= MAX_CONSECUTIVE_SERVICE_FAILURES:
                    stop_reason = (f"{provider_state['consecutive']} consecutive service-failure requests "
                                   "(likely a quota or rate cap)")
                if stop_reason:
                    break
                # Alternate which arm runs first so an outage mid-task does not always hit the same arm.
                arm_order = ARMS if index % 2 else ARMS[::-1]
                result = evaluate_pair(task, planner, config_snapshot=cfgs, event_sink=sink, arm_order=arm_order)
                results.write(result.model_dump_json() + "\n")
                results.flush()
                print(f"task {index}/{len(selected)} {task.metadata.task_id}: "
                      f"repair={result.repair.outcome} resample={result.resample.outcome}", flush=True)
    except BaseException as exc:
        # Includes KeyboardInterrupt: the partial report must be conspicuously incomplete.
        _write_json(run_dir / "run.json", {**meta, "status": "incomplete", "error_type": type(exc).__name__})
        write_report(run_dir)
        raise
    if stop_reason:
        _write_json(run_dir / "run.json", {**meta, "status": "incomplete", "stop_reason": stop_reason})
        write_report(run_dir)
        raise PilotError(f"run stopped: {stop_reason}; partial evidence and an INCOMPLETE report are in {run_dir}")
    _write_json(run_dir / "run.json", {**meta, "status": "complete", "finished_utc": _utc_now()})
    write_report(run_dir)
    return run_dir


def _succeeded(record) -> bool:
    e = record.evaluation
    return e is not None and e.outcome == "executed" and e.judge is not None and e.judge.success


def _rate(count: int, denominator: int) -> dict:
    return {"count": count, "denominator": denominator, "rate": count / denominator if denominator else None}


def _median(values):
    return statistics.median(values) if values else None


def summarize(results: list[PairedResult], n_expected: int) -> dict:
    """Counts over every completed task; service failures and errors stay in the denominator."""
    n = len(results)
    first = [_succeeded(r.initial) for r in results]
    won = {arm: [_succeeded(getattr(r, arm).rounds[-1]) for r in results] for arm in ARMS}
    pairs = list(zip(won["repair"], won["resample"]))
    unique_rounds = [round_ for r in results
                     for round_ in (r.initial,) + r.repair.rounds[1:] + r.resample.rounds[1:]]

    def per_arm(count):
        return {arm: dict(Counter(x for r in results for x in count(getattr(r, arm)))) for arm in ARMS}

    def failure_status(rounds):
        return [str(x.call.status_code or "transport") for x in rounds if x.call.outcome == "service_failure"]

    initial_terminal = [_terminal(r.initial) for r in results]

    return {
        "scope": HEADER, "tasks_expected": n_expected, "tasks_completed": n, "complete": n == n_expected,
        "success": {"first_attempt": _rate(sum(first), n), "repair": _rate(sum(won["repair"]), n),
                    "resample": _rate(sum(won["resample"]), n)},
        "paired": {"repair_only": sum(a and not b for a, b in pairs), "resample_only": sum(b and not a for a, b in pairs),
                   "both": sum(a and b for a, b in pairs), "neither": sum(not a and not b for a, b in pairs),
                   # Tasks whose arms actually continued past the shared initial call.
                   "repair_opportunities": sum(t is None for t in initial_terminal)},
        "initial_terminal_failures": dict(Counter("judge_failure" if t == "executed" else t
                                                  for t, ok in zip(initial_terminal, first) if t and not ok)),
        "service_failures": {"total": len(failure_status(unique_rounds)),
                             "initial": dict(Counter(failure_status([r.initial for r in results]))),
                             **per_arm(lambda arm: failure_status(arm.rounds))},
        "outcomes": per_arm(lambda arm: [arm.outcome]),
        # Arm counts include the shared initial round, so the two arms are directly comparable.
        "finding_checks": {"first_round": dict(Counter(f.check for r in results if r.initial.evaluation
                                                       for f in r.initial.evaluation.findings)),
                           **per_arm(lambda arm: [f.check for x in arm.rounds if x.evaluation
                                                  for f in x.evaluation.findings])},
        "observable_failures": per_arm(lambda arm: [f for x in arm.rounds for f in x.observable_failures]),
        "repeat_plans": {arm: sum(x.repeat_plan for r in results for x in getattr(r, arm).rounds) for arm in ARMS},
        "cost": {
            "logical_calls": {arm: sum(getattr(r, arm).call_count for r in results) for arm in ARMS},
            "unique_requests": sum(r.unique_call_count for r in results),
            "tokens": {**{arm: {"known_total": sum(getattr(r, arm).known_total_tokens for r in results),
                                "missing_usage_calls": sum(getattr(r, arm).missing_usage_calls for r in results)}
                          for arm in ARMS},
                       "unique": {"known_total": sum(r.unique_known_total_tokens for r in results),
                                  "missing_usage_calls": sum(r.unique_missing_usage_calls for r in results)}},
            "latency_s": {**{arm: {"total": sum(getattr(r, arm).latency_s for r in results),
                                   "median_per_task": _median([getattr(r, arm).latency_s for r in results])}
                             for arm in ARMS},
                          "unique": {"total": sum(r.unique_latency_s for r in results),
                                     "median_per_task": _median([r.unique_latency_s for r in results]),
                                     "median_per_request": _median([x.call.latency_s for x in unique_rounds])}}},
    }


def _render(summary: dict, meta: dict) -> str:
    failures = summary["service_failures"]
    prefixes = []
    if not summary["finished"]:
        prefixes.append(f"INCOMPLETE: {summary['tasks_completed']}/{summary['tasks_expected']} tasks")
    if summary["limited"]:
        prefixes.append(f"LIMITED: first {meta['n_tasks']} of {meta['n_prepared']} tasks")
    if failures["total"]:
        prefixes.append(f"DEGRADED: {failures['total']} service failures")
    lines = [f"# {' — '.join(prefixes + [HEADER])}", ""]
    if not summary["finished"]:
        lines += [f"**This run did not finish (status: {meta['status']}"
                  + (f"; stopped: {meta['stop_reason']}" if meta.get("stop_reason") else "")
                  + "). Do not read these counts as a result.**", ""]
    if failures["total"]:
        lines += [f"**{failures['total']} provider requests failed (by status, shared initial call: "
                  f"{failures['initial']}; repair arm: {failures['repair']}; resample arm: {failures['resample']}). "
                  "These failures stay in the denominators, so success counts partly measure provider "
                  "availability, not planning.**", ""]
    if summary.get("truncated_result_lines"):
        lines += ["**The last line of results.jsonl was cut off (killed mid-write) and was skipped.**", ""]
    if meta.get("limit") is not None:
        lines += [f"**Limited run: first {meta['n_tasks']} of {meta['n_prepared']} prepared tasks (smoke test).**", ""]
    manipulapy = (meta.get("versions") or {}).get("ManipulaPy")
    if manipulapy != meta.get("manifest_manipulapy_version", manipulapy):
        lines += [f"**ManipulaPy version differs from preparation: {meta['manifest_manipulapy_version']} "
                  f"at prepare, {manipulapy} at run.**", ""]
    if meta.get("source_drift"):
        lines += ["**Source drift since preparation (allowed by flag):** " + ", ".join(meta["source_drift"]), ""]
    lines += [f"Run `{meta['run_id']}`: model `{meta['model']}` via {meta['provider']} ({meta['base_url_host']}), "
              f"temperature {meta['temperature']} for every call in both arms; manifest sha256 "
              f"`{meta['manifest_sha256']}`.", "",
              "Counts only, no significance testing: a small feasible-only development cohort is diagnostic.", "",
              "## Success over all completed tasks", "", "| | successes | tasks | rate |", "|---|---|---|---|"]
    for name, value in summary["success"].items():
        rate = "n/a" if value["rate"] is None else f"{value['rate']:.2f}"
        lines.append(f"| {name} | {value['count']} | {value['denominator']} | {rate} |")
    lines += ["", "## Paired outcomes", ""] + [f"- {k}: {v}" for k, v in summary["paired"].items()]
    for section in ("initial_terminal_failures", "service_failures", "outcomes", "finding_checks", "observable_failures", "repeat_plans", "cost"):
        lines += ["", f"## {section.replace('_', ' ').capitalize()}", "", "```json",
                  json.dumps(summary[section], indent=2), "```"]
    return "\n".join(lines) + "\n"


def write_report(run_dir: Path) -> dict:
    """(Re)build summary.json and report.md from saved evidence; usable after a hard crash."""
    run_dir = Path(run_dir)
    if not (run_dir / "run.json").is_file():
        raise PilotError(f"{run_dir / 'run.json'} is missing: not a pilot run directory")
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    path = run_dir / "results.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    results, truncated = [], 0
    for i, line in enumerate(lines):
        try:
            results.append(PairedResult.model_validate_json(line))
        except ValueError:
            if i != len(lines) - 1:  # only a kill mid-append can damage the last line; anything else is corruption
                raise
            truncated = 1
    summary = summarize(results, meta["n_tasks"])
    summary["truncated_result_lines"] = truncated
    summary["finished"] = summary["complete"] and meta["status"] == "complete" and not truncated
    # A --limit smoke test answers a different question than the cohort; it is never "complete".
    summary["limited"] = meta["n_tasks"] < meta["n_prepared"]
    summary["complete"] = summary["finished"] and not summary["limited"]
    summary["run"] = meta
    _write_json(run_dir / "summary.json", summary)
    (run_dir / "report.md").write_text(_render(summary, meta), encoding="utf-8")
    return summary


def _read_results(run_dir: Path) -> list[PairedResult]:
    path = Path(run_dir) / "results.jsonl"
    if not path.is_file():
        raise PilotError(f"{path} is missing: not a pilot run directory")
    # a killed run may leave a truncated final line; write_report flags it, viewing just skips it
    results = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            results.append(PairedResult.model_validate_json(line))
        except ValueError:
            continue
    return results


def _round_label(rnd: RoundRecord) -> str:
    return rnd.evaluation.outcome if rnd.evaluation is not None else rnd.call.outcome


def list_rounds(run_dir: Path) -> list[str]:
    """One line per task: the outcome of every round in each arm (round 0 is the shared initial call)."""
    return [f"{r.task_id}  " + "  ".join(f"{arm.arm}: " + " -> ".join(_round_label(x) for x in arm.rounds)
                                         for arm in (r.repair, r.resample))
            for r in _read_results(run_dir)]


def select_round(run_dir: Path, task_id: str, arm: str, round_index: int | None = None
                 ) -> tuple[GeneratedTask, RoundRecord]:
    """The recorded plan of one round plus its task, for re-execution. The run lives at <cohort>/runs/<run_id>;
    source drift is tolerated because viewing produces no evidence."""
    result = next((r for r in _read_results(run_dir) if r.task_id == task_id), None)
    if result is None:
        raise PilotError(f"no task {task_id} in {run_dir}")
    rounds = getattr(result, arm).rounds
    if round_index is None:
        round_index = rounds[-1].round_index
    rnd = next((x for x in rounds if x.round_index == round_index), None)
    if rnd is None:
        raise PilotError(f"{arm} arm of {task_id} has no round {round_index} (rounds 0-{rounds[-1].round_index})")
    tasks = load_prepared(Path(run_dir).parent.parent, allow_source_drift=True).tasks
    return next(t for t in tasks if t.metadata.task_id == task_id), rnd
