"""Recorded attempts for keyless replay (spec §15 replay mode, D41): the AttemptRecord, its task and the seed its
motion layer used. Nothing here reads an env file, imports a planner or touches the network.

Sources:
- a W3.1 run directory (an eval_run manifest.json and attempts.jsonl, records.EvalAttempt): one case per row. The task
  comes from the seed set the manifest names (or seed_dir), whose tasks.jsonl and oracle.jsonl must hash to the run's
  seed-set hash;
- a pilot run directory (<cohort>/runs/<run_id>/ with run.json and results.jsonl, records.PairedResult): one case per
  task and arm. The task comes from the cohort two levels up, whose files must match its manifest;
- a UI recording (ui.live.save_bundle), or a directory of them such as runs/ui.

Replay is display, not evidence, so unlike `eval run` and `report build` no configuration or source change refuses it;
replay.replay ties every played-back ExecutionRecord to its scene by hash.

Paths: network (UNC) paths are always refused, since opening one on Windows sends the user's credentials to that
server. With roots (the UI passes the working directory, --record-dir, --replay and --seed-dir), every path read,
including a seed set or cohort a run points to, must resolve under one of them. Damaged files raise ReplayError.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from vplan.records import (AttemptRecord, EvalAttempt, PairedResult, SeedSetOracle, SeedSetTask, TaskOracle,
                           TaskSpec)

BUNDLE_KIND = "vplan_ui_attempt"  # ui.live writes it
EVAL_KIND = "eval_run"  # evaluate.runner.KIND


class ReplayError(RuntimeError):
    """A path or case that cannot be replayed; the message is safe to show."""


Roots = Sequence[str | Path] | None
DAMAGED = (KeyError, IndexError, TypeError, AttributeError, ValueError)  # pydantic's ValidationError is a ValueError


def is_network_path(value: str | Path) -> bool:
    text = str(value).replace("\\", "/")
    return text.startswith("//")


def check_path(value: str | Path, roots: Roots = None) -> Path:
    """value as a Path; ReplayError for a network path, or (with roots) for one that does not resolve under a root."""
    if is_network_path(value):
        raise ReplayError(f"{value} is a network (UNC) path; replay reads local files only")
    path = Path(value)
    if roots is not None:
        resolved = path.resolve()
        bases = [Path(r).resolve() for r in roots if r and not is_network_path(r)]
        if not any(resolved == b or b in resolved.parents for b in bases):
            raise ReplayError(f"{value} is outside the replay roots (the working directory, --record-dir, --replay "
                              "and --seed-dir)")
    return path


@dataclass(frozen=True)
class CaseRef:
    case_id: str
    label: str


@dataclass(frozen=True)
class ReplayCase:
    case_id: str
    label: str
    task: TaskSpec
    oracle: TaskOracle | None
    record: AttemptRecord
    seed: int  # what run_task passed to the motion layer (mod 2**32)
    source: str
    provenance: tuple[tuple[str, str], ...] = ()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ReplayError(f"{path} is not readable JSON ({type(exc).__name__})") from None
    if not isinstance(data, dict):
        raise ReplayError(f"{path} does not hold a JSON object")
    return data


def _lines(path: Path):
    """(line number, raw line, JSON object) per line; a cut-off final line (a killed run) is skipped."""
    if not path.is_file():
        raise ReplayError(f"{path} is missing")
    lines = path.read_bytes().splitlines()
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            if number == len(lines):
                continue
            raise ReplayError(f"{path.name} line {number} is not JSON") from None
        if not isinstance(row, dict):
            raise ReplayError(f"{path.name} line {number} is not a JSON object")
        yield number, line, row


def _text(row: dict, key: str, where: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ReplayError(f"{where} has no {key}")
    return value


def _dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def source_kind(path: str | Path) -> str:
    path = Path(path)
    if path.is_file():
        return "recording"
    if not path.is_dir():
        raise ReplayError(f"{path} does not exist")
    if (path / "manifest.json").is_file():
        if _json(path / "manifest.json").get("kind") == EVAL_KIND:
            return "eval"
        raise ReplayError(f"{path} has a manifest.json but is not a W3.1 run directory (a seed set or cohort holds "
                          "tasks, not attempts)")
    if (path / "run.json").is_file() and (path / "results.jsonl").is_file():
        return "pilot"
    if any(path.glob("*.json")):
        return "recordings"
    raise ReplayError(f"{path} is neither a run directory, a pilot run, nor a UI recording (or a folder of them)")


# ---- W3.1 evaluation runs -------------------------------------------------------------------------------------------

def _eval_cases(run_dir: Path) -> list[CaseRef]:
    out = []
    for number, _, row in _lines(run_dir / "attempts.jsonl"):
        attempt_id = _text(row, "attempt_id", f"attempts.jsonl line {number}")
        record = _dict(row.get("record"))
        success = record.get("outcome") in ("executed", "refused") and _dict(record.get("judge")).get("success")
        out.append(CaseRef(attempt_id, f"{attempt_id}  {row.get('task_id', '')}  {row.get('outcome', '')}"
                                       + ("  (success)" if success else "")))
    return out


def _seed_set_task(seed_dir: Path, index: int, task_id: str, expected_hash: str) -> tuple[SeedSetTask, TaskOracle]:
    from vplan.evaluate.seedsets import seed_set_hash

    tasks, oracles = seed_dir / "tasks.jsonl", seed_dir / "oracle.jsonl"
    if not tasks.is_file() or not oracles.is_file():
        raise ReplayError(f"no seed set at {seed_dir} (tasks.jsonl and oracle.jsonl); pass the seed-set directory")
    found = seed_set_hash(_sha256(tasks), _sha256(oracles))
    if found != expected_hash:
        raise ReplayError(f"the seed set at {seed_dir} has hash {found[:12]}, but the run was made on "
                          f"{expected_hash[:12]}")
    task_line = tasks.read_bytes().splitlines()[index]
    oracle_line = oracles.read_bytes().splitlines()[index]
    try:
        task, oracle = SeedSetTask.model_validate_json(task_line), SeedSetOracle.model_validate_json(oracle_line)
    except ValidationError as exc:
        raise ReplayError(f"seed-set task {index} does not validate ({exc.error_count()} errors)") from None
    if task.task_id != task_id or oracle.task_id != task_id:
        raise ReplayError(f"seed-set task {index} is {task.task_id!r}, but the run recorded {task_id!r}")
    return task, oracle.oracle


def _eval_case(run_dir: Path, case_id: str | None, seed_dir: str | Path | None, roots: Roots) -> ReplayCase:
    manifest = _json(run_dir / "manifest.json")
    for number, line, row in _lines(run_dir / "attempts.jsonl"):
        if case_id is None or _text(row, "attempt_id", f"attempts.jsonl line {number}") == case_id:
            break
    else:
        raise ReplayError(f"{run_dir.name} has no attempt {case_id!r}" if case_id else f"{run_dir.name} has no "
                          "recorded attempts yet")
    try:
        attempt = EvalAttempt.model_validate_json(line)
    except ValidationError as exc:
        raise ReplayError(f"attempt {row.get('attempt_id')} does not validate ({exc.error_count()} errors)") from None
    if attempt.record is None:
        raise ReplayError(f"attempt {attempt.attempt_id} has no AttemptRecord to replay ({attempt.outcome}: "
                          f"{attempt.error_type or 'no record'})")
    recorded = _dict(manifest.get("seed_set"))
    if seed_dir:
        candidates = [check_path(seed_dir, roots)]
    else:
        named = [p for p in (recorded.get("dir"), recorded.get("resolved")) if isinstance(p, str) and p]
        candidates = []
        for p in named:
            try:
                candidates.append(check_path(p, roots))
            except ReplayError:
                continue  # a recorded path outside the roots, or on a network share, is never opened
        if named and not candidates:
            raise ReplayError(f"{run_dir.name}'s seed set ({named[0]}) is outside the replay roots or on a network "
                              "share; pass the seed-set directory")
    where = next((c for c in candidates if (c / "tasks.jsonl").is_file()), candidates[0] if candidates else None)
    if where is None:
        raise ReplayError(f"{run_dir.name} does not record its seed set; pass the seed-set directory")
    task, oracle = _seed_set_task(where, attempt.task_index, attempt.task_id, str(manifest.get("seed_set_hash")))
    provenance = (("source", f"W3.1 run {run_dir.name}"),
                  ("config", f"{manifest.get('config')} ({manifest.get('mode')}, k={manifest.get('k')})"),
                  ("planner", f"{manifest.get('provider')}/{manifest.get('model')}"),
                  ("seed set", f"{manifest.get('split')} {str(manifest.get('seed_set_hash'))[:12]}, task "
                               f"{attempt.task_index} ({attempt.task_id})"))
    return ReplayCase(attempt.attempt_id, f"{attempt.attempt_id} {attempt.task_id}", task.task, oracle,
                      attempt.record, task.seed, "W3.1 evaluation run", provenance)


# ---- feedback pilot runs --------------------------------------------------------------------------------------------

def _pilot_cases(run_dir: Path) -> list[CaseRef]:
    out = []
    for number, _, row in _lines(run_dir / "results.jsonl"):
        task_id = _text(row, "task_id", f"results.jsonl line {number}")
        for arm in ("repair", "resample"):
            out.append(CaseRef(f"{task_id}:{arm}", f"{task_id}  {arm}: {_dict(row.get(arm)).get('outcome', '')}"))
    return out


def _cohort(cohort: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    manifest = _json(cohort / "manifest.json") if (cohort / "manifest.json").is_file() else None
    if manifest is None:
        raise ReplayError(f"the pilot cohort {cohort} has no manifest.json")
    for name in ("tasks.jsonl", "oracle.jsonl"):
        if not (cohort / name).is_file() or _sha256(cohort / name) != (manifest.get("files") or {}).get(name):
            raise ReplayError(f"the pilot cohort's {name} does not match its manifest.json")
    tasks = {row["task_id"]: row for _, _, row in _lines(cohort / "tasks.jsonl")}
    oracles = {row["task_id"]: row for _, _, row in _lines(cohort / "oracle.jsonl")}
    return tasks, oracles


def _pilot_case(run_dir: Path, case_id: str | None, roots: Roots) -> ReplayCase:
    task_id, _, arm = (case_id or "").rpartition(":")
    for number, line, row in _lines(run_dir / "results.jsonl"):
        if case_id is None or _text(row, "task_id", f"results.jsonl line {number}") == task_id:
            break
    else:
        raise ReplayError(f"{run_dir.name} has no task {task_id!r}")
    arm = arm if case_id else "repair"
    if arm not in ("repair", "resample"):
        raise ReplayError(f"unknown pilot arm {arm!r}")
    try:
        result = PairedResult.model_validate_json(line)
    except ValidationError as exc:
        raise ReplayError(f"pilot result {row.get('task_id')} does not validate ({exc.error_count()} errors)") from None
    tasks, oracles = _cohort(check_path(run_dir.parent.parent, roots))
    if result.task_id not in tasks:
        raise ReplayError(f"the pilot cohort has no task {result.task_id!r}")
    meta = _json(run_dir / "run.json")
    a = getattr(result, arm)
    last = a.rounds[-1].evaluation
    record = AttemptRecord(attempt_id=f"{result.task_id}:{arm}", mode=arm, k=3, outcome=a.outcome, rounds=a.rounds,
                           call_count=a.call_count,
                           judge=last.judge if last is not None and a.outcome in ("executed", "refused") else None,
                           total_tokens=a.total_tokens, known_total_tokens=a.known_total_tokens,
                           missing_usage_calls=a.missing_usage_calls, latency_s=a.latency_s,
                           config_hashes=result.config_hashes)
    entry = tasks[result.task_id]
    provenance = (("source", f"pilot run {run_dir.name} (development pilot, feasible tasks only)"),
                  ("arm", arm), ("planner", f"{meta.get('provider')}/{meta.get('model')}"))
    oracle = TaskOracle.model_validate(oracles[result.task_id]["oracle"]) if result.task_id in oracles else None
    return ReplayCase(f"{result.task_id}:{arm}", f"{result.task_id} {arm}", TaskSpec.model_validate(entry["task"]),
                      oracle, record, int(entry["seed"]), "feedback pilot run", provenance)


# ---- UI recordings --------------------------------------------------------------------------------------------------

def _recording(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("kind") == BUNDLE_KIND else None


def _recording_cases(folder: Path) -> list[CaseRef]:
    out = []
    for path in sorted(folder.glob("*.json"), reverse=True):  # newest first: names carry a UTC time
        data = _recording(path)
        if data is not None:  # a damaged recording is still listed; loading it explains what is wrong
            out.append(CaseRef(path.name, f"{path.name}  {data.get('label', '')}  "
                                          f"{_dict(data.get('record')).get('outcome', '')}"))
    return out


def _recording_case(path: Path) -> ReplayCase:
    data = _recording(path)
    if data is None:
        raise ReplayError(f"{path.name} is not a UI recording")
    if data.get("schema_version") != 1:
        raise ReplayError(f"{path.name}: unsupported recording schema {data.get('schema_version')!r}")
    seed, label = data.get("seed", 0), data.get("label", path.name)
    if type(seed) is not int or seed < 0 or not isinstance(label, str):
        raise ReplayError(f"{path.name}: its seed must be a non-negative integer and its label text")
    try:
        task = TaskSpec.model_validate(data["task"])
        oracle = TaskOracle.model_validate(data["oracle"]) if data.get("oracle") else None
        record = AttemptRecord.model_validate(data["record"])
    except (KeyError, ValidationError) as exc:
        raise ReplayError(f"{path.name} does not validate ({type(exc).__name__})") from None
    provenance = (("source", f"UI recording {path.name} ({data.get('created_utc', 'unknown time')})"),
                  ("planner", str(data.get("planner", ""))),
                  ("Cosmos gate", f"on ({data['cosmos_host']})" if data.get("cosmos_host") else "off"))
    return ReplayCase(path.name, label, task, oracle, record, seed, "UI recording", provenance)


# ---- entry points ---------------------------------------------------------------------------------------------------

def list_cases(path: str | Path, *, roots: Roots = None) -> tuple[CaseRef, ...]:
    """Every recorded attempt at path. roots: see the module docstring (None: any local path)."""
    path = check_path(path, roots)
    try:
        kind = source_kind(path)
        if kind == "eval":
            cases = _eval_cases(path)
        elif kind == "pilot":
            cases = _pilot_cases(path)
        elif kind == "recordings":
            cases = _recording_cases(path)
        else:
            data = _recording(path)
            if data is None:
                raise ReplayError(f"{path.name} is not a UI recording")
            cases = [CaseRef(path.name, f"{path.name}  {data.get('label', '')}")]
    except DAMAGED as exc:
        raise ReplayError(f"{path} is not in the expected format ({type(exc).__name__}: {exc})"[:300]) from None
    if not cases:
        raise ReplayError(f"{path} holds no recorded attempts")
    return tuple(cases)


def load_case(path: str | Path, case_id: str | None = None, *, seed_dir: str | Path | None = None,
              roots: Roots = None) -> ReplayCase:
    """One case; case_id None is the first listed."""
    path = check_path(path, roots)
    if seed_dir:
        check_path(seed_dir, roots)
    try:
        kind = source_kind(path)
        if kind == "eval":
            return _eval_case(path, case_id, seed_dir, roots)
        if kind == "pilot":
            return _pilot_case(path, case_id, roots)
        if kind == "recordings":
            name = case_id or list_cases(path, roots=roots)[0].case_id
            if Path(name).name != name:
                raise ReplayError(f"{name!r} is not a file name in {path}")
            return _recording_case(check_path(path / name, roots))
        return _recording_case(path)
    except DAMAGED as exc:
        raise ReplayError(f"{path} is not in the expected format ({type(exc).__name__}: {exc})"[:300]) from None
