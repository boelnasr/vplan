"""W3.1 evaluation runner (spec §14 "Runner" and "Run directory"; D33, D40, D42, D48): `vplan eval run`.

Each attempt is repair.run_task on one seed-set task, in its own supervised worker subprocess (the "run_task" job);
attempts run one after another, and the parent is the only writer of attempts.jsonl.

Run directory RUN_DIR (default runs/c<config>-<split>-<UTC time>):
- manifest.json: written once when the run is created; the run's identity and first-session provenance.
- sessions.jsonl: a "start" line (that session's provenance) and an "end" line (status) per session.
- attempts.jsonl: one EvalAttempt per finished attempt (see records.EvalAttempt for the row shape).
- workers/: per attempt STEM.params.json, STEM.tryT.{result.json, stdout.log, stderr.log} and the pacing file.
  STEM is the task index (NNNN) in session 0 and NNNN.sS in resumed session S, so a resumed attempt never shares
  files with a killed session's attempt (or its orphaned worker). Attempt IDs contain ':', so they never name files.
- frames/: per attempt frames/STEM/ with the visual-gate key frames of its executed rounds and index.json
  (evaluate.frames), the input of the config-4 shadow study (spec §11, §14). Config 3 renders them by default;
  --frames/--no-frames overrides that for any config, and the manifest records the choice. The worker renders after
  the attempt's result is final, and each row's frames field (records.FrameExport) records the export or why it
  failed; a failed export never changes the attempt.

Configs (spec §14, D40, D48): 1 Nemotron unverified k=0; 3 Nemotron repair k=3 (config 2 is its round 1); 5 heuristic
repair k=3, no LLM (its later rounds end as repeat_plan, D39); 6 a frontier model, unverified k=0, named with --model
and paid (--allow-paid); 7 Nemotron resample k=3. LLM configs read LLM_BASE_URL and LLM_MODEL from --env-file in the
parent (for the free-model guard and the manifest); only workers read LLM_API_KEY, from the same file, so the key is
never in params, results, logs, the manifest or attempts.jsonl.

Resume (--resume RUN_DIR) discards a partial final line, skips recorded attempt IDs and refuses a run whose config,
mode, k, planner, seed-set hash, config hashes, task count or source (unless --allow-source-drift) differ.

Git provenance (spec §14): a session starts only when the checkout's git SHA is known (pilot._git: the git command,
then the repository files, then VPLAN_GIT_SHA), unless --allow-no-git, which the manifest and session line record.

Stop rules (D33, D42): a 401/402/403/404 stops the run at once; three consecutive failed provider requests stop it.
Requests are counted from each AttemptRecord's rounds, in order, as D33 counts them: run_task ends an attempt at its
first service failure, and a request that reached the provider resets the count. A worker fault without a record
resets it too (D42). A change to any source or config file between attempts also stops the run, because each worker
imports the current checkout.

Rate limit (spec §14): --max-rpm spaces provider request starts across the sequential workers (RequestPacer); it is off
by default because at the R4 median of 34 s per call a sequential run stays under 2 requests per minute.
"""
from __future__ import annotations

import importlib.metadata
import json
import math
import os
import platform
import subprocess
import threading
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, NamedTuple

from pydantic import ValidationError

from vplan.config import config_hash, config_snapshot
from vplan.evaluate.pilot import (PilotError, _git, _manipulapy, _utc_now, _versions, _write_json, llm_settings,
                                  provider_guard, read_env_file, source_digest, source_hashes)
from vplan.evaluate.seedsets import SPLITS, EvalError, SeedSet, default_seed_dir, load_seed_set
from vplan.evaluate.supervisor import FATAL_STATUS, MAX_CONSECUTIVE_SERVICE_FAILURES, AttemptJob, supervise
from vplan.evaluate.worker import AFTER
from vplan.planners.heuristic import HeuristicPlanner
from vplan.planners.nemotron import CALL_DEADLINE_S
from vplan.records import (AttemptMode, AttemptRecord, EvalAttempt, FrameExport, SupervisedAttempt, TaskOracle,
                           TaskSpec)

SCHEMA_VERSION = 1
KIND = "eval_run"
JOB = "run_task"
DEFAULT_TIMEOUT_S = 600.0  # k + 1 = 4 calls at the 60 s call deadline plus simulation stay well inside it (D42, D63)
SIMULATION_ALLOWANCE_S = DEFAULT_TIMEOUT_S - 4 * CALL_DEADLINE_S  # what the default leaves beyond the calls (D65)
DEFAULT_TEMPERATURE = 0.7  # as in the pilot (D33): resampling only means something if samples differ
RUNS_ROOT = Path("runs")  # relative to the working directory, which is the repo root for every command


@dataclass(frozen=True)
class EvalConfig:
    number: int
    planner: Literal["nemotron", "heuristic", "frontier"]
    mode: AttemptMode
    k: int


CONFIGS: dict[int, EvalConfig] = {
    1: EvalConfig(1, "nemotron", "unverified", 0),  # ablation without the gate; one blind attempt (D48)
    3: EvalConfig(3, "nemotron", "repair", 3),  # headline "after"; its round 1 is config 2
    5: EvalConfig(5, "heuristic", "repair", 3),  # baseline without an LLM
    6: EvalConfig(6, "frontier", "unverified", 0),  # optional paid reference, excluded from the headline
    7: EvalConfig(7, "nemotron", "resample", 3),  # equal-budget resampling
}
NOT_RUNS = {2: "config 2 is round 1 of config 3 (spec §14); run --config 3",
            4: "config 4 is the Cosmos shadow study over config 3's frames (spec §14), not a runner config"}


def eval_config(number: int) -> EvalConfig:
    if number in NOT_RUNS:
        raise EvalError(NOT_RUNS[number])
    if number not in CONFIGS:
        raise EvalError(f"unknown config {number}; runnable configs are {', '.join(map(str, CONFIGS))}")
    return CONFIGS[number]


@dataclass(frozen=True)
class PlannerSettings:
    """Who plans: recorded in the manifest and sent to workers. It never holds the API key."""
    kind: Literal["heuristic", "llm"]
    provider: str
    model: str
    temperature: float
    base_url: str | None = None
    base_url_host: str | None = None
    env_file: str | None = None
    call_deadline_s: float = CALL_DEADLINE_S  # wall-clock limit per planner call (D63); other values dev only (D65)

    def describe(self) -> dict:
        """The manifest form, compared on resume."""
        out = {"kind": self.kind, "provider": self.provider, "model": self.model, "temperature": self.temperature,
               "base_url_host": self.base_url_host}
        if self.kind == "llm":
            out["call_deadline_s"] = self.call_deadline_s
        return out

    def params(self) -> dict:
        """The worker form: the env-file path to read the key from, never the key."""
        return {**self.describe(), "base_url": self.base_url, "env_file": self.env_file}


def planner_settings(config: EvalConfig, *, env_file: str | Path | None = None, model: str | None = None,
                     temperature: float | None = None, allow_paid: bool = False,
                     call_deadline_s: float | None = None) -> PlannerSettings:
    """Resolve and guard the planner before anything runs (D33 free-model guard)."""
    if config.planner == "heuristic":
        given = [flag for flag, value in (("--env-file", env_file), ("--model", model), ("--temperature", temperature),
                                          ("--call-deadline", call_deadline_s)) if value is not None]
        given += ["--allow-paid"] if allow_paid else []
        if given:
            raise EvalError(f"config {config.number} runs the heuristic planner without an LLM; drop "
                            + ", ".join(given))
        return PlannerSettings("heuristic", HeuristicPlanner.provider, HeuristicPlanner.model,
                               HeuristicPlanner.temperature)
    if env_file is None:
        raise EvalError(f"config {config.number} calls an LLM: pass --env-file naming LLM_BASE_URL, LLM_API_KEY "
                        "and LLM_MODEL")
    if config.planner == "frontier":
        if not model:
            raise EvalError("config 6 (frontier reference) needs the model named explicitly with --model")
        if not allow_paid:
            raise EvalError("config 6 calls a paid frontier model: pass --allow-paid to approve the spend")
    try:
        base_url, _, model = llm_settings(Path(env_file), model)  # the key is only checked for presence here
        host, provider = provider_guard(base_url, model, allow_paid=allow_paid)
    except PilotError as exc:
        raise EvalError(str(exc)) from None
    temperature = DEFAULT_TEMPERATURE if temperature is None else float(temperature)
    if not (math.isfinite(temperature) and temperature >= 0):
        raise EvalError("--temperature must be a finite number >= 0")
    deadline = CALL_DEADLINE_S if call_deadline_s is None else call_deadline_s
    if type(deadline) not in (int, float) or not 0 < deadline <= threading.TIMEOUT_MAX:
        raise EvalError("--call-deadline must be a positive number of seconds")
    return PlannerSettings("llm", provider, model, temperature, base_url, host, str(Path(env_file).resolve()),
                           float(deadline))


def default_timeout(spec: EvalConfig, settings: PlannerSettings) -> float:
    """The attempt limit when --timeout is not given: (k + 1) calls at the call deadline plus the simulation
    allowance, never below the 600 s default (D42, D65)."""
    return max(DEFAULT_TIMEOUT_S, (spec.k + 1) * settings.call_deadline_s + SIMULATION_ALLOWANCE_S)


# ---- worker side ------------------------------------------------------------------------------------------
class RequestPacer:
    """Minimum spacing between provider request starts, shared by a run's sequential workers (spec §14's
    process-shared rate limiter, only as elaborate as sequential runs need): the last start time (Unix s) lives in a
    small JSON file, and one worker runs at a time, so there is no lock. A wait never exceeds one interval, so a
    clock jump cannot stall a run."""

    def __init__(self, path: Path, min_interval_s: float, *, clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep):
        self.path, self.min_interval_s, self.clock, self.sleep = Path(path), float(min_interval_s), clock, sleep

    def _last(self) -> float | None:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))["last_request_start_unix_s"]
        except (OSError, ValueError, KeyError, TypeError):
            return None
        return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None

    def wait(self) -> float:
        """Sleep until one interval after the previous request start; returns the seconds slept."""
        last = self._last()
        delay = 0.0 if last is None else min(self.min_interval_s, max(0.0, last + self.min_interval_s - self.clock()))
        if delay > 0:
            self.sleep(delay)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps({"last_request_start_unix_s": self.clock()}), encoding="utf-8")
        os.replace(tmp, self.path)
        return delay


class PacedPlanner:
    """A planner whose every request waits for the pacer first; everything else passes through (model, _secret)."""

    def __init__(self, planner, pacer: RequestPacer):
        self._planner, self._pacer = planner, pacer

    def plan(self, task, **kwargs):
        self._pacer.wait()
        return self._planner.plan(task, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._planner, name)


def build_planner(params: dict, pacing: dict | None = None):
    """The worker's planner and the secret to scrub from error text. The key is read from the env file here only."""
    if params["kind"] == "heuristic":
        return HeuristicPlanner(), None
    from vplan.planners.nemotron import NemotronPlanner
    key = read_env_file(Path(params["env_file"])).get("LLM_API_KEY", "")
    if not key:
        raise EvalError(f"{params['env_file']} does not set LLM_API_KEY")
    planner = NemotronPlanner(model=params["model"], provider=params["provider"], base_url=params["base_url"],
                              api_key=key, temperature=params["temperature"],
                              call_deadline_s=params.get("call_deadline_s", CALL_DEADLINE_S))
    if pacing is not None:
        planner = PacedPlanner(planner, RequestPacer(Path(pacing["path"]), pacing["min_interval_s"]))
    return planner, key


def run_task_job(params: dict) -> dict:
    """Worker side of one attempt (registered as the "run_task" job in vplan.evaluate.worker). With params["frames"]
    ({"dir", "rel"}), the key-frame export (evaluate.frames) is the job's after step: the worker runs it only once the
    attempt's result is written, and it reads the finished record without changing it."""
    from vplan.repair import _simulation_error, run_task
    secret = None
    try:
        spec = TaskSpec.model_validate(params["task"])
        oracle = TaskOracle.model_validate(params["oracle"])
        planner, secret = build_planner(params["planner"], params.get("pacing"))
        record = run_task(spec, oracle, planner, k=params["k"], mode=params["mode"],
                          config_snapshot=params["config_snapshot"], attempt_id=params["attempt_id"],
                          seed=params["seed"])
    except Exception as exc:  # anything escaping run_task is a sim_error, scrubbed of the key like any evidence
        error = _simulation_error(exc, "sim_error", secret)
        return {"outcome": "sim_error", "error_type": error.error_type, "error_message": error.error_message,
                "error_traceback": error.error_traceback}
    failed = next((r.call.status_code for r in record.rounds if r.call.outcome == "service_failure"), None)
    out = {"outcome": record.outcome, "status_code": failed, "payload": {"record": record.model_dump(mode="json")}}
    frames = params.get("frames")
    if frames is not None:
        def export() -> dict:
            # Timed from before the import: loading the rendering stack (about 0.4 s in a fresh worker) is part of
            # the export, which report rows leave out of the attempt's wall time (collect.report_wall_s).
            start = time.perf_counter()
            from vplan.evaluate.frames import export_frames
            result = export_frames(record, spec, seed=params["seed"], config_snapshot=params["config_snapshot"],
                                   out_dir=Path(frames["dir"]), rel_dir=frames["rel"])
            return {"frames": result.model_copy(update={"render_s": time.perf_counter() - start}).model_dump(
                mode="json")}
        out[AFTER] = export
    return out


# ---- parent side ------------------------------------------------------------------------------------------
class RunSummary(NamedTuple):
    run_dir: Path
    status: Literal["complete", "incomplete"]
    stop_reason: str | None
    planned: int
    recorded: int  # attempts.jsonl rows, all sessions
    session: int
    outcomes: dict[str, int]  # all sessions
    successes: int  # AttemptRecord.success, all sessions


def _manipulapy_provenance() -> dict:
    info = _manipulapy()
    git = None
    if info["location"]:
        init = Path(info["location"])
        try:  # only a checkout that tracks the package: a venv inside another repository must not borrow its SHA
            tracked = subprocess.run(["git", "ls-files", "--", init.name], cwd=init.parent, capture_output=True,
                                     text=True, timeout=10, check=True).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            tracked = ""
        git = _git(init.parent) if tracked else None
    return {**info, "git": git}


def _provenance() -> dict:
    versions = dict(_versions())
    for name in ("numpy", "pydantic"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"git": _git(), "python": platform.python_version(), "os": platform.platform(), "versions": versions,
            "manipulapy": _manipulapy_provenance(), "image_digest": os.environ.get("VPLAN_IMAGE_DIGEST") or None}


def _drift(current: dict[str, str], recorded: dict[str, str]) -> list[str]:
    return sorted(p for p in set(current) | set(recorded) if current.get(p) != recorded.get(p))


def _append(path: Path, line: str) -> None:
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def _discard_partial_line(path: Path) -> int:
    """Truncate a final line cut off by a kill mid-write (no trailing newline); returns the bytes dropped."""
    if not path.exists():
        return 0
    with open(path, "r+b") as f:
        size = f.seek(0, os.SEEK_END)
        if size == 0:
            return 0
        f.seek(size - 1)
        if f.read(1) == b"\n":
            return 0
        keep, end = 0, size
        while end > 0:  # scan back for the last newline without reading a large file into memory
            start = max(0, end - 65536)
            f.seek(start)
            i = f.read(end - start).rfind(b"\n")
            if i >= 0:
                keep = start + i + 1
                break
            end = start
        f.truncate(keep)
    return size - keep


class _Recorded(NamedTuple):
    ids: set[str]
    outcomes: Counter
    successes: int


def _read_attempts(path: Path, planned: set[str]) -> _Recorded:
    ids, outcomes, successes = set(), Counter(), 0
    if not path.exists():
        return _Recorded(ids, outcomes, successes)
    with open(path, "rb") as f:
        for number, line in enumerate(f, 1):  # streamed: a test-split file holds hundreds of MB of records
            try:
                row = EvalAttempt.model_validate_json(line)
            except ValidationError:
                raise EvalError(f"attempts.jsonl line {number} is corrupt; refusing to resume") from None
            if row.attempt_id in ids:
                raise EvalError(f"attempts.jsonl records {row.attempt_id} twice (duplicate); refusing to resume")
            if row.attempt_id not in planned:
                raise EvalError(f"attempts.jsonl holds {row.attempt_id}, which is not part of this run's plan")
            ids.add(row.attempt_id)
            outcomes[row.outcome] += 1
            successes += row.record is not None and row.record.success
    return _Recorded(ids, outcomes, successes)


def _sessions_started(path: Path) -> int:
    count = 0
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                count += json.loads(line).get("event") == "start"
            except (ValueError, AttributeError):
                raise EvalError(f"{path} is corrupt; refusing to resume") from None
    return count


def _frames(supervised: SupervisedAttempt, requested: bool) -> FrameExport | None:
    """The row's key-frame export: None when the run renders no frames or the worker returned no record; failed when
    the worker returned a record but no valid export (it ended or raised in its after step)."""
    payload = supervised.payload
    if not requested or payload.get("record") is None:
        return None
    raw = payload.get("frames")
    if raw is None:
        error = payload.get("after_error") if isinstance(payload.get("after_error"), dict) else {}
        return FrameExport(status="failed", error_type=str(error.get("error_type") or "frames_not_reported"),
                           error_message=str(error.get("error_message") or "the worker ended before reporting its "
                                             "frames (a crash or timeout while rendering)")[:500])
    try:
        return FrameExport.model_validate(raw)
    except ValidationError as exc:
        return FrameExport(status="failed", error_type="invalid_frame_export",
                           error_message=f"the worker's frame export does not validate ({exc.error_count()} errors)")


def _row(supervised: SupervisedAttempt, config: EvalConfig, split: str, index: int, task_id: str, session: int,
         config_hashes: dict[str, str], frames: bool = False) -> EvalAttempt:
    """The attempts.jsonl row; a record that does not belong to this attempt and run becomes sim_error, and its frame
    export failed (record_rejected)."""
    common = dict(attempt_id=supervised.attempt_id, config=config.number, split=split, task_index=index,
                  task_id=task_id, session=session, tries=supervised.tries, wall_s=supervised.wall_s,
                  exit_code=supervised.exit_code, status_code=supervised.status_code,
                  frames=_frames(supervised, frames))
    raw = supervised.payload.get("record")
    if raw is None:
        return EvalAttempt(**common, outcome=supervised.outcome, error_type=supervised.error_type,
                           error_message=supervised.error_message, error_traceback=supervised.error_traceback)
    try:
        record = AttemptRecord.model_validate(raw)
    except ValidationError as exc:
        problem = f"the worker's AttemptRecord does not validate ({exc.error_count()} errors)"
    else:
        problem = (
            "the record belongs to another attempt or outcome"
            if (record.attempt_id, record.outcome) != (supervised.attempt_id, supervised.outcome) else
            f"mode/k {record.mode}/{record.k} differ from config {config.number}"
            if (record.mode, record.k) != (config.mode, config.k) else
            "config hashes differ from the run's snapshot" if record.config_hashes != config_hashes else None)
    if problem is not None:
        exported = common["frames"]
        if exported is not None and exported.status != "failed":  # frames of a rejected record are not evidence
            common["frames"] = FrameExport(status="failed", bytes=exported.bytes, render_s=exported.render_s,
                                           error_type="record_rejected", error_message="the attempt's record was "
                                           "rejected (invalid_attempt_record), so its frames are not evidence")
        return EvalAttempt(**common, outcome="sim_error", error_type="invalid_attempt_record", error_message=problem)
    return EvalAttempt(**common, outcome=record.outcome, record=record)


class _ProviderWatch:
    """The D33 stop rules over provider requests, in request order."""

    def __init__(self) -> None:
        self.consecutive = 0

    def update(self, row: EvalAttempt) -> str | None:
        calls = ([(r.call.outcome, r.call.status_code) for r in row.record.rounds] if row.record is not None
                 else [(row.outcome, row.status_code)])  # no record: a worker fault (resets, D42) or a bare failure
        for outcome, status in calls:
            if outcome != "service_failure":
                self.consecutive = 0
                continue
            self.consecutive += 1
            if status in FATAL_STATUS:
                return f"fatal provider status {status} at {row.attempt_id} (auth, billing or endpoint path)"
        if self.consecutive >= MAX_CONSECUTIVE_SERVICE_FAILURES:
            return f"{self.consecutive} consecutive service-failure requests (likely a quota or rate cap)"
        return None


def _incompatible(what: str, recorded, now) -> EvalError:
    return EvalError(f"incompatible run: {what} is {recorded!r} in its manifest.json but {now!r} now")


class _Session(NamedTuple):
    run_dir: Path
    manifest: dict
    seeds: SeedSet
    n: int  # planned attempts: task indices 0..n-1
    hashes: dict[str, str]  # source and config files at session start
    drift: list[str]  # vs the seed set (new run) or the run manifest (resume), allowed by flag
    number: int  # 0 for a new run
    previous: _Recorded  # rows of earlier sessions
    discarded: int  # bytes of a partial final attempts.jsonl line
    frames: bool  # workers render key frames (evaluate.frames)


def _create(spec: EvalConfig, split: str, settings: PlannerSettings, *, seed_dir: str | Path | None,
            run_dir: str | Path | None, n: int | None, allow_source_drift: bool, timeout_s: float,
            max_rpm: float | None, provenance: dict, allow_no_git: bool, frames: bool | None) -> _Session:
    seed_dir = Path(seed_dir) if seed_dir is not None else default_seed_dir(split)
    seeds = load_seed_set(seed_dir, allow_source_drift=allow_source_drift)
    if seeds.split != split:
        raise EvalError(f"{seed_dir} holds the {seeds.split} split, not {split}")
    total = len(seeds.tasks)
    n = total if n is None else n
    if not 1 <= n <= total:
        raise EvalError(f"--n must be between 1 and the seed set's {total} tasks")
    hashes = source_hashes()
    drift = _drift(hashes, seeds.manifest["source_hashes"])
    if drift and not allow_source_drift:
        raise EvalError("source changed since the seed set was prepared (pass --allow-source-drift to record and "
                        "proceed): " + ", ".join(drift))
    run_dir = (Path(run_dir) if run_dir is not None else
               RUNS_ROOT / f"c{spec.number}-{split}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    if run_dir.exists() and (not run_dir.is_dir() or any(run_dir.iterdir())):
        raise EvalError(f"run directory {run_dir} is not empty; refusing to overwrite evidence (--resume continues "
                        "a run)")
    run_dir.mkdir(parents=True, exist_ok=True)
    frames = spec.number == 3 if frames is None else frames  # config 3's frames feed the config-4 shadow study
    manifest = {
        "schema_version": SCHEMA_VERSION, "kind": KIND, "run_id": run_dir.name, "created_utc": _utc_now(),
        "config": spec.number, "mode": spec.mode, "k": spec.k, "planner": settings.describe(),
        "provider": settings.provider, "model": settings.model, "temperature": settings.temperature,
        "split": split, "seed_set_hash": seeds.seed_set_hash,
        "seed_set": {"dir": str(seed_dir), "resolved": str(seed_dir.resolve()),
                     "manifest_sha256": seeds.manifest_sha256, "n_tasks": total},
        "n": n, "n_prepared": total, "limited": n < total,
        "config_hashes": {key: config_hash(value) for key, value in seeds.config.items()},
        "source_hashes": hashes, "source_digest": source_digest(hashes), "source_drift": drift,
        "timeout_s": timeout_s, "max_rpm": max_rpm, "allow_no_git": allow_no_git, "frames": frames, **provenance}
    _write_json(run_dir / "manifest.json", manifest)
    return _Session(run_dir, manifest, seeds, n, hashes, drift, 0, _Recorded(set(), Counter(), 0), 0, frames)


def _resume(spec: EvalConfig, split: str, settings: PlannerSettings, run_dir: str | Path, *,
            seed_dir: str | Path | None, n: int | None, allow_source_drift: bool, frames: bool | None) -> _Session:
    run_dir = Path(run_dir)
    try:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        manifest = None
    if not isinstance(manifest, dict) or manifest.get("kind") != KIND:
        raise EvalError(f"{run_dir} is not an evaluation run (no eval_run manifest.json)")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise EvalError(f"unsupported run manifest schema {manifest.get('schema_version')!r}")
    for what, now in (("config", spec.number), ("split", split), ("mode", spec.mode), ("k", spec.k)):
        if manifest.get(what) != now:
            raise _incompatible(what, manifest.get(what), now)
    planner = manifest.get("planner") or {}
    if settings.kind == "llm":  # runs from before D65 recorded no deadline and ran at the spec's
        planner = {"call_deadline_s": CALL_DEADLINE_S, **planner}
    for key, now in settings.describe().items():
        if planner.get(key) != now:
            raise _incompatible(f"planner {key}", planner.get(key), now)
    if not isinstance(manifest.get("n"), int) or (n is not None and n != manifest["n"]):
        raise _incompatible("--n (the planned task count)", manifest.get("n"), n)
    recorded_frames = manifest.get("frames", False)  # runs from before frames rendered none
    if frames is not None and frames != recorded_frames:  # a shadow study needs every attempt's frames, or none
        raise _incompatible("frames (--frames/--no-frames)", recorded_frames, frames)
    seed_dir = Path(seed_dir) if seed_dir is not None else Path((manifest.get("seed_set") or {}).get("dir", ""))
    # Drift against the seed set was accepted when the run was created; only drift since then matters now.
    seeds = load_seed_set(seed_dir, allow_source_drift=True)
    if seeds.seed_set_hash != manifest.get("seed_set_hash"):
        raise _incompatible("the seed-set hash", manifest.get("seed_set_hash"), seeds.seed_set_hash)
    current_cfg = {key: config_hash(value) for key, value in config_snapshot().items()}
    if current_cfg != manifest.get("config_hashes"):
        raise EvalError("incompatible run: the configuration changed since the run started: "
                        + ", ".join(_drift(current_cfg, manifest.get("config_hashes") or {})))
    hashes = source_hashes()
    drift = _drift(hashes, manifest.get("source_hashes") or {})
    if drift and not allow_source_drift:
        raise EvalError("incompatible run: source changed since the run started (pass --allow-source-drift to "
                        "record and proceed): " + ", ".join(drift))
    discarded = _discard_partial_line(run_dir / "attempts.jsonl")
    _discard_partial_line(run_dir / "sessions.jsonl")
    n = manifest["n"]
    previous = _read_attempts(run_dir / "attempts.jsonl", {f"{spec.number}:{split}:{i}" for i in range(n)})
    return _Session(run_dir, manifest, seeds, n, hashes, drift, _sessions_started(run_dir / "sessions.jsonl"),
                    previous, discarded, bool(recorded_frames))


def run_eval(config: int, split: str, *, seed_dir: str | Path | None = None, run_dir: str | Path | None = None,
             resume: str | Path | None = None, n: int | None = None, env_file: str | Path | None = None,
             model: str | None = None, temperature: float | None = None, allow_paid: bool = False,
             allow_source_drift: bool = False, timeout_s: float | None = None, max_rpm: float | None = None,
             allow_no_git: bool = False, frames: bool | None = None, call_deadline_s: float | None = None,
             import_modules: Sequence[str] = (),
             supervise_attempt: Callable[..., SupervisedAttempt] = supervise,
             echo: Callable[[str], None] = lambda line: print(line, flush=True)) -> RunSummary:
    """Run (or with resume, continue) one config over a prepared seed set. Everything is validated before the first
    worker starts. seed_dir defaults to runs/seeds/<split>, or on resume to the run's own seed set. Without a git SHA
    for the checkout the run is refused unless allow_no_git. frames (None: config 3 only; on resume, the run's own
    setting) makes workers render the key frames of executed rounds. call_deadline_s (default the spec's 60 s) bounds
    each planner call; another value is for dev runs only, and timeout_s then defaults to default_timeout (D65)."""
    spec = eval_config(config)
    if split not in SPLITS:
        raise EvalError(f"unknown split {split!r}; expected one of {', '.join(SPLITS)}")
    if timeout_s is not None and not (math.isfinite(timeout_s) and timeout_s > 0):
        raise EvalError("--timeout must be a positive number of seconds")
    if max_rpm is not None and not (math.isfinite(max_rpm) and max_rpm > 0):
        raise EvalError("--max-rpm must be a positive number")
    if resume is not None and run_dir is not None:
        raise EvalError("pass a new run directory or --resume RUN_DIR, not both")
    if frames is not None and not isinstance(frames, bool):
        raise EvalError("frames must be True, False or None (the config's default)")
    settings = planner_settings(spec, env_file=env_file, model=model, temperature=temperature,
                                allow_paid=allow_paid, call_deadline_s=call_deadline_s)
    if split == "test" and settings.call_deadline_s != CALL_DEADLINE_S:
        raise EvalError(f"a call deadline other than spec §10's {CALL_DEADLINE_S:g} s is for dev runs only (D65)")
    if timeout_s is None:
        timeout_s = default_timeout(spec, settings)
    provenance = _provenance()
    if provenance["git"] is None and not allow_no_git:
        raise EvalError("no git SHA for this checkout: the git command failed, HEAD could not be read from the "
                        "repository files, and VPLAN_GIT_SHA is unset or not a SHA; pass --allow-no-git to run "
                        "without one (the manifest records it)")
    if resume is None:
        state = _create(spec, split, settings, seed_dir=seed_dir, run_dir=run_dir, n=n,
                        allow_source_drift=allow_source_drift, timeout_s=timeout_s, max_rpm=max_rpm,
                        provenance=provenance, allow_no_git=allow_no_git, frames=frames)
    else:
        state = _resume(spec, split, settings, resume, seed_dir=seed_dir, n=n, allow_source_drift=allow_source_drift,
                        frames=frames)
    run_dir, manifest, seeds, n, hashes, session, previous = (state.run_dir, state.manifest, state.seeds, state.n,
                                                              state.hashes, state.number, state.previous)

    pending = [i for i in range(n) if f"{spec.number}:{split}:{i}" not in previous.ids]
    workers = run_dir / "workers"
    workers.mkdir(exist_ok=True)
    pacing = None if max_rpm is None else {"path": str((workers / "pacing.json").resolve()),
                                           "min_interval_s": 60.0 / max_rpm}
    cfg_hashes = manifest["config_hashes"]
    base = {"config_snapshot": seeds.config, "mode": spec.mode, "k": spec.k, "planner": settings.params(),
            "pacing": pacing}
    _append(run_dir / "sessions.jsonl", json.dumps({
        "event": "start", "session": session, "started_utc": _utc_now(), "resumed": resume is not None,
        "pending": len(pending), "discarded_partial_bytes": state.discarded, "source_digest": source_digest(hashes),
        "source_drift": state.drift, "timeout_s": timeout_s, "max_rpm": max_rpm, "allow_no_git": allow_no_git,
        "frames": state.frames, **provenance}))
    echo(f"run {run_dir}: config {spec.number} ({spec.mode}, k={spec.k}, {settings.provider}/{settings.model}) on "
         f"{split} seed set {manifest['seed_set_hash'][:12]}; {len(pending)} of {n} attempts to run "
         f"(session {session}; resume with --resume {run_dir})")

    outcomes, successes, recorded = Counter(previous.outcomes), previous.successes, len(previous.ids)
    watch, stop_reason = _ProviderWatch(), None
    try:
        with open(run_dir / "attempts.jsonl", "a", encoding="utf-8", newline="\n") as sink:
            for index in pending:
                changed = _drift(source_hashes(), hashes)
                if changed:  # workers import the checkout: evidence must not mix snapshots (D33)
                    stop_reason = "source changed during the run: " + ", ".join(changed)
                    break
                task = seeds.tasks[index]
                attempt_id = f"{spec.number}:{split}:{index}"
                stem = workers / (f"{index:04d}" if session == 0 else f"{index:04d}.s{session}")
                frame_dir = None if not state.frames else {"dir": str((run_dir / "frames" / stem.name).resolve()),
                                                           "rel": f"frames/{stem.name}"}
                job = AttemptJob(attempt_id, JOB, {
                    **base, "attempt_id": attempt_id, "seed": task.seed, "task": task.task.model_dump(mode="json"),
                    "oracle": seeds.oracles[index].model_dump(mode="json"), "frames": frame_dir})
                supervised = supervise_attempt(job, stem, timeout_s=timeout_s, import_modules=import_modules)
                row = _row(supervised, spec, split, index, task.task_id, session, cfg_hashes, state.frames)
                sink.write(row.model_dump_json() + "\n")
                sink.flush()
                os.fsync(sink.fileno())  # a killed parent loses at most the attempt in flight
                recorded += 1
                outcomes[row.outcome] += 1
                successes += row.record is not None and row.record.success
                echo(f"{attempt_id} {row.outcome}{' success' if row.record is not None and row.record.success else ''}"
                     f" ({row.wall_s:.1f} s{'' if row.frames is None else '; frames ' + row.frames.status})")
                stop_reason = watch.update(row)
                if stop_reason is not None:
                    break
    except BaseException as exc:  # includes KeyboardInterrupt: the session log must say it did not finish
        _append(run_dir / "sessions.jsonl", json.dumps({
            "event": "end", "session": session, "finished_utc": _utc_now(), "status": "interrupted",
            "error_type": type(exc).__name__, "recorded": recorded, "planned": n}))
        raise
    status = "complete" if recorded == n else "incomplete"
    stop_reason = None if status == "complete" else stop_reason
    _append(run_dir / "sessions.jsonl", json.dumps({
        "event": "end", "session": session, "finished_utc": _utc_now(), "status": status, "stop_reason": stop_reason,
        "recorded": recorded, "planned": n, "outcomes": dict(outcomes), "successes": successes}))
    return RunSummary(run_dir, status, stop_reason, n, recorded, session, dict(outcomes), successes)
