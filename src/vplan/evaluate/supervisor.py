"""R9 supervisor: one worker subprocess per attempt try, a wall-clock timeout per try, and the parent as the only
writer of RUN_DIR/results.jsonl (spec §14, §16). Crashes, kills and hangs become sim_error; provider failures
reported by the worker stay service_failure in the denominator, with the D33 stop rules.

A try that outlives its timeout is killed; it is a worker_timeout unless its result file was already written (the
worker writes it before a job's after step), in which case that result stands.

Layout: RUN_DIR/results.jsonl (parent) and RUN_DIR/workers/NNNN.{params.json, tryT.result.json,
tryT.stdout.log, tryT.stderr.log}. Files are named by batch index because attempt IDs such as "3:dev:7"
contain ':', which Windows does not allow in file names. supervise() is one attempt's try loop; the W3.1
evaluation runner reuses it with its own sink (attempts.jsonl) and resume.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Sequence

from pydantic import ValidationError

import vplan
from vplan.records import SupervisedAttempt, WorkerResult

# Mirror vplan.evaluate.pilot (D33); duplicated so the parent never imports the planner/simulation stack.
FATAL_STATUS = frozenset({401, 402, 403, 404})
MAX_CONSECUTIVE_SERVICE_FAILURES = 3
STDERR_TAIL_CHARS = 4000
_PACKAGE_PARENT = str(Path(vplan.__file__).resolve().parents[1])


class SupervisorError(RuntimeError):
    """A refused batch (bad job list or a reused run directory)."""


@dataclass(frozen=True)
class AttemptJob:
    attempt_id: str
    job: str
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class BatchSummary:
    status: Literal["complete", "incomplete"]
    stop_reason: str | None
    attempts: tuple[SupervisedAttempt, ...]


def child_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Explicit UTF-8 mode (the ManipulaPy banner crashes cp1252 stdio) and the package on PYTHONPATH."""
    env = dict(os.environ if base is None else base)
    env["PYTHONUTF8"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(p for p in (_PACKAGE_PARENT, env.get("PYTHONPATH", "")) if p)
    return env


def worker_command(job: str, attempt_id: str, params_path: str, out_path: str,
                   import_modules: Sequence[str] = ()) -> list[str]:
    cmd = [sys.executable, "-m", "vplan.evaluate.worker", "--job", job, "--attempt-id", attempt_id,
           "--params", str(params_path), "--out", str(out_path)]
    for module in import_modules:
        cmd += ["--import", module]
    return cmd


def read_result(path: Path, attempt_id: str) -> WorkerResult | None:
    """None for a missing, unparsable or foreign result file; the caller classifies the try as a worker fault."""
    try:
        result = WorkerResult.model_validate_json(path.read_bytes())
    except (OSError, ValidationError):
        return None
    return result if result.attempt_id == attempt_id else None


def _tail(path: Path) -> str | None:
    try:
        text = path.read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return None
    return text[-STDERR_TAIL_CHARS:] or None


def _run_try(job: AttemptJob, stem: Path, attempt: int, timeout_s: float, import_modules: Sequence[str],
             cwd: str | None) -> tuple[WorkerResult, int | None]:
    out = stem.with_name(f"{stem.name}.try{attempt}.result.json")
    stdout_log = stem.with_name(f"{stem.name}.try{attempt}.stdout.log")
    stderr_log = stem.with_name(f"{stem.name}.try{attempt}.stderr.log")
    params = stem.with_name(f"{stem.name}.params.json")
    cmd = worker_command(job.job, job.attempt_id, str(params), str(out), import_modules)
    with open(stdout_log, "wb") as so, open(stderr_log, "wb") as se:
        # Files, not pipes: a chatty child can never block on a full pipe buffer, and stdout is never parsed.
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=so, stderr=se, env=child_env(), cwd=cwd)
        try:
            exit_code: int | None = proc.wait(timeout=timeout_s)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            if proc.poll() is None:  # timeout, or the parent itself is being interrupted
                proc.kill()
                proc.wait()
    if timed_out:
        # A result written before the hang is final: the worker writes it before any after step (worker.AFTER), and
        # a hang there or in interpreter teardown must not turn a finished attempt into a sim_error.
        result = read_result(out, job.attempt_id)
        if result is not None:
            return result, None
        return (WorkerResult(attempt_id=job.attempt_id, job=job.job, outcome="sim_error", error_type="worker_timeout",
                             error_message=f"worker killed after the {timeout_s:g} s attempt timeout",
                             error_traceback=_tail(stderr_log)), None)
    result = read_result(out, job.attempt_id)
    if result is not None:  # a valid result outlives a crash during interpreter teardown
        return result, exit_code
    kind = "worker_crash" if exit_code != 0 else "worker_bad_result"
    return (WorkerResult(attempt_id=job.attempt_id, job=job.job, outcome="sim_error", error_type=kind,
                         error_message=f"worker exited with code {exit_code} without a valid result file",
                         error_traceback=_tail(stderr_log)), exit_code)


def _retryable(result: WorkerResult) -> bool:
    return result.outcome == "service_failure" and result.status_code is not None and (
        result.status_code == 429 or result.status_code >= 500)


def supervise(job: AttemptJob, stem: Path, *, timeout_s: float = 600.0, service_retries: int = 0,
              backoff_s: float = 5.0, max_backoff_s: float = 60.0, import_modules: Sequence[str] = (),
              cwd: str | None = None, sleep: Callable[[float], None] = time.sleep) -> SupervisedAttempt:
    """One attempt: write STEM.params.json, then run supervised tries (STEM.tryT.*) until a result that is not
    retried. The caller owns the results sink; stem's directory must exist."""
    stem = Path(stem)
    stem.with_name(f"{stem.name}.params.json").write_text(json.dumps(job.params), encoding="utf-8")
    start = time.monotonic()
    tries = 0
    while True:
        tries += 1
        result, exit_code = _run_try(job, stem, tries, timeout_s, import_modules, cwd)
        if tries > service_retries or not _retryable(result) or result.status_code in FATAL_STATUS:
            break
        delay = backoff_s * 2 ** (tries - 1)
        sleep(min(max_backoff_s, max(delay, result.retry_after_s or 0.0)))
    return SupervisedAttempt(
        attempt_id=job.attempt_id, job=job.job, outcome=result.outcome, tries=tries,
        wall_s=time.monotonic() - start, exit_code=exit_code, status_code=result.status_code,
        payload=result.payload, error_type=result.error_type, error_message=result.error_message,
        error_traceback=result.error_traceback)


def run_batch(jobs: Sequence[AttemptJob], run_dir: Path, *, timeout_s: float = 600.0, service_retries: int = 0,
              backoff_s: float = 5.0, max_backoff_s: float = 60.0, import_modules: Sequence[str] = (),
              cwd: str | None = None, sleep: Callable[[float], None] = time.sleep) -> BatchSummary:
    """Run jobs sequentially, one supervised worker per try. service_retries re-runs a whole attempt after a 429
    or 5xx, which re-spends planner budget, so it defaults to 0 (spec §16 leaves in-call retries to the SDK)."""
    ids = [j.attempt_id for j in jobs]
    if len(set(ids)) != len(ids):
        raise SupervisorError("duplicate attempt IDs in the job list")
    if timeout_s <= 0 or service_retries < 0:
        raise SupervisorError("timeout_s must be positive and service_retries non-negative")
    run_dir = Path(run_dir)
    results_path = run_dir / "results.jsonl"
    if results_path.exists():
        raise SupervisorError(f"{results_path} already exists; refusing to reuse a run directory")
    workers = run_dir / "workers"
    workers.mkdir(parents=True, exist_ok=True)

    done: list[SupervisedAttempt] = []
    stop_reason: str | None = None
    consecutive = 0
    with open(results_path, "x", encoding="utf-8") as sink:
        for index, job in enumerate(jobs):
            record = supervise(job, workers / f"{index:04d}", timeout_s=timeout_s, service_retries=service_retries,
                               backoff_s=backoff_s, max_backoff_s=max_backoff_s, import_modules=import_modules,
                               cwd=cwd, sleep=sleep)
            sink.write(record.model_dump_json() + "\n")
            sink.flush()
            os.fsync(sink.fileno())  # a crashed parent loses at most the attempt in flight
            done.append(record)

            if record.outcome == "service_failure":
                consecutive += 1
                if record.status_code in FATAL_STATUS:
                    stop_reason = f"fatal provider status {record.status_code} at {job.attempt_id}"
                elif consecutive >= MAX_CONSECUTIVE_SERVICE_FAILURES:
                    stop_reason = f"{consecutive} consecutive service failures (likely a quota or rate cap)"
            else:
                consecutive = 0
            if stop_reason is not None:
                break
    return BatchSummary(status="incomplete" if stop_reason else "complete", stop_reason=stop_reason,
                        attempts=tuple(done))
