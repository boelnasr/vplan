"""R9 worker: runs one attempt try in its own process and writes a WorkerResult JSON file.

Usage (argument list, never a shell): python -m vplan.evaluate.worker --job NAME --attempt-id ID
    --params PARAMS.json --out RESULT.json [--import MODULE ...]
stdout/stderr are only logs: ManipulaPy prints a banner on import, so the result travels via --out.
A job is `fn(params: dict) -> dict` with WorkerResult's outcome/status_code/retry_after_s/payload keys;
W2.5's run_task registers here as another job kind.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Callable

from vplan.records import WorkerResult

Job = Callable[[dict], dict]
JOBS: dict[str, Job] = {}
EXIT_UNKNOWN_JOB = 2


def register(name: str) -> Callable[[Job], Job]:
    def deco(fn: Job) -> Job:
        JOBS[name] = fn
        return fn
    return deco


@register("basic_attempt")
def basic_attempt(params: dict) -> dict:
    """Heuristic plan on make_basic_scene, one round (k = 0): a blocked gate is budget_exhausted."""
    from vplan.pipeline import run_attempt
    from vplan.planners.heuristic import heuristic_geometry
    from vplan.tasks.scenes import make_basic_scene

    target = params.get("target_id", "cube_red")
    scene = make_basic_scene(with_obstacle=bool(params.get("obstacle", False)))
    res = run_attempt(heuristic_geometry(scene, target), scene, target, seed=int(params.get("seed", 0)))
    return {"outcome": "executed" if res.outcome == "executed" else "budget_exhausted",
            "payload": {"judge_success": None if res.judge is None else res.judge.success,
                        "findings": [f.model_dump(mode="json") for f in res.findings]}}


def write_result(path: Path, result: WorkerResult) -> None:
    # Replace, never truncate in place: the parent must never read a half-written file.
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(result.model_dump_json(), encoding="utf-8")
    os.replace(tmp, path)


def run(job: str, attempt_id: str, params: dict) -> WorkerResult:
    fn = JOBS.get(job)
    if fn is None:
        return WorkerResult(attempt_id=attempt_id, job=job, outcome="sim_error", error_type="unknown_job",
                            error_message=f"no job registered as {job!r}")
    try:
        return WorkerResult(attempt_id=attempt_id, job=job, **fn(params))
    except Exception as exc:  # spec §16: every exception becomes an outcome with its traceback
        return WorkerResult(attempt_id=attempt_id, job=job, outcome="sim_error", error_type=type(exc).__name__,
                            error_message=str(exc)[:2000], error_traceback=traceback.format_exc())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vplan.evaluate.worker")
    ap.add_argument("--job", required=True)
    ap.add_argument("--attempt-id", required=True)
    ap.add_argument("--params", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--import", dest="imports", action="append", default=[])
    args = ap.parse_args(argv)
    for module in args.imports:
        importlib.import_module(module)  # registers extra jobs (test fault jobs, later run_task)
    params = json.loads(args.params.read_text(encoding="utf-8"))
    result = run(args.job, args.attempt_id, params)
    write_result(args.out, result)
    return EXIT_UNKNOWN_JOB if result.error_type == "unknown_job" else 0


if __name__ == "__main__":
    # Under -m this file is __main__, a second copy; job modules register into vplan.evaluate.worker.JOBS.
    from vplan.evaluate.worker import main as _main
    sys.exit(_main())
