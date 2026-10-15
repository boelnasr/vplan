"""W3.1 run directories as report input (spec §15 `vplan report build --run ...`).

One run directory per configuration (runner.run_eval). Each attempts.jsonl row (records.EvalAttempt) becomes one
records.AttemptRow: config, task index and id, outcome, error type and wall time from the row; the seed-set name from
its split; the AttemptRecord from its record; the seed-set hash, provider and model from the run's manifest.json; the
TaskSpec and scene counts from the seed set's tasks[task_index], and the oracle from oracles[task_index].

The seed set is loaded and validated with the W3.1 loader (seedsets.load_seed_set), from --seed-dir or from the
directory the first run recorded, and its hash must match every run manifest. Reporting code is not evidence, so source
drift since preparation is allowed; a configuration change is refused, as everywhere in W3.1.

Refused, with a message that is safe to print (EvalError, exit code 2 in the CLI): runs over different seed sets, two
runs of one configuration, a run directory without a valid manifest, and any row that does not validate or does not
belong to its run and seed set. A run that recorded fewer rows than it planned, or that covered only an --n prefix,
is reported with its INCOMPLETE or LIMITED label (D33), never refused.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

from pydantic import ValidationError

from vplan.evaluate import runner
from vplan.evaluate.report import ReportFiles, build_report
from vplan.evaluate.seedsets import EvalError, SeedSet, load_seed_set
from vplan.records import AttemptRow, EvalAttempt

MANIFEST_KEYS = ("config", "split", "seed_set_hash", "provider", "model", "n", "n_prepared")

__all__ = ["Collected", "EvalError", "build", "collect_runs", "read_run_manifest"]


class Collected(NamedTuple):
    rows: list[AttemptRow]
    manifests: list[dict[str, Any]]  # as read; metrics.public_manifest drops their local paths
    seeds: SeedSet
    undetermined: int  # the seed-set manifest's excluded candidates (spec §6)
    seed_set_size: int  # the seed-set manifest's task count


def read_run_manifest(run_dir: Path) -> dict[str, Any]:
    path = Path(run_dir) / "manifest.json"
    if not path.is_file():
        raise EvalError(f"{run_dir} has no manifest.json: not an evaluation run directory")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        raise EvalError(f"{path} is not valid JSON") from None
    if not isinstance(manifest, dict) or manifest.get("kind") != runner.KIND:
        raise EvalError(f"{run_dir} is not an evaluation run (no eval_run manifest.json)")
    if manifest.get("schema_version") != runner.SCHEMA_VERSION:
        raise EvalError(f"{path}: unsupported run manifest schema {manifest.get('schema_version')!r}")
    missing = [key for key in MANIFEST_KEYS if key not in manifest]
    if missing:
        raise EvalError(f"{path} lacks required keys: {', '.join(missing)}")
    return manifest


def _rows(run_dir: Path, manifest: dict[str, Any], seeds: SeedSet) -> list[AttemptRow]:
    path = run_dir / "attempts.jsonl"
    if not path.exists():
        return []  # a run that recorded nothing yet is reported as INCOMPLETE 0/n
    where = f"{run_dir.name}/attempts.jsonl"
    rows, seen = [], set()
    with open(path, "rb") as f:
        for number, line in enumerate(f, 1):  # streamed: a test-split file holds hundreds of MB of records
            try:
                attempt = EvalAttempt.model_validate_json(line)
            except ValidationError as exc:
                raise EvalError(f"{where} line {number} does not validate ({exc.error_count()} errors); a partial "
                                "final line from a killed run is dropped by eval run --resume") from None
            if (attempt.config, attempt.split) != (manifest["config"], manifest["split"]):
                raise EvalError(f"{where} line {number} is a config {attempt.config} {attempt.split} attempt, but the "
                                f"run's manifest says config {manifest['config']} {manifest['split']}")
            if attempt.task_index >= manifest["n"]:
                raise EvalError(f"{where} line {number} holds {attempt.attempt_id}, which is not part of the run's "
                                f"plan ({manifest['n']} tasks)")
            if attempt.attempt_id in seen:
                raise EvalError(f"{where} records {attempt.attempt_id} twice (duplicate)")
            seen.add(attempt.attempt_id)
            task = seeds.tasks[attempt.task_index]
            if attempt.task_id != task.task_id:
                raise EvalError(f"{where} line {number}: task id {attempt.task_id!r} differs from the seed set's "
                                f"{task.task_id!r} at index {attempt.task_index}")
            try:
                rows.append(AttemptRow(
                    config=attempt.config, seed_set=attempt.split, seed_set_hash=manifest["seed_set_hash"],
                    task_index=attempt.task_index, task_id=attempt.task_id, task=task.task,
                    oracle=seeds.oracles[attempt.task_index], distractors=task.distractors, obstacles=task.obstacles,
                    outcome=attempt.outcome, attempt=attempt.record, provider=manifest["provider"],
                    model=manifest["model"], error_type=attempt.error_type, wall_s=attempt.wall_s))
            except ValidationError as exc:
                first = exc.errors()[0]["msg"]
                raise EvalError(f"{where} line {number} does not join its seed-set task and run "
                                f"({exc.error_count()} errors; first: {first})") from None
    return rows


def collect_runs(run_dirs: Sequence[str | Path], seed_dir: str | Path | None = None) -> Collected:
    """Validate W3.1 run directories against each other and their seed set, and join every attempt with its task."""
    if not run_dirs:
        raise EvalError("pass at least one run directory")
    dirs = [Path(d) for d in run_dirs]
    manifests = [read_run_manifest(d) for d in dirs]
    hashes = {m["seed_set_hash"] for m in manifests}
    if len(hashes) > 1:
        raise EvalError("the runs were made over different seed sets: " + ", ".join(
            f"{d} ({str(m['seed_set_hash'])[:12]})" for d, m in zip(dirs, manifests)))
    by_config: dict[Any, Path] = {}
    for d, m in zip(dirs, manifests):
        if m["config"] in by_config:
            raise EvalError(f"two runs of config {m['config']}: {by_config[m['config']]} and {d}; a report takes one "
                            "run per configuration")
        by_config[m["config"]] = d
    if seed_dir is None:
        seed_dir = (manifests[0].get("seed_set") or {}).get("dir")
        if not seed_dir:
            raise EvalError(f"{dirs[0]} does not record its seed-set directory; pass --seed-dir")
    seeds = load_seed_set(Path(seed_dir), allow_source_drift=True)
    (run_hash,) = hashes
    if seeds.seed_set_hash != run_hash:
        raise EvalError(f"the seed set in {seed_dir} has seed-set hash {seeds.seed_set_hash[:12]}, but the runs were "
                        f"made on {str(run_hash)[:12]}")
    rows = [row for d, m in zip(dirs, manifests) for row in _rows(d, m, seeds)]
    return Collected(rows, manifests, seeds, seeds.manifest["excluded"], seeds.manifest["n"])


def build(run_dirs: Sequence[str | Path], output: str | Path, seed_dir: str | Path | None = None
          ) -> tuple[ReportFiles, dict[str, Any]]:
    """Collect the runs and write OUTPUT/metrics.json and OUTPUT/report.html; returns the files and the report's
    status (complete, incomplete, limited, title prefixes)."""
    got = collect_runs(run_dirs, seed_dir)
    files = build_report(got.rows, Path(output), manifests=got.manifests, undetermined=got.undetermined,
                         seed_set_size=got.seed_set_size)
    return files, json.loads(files.metrics.read_text(encoding="utf-8"))["status"]
