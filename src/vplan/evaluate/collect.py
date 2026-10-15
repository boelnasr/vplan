"""W3.1 run directories as report input (spec §15 `vplan report build --run ...`).

One run directory per configuration (runner.run_eval). Each attempts.jsonl row (records.EvalAttempt) becomes one
records.AttemptRow: config, task index and id, outcome, error type and wall time from the row; the seed-set name from
its split; the AttemptRecord from its record, without the per-step ExecutionRecord streams (without_streams); the
seed-set hash, provider and model from the run's manifest.json; the TaskSpec and scene counts from the seed set's
tasks[task_index], and the oracle from oracles[task_index].

The seed set is loaded and validated with the W3.1 loader (seedsets.load_seed_set), from --seed-dir or from the
directory the first run recorded, and its hash must match every run manifest. Reporting code is not evidence, so source
drift since preparation is allowed; a configuration change is refused, as everywhere in W3.1.

Refused, with a message that is safe to print (EvalError, exit code 2 in the CLI): runs over different seed sets, two
runs of one configuration, a run directory without a valid manifest, a manifest whose planned or prepared task count
does not fit the seed set, any row that does not validate or does not belong to its run and seed set, and an --output
that is, or lies under, an existing file. A run that recorded fewer rows than it planned, or that covered only an --n prefix,
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
from vplan.records import AttemptRecord, AttemptRow, EvalAttempt, RoundRecord

# The run-manifest fields the loader uses, with their types: a hand-edited manifest is refused here with a message,
# not deeper in with a TypeError.
MANIFEST_KEYS: dict[str, type] = {"config": int, "split": str, "seed_set_hash": str, "provider": str, "model": str,
                                  "n": int, "n_prepared": int}

# Per-step ExecutionRecord series that no metric reads: the 60 Hz replay streams (D41) and the 240 Hz Judge inputs
# (sigma_min per step, the dwell histories), which the Judge has already scored. They are most of an executed record
# (about 0.4 of 0.45 MB as JSON, far more as Python objects), so report rows replace them with empty values; the
# per-execution summaries (events, contacts, joint margins, tracking errors, final poses) stay.
STREAMS: dict[str, Any] = {"q_replay": (), "finger_replay": (), "object_replay": {}, "sigma_min_steps": (),
                           "rest_lin_hist": {}, "rest_ang_hist": {}, "rest_pose_hist": {}}

__all__ = ["Collected", "EvalError", "STREAMS", "build", "collect_runs", "read_run_manifest", "without_streams"]


def without_streams(record: AttemptRecord | None) -> AttemptRecord | None:
    """The record with every ExecutionRecord's per-step streams emptied (STREAMS); a record without an execution is
    returned as it is. Replay needs attempts.jsonl itself, never a report row."""
    def slim(r: RoundRecord) -> RoundRecord:
        if r.evaluation is None or r.evaluation.record is None:
            return r
        return r.model_copy(update={"evaluation": r.evaluation.model_copy(
            update={"record": r.evaluation.record.model_copy(update=STREAMS)})})

    if record is None or all(r.evaluation is None or r.evaluation.record is None for r in record.rounds):
        return record
    return record.model_copy(update={"rounds": tuple(slim(r) for r in record.rounds)})


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
    for key, kind in MANIFEST_KEYS.items():
        value = manifest[key]
        if not isinstance(value, kind) or isinstance(value, bool):
            raise EvalError(f"{path}: {key} must be {'an integer' if kind is int else 'a string'}, got {value!r}")
    return manifest


def _rows(run_dir: Path, manifest: dict[str, Any], seeds: SeedSet) -> list[AttemptRow]:
    path = run_dir / "attempts.jsonl"
    if not path.exists():
        return []  # a run that recorded nothing yet is reported as INCOMPLETE 0/n
    where = f"{run_dir.name}/attempts.jsonl"
    rows, seen = [], set()
    with open(path, "rb") as f:
        # Line by line, and each row drops its streams once validated, so only one full record is in memory at a time.
        for number, line in enumerate(f, 1):
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
                    outcome=attempt.outcome, attempt=without_streams(attempt.record), provider=manifest["provider"],
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
    size = seeds.manifest["n"]
    for d, m in zip(dirs, manifests):  # before any row is read: a row index is checked against n, then indexes tasks
        if not (1 <= m["n"] <= m["n_prepared"] == size):
            raise EvalError(f"{d / 'manifest.json'}: n {m['n']} and n_prepared {m['n_prepared']} do not fit the seed "
                            f"set's {size} tasks (expected 1 <= n <= n_prepared = {size})")
    excluded = seeds.manifest.get("excluded")  # not among the W3.1 loader's required keys
    if not isinstance(excluded, int) or isinstance(excluded, bool) or excluded < 0:
        raise EvalError(f"the seed set in {seed_dir} does not record its excluded candidates as a count: {excluded!r}")
    rows = [row for d, m in zip(dirs, manifests) for row in _rows(d, m, seeds)]
    return Collected(rows, manifests, seeds, excluded, size)


def build(run_dirs: Sequence[str | Path], output: str | Path, seed_dir: str | Path | None = None
          ) -> tuple[ReportFiles, dict[str, Any]]:
    """Collect the runs and write OUTPUT/metrics.json and OUTPUT/report.html; returns the files and the report's
    status (complete, incomplete, limited, degraded, title prefixes)."""
    output = Path(output)
    taken = next((p for p in (output, *output.parents) if p.exists() and not p.is_dir()), None)
    if taken is not None:  # refused before any work, instead of a FileExistsError from mkdir at the end
        raise EvalError(f"--output {output} names a report directory, but {taken} is an existing file")
    got = collect_runs(run_dirs, seed_dir)
    files = build_report(got.rows, Path(output), manifests=got.manifests, undetermined=got.undetermined,
                         seed_set_size=got.seed_set_size)
    return files, json.loads(files.metrics.read_text(encoding="utf-8"))["status"]
