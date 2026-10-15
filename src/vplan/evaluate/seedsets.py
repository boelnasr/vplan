"""Evaluation seed sets (W3.1; spec §6 "Seed sets", D51): offline, proof- or witness-labelled tasks for `eval run`.

Layout, like the pilot cohort (D33): DIR/tasks.jsonl (SeedSetTask, planner-visible), DIR/oracle.jsonl (SeedSetOracle:
feasibility and the proof or witness reference, private), DIR/exclusions.jsonl (SeedSetExclusion: undetermined
candidates) and DIR/manifest.json, written last, so a crashed preparation never loads.

Mix: dev = 75 feasible + 25 impossible (7, 6, 6, 6 for types 1-4), test = 300 + 100 (25 per type); both counts can be
overridden for small sets. Feasible tasks spread as evenly as possible over the 16 (distractors, obstacles) cells, the
remainder going to cells in diagonal order (CELLS), so every distractor count and every obstacle count are also within
one task of each other. Impossible tasks split as evenly as possible over types 1-4, lower types first.

Task order is the candidate-seed order. Feasible and impossible slots interleave evenly, feasible slots cycle through
CELLS and impossible slots through types 1-4, so any prefix (`eval run --n`) is a stratified sample. Candidate seeds
start at SEED_BASE[split] and are consumed one at a time; a candidate that yields neither a witness nor a proof goes
to exclusions.jsonl and the next seed is tried for the same slot.

seed_set_hash = sha256("vplan-seed-set-v1:" + sha256(tasks.jsonl) + ":" + sha256(oracle.jsonl)) names the task
content alone, not when or where it was prepared. Files are written with LF line endings on every OS.
"""
from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple

from pydantic import ValidationError

from vplan.config import config_hash, config_snapshot
from vplan.evaluate.pilot import (PilotError, _git, _manipulapy, _sha256, _utc_now, _write_json, _write_jsonl,
                                  source_digest, source_hashes)
from vplan.records import SeedSetExclusion, SeedSetOracle, SeedSetTask, TaskOracle, TaskSpec, WitnessEvaluation
from vplan.tasks.generate import GenerationRejected, draft_task, generate_task, run_witness, witness_rejection

SCHEMA_VERSION = 1
KIND = "seed_set"
GENERATOR = "tasks.generate.generate_task"  # also makes `pilot run` refuse a seed set (unknown generator)
SPLITS = ("dev", "test")
SEED_BASE = {"dev": 0, "test": 10_000_000}  # test starts in a range no earlier experiment used (D51)
DEFAULT_MIX = {"dev": (75, 25), "test": (300, 100)}  # (feasible, impossible), D51 and spec §6
IMPOSSIBLE_TYPES = (1, 2, 3, 4)
# (distractors, obstacles) in diagonal order: each run of four covers every distractor and every obstacle count once.
CELLS: tuple[tuple[int, int], ...] = tuple((d, (d + k) % 4) for k in range(4) for d in range(4))
FILES = ("tasks.jsonl", "oracle.jsonl", "exclusions.jsonl")
MANIFEST_KEYS = ("schema_version", "kind", "split", "n", "files", "seed_set_hash", "config_hashes", "source_hashes")
SEED_ROOT = Path("runs") / "seeds"  # `eval prepare/run --seeds S` default to SEED_ROOT/S (relative to the repo root)

WitnessEvaluator = Callable[[TaskSpec, str], WitnessEvaluation]


class EvalError(PilotError):
    """A refused seed-set preparation or evaluation run. Like PilotError, the message is safe to print."""


@dataclass(frozen=True)
class Mix:
    feasible: dict[tuple[int, int], int]  # (distractors, obstacles) -> tasks, in CELLS order
    impossible: dict[int, int]  # impossibility type -> tasks

    @property
    def n(self) -> int:
        return sum(self.feasible.values()) + sum(self.impossible.values())


@dataclass(frozen=True)
class Slot:
    """What one task index must hold: a feasible task in one cell, or an impossible task of one type."""
    kind: Literal["feasible", "impossible"]
    distractors: int | None = None
    obstacles: int | None = None
    impossible_type: int | None = None


def default_seed_dir(split: str) -> Path:
    return SEED_ROOT / split


def _spread(total: int, keys: tuple) -> dict:
    base, extra = divmod(total, len(keys))
    return {key: base + (i < extra) for i, key in enumerate(keys)}


def mix_for(split: str, feasible: int | None = None, impossible: int | None = None) -> Mix:
    """The D51 mix for a split, or the same spread over explicit feasible/impossible counts."""
    if split not in SPLITS:
        raise EvalError(f"unknown split {split!r}; expected one of {', '.join(SPLITS)}")
    if (feasible is None) != (impossible is None):
        raise EvalError("pass both the feasible and the impossible count, or neither (the D51 mix)")
    if feasible is None:
        feasible, impossible = DEFAULT_MIX[split]
    for name, value in (("feasible", feasible), ("impossible", impossible)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise EvalError(f"the {name} count must be a non-negative integer")
    if feasible + impossible == 0:
        raise EvalError("a seed set needs at least one task")
    return Mix(feasible=_spread(feasible, CELLS), impossible=_spread(impossible, IMPOSSIBLE_TYPES))


def _cycle(quota: dict) -> Iterator:
    """Keys in order, round after round, skipping keys whose quota is used up."""
    left = dict(quota)
    while any(left.values()):
        for key in left:
            if left[key]:
                left[key] -= 1
                yield key


def slot_order(mix: Mix) -> list[Slot]:
    """Task index -> slot. Impossible slots sit where floor(i * n_impossible / n) steps up (an even interleave)."""
    n_impossible = sum(mix.impossible.values())
    cells, types = _cycle(mix.feasible), _cycle(mix.impossible)
    slots = []
    for i in range(mix.n):
        if (i + 1) * n_impossible // mix.n > i * n_impossible // mix.n:
            slots.append(Slot("impossible", impossible_type=next(types)))
        else:
            d, o = next(cells)
            slots.append(Slot("feasible", distractors=d, obstacles=o))
    return slots


def seed_set_hash(tasks_sha256: str, oracle_sha256: str) -> str:
    """The seed set's identity: its planner-visible and private task content."""
    return hashlib.sha256(f"vplan-seed-set-v1:{tasks_sha256}:{oracle_sha256}".encode("ascii")).hexdigest()


def _candidate(seed: int, slot: Slot, cfgs: dict[str, dict], evaluator: WitnessEvaluator | None
               ) -> tuple[tuple[TaskSpec, TaskOracle] | None, str | None, WitnessEvaluation | None]:
    """(task pair, None, None) for an accepted seed, else (None, reason, witness or None)."""
    seen: list[tuple[WitnessEvaluation, str]] = []

    def witness(task: TaskSpec, target_id: str) -> WitnessEvaluation:
        # The same default witness as generate_task's; wrapped only to keep the rejection evidence.
        result = (run_witness(task, target_id, seed=seed, config_snapshot=cfgs) if evaluator is None
                  else evaluator(task, target_id))
        seen.append((result, target_id))
        return result

    try:
        pair = generate_task(seed, slot.kind, slot.impossible_type, distractors=slot.distractors,
                             obstacles=slot.obstacles, witness_evaluator=witness, config_snapshot=cfgs)
    except Exception as exc:  # one broken candidate must not end a long preparation; it stays as evidence
        return None, f"generation_error: {type(exc).__name__}: {exc}", None
    if pair is not None:
        return pair, None, None
    if seen:
        result, target_id = seen[-1]
        return None, witness_rejection(result, target_id), result
    try:  # generate_task drops the reason; the draft is deterministic and cheap to redo
        draft_task(seed, slot.kind, slot.impossible_type, distractors=slot.distractors, obstacles=slot.obstacles)
    except GenerationRejected as exc:
        return None, exc.reason, None
    return None, "proof_check_failed", None


def prepare_seed_set(output: Path, split: str, *, feasible: int | None = None, impossible: int | None = None,
                     max_candidates: int | None = None, allow_test_split: bool = False,
                     witness_evaluator: WitnessEvaluator | None = None,
                     progress: Callable[[int, int, int], None] | None = None) -> dict:
    """Generate and freeze a seed set (no model calls); returns the manifest.

    max_candidates bounds the candidate seeds tried in total (default 2n + 20). progress receives (attempted,
    accepted, n). witness_evaluator replaces the simulated heuristic witness in offline tests.
    """
    mix = mix_for(split, feasible, impossible)
    if split == "test" and not allow_test_split:
        raise EvalError("refusing to prepare the held-out test split without --allow-test-split: spec §6 generates it "
                        "only after the W3.3 generator freeze")
    n = mix.n
    max_candidates = 2 * n + 20 if max_candidates is None else max_candidates
    if not isinstance(max_candidates, int) or isinstance(max_candidates, bool) or max_candidates < n:
        raise EvalError(f"max candidates must be an integer >= the {n} requested tasks")
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise EvalError(f"output directory {output} is not empty; refusing to overwrite")
    hashes = source_hashes()
    cfgs = config_snapshot()
    cfg_hashes = {key: config_hash(value) for key, value in cfgs.items()}
    base = SEED_BASE[split]
    tasks: list[SeedSetTask] = []
    oracles: list[SeedSetOracle] = []
    exclusions: list[SeedSetExclusion] = []
    attempted = 0
    for index, slot in enumerate(slot_order(mix)):
        while True:
            if attempted >= max_candidates:
                # Keep the rejection log for diagnosis; without a manifest the directory can never be loaded.
                output.mkdir(parents=True, exist_ok=True)
                _write_jsonl(output / "exclusions.jsonl", (e.model_dump_json() for e in exclusions), newline="\n")
                raise EvalError(f"prepared {len(tasks)}/{n} tasks after {attempted} candidates (limit "
                                f"{max_candidates}); {output / 'exclusions.jsonl'} lists the rejected seeds")
            seed = base + attempted
            attempted += 1
            pair, reason, witness = _candidate(seed, slot, cfgs, witness_evaluator)
            if pair is None:
                exclusions.append(SeedSetExclusion(seed=seed, split=split, slot=index, kind=slot.kind,
                                                   distractors=slot.distractors, obstacles=slot.obstacles,
                                                   impossible_type=slot.impossible_type, reason=reason,
                                                   witness=witness))
            else:
                spec, oracle = pair
                task_id = f"{split}_{seed:08d}"
                tasks.append(SeedSetTask(task_id=task_id, index=index, seed=seed, split=split,
                                         distractors=len(spec.scene.objects) - 1,
                                         obstacles=len(spec.scene.obstacles), task=spec))
                oracles.append(SeedSetOracle(task_id=task_id, index=index, oracle=oracle))
            if progress is not None:
                progress(attempted, len(tasks), n)
            if pair is not None:
                break
    if (source_hashes() != hashes
            or {key: config_hash(value) for key, value in config_snapshot().items()} != cfg_hashes):
        raise EvalError("source or config files changed during preparation; discard and prepare again")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise EvalError(f"output directory {output} became non-empty during preparation")
    _write_jsonl(output / "tasks.jsonl", (t.model_dump_json() for t in tasks), newline="\n")
    _write_jsonl(output / "oracle.jsonl", (o.model_dump_json() for o in oracles), newline="\n")
    _write_jsonl(output / "exclusions.jsonl", (e.model_dump_json() for e in exclusions), newline="\n")
    files = {name: _sha256(output / name) for name in FILES}
    manifest = {
        "schema_version": SCHEMA_VERSION, "kind": KIND, "generator": GENERATOR, "created_utc": _utc_now(),
        "split": split, "n": n,
        "requested": {"feasible": sum(mix.feasible.values()), "impossible": sum(mix.impossible.values()),
                      # [distractors][obstacles]
                      "feasible_cells": [[mix.feasible[(d, o)] for o in range(4)] for d in range(4)],
                      "impossible_types": {str(t): count for t, count in mix.impossible.items()}},
        "seeds": {"first": base, "next": base + attempted, "attempted": attempted, "max_candidates": max_candidates},
        "excluded": len(exclusions), "config_hashes": cfg_hashes, "files": files,
        "seed_set_hash": seed_set_hash(files["tasks.jsonl"], files["oracle.jsonl"]),
        "source_hashes": hashes, "source_digest": source_digest(hashes),
        "python": platform.python_version(), "manipulapy": _manipulapy(), "git": _git()}
    _write_json(output / "manifest.json", manifest)
    return manifest


class SeedSet(NamedTuple):
    path: Path
    split: str
    tasks: tuple[SeedSetTask, ...]
    oracles: tuple[TaskOracle, ...]  # oracles[i] belongs to tasks[i]
    manifest: dict
    manifest_sha256: str
    seed_set_hash: str
    config: dict[str, dict]  # the current snapshot, equal to the prepared one
    source_hashes: dict[str, str]  # the current checkout
    drift: list[str]  # source files changed since preparation (allowed by flag)


def _rows(path: Path, model):
    out = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            out.append(model.model_validate_json(line))
        except ValidationError as exc:
            raise EvalError(f"{path.name} line {number} does not validate: {exc.error_count()} errors") from None
    return out


def load_seed_set(path: Path, *, allow_source_drift: bool = False) -> SeedSet:
    """Validate a prepared seed set against the current checkout before any model call."""
    path = Path(path)
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise EvalError(f"{manifest_path} is missing: not a prepared seed set, or preparation did not finish")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except ValueError:
        raise EvalError(f"{manifest_path} is not valid JSON") from None
    if not isinstance(manifest, dict) or manifest.get("kind") != KIND:
        raise EvalError(f"{path} is not an evaluation seed set (manifest kind "
                        f"{manifest.get('kind') if isinstance(manifest, dict) else None!r})")
    missing = [key for key in MANIFEST_KEYS if key not in manifest]
    if missing:
        raise EvalError(f"manifest.json lacks required keys: {', '.join(missing)}")
    if manifest["schema_version"] != SCHEMA_VERSION or manifest["split"] not in SPLITS:
        raise EvalError(f"unsupported seed-set manifest (schema {manifest['schema_version']!r}, "
                        f"split {manifest['split']!r})")
    for name in FILES:
        file = path / name
        if not file.is_file() or _sha256(file) != manifest["files"].get(name):
            raise EvalError(f"{name} does not match manifest.json (tampered or partial file)")
    if seed_set_hash(manifest["files"]["tasks.jsonl"], manifest["files"]["oracle.jsonl"]) != manifest["seed_set_hash"]:
        raise EvalError("manifest.json seed-set hash does not match its files")
    cfgs = config_snapshot()
    changed = sorted(key for key in set(cfgs) | set(manifest["config_hashes"])
                     if key not in cfgs or config_hash(cfgs[key]) != manifest["config_hashes"].get(key))
    if changed:
        raise EvalError(f"configuration changed since preparation: {', '.join(changed)}")
    current, prepared = source_hashes(), manifest["source_hashes"]
    drift = sorted(p for p in set(current) | set(prepared) if current.get(p) != prepared.get(p))
    if drift and not allow_source_drift:
        raise EvalError("source changed since preparation (pass --allow-source-drift to record and proceed): "
                        + ", ".join(drift))
    tasks = _rows(path / "tasks.jsonl", SeedSetTask)
    private = _rows(path / "oracle.jsonl", SeedSetOracle)
    if [t.index for t in tasks] != list(range(manifest["n"])):
        raise EvalError("tasks.jsonl must hold task indices 0..n-1 in order")
    if [(o.index, o.task_id) for o in private] != [(t.index, t.task_id) for t in tasks]:
        raise EvalError("tasks.jsonl and oracle.jsonl disagree on task ids or indices")
    if any(t.split != manifest["split"] for t in tasks):
        raise EvalError("a task's split differs from the manifest's")
    return SeedSet(path, manifest["split"], tuple(tasks), tuple(o.oracle for o in private), manifest,
                   _sha256(manifest_path), manifest["seed_set_hash"], cfgs, current, drift)
