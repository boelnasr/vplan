"""Config-4 Cosmos shadow study (spec §11 "Shadow study", §14; D8 item 4, D59): `vplan eval shadow`.

Cosmos never influenced the run it annotates, so nothing here measures incremental success (D23 item 7); like the live
gate it is a learned visual check, never verification. Every executed round of a W3.1 run with key frames
(evaluate.frames) gets the live gate's stage questions (visual.cosmos with the D47 protocol: thinking on, the gate
views first and every view after `uncertain`), asked about three kinds of frames:
- natural: the run's own frames of the executed plan;
- wrong_object (injected): the same plan retargeted at another object, chosen at random (seeded by the task seed and
  round) among the objects that are neither the oracle target nor the plan's target, rebuilt and posed as the gate
  poses it (visual.gate);
- misaligned_grasp (injected): the grasp key pose slid 3-4 cm along the TCP x axis, normal to the finger plane, as the
  D34 pilot's offset grasps, so the fingers straddle the table beside the object.
Truth is the spec §11 state predicate for the seed set's oracle target, read back from the posed simulator state at
shadow time (D59). An injected pose that misses its IK target, or whose predicate holds, is recorded as skipped: it is
not a negative. Cosmos hears the object only as the instruction's '<color> <shape>'; an attempt whose instruction names
no single object is not asked about, as the gate skips it.

Output directory OUT (default runs/shadow-<run directory name>), never inside the run directory, which is only read:
- manifest.json: the run, seed set and Cosmos settings; a rerun with other settings is refused.
- shadow.jsonl: one records.ShadowRecord per question, appended as each is answered.
- injected/<frames stem>/r<round>-<stage>-<source>-<view>.png: the injected renders.
- summary.json and summary.md: summarize() over shadow.jsonl, rewritten at the end of every session.
Rerun the same command to continue: answered and skipped questions are kept; endpoint errors and a partial final line
are dropped and asked again. Three endpoint errors in a row stop the session, as three failed requests stop a run (D33).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pybullet as p
from pydantic import ValidationError

from vplan.evaluate.collect import read_run_manifest
from vplan.evaluate.frames import FrameMismatch, rebuild_trajectory
from vplan.evaluate.pilot import _git, _utc_now, _write_json
from vplan.evaluate.runner import RUNS_ROOT
from vplan.evaluate.seedsets import EvalError, load_seed_set
from vplan.geometry import pose_to_T, rot_angle
from vplan.records import (CandidateTrajectory, EvalAttempt, FrameIndex, PickPlacePlan, Scene, SeedSetTask,
                           ShadowRecord, TaskOracle, VisualCheck)
from vplan.sim.scene import SceneHandles, build_scene
from vplan.visual.cosmos import DEFAULT_MODEL, CosmosUnavailable, ask_cosmos, score, subject_question
from vplan.visual.gate import SEGMENT, STAGES, TIMEOUT_S, _pose, _tcp, _with_client
from vplan.visual.predicates import grasp_truth, pregrasp_truth, prerelease_truth
from vplan.visual.render import render_views, to_png

KIND, SCHEMA_VERSION = "shadow_study", 1
SHADOW, SUMMARY_JSON, SUMMARY_MD, INJECTED = "shadow.jsonl", "summary.json", "summary.md", "injected"
ENABLE_THINKING = True  # the live gate's default (D47), the better D34 mode
MAX_CONSECUTIVE_ERRORS = 3
OFFSET_M = (0.03, 0.04)  # misaligned grasp: the D34 pilot's offset range along TCP x, m
IK_ITERATIONS = 500  # as the D34 pilot's single-pose IK
CLASSES = ("predicate_true", "natural_negative", "injected_negative", "wrong_object", "misaligned_grasp")
CLASS_LABELS = {"predicate_true": "predicate true (natural)", "natural_negative": "natural negatives",
                "injected_negative": "injected negatives, all", "wrong_object": "injected: wrong object",
                "misaligned_grasp": "injected: misaligned grasp"}
VERDICTS = ("pass", "fail", "uncertain")
NOTE = ("A learned visual check, not verification. Cosmos never influenced the run, so this measures what a visual "
        "gate would have caught or falsely rejected, never incremental success (spec §11, D8 item 4).")
READING = ("For predicate-true frames a fail is a false rejection and a pass a correct acceptance. For negatives a "
           "fail is a true rejection (a catch) and a pass a catch the gate missed. In the live gate a final uncertain "
           "also stops execution (visual_uncertain).")


class ShadowResult(NamedTuple):
    status: str  # complete: every question has an answer or a skip reason; incomplete: rerun to continue
    stop_reason: str | None
    asked: int  # questions sent to Cosmos in this session
    summary: dict[str, Any]
    out_dir: Path


class _Question(NamedTuple):
    fields: dict[str, Any]  # ShadowRecord fields without the result
    images: dict[str, bytes]  # view -> PNG, in retry-view order
    skipped: str | None


def default_output(run_dir: str | Path) -> Path:
    return RUNS_ROOT / f"shadow-{Path(run_dir).name}"


def question_id(attempt_id: str, round_index: int, stage: str, source: str) -> str:
    return f"{attempt_id}/r{round_index}/{stage}/{source}"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---- truth and injected poses -------------------------------------------------------------------------------
def _truth(h: SceneHandles, scene: Scene, stage: str, target_id: str | None, carried: str | None, cfg: dict) -> bool:
    """The spec §11 state predicate for the oracle target in h's current state; carried is the object the fingers hold
    at pre-release."""
    if target_id is None:
        return False  # no object matches the instruction (spec §6 type 4), so no frame can show it
    poses = {o.id: pose_to_T(*p.getBasePositionAndOrientation(h.bodies[o.id], physicsClientId=h.client))
             for o in scene.objects}
    sizes = {o.id: o.size for o in scene.objects}
    T_tcp = _tcp(h)
    if stage == "pregrasp":
        return pregrasp_truth(T_tcp, {k: tuple(T[:3, 3]) for k, T in poses.items()}, target_id)
    if stage == "grasp":
        return grasp_truth(T_tcp, poses[target_id], sizes[target_id], cfg)
    return prerelease_truth(carried, poses.get(carried), sizes.get(carried), target_id, scene.bin)


def _goal(traj: CandidateTrajectory, stage: str) -> np.ndarray:
    """The key pose (TCP, world frame) the stage's key frame was solved for: its segment's last IK target."""
    return np.reshape(traj.segments[SEGMENT[stage]].ik_targets[-1], (4, 4))


def _look(h: SceneHandles, scene: Scene, stage: str, target_id: str | None, carried: str | None, cfg: dict,
          goal: np.ndarray | None = None, views: tuple[str, ...] = ()) -> tuple[bool, bool, dict[str, bytes]]:
    """(truth, reached, PNGs) of h's current state. reached: the simulator TCP is within the Verifier's IK tolerances
    of goal (V-2), so the frame shows the pose it was meant to."""
    T = _tcp(h)
    reached = goal is None or bool(np.linalg.norm(T[:3, 3] - goal[:3, 3]) <= cfg["ik_pos_tol_m"] and
                                   np.degrees(rot_angle(T[:3, :3], goal[:3, :3])) <= cfg["ik_rot_tol_deg"])
    pngs = {k: to_png(v) for k, v in render_views(h, views).items()} if views else {}
    return _truth(h, scene, stage, target_id, carried, cfg), reached, pngs


def _misaligned(traj: CandidateTrajectory, scene: Scene, target_id: str | None, cfg: dict, rng: np.random.Generator,
                seed: int, views: tuple[str, ...]) -> tuple[bool, bool, dict[str, bytes]]:
    from vplan.robot import load_panda
    goal = _goal(traj, "grasp").copy()
    goal[:3, 3] += goal[:3, 0] * rng.choice((-1.0, 1.0)) * rng.uniform(*OFFSET_M)
    q_grasp = np.asarray(traj.segments[SEGMENT["grasp"]].q[-1])
    np.random.seed(seed % 2 ** 32)  # ManipulaPy IK perturbations use the global RNG
    q, _ = load_panda().ik(goal, q_grasp, max_iterations=IK_ITERATIONS)
    q = q if np.all(np.isfinite(q)) else q_grasp  # posed unchanged, so it misses the goal and is skipped
    return _with_client(lambda c: _look(build_scene(scene, c, q), scene, "grasp", target_id, None, cfg, goal, views))


# ---- reading the run ------------------------------------------------------------------------------------------
def _attempts(run_dir: Path) -> Iterator[EvalAttempt]:
    """attempts.jsonl rows with written key frames, one at a time (an executed record is about 0.5 MB)."""
    path = run_dir / "attempts.jsonl"
    if not path.is_file():
        raise EvalError(f"{run_dir} has no attempts.jsonl")
    with open(path, "rb") as f:
        for number, line in enumerate(f, 1):
            try:
                attempt = EvalAttempt.model_validate_json(line)
            except ValidationError as exc:
                raise EvalError(f"{run_dir.name}/attempts.jsonl line {number} does not validate "
                                f"({exc.error_count()} errors)") from None
            if attempt.record is not None and attempt.frames is not None and attempt.frames.status == "written":
                yield attempt


def _read_frames(run_dir: Path, attempt: EvalAttempt) -> tuple[FrameIndex, str, dict[str, bytes]]:
    """The attempt's frame index, its run-relative folder and every PNG it names, each checked against its sha256."""
    rel = Path(attempt.frames.index).parent.as_posix()
    try:
        data = (run_dir / attempt.frames.index).read_bytes()
        if _sha256(data) != attempt.frames.index_sha256:
            raise EvalError(f"{attempt.frames.index} does not match its attempts.jsonl row (sha256)")
        index = FrameIndex.model_validate_json(data)
        if index.attempt_id != attempt.attempt_id:
            raise EvalError(f"{attempt.frames.index} belongs to {index.attempt_id}, not {attempt.attempt_id}")
        pngs = {}
        for entry in index.frames:
            pngs[entry.file] = (run_dir / rel / entry.file).read_bytes()
            if _sha256(pngs[entry.file]) != entry.sha256:
                raise EvalError(f"{rel}/{entry.file} does not match its index (sha256)")
    except OSError as exc:
        raise EvalError(f"{attempt.attempt_id}: cannot read its frames ({exc.strerror}: {exc.filename})") from None
    except ValidationError:
        raise EvalError(f"{attempt.frames.index} does not validate") from None
    return index, rel, pngs


def _questions(attempt: EvalAttempt, index: FrameIndex, rel: str, pngs: dict[str, bytes], task: SeedSetTask,
               oracle: TaskOracle, cfgs: dict[str, dict], out_dir: Path, done: set[str]) -> Iterator[_Question]:
    """The attempt's questions in order: per executed round the natural stages, the wrong-object stages, then the
    misaligned grasp. A round whose questions are all done is not rebuilt."""
    scene, target, cfg, views = task.task.scene, oracle.target_id, cfgs["verifier"], index.retry_views
    entries = {(e.round_index, e.stage, e.view): e for e in index.frames}
    stem = Path(rel).name
    for rnd in attempt.record.rounds:
        if rnd.evaluation is None or rnd.evaluation.outcome != "executed":
            continue
        plan, executed, r = rnd.call.plan, rnd.evaluation.record, rnd.round_index
        if not isinstance(plan, PickPlacePlan) or executed is None:
            raise EvalError(f"{attempt.attempt_id} round {r} was executed without a plan or an execution record")
        others = sorted(o.id for o in scene.objects if o.id not in (target, plan.target_id))
        planned = [question_id(attempt.attempt_id, r, s, "natural") for s in STAGES]
        planned += [question_id(attempt.attempt_id, r, s, "wrong_object") for s in STAGES] if others else []
        planned += [question_id(attempt.attempt_id, r, "grasp", "misaligned_grasp")]
        if done.issuperset(planned):
            continue
        if any((r, s, v) not in entries or entries[r, s, v].approval != executed.approval
               for s in STAGES for v in views):
            raise EvalError(f"{attempt.frames.index} lacks frames of round {r}'s executed trajectory")
        try:
            traj = rebuild_trajectory(plan, executed, scene, seed=task.seed, config_snapshot=cfgs, round_index=r)
        except FrameMismatch as exc:
            raise EvalError(f"{attempt.attempt_id}: {exc}") from None

        def question(stage, source, posed, frames, images, truth, skipped=None):
            return _Question({"question_id": question_id(attempt.attempt_id, r, stage, source),
                              "attempt_id": attempt.attempt_id, "task_index": attempt.task_index, "round_index": r,
                              "stage": stage, "source": source, "subject": index.subject, "posed": posed,
                              "frames": frames, "truth": truth}, images, skipped)

        def injected(stage, source, posed, look, grasp_reached=True):
            truth, reached, images = look
            frames = tuple(f"{INJECTED}/{stem}/r{r}-{stage}-{source}-{v}.png" for v in views)
            for name, data in zip(frames, images.values()):
                (out_dir / name).parent.mkdir(parents=True, exist_ok=True)
                (out_dir / name).write_bytes(data)
            skipped = ("the pose missed its IK target" if not (reached and grasp_reached)
                       else "the state predicate holds" if truth else None)
            return question(stage, source, posed, frames, images, truth, skipped)

        for stage in STAGES:
            truth = _with_client(lambda c: _look(_pose(c, plan, traj, scene, stage), scene, stage, target,
                                                 plan.target_id, cfg)[0])
            files = [entries[r, stage, v].file for v in views]
            yield question(stage, "natural", plan.target_id, tuple(f"{rel}/{f}" for f in files),
                           {v: pngs[f] for v, f in zip(views, files)}, truth)
        rng = np.random.default_rng([task.seed, r])
        if others:
            from vplan.motion.trajectory import build_trajectory
            from vplan.robot import HOME_Q
            posed = str(rng.choice(others))
            wrong = plan.model_copy(update={"target_id": posed})
            wtraj = build_trajectory(wrong, scene, HOME_Q, seed=task.seed % 2 ** 32, cfg=cfg)
            looks = {stage: _with_client(lambda c: _look(_pose(c, wrong, wtraj, scene, stage), scene, stage, target,
                                                         posed, cfg, _goal(wtraj, stage), views)) for stage in STAGES}
            for stage in STAGES:
                # at pre-release the object hangs at its grasp-frame offset, so a missed grasp pose misplaces it too
                yield injected(stage, "wrong_object", posed, looks[stage], stage != "prerelease" or looks["grasp"][1])
        yield injected("grasp", "misaligned_grasp", plan.target_id,
                       _misaligned(traj, scene, target, cfg, rng, task.seed, views))


# ---- asking -----------------------------------------------------------------------------------------------------
def _ask(q: _Question, gate_views: tuple[str, ...], retry_views: tuple[str, ...], ask: Callable, base_url: str,
         model: str, timeout_s: float) -> ShadowRecord:
    """One question with the gate's protocol: the gate views, and every view once more after `uncertain`."""
    stage = q.fields["stage"]
    text = subject_question(stage, q.fields["subject"])
    checks: list[VisualCheck] = []
    for views in (gate_views, retry_views):
        try:
            v = ask({k: q.images[k] for k in views}, text, base_url, model=model, enable_thinking=ENABLE_THINKING,
                    timeout=timeout_s)
        except CosmosUnavailable as exc:
            return ShadowRecord(**q.fields, checks=tuple(checks), error=str(exc)[:500])
        checks.append(VisualCheck(stage=stage, views=views, verdict=v))
        if v.verdict != "uncertain":
            break
    return ShadowRecord(**q.fields, checks=tuple(checks), verdict=v.verdict)


def _kept(path: Path) -> list[ShadowRecord]:
    """The records a rerun keeps (answered or skipped), in file order. The file is rewritten without endpoint errors
    and without a partial final line (a session killed mid-write), so each question keeps one line."""
    if not path.is_file():
        return []
    raw = path.read_bytes()
    complete, _, partial = raw.rpartition(b"\n")
    kept, seen = [], set()
    for number, line in enumerate(complete.split(b"\n"), 1):
        if not line.strip():
            continue
        try:
            rec = ShadowRecord.model_validate_json(line)
        except ValidationError:
            raise EvalError(f"{path} line {number} does not validate") from None
        if rec.question_id in seen:
            raise EvalError(f"{path} records {rec.question_id} twice")
        seen.add(rec.question_id)
        if rec.error is None:
            kept.append(rec)
    if partial or len(kept) != len(seen):
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text("".join(r.model_dump_json() + "\n" for r in kept), encoding="utf-8", newline="\n")
        os.replace(tmp, path)
    return kept


def _records(path: Path) -> list[ShadowRecord]:
    return [ShadowRecord.model_validate_json(line) for line in path.read_bytes().splitlines() if line.strip()]


def _open(out_dir: Path, identity: dict[str, Any], extra: dict[str, Any]) -> None:
    path = out_dir / "manifest.json"
    if path.is_file():
        recorded = json.loads(path.read_text(encoding="utf-8"))
        changed = [f"{k} {recorded.get(k)!r}, now {v!r}" for k, v in identity.items() if recorded.get(k) != v]
        if changed:
            raise EvalError(f"{out_dir} holds a shadow study with other settings ({'; '.join(changed)}); pass another "
                            "--output")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(path, {**identity, **extra})


def run_shadow(run_dir: str | Path, out_dir: str | Path, *, base_url: str, seed_dir: str | Path | None = None,
               model: str = DEFAULT_MODEL, timeout_s: float = TIMEOUT_S, limit: int | None = None,
               ask: Callable = ask_cosmos, echo: Callable[[str], None] = print) -> ShadowResult:
    """Ask Cosmos about every executed round of run_dir (a W3.1 run with frames) and write OUT (see the module
    docstring). Refusals raise EvalError before anything is written. limit caps the questions asked in this session.
    ask is ask_cosmos's seam for tests."""
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    if out_dir.resolve().is_relative_to(run_dir.resolve()):
        raise EvalError(f"--output {out_dir} is inside the run directory {run_dir}, which the shadow study only reads")
    if limit is not None and limit < 1:
        raise EvalError("--limit must be at least 1")
    if not (math.isfinite(timeout_s) and timeout_s > 0):
        raise EvalError("--timeout must be a positive number of seconds")
    manifest = read_run_manifest(run_dir)
    if not manifest.get("frames"):
        raise EvalError(f"{run_dir} rendered no key frames (its manifest says frames: false); the shadow study needs "
                        "a run made with frames, as config 3 is by default")
    if seed_dir is None:
        seed_dir = (manifest.get("seed_set") or {}).get("dir")
        if not seed_dir:
            raise EvalError(f"{run_dir} does not record its seed-set directory; pass --seed-dir")
    seeds = load_seed_set(Path(seed_dir), allow_source_drift=True)
    if seeds.seed_set_hash != manifest["seed_set_hash"]:
        raise EvalError(f"the seed set in {seed_dir} has seed-set hash {seeds.seed_set_hash[:12]}, but the run was "
                        f"made on {manifest['seed_set_hash'][:12]}")
    identity = {"kind": KIND, "schema_version": SCHEMA_VERSION, "run_id": manifest.get("run_id", run_dir.name),
                "config": manifest["config"], "split": manifest["split"], "seed_set_hash": manifest["seed_set_hash"],
                "model": model, "enable_thinking": ENABLE_THINKING}
    _open(out_dir, identity, {"created_utc": _utc_now(), "base_url": base_url, "timeout_s": timeout_s,
                              "git": _git()})
    done = {r.question_id for r in _kept(out_dir / SHADOW)}
    asked, in_row, stop = 0, 0, None
    with open(out_dir / SHADOW, "a", encoding="utf-8", newline="\n") as f:
        for attempt in _attempts(run_dir):
            index, rel, pngs = _read_frames(run_dir, attempt)
            if index.subject is None:
                echo(f"{attempt.attempt_id}: skipped, the instruction names no single object")
                continue
            for q in _questions(attempt, index, rel, pngs, seeds.tasks[attempt.task_index],
                                seeds.oracles[attempt.task_index], seeds.config, out_dir, done):
                if q.fields["question_id"] in done:
                    continue
                if q.skipped is not None:
                    rec = ShadowRecord(**q.fields, skipped=q.skipped)
                elif limit is not None and asked >= limit:
                    stop = f"--limit {limit} reached"
                    break
                else:
                    rec = _ask(q, index.gate_views, index.retry_views, ask, base_url, model, timeout_s)
                    asked += 1
                f.write(rec.model_dump_json() + "\n")
                f.flush()
                result = rec.verdict or ("ERROR " + rec.error if rec.error else "skipped: " + rec.skipped)
                echo(f"{rec.question_id} (truth {rec.truth}): {result} "
                     f"({sum(c.verdict.latency_s for c in rec.checks):.1f} s)")
                in_row = in_row + 1 if rec.error else 0 if rec.verdict else in_row
                if in_row >= MAX_CONSECUTIVE_ERRORS:
                    stop = f"{in_row} endpoint errors in a row at {base_url} (last: {rec.error})"
                    break
            if stop:
                break
    records = _records(out_dir / SHADOW)
    errors = sum(r.error is not None for r in records)
    if stop is None and errors:
        stop = f"{errors} questions ended in an endpoint error"
    status = "complete" if stop is None else "incomplete"
    summary = {**identity, "status": status, "stop_reason": stop, **summarize(records)}
    _write_json(out_dir / SUMMARY_JSON, summary)
    (out_dir / SUMMARY_MD).write_text(summary_markdown(summary), encoding="utf-8", newline="\n")
    return ShadowResult(status, stop, asked, json.loads((out_dir / SUMMARY_JSON).read_text(encoding="utf-8")), out_dir)


# ---- the summary ------------------------------------------------------------------------------------------------
def _classes(r: ShadowRecord) -> tuple[str, ...]:
    if r.source == "natural":
        return ("predicate_true",) if r.truth else ("natural_negative",)
    return ("injected_negative", r.source) if not r.truth else ()  # a true injected frame is skipped, never asked


def _latency(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "median": None, "p90": None, "max": None}
    return {"n": len(values), "median": float(np.median(values)), "p90": float(np.percentile(values, 90)),
            "max": float(max(values))}


def _cell(rs: Sequence[ShadowRecord]) -> dict[str, Any]:
    answered = [r for r in rs if r.verdict is not None]
    n, c = len(answered), Counter(r.verdict for r in answered)
    return {"n": n, **{v: c[v] for v in VERDICTS}, "errors": sum(r.error is not None for r in rs),
            **{f"{v}_rate": c[v] / n if n else None for v in VERDICTS},
            "latency_s": _latency([ch.verdict.latency_s for r in answered for ch in r.checks])}


def _balanced(pos: Sequence[ShadowRecord], neg: Sequence[ShadowRecord]) -> float | None:
    """D34's pilot measure: pass <=> predicate true, a final uncertain counts as wrong in both classes."""
    if not pos or not neg:
        return None
    return score([True] * len(pos) + [False] * len(neg), [r.verdict for r in [*pos, *neg]])["balanced_accuracy"]


def _gate(records: Sequence[ShadowRecord]) -> dict[str, Any]:
    """The live gate replayed on the natural frames of every round with all three stages answered: it stops at the
    first stage that does not pass. A round is predicate true when all three of its predicates hold."""
    out = {k: {"n": 0, **{v: 0 for v in VERDICTS}, "stopped_at": {s: 0 for s in STAGES}}
           for k in ("predicate_true", "predicate_false")}
    rounds: dict[tuple[str, int], dict[str, ShadowRecord]] = {}
    for r in records:
        if r.source == "natural" and r.verdict is not None:
            rounds.setdefault((r.attempt_id, r.round_index), {})[r.stage] = r
    for stages in rounds.values():
        if len(stages) != len(STAGES):
            continue
        cell = out["predicate_true" if all(r.truth for r in stages.values()) else "predicate_false"]
        cell["n"] += 1
        first = next((stages[s] for s in STAGES if stages[s].verdict != "pass"), None)
        if first is None:
            cell["pass"] += 1
        else:
            cell[first.verdict] += 1
            cell["stopped_at"][first.stage] += 1
    return out


def summarize(records: Sequence[ShadowRecord]) -> dict[str, Any]:
    """Per class and stage: answered n, verdict counts and rates, endpoint errors and request latency; balanced
    accuracy against natural and injected negatives; the gate replayed on natural frames; overall latency."""
    by: dict[str, list[ShadowRecord]] = {k: [] for k in CLASSES}
    for r in records:
        for k in _classes(r):
            by[k].append(r)
    stages = (*STAGES, "all")

    def at(rs, stage):
        return [r for r in rs if stage in ("all", r.stage)]

    answered = [r for r in records if r.verdict is not None]
    positives = [r for r in by["predicate_true"] if r.verdict is not None]
    return {
        "note": NOTE, "reading": READING,
        "questions": {"recorded": len(records), "answered": len(answered),
                      "errors": sum(r.error is not None for r in records),
                      "skipped": dict(Counter(r.skipped for r in records if r.skipped is not None))},
        "attempts": len({r.attempt_id for r in records}),
        "classes": {k: {s: _cell(at(by[k], s)) for s in stages} for k in CLASSES},
        "balanced_accuracy": {
            name: {s: _balanced(at(positives, s), at([r for r in by[neg] if r.verdict is not None], s))
                   for s in stages}
            for name, neg in (("natural", "natural_negative"), ("injected", "injected_negative"))},
        "gate": _gate(records),
        "latency_s": {"request": _latency([c.verdict.latency_s for r in answered for c in r.checks]),
                      "question": _latency([sum(c.verdict.latency_s for c in r.checks) for r in answered])},
    }


def _pct(rate: float | None) -> str:
    return "—" if rate is None else f"{100 * rate:.1f}%"


def _s(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def summary_markdown(s: dict[str, Any]) -> str:
    q, lat = s["questions"], s["latency_s"]
    status = "complete" if s["status"] == "complete" else f"INCOMPLETE ({s['stop_reason']}; rerun to continue)"
    skipped = ", ".join(f"{n} {why}" for why, n in sorted(q["skipped"].items())) or "none"
    lines = [f"# Cosmos shadow study of {s['run_id']}: a learned visual check, not verification", "", s["note"], "",
             f"- Status: {status}. Model `{s['model']}`, thinking {'on' if s['enable_thinking'] else 'off'}; config "
             f"{s['config']} on the {s['split']} seed set `{s['seed_set_hash'][:12]}`; {s['attempts']} attempts.",
             f"- Questions: {q['recorded']} recorded, {q['answered']} answered, {q['errors']} endpoint errors; "
             f"injected frames skipped: {skipped}.",
             f"- Latency per request: n {lat['request']['n']}, median {_s(lat['request']['median'])} s, p90 "
             f"{_s(lat['request']['p90'])} s, max {_s(lat['request']['max'])} s. Per question (with re-asks): median "
             f"{_s(lat['question']['median'])} s, p90 {_s(lat['question']['p90'])} s.", "",
             "## Per stage", "", s["reading"], "",
             "| Frames | Stage | N | pass | fail | uncertain | fail rate | pass rate | uncertain rate | errors | "
             "latency median / p90 (s) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in CLASSES:
        for stage, c in s["classes"][k].items():
            lines.append(f"| {CLASS_LABELS[k]} | {stage} | {c['n']} | {c['pass']} | {c['fail']} | {c['uncertain']} | "
                         f"{_pct(c['fail_rate'])} | {_pct(c['pass_rate'])} | {_pct(c['uncertain_rate'])} | "
                         f"{c['errors']} | {_s(c['latency_s']['median'])} / {_s(c['latency_s']['p90'])} |")
    ba = s["balanced_accuracy"]
    lines += ["", "## Balanced accuracy", "",
              "The D34 pilot's measure (pass for predicate-true frames, fail for negatives; a final uncertain counts "
              "as wrong). The D34 pilot measured 0.70 with thinking on, on 30 injected-negative frames.", "",
              "| Negatives | " + " | ".join(ba["natural"]) + " |", "|---|" + "---|" * len(ba["natural"])]
    lines += [f"| {name} | " + " | ".join(_s(v) for v in ba[name].values()) + " |" for name in ba]
    lines += ["", "## The live gate replayed on the natural frames", "",
              "The gate asks the stages in order and stops at the first that does not pass. A round's frames are "
              "predicate true when all three stage predicates hold.", "",
              "| Round's frames | N | pass (executes) | stopped by fail | stopped by uncertain | stopped at pregrasp / "
              "grasp / prerelease |", "|---|---|---|---|---|---|"]
    for k, c in s["gate"].items():
        at = " / ".join(str(c["stopped_at"][stage]) for stage in STAGES)
        lines.append(f"| {k.replace('_', ' ')} | {c['n']} | {c['pass']} | {c['fail']} | {c['uncertain']} | {at} |")
    return "\n".join(lines) + "\n"
