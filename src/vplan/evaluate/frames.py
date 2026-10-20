"""Key frames for the config-4 Cosmos shadow study (spec §11 "Shadow study"; spec §14: the run directory holds
frames/, and the shadow study runs Cosmos on config 3's frames). D53 deferred them to replay; they are rendered at run
time instead, so they cannot drift from later code changes.

`vplan eval run` renders them in the worker, as the run_task job's after step (worker.AFTER): once the attempt's
result is final and written, for every executed round. Each frame is a visual-gate key frame (visual.gate: the
pre-grasp, grasp and pre-release stages of D34/D47) in the gate's two views plus the side view of its `uncertain`
re-ask, posed from the round's plan and its rebuilt trajectory on the task scene by visual.gate.render_stage, which
opens its own DIRECT client per stage. Approval, execution and the Judge are therefore untouched. The rebuilt
trajectory must reproduce the executed record's approval token (it hashes the trajectory, scene, target and configs),
so the frames show exactly what was executed.

Layout: RUN_DIR/frames/<attempt stem>/r<round>-<stage>-<view>.png and index.json (records.FrameIndex), written after
every PNG is rendered, the index last. Any failure comes back as a FrameExport with status "failed"; it never raises.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

from vplan.records import (AttemptRecord, CandidateTrajectory, ExecutionRecord, FrameEntry, FrameExport, FrameIndex,
                           PickPlacePlan, Scene, TaskSpec, scene_hash)
from vplan.visual.gate import RETRY_VIEWS, STAGES, VIEWS, render_stage, subject

FRAME_VIEWS: tuple[str, ...] = RETRY_VIEWS  # the gate's two views, then the side view of the re-ask (spec §11)
INDEX = "index.json"
SEED_MODULUS = 2 ** 32  # run_task passes seed % 2**32 to the motion layer; the rebuild must use the same seed


class FrameMismatch(RuntimeError):
    """The rebuilt trajectory is not the executed one, so its frames would show something else."""


def plan_hash(plan: PickPlacePlan) -> str:
    return hashlib.sha256(plan.model_dump_json().encode("utf-8")).hexdigest()


def frame_name(round_index: int, stage: str, view: str) -> str:
    return f"r{round_index}-{stage}-{view}.png"  # never the attempt id: it contains ':'


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rebuild_trajectory(plan: PickPlacePlan, executed: ExecutionRecord, scene: Scene, *, seed: int,
                       config_snapshot: dict[str, dict], round_index: int) -> CandidateTrajectory:
    """The executed round's trajectory, built as pipeline.run_attempt built it (same plan, scene, start, seed and
    verifier config). Raises FrameMismatch unless it reproduces the executed approval token."""
    from vplan.execute.executor import approval_token
    from vplan.motion.trajectory import build_trajectory
    from vplan.robot import HOME_Q

    cfgs = deepcopy(config_snapshot)
    traj = build_trajectory(plan, scene, HOME_Q, seed=seed % SEED_MODULUS, cfg=cfgs["verifier"])
    if approval_token(traj, scene, plan.target_id, cfgs) != executed.approval:
        raise FrameMismatch(f"round {round_index}: the rebuilt trajectory's approval token differs from the executed "
                            "one")
    return traj


def export_frames(record: AttemptRecord, spec: TaskSpec, *, seed: int, config_snapshot: dict[str, dict],
                  out_dir: Path, rel_dir: str, render: Callable[..., dict[str, bytes]] = render_stage,
                  clock: Callable[[], float] = time.perf_counter) -> FrameExport:
    """Render and write the key frames of record's executed rounds into out_dir (rel_dir is its run-relative POSIX
    path, for the index field). seed and config_snapshot are the attempt's, as run_task received them."""
    start = clock()
    executed = [r for r in record.rounds if r.evaluation is not None and r.evaluation.outcome == "executed"]
    if not executed:
        return FrameExport(status="none", render_s=clock() - start)
    try:
        scene, shash = spec.scene, scene_hash(spec.scene)
        images: list[tuple[FrameEntry, bytes]] = []
        for rnd in executed:
            plan, executed_record = rnd.call.plan, rnd.evaluation.record
            if not isinstance(plan, PickPlacePlan) or executed_record is None:
                raise FrameMismatch(f"round {rnd.round_index} was executed without a plan or an execution record")
            traj = rebuild_trajectory(plan, executed_record, scene, seed=seed, config_snapshot=config_snapshot,
                                      round_index=rnd.round_index)
            phash = plan_hash(plan)
            for stage in STAGES:
                pngs = render(plan, traj, scene, stage, FRAME_VIEWS)
                for view in FRAME_VIEWS:
                    data = pngs[view]
                    images.append((FrameEntry(attempt_id=record.attempt_id, round_index=rnd.round_index, stage=stage,
                                              view=view, file=frame_name(rnd.round_index, stage, view),
                                              sha256=_sha256(data), scene_hash=shash, plan_hash=phash,
                                              approval=executed_record.approval), data))
        index = FrameIndex(attempt_id=record.attempt_id, subject=subject(spec), gate_views=VIEWS,
                           retry_views=FRAME_VIEWS, frames=tuple(entry for entry, _ in images))
        index_bytes = (json.dumps(index.model_dump(mode="json"), indent=2) + "\n").encode("utf-8")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for entry, data in images:
            (out_dir / entry.file).write_bytes(data)
        tmp = out_dir / (INDEX + ".tmp")
        tmp.write_bytes(index_bytes)
        os.replace(tmp, out_dir / INDEX)  # last: an index names only frames that are already on disk
    except Exception as exc:  # spec §16 style: recorded in the attempt row, never fatal to the attempt
        return FrameExport(status="failed", render_s=clock() - start, error_type=type(exc).__name__,
                           error_message=str(exc)[:500])
    return FrameExport(status="written", index=f"{rel_dir}/{INDEX}", index_sha256=_sha256(index_bytes),
                       rounds=tuple(r.round_index for r in executed), files=len(images),
                       bytes=sum(len(data) for _, data in images) + len(index_bytes), render_s=clock() - start)
