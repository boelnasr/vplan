"""Live Cosmos gate for the interactive demo (D47; spec §11, D5/D20). It runs only after the deterministic Verifier
has passed a plan, so it can add rejections but never approve a geometric failure. It is a learned visual check,
not verification: the D34 pilot measured balanced accuracy 0.70 (thinking on), so evaluation stays physics-only.

Key frames are the verified trajectory's own joint states (end of approach, descend and descend_place), posed
kinematically in a private DIRECT client so the Verifier's preview client and the physics client are untouched."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pybullet as p

from vplan.geometry import T_to_pose, pose_to_T
from vplan.records import (CandidateTrajectory, Finding, PickPlacePlan, Scene, TaskSpec, VisualCheck,
                           VisualGateReport, VisualStage)
from vplan.sim.scene import SceneHandles, build_scene, set_arm
from vplan.tasks.generator import resolve_instruction
from vplan.visual.cosmos import DEFAULT_MODEL, CosmosUnavailable, ask_cosmos, subject_question
from vplan.visual.render import render_views, to_png

STAGES: tuple[VisualStage, ...] = ("pregrasp", "grasp", "prerelease")
SEGMENT = {"pregrasp": 0, "grasp": 1, "prerelease": -2}  # key frame = last sample of approach, descend, descend_place
PLAN_FIELD = {"pregrasp": "target_id", "grasp": "grasp", "prerelease": "place.position"}
# Fixed feedback text: Cosmos's free-text reason may read as a suggested fix (D17), so it stays in the report only.
DIRECTION = {"pregrasp": "learned visual check (Cosmos): approach to the instructed object not confirmed",
             "grasp": "learned visual check (Cosmos): object between the fingers not confirmed",
             "prerelease": "learned visual check (Cosmos): object over the bin not confirmed"}
VIEWS = ("front_oblique", "top")
RETRY_VIEWS = ("front_oblique", "top", "side")  # spec §11: uncertain -> one re-ask with the third view


def subject(spec: TaskSpec) -> str | None:
    """The instructed object as '<color> <shape>', or None when the instruction names no single object."""
    try:
        target = resolve_instruction(spec.instruction, spec.scene)
    except ValueError:
        return None
    obj = next(o for o in spec.scene.objects if o.id == target)
    return f"{obj.color} {obj.shape}"


def _tcp(h: SceneHandles) -> np.ndarray:
    ls = p.getLinkState(h.robot, h.tcp_link, computeForwardKinematics=True, physicsClientId=h.client)
    return pose_to_T(ls[4], ls[5])


def _pose(client: int, plan: PickPlacePlan, traj: CandidateTrajectory, scene: Scene,
          stage: VisualStage) -> SceneHandles:
    """Build the scene at the stage's key frame. At pre-release the planned object hangs from the closed fingers
    with the TCP-relative offset it has at the grasp frame, as the executor's fixed constraint would hold it."""
    q_grasp = np.array(traj.segments[SEGMENT["grasp"]].q[-1])
    if stage != "prerelease":
        return build_scene(scene, client, np.array(traj.segments[SEGMENT[stage]].q[-1]))
    h = build_scene(scene, client, q_grasp)
    obj = next(o for o in scene.objects if o.id == plan.target_id)
    attach = np.linalg.inv(_tcp(h)) @ pose_to_T(obj.pose.position, obj.pose.orientation)
    # TCP y is the closing axis: project all object half-extents on it, as capture_check does. The footprint's
    # smaller half-size is wrong for a yawed top grasp or a side grasp, and the fingers would cut into the object.
    half_width = float(np.abs(attach[1, :3]) @ (np.asarray(obj.size) / 2))
    set_arm(h, np.array(traj.segments[SEGMENT["prerelease"]].q[-1]))
    pos, orn = T_to_pose(_tcp(h) @ attach)
    p.resetBasePositionAndOrientation(h.bodies[obj.id], pos, orn, physicsClientId=client)
    for j in h.finger_joints:
        p.resetJointState(h.robot, j, half_width, physicsClientId=client)
    return h


def _with_client(fn):
    client = p.connect(p.DIRECT)
    if client < 0:
        raise RuntimeError("could not connect to a PyBullet DIRECT physics server")
    try:
        return fn(client)
    finally:
        p.disconnect(client)


def key_frame_state(plan: PickPlacePlan, traj: CandidateTrajectory, scene: Scene,
                    stage: VisualStage) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """(TCP position, planned object position) at the key frame, world frame, m. For tests and diagnostics."""
    def read(client):
        h = _pose(client, plan, traj, scene, stage)
        pos = p.getBasePositionAndOrientation(h.bodies[plan.target_id], physicsClientId=client)[0]
        return tuple(_tcp(h)[:3, 3]), tuple(pos)
    return _with_client(read)


def render_stage(plan: PickPlacePlan, traj: CandidateTrajectory, scene: Scene, stage: VisualStage,
                 views: tuple[str, ...]) -> dict[str, bytes]:
    """PNG bytes per view, in the given order."""
    return _with_client(lambda client: {k: to_png(v) for k, v in
                                        render_views(_pose(client, plan, traj, scene, stage), views).items()})


class CosmosGate:
    """Callable gate: gate(plan, traj, scene, subject=...) -> VisualGateReport. Stops at the first non-pass stage.
    ask and render are seams for tests; on_check sees every verdict as it arrives (live demo output).
    timeout (s) covers a full thinking reply (1024 tokens) at the ~22 tokens/s measured under GPU contention.
    Any gate-side error (endpoint, rendering, callback) means unavailable: the demo falls back to the geometric
    checks instead of blaming the simulation (spec §11)."""

    def __init__(self, base_url: str, *, model: str = DEFAULT_MODEL, enable_thinking: bool = True,
                 timeout: float = 90.0, ask: Callable = ask_cosmos, render: Callable = render_stage,
                 on_check: Callable[[VisualCheck], None] | None = None):
        self.base_url, self.model, self.enable_thinking, self.timeout = base_url, model, enable_thinking, timeout
        self._ask, self._render, self._on_check = ask, render, on_check

    def __call__(self, plan: PickPlacePlan, traj: CandidateTrajectory, scene: Scene, *,
                 subject: str) -> VisualGateReport:
        checks: list[VisualCheck] = []
        for stage in STAGES:
            for views in (VIEWS, RETRY_VIEWS):
                try:
                    v = self._ask(self._render(plan, traj, scene, stage, views), subject_question(stage, subject),
                                  self.base_url, model=self.model, enable_thinking=self.enable_thinking,
                                  timeout=self.timeout)
                    checks.append(VisualCheck(stage=stage, views=views, verdict=v))
                    if self._on_check is not None:
                        self._on_check(checks[-1])
                except CosmosUnavailable as exc:
                    return VisualGateReport(status="unavailable", checks=tuple(checks), note=str(exc))
                except Exception as exc:  # a gate fault must not stop the demo or count as a sim_error
                    return VisualGateReport(status="unavailable", checks=tuple(checks),
                                            note=f"gate error {type(exc).__name__}: {exc}"[:300])
                if v.verdict != "uncertain":
                    break
            if v.verdict == "fail":
                finding = Finding(stage=stage, check="visual", plan_field=PLAN_FIELD[stage], measured=None,
                                  threshold=None, direction=DIRECTION[stage])
                return VisualGateReport(status="fail", checks=tuple(checks), findings=(finding,))
            if v.verdict == "uncertain":
                return VisualGateReport(status="uncertain", checks=tuple(checks))
        return VisualGateReport(status="pass", checks=tuple(checks))
