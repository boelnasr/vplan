"""Run V-1..V-8 in trajectory order; stop segment checks after IK/continuity failure (D18). <=1 finding per check."""
from __future__ import annotations

import numpy as np
import pybullet as p

from vplan.geometry import pose_to_T
from vplan.motion.poses import key_poses
from vplan.records import CandidateTrajectory, Finding, PickPlacePlan, Scene
from vplan.robot import load_panda
from vplan.sim.scene import build_scene
from vplan.verify import checks


def _order_findings(findings: list[Finding], segment_labels: list[str]) -> list[Finding]:
    """Plan-stage findings first, then in trajectory order (first segment with the finding's stage, then sample).
    Stable, so ties keep their insertion order. Needed so the 5-finding cap never drops an earlier collision."""
    def key(f: Finding) -> tuple[int, int, int]:
        if f.stage == "plan":
            return (0, 0, 0)
        seg = segment_labels.index(f.stage) if f.stage in segment_labels else len(segment_labels)
        return (1, seg, f.sample_index or 0)

    return sorted(findings, key=key)


def verify(plan: PickPlacePlan, traj: CandidateTrajectory, scene: Scene, client: int, cfg: dict) -> list[Finding]:
    """Run V-1..V-8. Calls p.resetSimulation on `client` (wiping every body in it), so callers must pass a
    dedicated preview client, never one that holds state they still need."""
    m = load_panda()
    findings: list[Finding] = list(checks.grounding(plan, scene, cfg))
    if any(f.plan_field == "target_id" for f in findings):
        return findings
    kps = key_poses(plan, scene, m.fk(np.array(traj.q_start)))  # start, pregrasp, grasp, lift, via*, preplace, release, retreat
    if len(traj.segments) != len(kps) - 1:
        raise ValueError("trajectory segments do not match the plan's key poses")
    p.resetSimulation(physicsClientId=client)
    h = build_scene(scene, client, np.array(traj.q_start))
    obj = next(o for o in scene.objects if o.id == plan.target_id)
    # attachment from the planned grasp pose, not FK of a segment endpoint: if IK failed there, FK would give a
    # bogus attachment and spurious release findings (CR#9)
    attach = np.linalg.inv(kps[2].T) @ pose_to_T(obj.pose.position, obj.pose.orientation)
    for f in (checks.grasp_preconditions(plan, scene, kps[2].T, cfg), checks.release(plan, scene, kps[-2].T, attach, cfg)):
        if f:
            findings.append(f)
    for seg in traj.segments:
        stop = [f for f in (checks.ik_accuracy(seg, m, cfg), checks.continuity(seg, cfg)) if f]
        findings += stop
        if stop:
            break  # later joint states are undefined
        findings += [f for f in (checks.joint_limits(seg, m, cfg), checks.singularity(seg, m, cfg),
                                 checks.collision(seg, h, plan, scene, attach, m, cfg)) if f]
    seen, ordered = set(), []
    for f in _order_findings(findings, [sg.label for sg in traj.segments]):
        if f.check not in seen:
            seen.add(f.check)
            ordered.append(f)
    return ordered[:5]
