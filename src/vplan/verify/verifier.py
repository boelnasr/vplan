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


def verify(plan: PickPlacePlan, traj: CandidateTrajectory, scene: Scene, client: int, cfg: dict) -> list[Finding]:
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
    # attachment from the planned grasp endpoint, independent of collision-loop state (CR#9)
    attach = np.linalg.inv(m.fk(np.array(traj.segments[1].q[-1]))) @ pose_to_T(obj.pose.position, obj.pose.orientation)
    for f in (checks.grasp_preconditions(plan, scene, kps[2].T, cfg), checks.release(plan, scene, kps[-2].T, attach, cfg)):
        if f:
            findings.append(f)
    for seg in traj.segments:
        stop = [f for f in (checks.ik_accuracy(seg, m, cfg), checks.continuity(seg, cfg)) if f]
        findings += stop
        if stop:
            break  # later joint states are undefined
        findings += [f for f in (checks.joint_limits(seg, m, cfg), checks.singularity(seg, m, cfg),
                                 checks.collision(seg, h, plan, attach, m, cfg)) if f]
    seen, ordered = set(), []
    for f in findings:
        if f.check not in seen:
            seen.add(f.check)
            ordered.append(f)
    return ordered[:5]
