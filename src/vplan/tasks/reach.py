"""Conservative Panda reach bound for impossibility proofs (spec §6, D15, D23.3).

The bound is the maximum TCP distance from the shoulder point over the joint-limit box. Joint axes 1 and 2 both pass
through the shoulder, so q1 and q2 never change that distance and the reach set is a ball around the shoulder.
"""
from __future__ import annotations

import functools
import math

import numpy as np

# Joint-2 origin (= intersection of axes 1 and 2) in the base frame; checked against the URDF in _reach_samples.
SHOULDER: tuple[float, float, float] = (0.0, 0.0, 0.333)
# Added to the bound in every proof. Covers the finger pads' offset from the TCP and contact geometry at grasp.
REACH_MARGIN_M = 0.05
# Chain of points whose consecutive distances are rigid (q-independent): shoulder -> elbow (joint-4 origin) ->
# wrist (joint-6 origin, on axes 5 and 6) -> TCP (on axis 7). The triangle inequality bounds TCP distance by their sum.
_CHAIN = ("panda_link2", "panda_link4", "panda_link6", "panda_grasptarget")
N_SAMPLES = 20000


@functools.lru_cache(maxsize=1)
def _reach_samples() -> tuple[float, float]:
    """(max sampled shoulder-TCP distance, analytic chain bound), m."""
    # Lazy: vplan.tasks is imported by the Cosmos client, whose venv has neither pybullet nor ManipulaPy.
    import pybullet as p

    from vplan.robot import ARM_JOINTS, PANDA_URDF, TCP_LINK
    if _CHAIN[-1] != TCP_LINK:
        raise RuntimeError("the reach chain must end at the TCP link")
    client = p.connect(p.DIRECT)
    if client < 0:
        raise RuntimeError("could not connect to a PyBullet DIRECT physics server")
    try:
        robot = p.loadURDF(PANDA_URDF, useFixedBase=True, physicsClientId=client)
        infos = [p.getJointInfo(robot, i, physicsClientId=client) for i in range(p.getNumJoints(robot, physicsClientId=client))]
        joint = {i[1].decode(): i[0] for i in infos}
        link = {i[12].decode(): i[0] for i in infos}
        arm = [joint[n] for n in ARM_JOINTS]
        lower = np.array([infos[j][8] for j in arm])
        upper = np.array([infos[j][9] for j in arm])
        chain = [link[n] for n in _CHAIN]
        rng = np.random.default_rng(0)  # fixed: the bound is a constant of the robot model
        # Limit-box corners of q4 (the elbow, which sets the reach) are included explicitly, then uniform samples.
        qs = [np.where(np.arange(7) == 3, q4, (lower + upper) / 2) for q4 in (lower[3], upper[3])]
        qs.extend(rng.uniform(lower, upper) for _ in range(N_SAMPLES))
        best, segments = 0.0, []
        for q in qs:
            for j, qi in zip(arm, q):
                p.resetJointState(robot, j, float(qi), physicsClientId=client)
            pts = [np.array(p.getLinkState(robot, k, computeForwardKinematics=True, physicsClientId=client)[4])
                   for k in chain]
            if not np.allclose(pts[0], SHOULDER, atol=1e-6):
                raise RuntimeError(f"shoulder moved to {pts[0]}: the URDF no longer matches SHOULDER")
            best = max(best, float(np.linalg.norm(pts[-1] - pts[0])))
            segments.append([float(np.linalg.norm(b - a)) for a, b in zip(pts, pts[1:])])
    finally:
        p.disconnect(client)
    seg = np.array(segments)
    if float(np.ptp(seg, axis=0).max()) > 1e-5:
        raise RuntimeError("reach chain segments are not rigid; the analytic bound would be invalid")
    return best, float(seg[0].sum())


def reach_bound_m() -> float:
    """Max TCP distance from SHOULDER, m: the larger of the dense joint-space sample and the analytic chain bound."""
    return max(_reach_samples())


def target_clearance_m(position: tuple[float, float, float], size: tuple[float, float, float]) -> float:
    """Lower bound on the distance from SHOULDER to any point of an object (center minus half-diagonal), m."""
    return math.dist(position, SHOULDER) - math.hypot(*size) / 2


def bin_clearance_m(floor_center: tuple[float, float, float], inner_size_xy: tuple[float, float]) -> float:
    """Lower bound on the distance from SHOULDER to any point of the bin's interior column at any height, m.

    The column is unbounded in z, so the 0.15 m release-height limit and object height never need to be modelled."""
    return math.hypot(floor_center[0] - SHOULDER[0], floor_center[1] - SHOULDER[1]) - math.hypot(*inner_size_xy) / 2
