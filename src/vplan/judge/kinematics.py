"""Judge-side Jacobian from PyBullet. Independent of vplan.robot (ManipulaPy) by design (D21, D23.5).
calculateJacobian with localPosition = 0 is the link-frame origin (verified against finite differences, CR#16)."""
from __future__ import annotations

import numpy as np
import pybullet as p

from vplan.sim.scene import SceneHandles


def jacobian_s_pybullet(h: SceneHandles, q: np.ndarray, L: float) -> np.ndarray:
    """J_s at configuration q. Resets the arm kinematically: preview client and tests only."""
    for j, qi in zip(h.arm_joints, q):
        p.resetJointState(h.robot, j, float(qi), physicsClientId=h.client)
    return jacobian_s_current(h, L)


def jacobian_s_current(h: SceneHandles, L: float) -> np.ndarray:
    """J_s = [J_v/L; J_w] at the TCP, world frame, for the CURRENT state. Never touches joint state, so the
    Executor can call it every physics step (resetJointState would zero velocities)."""
    movable = [i for i in range(p.getNumJoints(h.robot, physicsClientId=h.client))
               if p.getJointInfo(h.robot, i, physicsClientId=h.client)[2] != p.JOINT_FIXED]
    q_all = [p.getJointState(h.robot, i, physicsClientId=h.client)[0] for i in movable]
    zeros = [0.0] * len(movable)
    lin, ang = p.calculateJacobian(h.robot, h.tcp_link, [0.0, 0.0, 0.0], q_all, zeros, zeros,
                                   physicsClientId=h.client)
    cols = [movable.index(j) for j in h.arm_joints]
    return np.vstack([np.asarray(lin)[:, cols] / L, np.asarray(ang)[:, cols]])
