"""Judge-side Jacobian from PyBullet (D21, D23.5).
Only the computation is independent of vplan.robot/ManipulaPy: J_s comes from PyBullet's calculateJacobian, never
from ManipulaPy. The import is not independent: vplan.sim.scene imports shared URDF/link constants from vplan.robot.
calculateJacobian's localPosition is a point in link coordinates; (0, 0, 0) is the link-frame origin (verified
against finite differences, CR#16). panda_grasptarget has a zero inertial offset, so the link-frame vs COM-frame
distinction does not matter here (verified by probe and by the ManipulaPy agreement test)."""
from __future__ import annotations

import numpy as np
import pybullet as p

from vplan.sim.scene import SceneHandles


def jacobian_s_pybullet(h: SceneHandles, q: np.ndarray, L: float) -> np.ndarray:
    """J_s at configuration q (7 arm joint angles, rad; ValueError otherwise). Resets the arm kinematically:
    preview client and tests only."""
    if len(q) != len(h.arm_joints):
        raise ValueError(f"q must have {len(h.arm_joints)} joint values, got {len(q)}")
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
