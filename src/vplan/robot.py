"""Panda kinematics for the Verifier side (ManipulaPy). The Judge uses PyBullet instead (judge/kinematics.py).
ManipulaPy 1.4.1 counts finger joints as arm DOF; we slice screw axes to panda_joint1..7 by name (D17).
Frames: ManipulaPy FK is in the robot base frame; Scene enforces base = world, so FK is world-frame (CR#16)."""
from __future__ import annotations

import functools
import os

import numpy as np
import pybullet_data
from ManipulaPy.kinematics import SerialManipulator
from ManipulaPy.urdf.core import URDF

PANDA_URDF = os.path.join(pybullet_data.getDataPath(), "franka_panda", "panda.urdf")
ARM_JOINTS = tuple(f"panda_joint{i}" for i in range(1, 8))
TCP_LINK = "panda_grasptarget"
HOME_Q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])  # rad, standard "ready" pose
PANDA_VEL_LIMITS = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])  # rad/s, Franka datasheet


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


class PandaModel:
    def __init__(self, manip: SerialManipulator, lower: np.ndarray, upper: np.ndarray) -> None:
        self.manip = manip
        self.lower = lower
        self.upper = upper
        self.vel_limits = PANDA_VEL_LIMITS

    def fk(self, q: np.ndarray) -> np.ndarray:
        """World-frame TCP pose (4x4) for arm joint angles q (rad)."""
        return np.asarray(self.manip.forward_kinematics(np.asarray(q, dtype=float), frame="space"))

    def ik(self, T: np.ndarray, q0: np.ndarray, max_iterations: int = 500) -> tuple[np.ndarray, bool]:
        """Damped-least-squares IK. Callers seed np.random; accuracy is re-checked by FK at every sample (V-2)."""
        q, ok, _ = self.manip.iterative_inverse_kinematics(
            T, np.asarray(q0, dtype=float), eomg=1e-4, ev=1e-4, max_iterations=max_iterations)
        return np.asarray(q, dtype=float), bool(ok)

    def jacobian_s(self, q: np.ndarray, L: float) -> np.ndarray:
        """J_s = [J_v/L; J_w]: J_v = TCP-point linear velocity, J_w = angular velocity, both world frame (D23.5)."""
        J = np.asarray(self.manip.jacobian(np.asarray(q, dtype=float), frame="space"))  # rows [w; v_space] (verified CR#15)
        Jw, Jv_space = J[:3, :], J[3:, :]
        p_tcp = self.fk(q)[:3, 3]
        return np.vstack([(Jv_space - _skew(p_tcp) @ Jw) / L, Jw])  # v_tcp = v_s + w x p


@functools.lru_cache(maxsize=1)
def load_panda() -> PandaModel:
    urdf = URDF.load(PANDA_URDF)
    params = urdf.extract_screw_axes(tip_link=TCP_LINK)
    names = [j.name for j in urdf.actuated_joints]
    idx = [names.index(n) for n in ARM_JOINTS]
    S = np.asarray(params["S_list"])[:, idx]
    B = np.asarray(params["B_list"])[:, idx]
    limits = [params["joint_limits"][i] for i in idx]
    # omega_list = rotation axes (3 x n); ManipulaPy's extract_omega_list returns the wrong shape for 6 x n input (CR#15)
    manip = SerialManipulator(M_list=np.asarray(params["M"]), omega_list=S[:3, :], S_list=S, B_list=B,
                              joint_limits=limits)
    lower = np.array([lo for lo, _ in limits], dtype=float)
    upper = np.array([hi for _, hi in limits], dtype=float)
    return PandaModel(manip, lower, upper)
