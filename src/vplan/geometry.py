"""Rigid-body helpers and the gripper capture-volume test.
Shared by the Verifier (planned grasp pose) and the Executor (actual grasp pose). The Judge must NOT import this
module: its predicates use independent code (D21). TCP frame: z = approach direction, y = finger closing axis."""
from __future__ import annotations

import numpy as np


def quat_to_R(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def R_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    """Shepperd's method; returns xyzw."""
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s]
    qa = np.array(q) / np.linalg.norm(q)
    return tuple(float(v) for v in qa)


def pose_to_T(pos, quat) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_to_R(quat)
    T[:3, 3] = pos
    return T


def T_to_pose(T: np.ndarray) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    return tuple(float(v) for v in T[:3, 3]), R_to_quat(T[:3, :3])


def rot_angle(R0: np.ndarray, R1: np.ndarray) -> float:
    """Geodesic angle between two rotations, rad, via quaternions (well-conditioned near pi)."""
    d = abs(float(np.dot(R_to_quat(R0), R_to_quat(R1))))
    return 2.0 * float(np.arccos(min(1.0, d)))


def slerp_R(R0: np.ndarray, R1: np.ndarray, s: float) -> np.ndarray:
    """Quaternion slerp. The old axis-angle form divided by sin(pi) at 180 deg (CR#8)."""
    q0, q1 = np.array(R_to_quat(R0)), np.array(R_to_quat(R1))
    d = float(q0 @ q1)
    if d < 0.0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + s * (q1 - q0)
    else:
        th = np.arccos(d)
        q = (np.sin((1 - s) * th) * q0 + np.sin(s * th) * q1) / np.sin(th)
    return quat_to_R(q / np.linalg.norm(q))


# Why capture_check rejected a grasp; callers compare by equality instead of matching message text.
REASON_OFF_AXIS = "object center is outside the capture volume (off the finger plane or depth)"
REASON_TOO_WIDE = "object is wider than the gripper opening along the closing axis"
REASON_NOT_BETWEEN = "object is not between the fingers along the closing axis"
REASON_APPROACH = "approach axis is not perpendicular to an object face"


def capture_check(T_tcp: np.ndarray, T_obj: np.ndarray, obj_size, cfg: dict) -> tuple[bool, dict[str, float], str]:
    """Is the object inside the finite gripper capture volume? Measured in the TCP frame (CR#5)."""
    R = T_tcp[:3, :3]
    o = R.T @ (T_obj[:3, 3] - T_tcp[:3, 3])  # object center in TCP frame: x = off-axis, y = closing, z = approach
    Ro = T_obj[:3, :3]
    half = np.asarray(obj_size, dtype=float) / 2
    half_w = float(sum(abs(R[:, 1] @ Ro[:, j]) * half[j] for j in range(3)))  # half-width along the closing axis
    max_half = cfg["gripper_max_opening_m"] / 2 - cfg["grasp_width_margin_m"]
    cos_face = max(abs(float(R[:, 2] @ Ro[:, j])) for j in range(3))
    approach_err = float(np.degrees(np.arccos(min(1.0, cos_face))))
    vals = {"off_x_m": float(o[0]), "off_closing_m": float(o[1]), "off_approach_m": float(o[2]),
            "half_width_m": half_w, "approach_err_deg": approach_err}
    if abs(o[0]) > cfg["grasp_band_m"] or abs(o[2]) > cfg["grasp_band_m"]:
        return False, vals, REASON_OFF_AXIS
    if half_w > max_half:
        return False, vals, REASON_TOO_WIDE
    if abs(o[1]) > max_half - half_w:
        return False, vals, REASON_NOT_BETWEEN
    if approach_err > cfg["approach_tol_deg"]:
        return False, vals, REASON_APPROACH
    return True, vals, ""
