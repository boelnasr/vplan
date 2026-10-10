"""Spec §11 state-predicate truth for the Cosmos shadow study and pilot, from simulator state only.
World frame = robot base frame, meters. Poses are 4x4 homogeneous transforms; T_tcp follows the motion-layer TCP
convention (z = approach direction, y = finger closing axis)."""
from __future__ import annotations

import numpy as np

from vplan.geometry import capture_check
from vplan.records import Bin

PREGRASP_AXIS_TOL_DEG = 10.0  # spec §11


def pregrasp_truth(T_tcp: np.ndarray, object_positions: dict[str, tuple[float, float, float]], target_id: str) -> bool:
    """TCP approach axis within 10 deg of the TCP->target-center vector, and the target is the object center
    nearest the TCP."""
    tcp = T_tcp[:3, 3]
    dist = {k: float(np.linalg.norm(np.asarray(v) - tcp)) for k, v in object_positions.items()}
    if min(dist, key=dist.get) != target_id:
        return False
    to_target = np.asarray(object_positions[target_id]) - tcp
    cos = float(T_tcp[:3, 2] @ to_target) / max(dist[target_id], 1e-12)
    return bool(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))) <= PREGRASP_AXIS_TOL_DEG)


def grasp_truth(T_tcp: np.ndarray, T_target: np.ndarray, target_size, cfg: dict) -> bool:
    """Target inside the gripper capture volume (closing band), the same test the Verifier (V-7) and the Executor's
    attach decision use. cfg: the verifier config."""
    return capture_check(T_tcp, T_target, target_size, cfg)[0]


def prerelease_truth(carried_id: str | None, T_carried: np.ndarray | None, carried_size, target_id: str,
                     bin_: Bin) -> bool:
    """The carried object is the target and all 8 box corners project inside the bin's inner xy rectangle
    (the Verifier's V-8 footprint rule; height is not part of the predicate)."""
    if carried_id != target_id or T_carried is None:
        return False
    half = np.asarray(carried_size, dtype=float) / 2
    signs = np.array([(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    corners = (T_carried[:3, :3] @ (signs * half).T).T + T_carried[:3, 3]
    c = np.asarray(bin_.floor_center[:2])
    return bool(np.all(np.abs(corners[:, :2] - c) <= np.asarray(bin_.inner_size_xy) / 2))
