"""Plan -> key TCP poses. TCP z axis = approach direction (motion toward the object); TCP y axis = finger closing
axis; yaw rotates the closing axis about the approach axis from world +x (top) or world +z (side) (spec §7)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vplan.records import PickPlacePlan, Scene

APPROACH_DIRS = {"top": (0, 0, -1), "side_+x": (-1, 0, 0), "side_-x": (1, 0, 0),
                 "side_+y": (0, -1, 0), "side_-y": (0, 1, 0)}
LIFT_M = 0.10
PREPLACE_CLEARANCE_M = 0.05
RETREAT_M = 0.10


@dataclass(frozen=True)
class KeyPose:
    name: str
    T: np.ndarray
    source_field: str


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def approach_rotation(approach: str, yaw_deg: float) -> np.ndarray:
    z = np.array(APPROACH_DIRS[approach], dtype=float)
    y_ref = np.array([1.0, 0.0, 0.0]) if approach == "top" else np.array([0.0, 0.0, 1.0])
    y = _rodrigues(z, np.deg2rad(yaw_deg)) @ y_ref
    return np.column_stack([np.cross(y, z), y, z])


def _T(R: np.ndarray, pos) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


def key_poses(plan: PickPlacePlan, scene: Scene, T_start: np.ndarray) -> list[KeyPose]:
    """The object is held with its center at the TCP, so place.position (+ release height) is the TCP target."""
    obj = next((o for o in scene.objects if o.id == plan.target_id), None)
    center = np.array(obj.pose.position) if obj is not None else np.array(plan.place.position)  # V-1 reports unknown ids
    Rg = approach_rotation(plan.grasp.approach, plan.grasp.yaw_deg)
    up = np.array([0.0, 0.0, 1.0])
    Rp = _rodrigues(up, np.deg2rad(plan.place.yaw_deg - plan.grasp.yaw_deg)) @ Rg
    release = np.array(plan.place.position) + up * plan.place.release_height_m
    poses = [KeyPose("start", T_start, "grasp"),
             KeyPose("pregrasp", _T(Rg, center - Rg[:, 2] * plan.grasp.pregrasp_offset_m), "grasp"),
             KeyPose("grasp", _T(Rg, center), "grasp"),
             KeyPose("lift", _T(Rg, center + up * LIFT_M), "grasp")]
    poses += [KeyPose(f"via_{i}", _T(Rg, np.array(v)), f"transport_via[{i}]") for i, v in enumerate(plan.transport_via)]
    poses += [KeyPose("preplace", _T(Rp, release + up * PREPLACE_CLEARANCE_M), "place"),
              KeyPose("release", _T(Rp, release), "place"),
              KeyPose("retreat", _T(Rp, release + up * RETREAT_M), "place")]
    return poses
