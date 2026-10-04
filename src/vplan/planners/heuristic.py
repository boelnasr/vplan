"""No-LLM baseline geometry (D16): top grasp aligned to the nearest cube face, vertical via-points above the tallest
obstacle, place at the bin center. The template parser for grounding is added in W2.4."""
from __future__ import annotations

import numpy as np

from vplan.records import Grasp, PickPlacePlan, Place, Scene

SAFE_CLEARANCE_M = 0.10


def heuristic_geometry(scene: Scene, target_id: str) -> PickPlacePlan:
    obj = next(o for o in scene.objects if o.id == target_id)
    x, y, z, w = obj.pose.orientation
    yaw_obj = float(np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))
    yaw = ((yaw_obj + 45.0) % 90.0) - 45.0  # nearest face alignment, in [-45, 45)
    tallest = max([ob.pose.position[2] + ob.half_extents[2] for ob in scene.obstacles]
                  + [scene.bin.floor_center[2] + scene.bin.wall_height, obj.pose.position[2]])
    via_z = tallest + SAFE_CLEARANCE_M
    bx, by, bz = scene.bin.floor_center
    ox, oy, _ = obj.pose.position
    return PickPlacePlan(
        target_id=target_id, grasp=Grasp(approach="top", yaw_deg=yaw, pregrasp_offset_m=0.10),
        transport_via=((ox, oy, via_z), (bx, by, via_z)),
        place=Place(position=(bx, by, bz + obj.size[2] / 2), yaw_deg=yaw, release_height_m=0.05),  # 0.03 put the hand within 1 cm of the bin wall (probe)
        rationale="heuristic: top grasp, vertical clearance above tallest obstacle, bin center")
