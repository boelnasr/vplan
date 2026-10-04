"""V-1..V-8. Kinematic only: joint states are set in the preview client, no physics steps."""
from __future__ import annotations

import numpy as np
import pybullet as p

from vplan.geometry import (REASON_APPROACH, REASON_NOT_BETWEEN, REASON_OFF_AXIS, REASON_TOO_WIDE, T_to_pose,
                            capture_check, pose_to_T, rot_angle)
from vplan.records import Finding, PickPlacePlan, Scene, Segment
from vplan.robot import PandaModel
from vplan.sim.scene import SceneHandles, set_arm

FINGERS = frozenset({"panda_leftfinger", "panda_rightfinger"})
RESTING_HEIGHT_TOL_M = 0.005  # spec §7/§9 V-8: place.position z must be the resting center height, +-5 mm


def grounding(plan: PickPlacePlan, scene: Scene, cfg: dict) -> list[Finding]:
    ids = {o.id for o in scene.objects}
    if plan.target_id not in ids:
        return [Finding(stage="plan", check="grounding", plan_field="target_id", measured=None, threshold=None,
                        direction=f"target_id '{plan.target_id}' is not an object in the scene; known ids: {sorted(ids)}")]
    obj = next(o for o in scene.objects if o.id == plan.target_id)
    margin = max(obj.size[0], obj.size[1]) / 2 + cfg["place_margin_m"]
    bx, by, _ = scene.bin.floor_center
    ix, iy = scene.bin.inner_size_xy[0] / 2 - margin, scene.bin.inner_size_xy[1] / 2 - margin
    px, py, _ = plan.place.position
    excess = max(abs(px - bx) - ix, abs(py - by) - iy)
    if excess > 0:
        return [Finding(stage="plan", check="grounding", plan_field="place.position", measured=float(excess),
                        threshold=0.0, direction=f"place.position is outside the usable bin interior by {excess:.3f} m "
                                                 f"(bin center ({bx:.2f}, {by:.2f}), usable half-size ({ix:.3f}, {iy:.3f}))")]
    return []


def ik_accuracy(seg: Segment, m: PandaModel, cfg: dict) -> Finding | None:
    """FK residual at EVERY IK sample vs its stored Cartesian target (CR#6)."""
    for idx, Tf in zip(seg.ik_idx, seg.ik_targets):
        target = np.array(Tf).reshape(4, 4)
        T = m.fk(np.array(seg.q[idx]))
        pos_err = float(np.linalg.norm(T[:3, 3] - target[:3, 3]))
        rot_err = float(np.degrees(rot_angle(T[:3, :3], target[:3, :3])))
        if pos_err > cfg["ik_pos_tol_m"] or rot_err > cfg["ik_rot_tol_deg"]:
            # report the quantity that failed (rotation only when position is within tolerance)
            rot_fail = pos_err <= cfg["ik_pos_tol_m"]
            return Finding(stage=seg.label, check="ik_accuracy", plan_field=seg.source_field,
                           measured=rot_err if rot_fail else pos_err,
                           threshold=cfg["ik_rot_tol_deg"] if rot_fail else cfg["ik_pos_tol_m"], sample_index=idx,
                           direction=f"pose not reachable: residual {pos_err * 1000:.1f} mm / {rot_err:.1f} deg")
    return None


def continuity(seg: Segment, cfg: dict) -> Finding | None:
    """Branch flips between consecutive IK solutions, measured BEFORE densification hides them (CR#6)."""
    S = np.array([seg.q[0]] + [seg.q[i] for i in seg.ik_idx])
    steps = np.max(np.abs(np.diff(S, axis=0)), axis=1)
    k = int(np.argmax(steps))
    if steps[k] > cfg["continuity_max_step_rad"]:
        return Finding(stage=seg.label, check="continuity", plan_field=seg.source_field, measured=float(steps[k]),
                       threshold=cfg["continuity_max_step_rad"], sample_index=seg.ik_idx[k],
                       direction="IK branch flip: large joint jump between consecutive Cartesian samples")
    return None


def joint_limits(seg: Segment, m: PandaModel, cfg: dict) -> Finding | None:
    Q = np.array(seg.q)
    margin = np.deg2rad(cfg["joint_limit_margin_deg"])
    worst = np.maximum((m.lower + margin) - Q, Q - (m.upper - margin))
    k, j = np.unravel_index(int(np.argmax(worst)), worst.shape)
    if worst[k, j] > 0:
        return Finding(stage=seg.label, check="joint_limits", plan_field=seg.source_field, measured=float(worst[k, j]),
                       threshold=float(margin), sample_index=int(k),
                       direction=f"panda_joint{j + 1} within {np.degrees(margin):.0f} deg of its limit")
    vel = np.abs(np.diff(Q, axis=0)) / np.diff(np.array(seg.t))[:, None]
    limit = m.vel_limits * cfg["velocity_limit_fraction"]
    ratio = vel / limit
    k, j = np.unravel_index(int(np.argmax(ratio)), ratio.shape)
    if ratio[k, j] > 1.0:
        return Finding(stage=seg.label, check="joint_limits", plan_field=seg.source_field, measured=float(vel[k, j]),
                       threshold=float(limit[j]), sample_index=int(k),
                       direction=f"panda_joint{j + 1} exceeds its velocity limit after timing")
    return None


def singularity(seg: Segment, m: PandaModel, cfg: dict) -> Finding | None:
    L, thr = cfg["jacobian_char_length_m"], cfg["sigma_min_threshold"]
    for k, q in enumerate(seg.q):
        s_min = float(np.linalg.svd(m.jacobian_s(np.array(q), L), compute_uv=False)[-1])
        if s_min < thr:
            return Finding(stage=seg.label, check="singularity", plan_field=seg.source_field, measured=s_min,
                           threshold=thr, sample_index=k, direction="near-singular configuration (sigma_min below threshold)")
    return None


def _robot_allowed(stage: str, link: str, other: str, target: str) -> bool:
    if link == "panda_link0" and other == "table":
        return True  # base rests on the table
    return link in FINGERS and other == target and stage in ("descend", "retreat")


def _carried_allowed(stage: str, other: str, dist: float) -> bool:
    """dist = closest distance (negative = penetration). Proximity to the bin floor is allowed while placing, but
    penetration is not: pressing the object into the floor is outside the manipulation model (spec §9)."""
    return (other == "table" and stage == "lift") or (other == "bin_floor" and stage == "descend_place" and dist >= 0.0)


def collision(seg: Segment, h: SceneHandles, plan: PickPlacePlan, attach: np.ndarray | None, m: PandaModel,
              cfg: dict) -> Finding | None:
    """Robot vs scene, carried object vs scene AND vs non-finger robot links (CR#9), and self-collision outside
    the topology exclusions (CR#1), at every densified sample."""
    margin, self_margin = cfg["collision_margin_m"], cfg["self_collision_margin_m"]
    target = plan.target_id
    tb = h.bodies.get(target)
    carrying = seg.carrying and tb is not None and attach is not None
    scene_bodies = {n: b for n, b in h.bodies.items() if not (carrying and n == target)}

    def hit(k: int, dist: float, a: str, b: str, what: str) -> Finding:
        return Finding(stage=seg.label, check="collision", plan_field=seg.source_field, measured=float(dist),
                       threshold=margin, sample_index=k, bodies=(a, b), direction=f"{what} ({dist * 100:.1f} cm)")

    for k, q in enumerate(seg.q):
        set_arm(h, np.array(q))
        if carrying:
            pos, orn = T_to_pose(m.fk(np.array(q)) @ attach)
            p.resetBasePositionAndOrientation(tb, pos, orn, physicsClientId=h.client)
        for name, bid in scene_bodies.items():
            for c in p.getClosestPoints(h.robot, bid, margin, physicsClientId=h.client):
                link = h.link_names[c[3]]
                if not _robot_allowed(seg.label, link, name, target):
                    return hit(k, c[8], f"robot:{link}", name, f"robot link {link} too close to '{name}'")
        if carrying:
            for c in p.getClosestPoints(tb, h.robot, margin, physicsClientId=h.client):
                link = h.link_names[c[4]]
                if link not in FINGERS:
                    return hit(k, c[8], target, f"robot:{link}", f"carried '{target}' too close to robot link {link}")
            for name, bid in scene_bodies.items():
                for c in p.getClosestPoints(tb, bid, margin, physicsClientId=h.client):
                    if not _carried_allowed(seg.label, name, c[8]):
                        return hit(k, c[8], target, name, f"carried '{target}' too close to '{name}'")
        for c in p.getClosestPoints(h.robot, h.robot, self_margin, physicsClientId=h.client):
            a, b = sorted((c[3], c[4]))
            if a != b and (a, b) not in h.self_excluded:
                return hit(k, c[8], f"robot:{h.link_names[a]}", f"robot:{h.link_names[b]}", "self-collision")
    return None


def grasp_preconditions(plan: PickPlacePlan, scene: Scene, T_grasp: np.ndarray, cfg: dict) -> Finding | None:
    obj = next((o for o in scene.objects if o.id == plan.target_id), None)
    if obj is None:
        return None  # reported by grounding
    ok, vals, why = capture_check(T_grasp, pose_to_T(obj.pose.position, obj.pose.orientation), obj.size, cfg)
    if ok:
        return None
    field = "grasp.yaw_deg" if why in (REASON_TOO_WIDE, REASON_NOT_BETWEEN) else "grasp"
    # report the evidence for the rejection reason (the repair prompt carries measured vs threshold, spec §9)
    max_half = cfg["gripper_max_opening_m"] / 2 - cfg["grasp_width_margin_m"]
    evidence = {REASON_OFF_AXIS: (max(abs(vals["off_x_m"]), abs(vals["off_approach_m"])), cfg["grasp_band_m"]),
                REASON_TOO_WIDE: (vals["half_width_m"], max_half),
                REASON_NOT_BETWEEN: (abs(vals["off_closing_m"]), max_half - vals["half_width_m"]),
                REASON_APPROACH: (vals["approach_err_deg"], cfg["approach_tol_deg"])}
    measured, threshold = evidence[why]
    return Finding(stage="descend", check="grasp_preconditions", plan_field=field,
                   measured=float(measured), threshold=float(threshold), direction=why)


def release(plan: PickPlacePlan, scene: Scene, T_release: np.ndarray, attach: np.ndarray, cfg: dict) -> Finding | None:
    """Release height within the manipulation model, place.position at the resting height (so the drop is exactly
    release_height_m), and the carried object's footprint inside the bin interior."""
    if plan.place.release_height_m > cfg["release_height_max_m"]:
        return Finding(stage="descend_place", check="release", plan_field="place.release_height_m",
                       measured=plan.place.release_height_m, threshold=cfg["release_height_max_m"],
                       direction="release height above the manipulation-model limit")
    obj = next((o for o in scene.objects if o.id == plan.target_id), None)
    if obj is None:
        return None
    rest_z = scene.bin.floor_center[2] + obj.size[2] / 2
    z_err = abs(plan.place.position[2] - rest_z)
    if z_err > RESTING_HEIGHT_TOL_M:
        return Finding(stage="descend_place", check="release", plan_field="place.position", measured=float(z_err),
                       threshold=RESTING_HEIGHT_TOL_M,
                       direction=f"place.position z must be the object's resting height {rest_z:.3f} m "
                                 f"(bin floor z + half the object height); release_height_m sets the drop")
    T_obj = T_release @ attach
    half = np.asarray(obj.size) / 2
    corners = [T_obj[:3, :3] @ (half * np.array(sg)) + T_obj[:3, 3]
               for sg in [(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]]
    bx, by, _ = scene.bin.floor_center
    ix, iy = scene.bin.inner_size_xy[0] / 2, scene.bin.inner_size_xy[1] / 2
    excess = max(max(abs(c[0] - bx) - ix, abs(c[1] - by) - iy) for c in corners)
    if excess > 0:
        return Finding(stage="descend_place", check="release", plan_field="place.position", measured=float(excess),
                       threshold=0.0, direction=f"carried object's footprint extends {excess:.3f} m outside the bin interior")
    return None
