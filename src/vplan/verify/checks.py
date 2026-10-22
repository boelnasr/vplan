"""V-1..V-8. Kinematic only: joint states are set in the preview client, no physics steps."""
from __future__ import annotations

import numpy as np
import pybullet as p

from vplan.geometry import (REASON_APPROACH, REASON_NOT_BETWEEN, REASON_OFF_AXIS, REASON_TOO_WIDE, T_to_pose,
                            capture_check, pose_to_T, rot_angle)
from vplan.records import RESERVED_BODY_IDS, Finding, PickPlacePlan, Scene, Segment
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


def sigma_min(q, m: PandaModel, L: float) -> float:
    """sigma_min(J_s) at one joint sample: the quantity V-4 compares with sigma_min_threshold (spec §9 Jacobian
    contract). The W3.3 calibration measures witnesses with this same function."""
    return float(np.linalg.svd(m.jacobian_s(np.array(q), L), compute_uv=False)[-1])


def singularity(seg: Segment, m: PandaModel, cfg: dict) -> Finding | None:
    L, thr = cfg["jacobian_char_length_m"], cfg["sigma_min_threshold"]
    for k, q in enumerate(seg.q):
        s_min = sigma_min(q, m, L)
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


_LINK_WORDS = {"panda_hand": "the hand", "panda_leftfinger": "the left finger", "panda_rightfinger": "the right finger",
               "panda_link0": "the arm base"}


def _part(link: str) -> str:
    """A robot link in plain words for a collision direction."""
    return _LINK_WORDS.get(link) or link.replace("panda_link", "arm link ")


def _kind(name: str, scene: Scene, target: str) -> tuple[str, str]:
    """What kind of scene body `name` is, in plain words, indefinite and definite: ('an obstacle', 'the obstacle')."""
    if name in {o.id for o in scene.obstacles}:
        return "an obstacle", "the obstacle"
    if name == target:
        return "the target object", "the target object"
    if name in RESERVED_BODY_IDS:  # build_scene's own bodies; an object id is free text, so no prefix match
        if name.startswith("bin_wall"):
            return "a bin wall", "the bin wall"
        return (f"the {name.replace('_', ' ')}",) * 2  # the table, the bin floor
    return "another object", "the other object"


# Segment labels as plain-word phases (spec §8 key poses).
PHASES = {"approach": "the approach", "descend": "the descent to the grasp", "lift": "the lift",
          "transport": "transport", "descend_place": "placement, descending into the bin",
          "retreat": "placement, retreating after the release"}


def _extreme_z(client: int, body: int, link: int, highest: bool) -> float:
    """World z, m, of the highest or lowest point of (body, link) at its current pose: its closest point to a wide slab
    1 m beyond its AABB, so it comes from the collision shape the clearance is measured on. getAABB pads convex meshes
    (measured: 3 mm for the Panda links, 1 mm for the cylinder prism), so it only places the slab, and wherever the body
    is the slab is outside it and covers it in x and y. The slab is a bare shape, never a body in the world. It is not
    freed here (removeCollisionShape only logs "TODO: dealloc"); verify's resetSimulation drops it."""
    lo, hi = p.getAABB(body, link, physicsClientId=client)
    slab = p.createCollisionShape(p.GEOM_BOX, halfExtents=((hi[0] - lo[0]) / 2 + 1.0, (hi[1] - lo[1]) / 2 + 1.0, 0.01),
                                  physicsClientId=client)
    centre = ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, hi[2] + 1.0 if highest else lo[2] - 1.0)
    return p.getClosestPoints(body, -1, 3.0, linkIndexA=link, collisionShapeB=slab, collisionShapePositionB=centre,
                              physicsClientId=client)[0][5][2]


def _vertical(client: int, low: tuple[int, int], top: int, top_name: str) -> str:
    """The measured vertical context of a collision, e.g. "; its bottom is 0.030 m below the obstacle's top (top at z
    0.136 m)": the lowest point of `low` (body, link) against the highest point of body `top` at their current poses.
    A measurement only, never a target height (D17)."""
    bottom_z, top_z = _extreme_z(client, *low, highest=False), _extreme_z(client, top, -1, highest=True)
    side = "below" if bottom_z < top_z else "above"
    return f"; its bottom is {abs(top_z - bottom_z):.3f} m {side} {top_name}'s top (top at z {top_z:.3f} m)"


def collision(seg: Segment, h: SceneHandles, plan: PickPlacePlan, scene: Scene, attach: np.ndarray | None,
              m: PandaModel, cfg: dict) -> Finding | None:
    """Robot vs scene, carried object vs scene AND vs non-finger robot links (CR#9), and self-collision outside
    the topology exclusions (CR#1), at every densified sample."""
    margin, self_margin = cfg["collision_margin_m"], cfg["self_collision_margin_m"]
    target = plan.target_id
    tb = h.bodies.get(target)
    carrying = seg.carrying and tb is not None and attach is not None
    scene_bodies = {n: b for n, b in h.bodies.items() if not (carrying and n == target)}

    # The leg into the drop-off approach ends at a pose built from place, but its path is the transport's: a
    # collision there names the waypoints (spec V-6, D74). Other checks keep the pose's field.
    field = "transport_via" if (seg.label, seg.source_field) == ("transport", "place") else seg.source_field

    def hit(k: int, dist: float, a: str, b: str, who: str, what: str, vertical: str = "") -> Finding:
        """The direction says who came too close to what kind of body, in which phase and by how much, then (against
        a scene body) the vertical context at this sample."""
        closest = f"penetration {-dist * 100:.2f} cm" if dist < 0 else f"clearance {dist * 100:.2f} cm"
        return Finding(stage=seg.label, check="collision", plan_field=field, measured=float(dist), threshold=margin,
                       sample_index=k, bodies=(a, b),
                       direction=f"{who} too close to {what} during {PHASES[seg.label]} ({closest}){vertical}")

    for k, q in enumerate(seg.q):
        set_arm(h, np.array(q))
        if carrying:
            pos, orn = T_to_pose(m.fk(np.array(q)) @ attach)
            p.resetBasePositionAndOrientation(tb, pos, orn, physicsClientId=h.client)
        for name, bid in scene_bodies.items():
            for c in p.getClosestPoints(h.robot, bid, margin, physicsClientId=h.client):
                link = h.link_names[c[3]]
                if not _robot_allowed(seg.label, link, name, target):
                    what, the = _kind(name, scene, target)
                    return hit(k, c[8], f"robot:{link}", name, _part(link), what,
                               _vertical(h.client, (h.robot, c[3]), bid, the))
        if carrying:
            for c in p.getClosestPoints(tb, h.robot, margin, physicsClientId=h.client):
                link = h.link_names[c[4]]
                if link not in FINGERS:
                    return hit(k, c[8], target, f"robot:{link}", "the carried object",
                               f"the robot itself ({_part(link)})")
            for name, bid in scene_bodies.items():
                for c in p.getClosestPoints(tb, bid, margin, physicsClientId=h.client):
                    if not _carried_allowed(seg.label, name, c[8]):
                        what, the = _kind(name, scene, target)
                        return hit(k, c[8], target, name, "the carried object", what,
                                   _vertical(h.client, (tb, -1), bid, the))
        for c in p.getClosestPoints(h.robot, h.robot, self_margin, physicsClientId=h.client):
            a, b = sorted((c[3], c[4]))
            if a != b and (a, b) not in h.self_excluded:
                la, lb = h.link_names[a], h.link_names[b]
                return hit(k, c[8], f"robot:{la}", f"robot:{lb}", _part(la), f"the robot itself ({_part(lb)})")
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
