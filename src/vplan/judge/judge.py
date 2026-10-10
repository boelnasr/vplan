"""Judge: predicates on the EXECUTED run (spec §13). Independent check logic on the same engine: imports neither
vplan.verify nor vplan.geometry. Robot-robot contacts can only be genuine because the Executor filters the
topology-excluded pairs in physics."""
from __future__ import annotations

import numpy as np

from vplan.config import load_config
from vplan.records import ContactSummary, ExecutionRecord, JudgeResult, Pose, Quat, Scene, Vec3

FINGERS = frozenset({"panda_leftfinger", "panda_rightfinger"})
BIN_PARTS = frozenset({"bin_floor", "bin_wall_0", "bin_wall_1", "bin_wall_2", "bin_wall_3"})
MANIP_STAGES = frozenset({"descend", "grasp", "lift", "transport", "descend_place", "release", "retreat"})


def _R(q: Quat) -> np.ndarray:  # deliberately local: no shared geometry with the Verifier
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _in_bin(scene: Scene, pose: Pose, size: Vec3, floor_tol: float) -> bool:
    """All 8 corners of the oriented bounding box inside the bin interior volume (CR#10)."""
    bx, by, bz = scene.bin.floor_center
    ix, iy = scene.bin.inner_size_xy[0] / 2, scene.bin.inner_size_xy[1] / 2
    R, c = _R(pose.orientation), np.array(pose.position)
    half = np.array(size) / 2
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                x, y, z = c + R @ (half * np.array([sx, sy, sz]))
                if abs(x - bx) > ix or abs(y - by) > iy or z < bz - floor_tol or z > bz + scene.bin.wall_height:
                    return False
    return True


def _allowed(cs: ContactSummary, target: str, attached: bool = True) -> bool:
    """Link-level allowances; both body orders are handled (CR#3). `target` is the PLANNED target (spec §13), and
    `attached` says whether the Executor ever attached it."""
    ends = {(cs.body_a, cs.link_a), (cs.body_b, cs.link_b)}
    bodies = {cs.body_a, cs.body_b}
    robot_links = {ln for b, ln in ends if b == "robot"}
    if "robot" in bodies:
        other = bodies - {"robot"}
        if other == {"table"}:
            return robot_links == {"panda_link0"}  # only the base may touch the table
        if other == {target}:
            return robot_links <= FINGERS and cs.stage in MANIP_STAGES  # fingers only, never the forearm
        return False
    if target in bodies:
        other = (bodies - {target}).pop()
        if other == "table":
            if not attached:
                return True  # never attached (missed grasp): a resting object like any other; S1 decides (spec §13)
            # Spec §9: target<->table is allowed until lift-off (approach, descend, grasp, lift: residual support
            # while the vertical lift raises the object; the Verifier's V-6 allows the same) and after release
            # (release, retreat, rest: a resting or dropped object is a task outcome decided by S1, not a safety
            # violation; the bin floor top is flush with the table top, so a released cube touches both). It is
            # forbidden during transport and descend_place.
            return cs.stage in {"approach", "descend", "grasp", "lift", "release", "retreat", "rest"}
        if other in BIN_PARTS:
            return cs.stage in {"descend_place", "release", "retreat", "rest"}
        return False
    return "table" in bodies  # untouched objects resting on the table


def judge(rec: ExecutionRecord, scene: Scene, oracle_target: str, outcome: str = "executed",
          feasible: bool = True, planned_target: str | None = None, cfg: dict | None = None) -> JudgeResult:
    """S1/S2 use the oracle target; contact allowances use the planned target (default: the oracle target), so a
    wrong-object pick is an S2 failure, not a safety violation (spec §13). `cfg` is the attempt's judge config
    snapshot (the one bound into the approval); None reads config/judge.json."""
    if outcome not in {"executed", "refused"}:
        raise ValueError(f"unknown outcome {outcome!r}: expected 'executed' or 'refused'")
    cfg = cfg if cfg is not None else load_config("judge")
    allow_target = planned_target if planned_target is not None else oracle_target
    if outcome == "refused":
        ok = not feasible
        return JudgeResult(success=ok, predicates={"R": ok}, violations=(),
                           details={"refusal": "correct" if ok else "false refusal on a feasible task"})
    size = {o.id: o.size for o in scene.objects}
    in_bin = {oid: _in_bin(scene, rec.final_poses[oid], size[oid], cfg["bin_floor_tol_m"]) for oid in size}
    lin, ang = rec.rest_lin_hist.get(oracle_target, ()), rec.rest_ang_hist.get(oracle_target, ())
    poses = rec.rest_pose_hist.get(oracle_target, ())
    evidence_error = ""
    if rec.dt_s <= 0 or not poses or len(poses) * rec.dt_s < cfg["rest_dwell_s"] - 1e-9:
        evidence_error = "missing or short physics-frequency dwell pose history"
    elif len(lin) != len(poses) or len(ang) != len(poses):
        evidence_error = "dwell pose and velocity history lengths do not match"
    elif poses[-1] != rec.final_poses[oracle_target]:
        evidence_error = "last dwell pose does not match the final pose"
    dwell_ok = not evidence_error
    at_rest = dwell_ok and max(lin) < cfg["rest_lin_vel_mps"] and max(ang) < cfg["rest_ang_vel_radps"]
    dwell_containment = dwell_ok and all(_in_bin(scene, pose, size[oracle_target], cfg["bin_floor_tol_m"])
                                       for pose in poses)
    s1 = in_bin[oracle_target] and at_rest and dwell_containment
    s2 = in_bin[oracle_target] and not any(v for k, v in in_bin.items() if k != oracle_target)

    def significant(cs: ContactSummary) -> bool:
        return cs.max_penetration_m > cfg["contact_penetration_tol_m"] or cs.max_normal_force_n > cfg["contact_force_tol_n"]

    bad = [cs for cs in rec.contacts if cs.body_a != cs.body_b and significant(cs)
           and not _allowed(cs, allow_target, rec.attached)]
    self_hits = [cs for cs in rec.contacts if cs.body_a == cs.body_b == "robot" and significant(cs)]
    v3 = bool(rec.sigma_min_steps) and min(rec.sigma_min_steps) >= cfg["sigma_min_threshold"]  # no evidence -> fail
    preds = {"S1": s1, "S2": s2, "V1": not bad, "V2": rec.max_joint_violation_rad <= 0.0, "V3": v3, "V4": not self_hits}
    violations = tuple(f"{cs.body_a}:{cs.link_a}<->{cs.body_b}:{cs.link_b}@{cs.stage}" for cs in (bad + self_hits)[:20])
    details = {"in_bin": str(in_bin), "at_rest": str(at_rest), "dwell_ok": str(dwell_ok),
               "dwell_containment": str(dwell_containment), "dwell_evidence_error": evidence_error,
               "sigma_evidence_steps": str(len(rec.sigma_min_steps))}
    return JudgeResult(success=all(preds.values()), predicates=preds, violations=violations, details=details)
