"""Key poses -> CandidateTrajectory. IK at every Cartesian sample (kept, with its target, for V-2/V-5), then
joint-space densification so the executed joint-interpolated path is what gets verified (spec §8, CR#6)."""
from __future__ import annotations

import numpy as np

from vplan.config import load_config
from vplan.geometry import rot_angle, slerp_R
from vplan.motion.poses import KeyPose, key_poses
from vplan.records import CandidateTrajectory, PickPlacePlan, Scene, Segment
from vplan.robot import HOME_Q, load_panda

CART_STEP_M = 0.01
CART_STEP_RAD = np.deg2rad(2.0)
VEL_FRACTION = 0.5  # quintic peak ds/dt = 1.875/T; 0.5 leaves 2x headroom for the time-inversion error
LABELS = {"pregrasp": ("approach", False), "grasp": ("descend", False), "lift": ("lift", True),
          "preplace": ("transport", True), "release": ("descend_place", True), "retreat": ("retreat", False)}


def _cartesian_samples(A: KeyPose, B: KeyPose) -> list[np.ndarray]:
    dp = float(np.linalg.norm(B.T[:3, 3] - A.T[:3, 3]))
    dth = rot_angle(A.T[:3, :3], B.T[:3, :3])
    n = max(1, int(np.ceil(max(dp / CART_STEP_M, dth / CART_STEP_RAD))))
    out = []
    for k in range(1, n + 1):
        s = k / n
        T = np.eye(4)
        T[:3, :3] = slerp_R(A.T[:3, :3], B.T[:3, :3], s)
        T[:3, 3] = (1 - s) * A.T[:3, 3] + s * B.T[:3, 3]
        out.append(T)
    return out


def densify(sols: list[np.ndarray], max_sum_dq: float) -> tuple[list[np.ndarray], list[int]]:
    """Joint-linear substeps with sum|dq| <= max_sum_dq; returns the dense list and the index of each IK solution."""
    out, idx = [sols[0]], []
    for q in sols[1:]:
        prev = out[-1]
        n = max(1, int(np.ceil(np.sum(np.abs(q - prev)) / max_sum_dq)))
        out += [prev + (q - prev) * (k / n) for k in range(1, n)] + [q]
        idx.append(len(out) - 1)
    return out, idx


def time_scale(qs: list[np.ndarray], vel_limits: np.ndarray) -> tuple[float, ...]:
    """Quintic s(t) over aggregate joint arc length; T from the largest local |dq_i/ds| (CR#8)."""
    dQ = np.diff(np.array(qs), axis=0)
    seg = np.abs(dQ).sum(axis=1)
    total = float(seg.sum())
    if total < 1e-12:
        return tuple(float(x) for x in np.linspace(0.0, 0.5, len(qs)))
    ds = seg / total
    slope = np.divide(np.abs(dQ), ds[:, None], out=np.zeros_like(dQ), where=ds[:, None] > 0)
    T = max(0.5, float(np.max(1.875 * slope / (VEL_FRACTION * vel_limits))))
    s = np.concatenate([[0.0], np.cumsum(ds)])
    s[-1] = 1.0
    taus = np.linspace(0.0, 1.0, 4001)
    t = np.interp(s, 10 * taus**3 - 15 * taus**4 + 6 * taus**5, taus) * T
    for k in range(1, len(t)):  # zero-length steps (identical IK solutions) still need increasing time
        if t[k] <= t[k - 1]:
            t[k] = t[k - 1] + 1e-6
    return tuple(float(x) for x in t)


def build_trajectory(plan: PickPlacePlan, scene: Scene, q_start: np.ndarray = HOME_Q, seed: int = 0,
                     cfg: dict | None = None) -> CandidateTrajectory:
    cfg = cfg if cfg is not None else load_config("verifier")
    max_sum_dq = cfg["densify_max_point_motion_m"] / cfg["lever_bound_m"]
    m = load_panda()
    np.random.seed(seed)  # ManipulaPy IK perturbations use the global RNG
    kps = key_poses(plan, scene, m.fk(q_start))
    q = np.asarray(q_start, dtype=float)
    segments = []
    for A, B in zip(kps[:-1], kps[1:]):
        label, carrying = LABELS.get(B.name, ("transport", True))  # via_i -> transport
        targets = _cartesian_samples(A, B)
        sols = [q]
        for T in targets:
            q_next, _ = m.ik(T, sols[-1], max_iterations=200)  # status unused: V-2 recomputes FK at every sample
            if not np.all(np.isfinite(q_next)):
                q_next = sols[-1].copy()  # keep the record finite; V-2 reports the residual
            sols.append(q_next)
        dense, ik_idx = densify(sols, max_sum_dq)
        segments.append(Segment(label=label, source_field=B.source_field, carrying=carrying,
                                q=tuple(tuple(float(x) for x in qq) for qq in dense),
                                t=time_scale(dense, m.vel_limits), ik_idx=tuple(ik_idx),
                                ik_targets=tuple(tuple(float(x) for x in T.reshape(-1)) for T in targets)))
        q = dense[-1]
    return CandidateTrajectory(q_start=tuple(float(x) for x in q_start), segments=tuple(segments))
