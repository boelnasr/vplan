"""Seeded feasible and impossible-by-construction tasks (spec §6, plan W2.2).

Feasible tasks are constructive: geometry is sampled, then the heuristic witness must pass run_attempt with a Judge
success under the attempt's config snapshot, or the seed is rejected (undetermined, never labelled impossible).
Impossible tasks carry a proof that check_impossibility re-derives from the scene alone.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from vplan.records import (IMPOSSIBLE_REFUSALS, Bin, ImpossibilityProof, Obstacle, Pose, RobotFacts, Scene,
                           SceneObject, TaskOracle, TaskSpec, WitnessEvaluation)
from vplan.tasks.reach import REACH_MARGIN_M, bin_clearance_m, reach_bound_m, target_clearance_m
from vplan.tasks.referents import instruction, resolve_referent

if TYPE_CHECKING:  # pipeline imports the verifier and the executor; generate imports it lazily
    from vplan.pipeline import AttemptResult

TaskKind = Literal["feasible", "impossible"]
COLORS = ("red", "green", "blue", "yellow", "purple", "orange")
SHAPES = ("cube", "cylinder")
# Footprint of the table built by sim.scene.build_scene: center (0.45, 0), half extents (0.7, 0.8).
TABLE_X, TABLE_Y = (-0.25, 1.15), (-0.8, 0.8)
GRIPPER_OPENING_M = 0.08
# D29 I-6: center top grasps need object heights of ~3.6-5.9 cm; edges also stay well under the 8 cm opening.
OBJECT_EDGE_M = (0.04, 0.05)
GRASP_HEIGHT_M = (0.036, 0.059)
BIN_INNER_M, BIN_WALL_M, BIN_THICKNESS_M = 0.16, 0.08, 0.01
OBJECT_GAP_M = 0.05  # between object footprints: room for the fingers on either side of the target
HAND_GAP_M = 0.08  # obstacles keep this from the target and the bin: the hand is ~0.2 m wide across the fingers
OBSTACLE_GAP_M = 0.03  # obstacles to distractors and to each other
CAVITY_CLEARANCE_M = 0.01  # type 3: no initial contact between the target and its enclosure
PANEL_M = 0.02  # type 3 panel thickness
_TRIES = 200


class GenerationRejected(Exception):
    """The sampled geometry could not satisfy the construction rules; reason names the failed step."""
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class Draft:
    """Geometry and label before any witness; target_id is private truth (None for an absent referent)."""
    task: TaskSpec
    target_id: str | None
    proof: ImpossibilityProof | None
    template: str


WitnessEvaluator = Callable[[TaskSpec, str], WitnessEvaluation]


# ---- footprints (xy AABB: cx, cy, hx, hy) -----------------------------------------------------
def _object_half(obj: SceneObject) -> float:
    # A yawed cube's circumscribed circle: conservative for any yaw.
    return obj.size[0] / 2 * (math.sqrt(2) if obj.shape == "cube" else 1.0)


def _gap(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Separation of two xy boxes: positive when apart along at least one axis."""
    return max(abs(a[0] - b[0]) - a[2] - b[2], abs(a[1] - b[1]) - a[3] - b[3])


def _on_table(fp: tuple[float, ...]) -> bool:
    return (TABLE_X[0] <= fp[0] - fp[2] and fp[0] + fp[2] <= TABLE_X[1]
            and TABLE_Y[0] <= fp[1] - fp[3] and fp[1] + fp[3] <= TABLE_Y[1])


def _bin_fp(b: Bin) -> tuple[float, ...]:
    return (b.floor_center[0], b.floor_center[1], b.inner_size_xy[0] / 2 + b.wall_thickness,
            b.inner_size_xy[1] / 2 + b.wall_thickness)


def _place(rng: np.random.Generator, xs: tuple[float, float], ys: tuple[float, float], hx: float, hy: float,
           keep: list[tuple[tuple[float, ...], float]], ok: Callable[[float, float], bool] = lambda x, y: True,
           what: str = "object") -> tuple[float, float]:
    """Rejection-sample a center whose footprint is on the table and clears every (footprint, gap) in keep."""
    for _ in range(_TRIES):
        x, y = float(rng.uniform(*xs)), float(rng.uniform(*ys))
        fp = (x, y, hx, hy)
        if _on_table(fp) and ok(x, y) and all(_gap(fp, other) > g for other, g in keep):
            return x, y
    raise GenerationRejected(f"no_room_for_{what}")


def _id(rng: np.random.Generator) -> str:
    return f"o_{rng.integers(0, 2**64, dtype=np.uint64):016x}"


def _span(side: int, lo: float, hi: float) -> tuple[float, float]:
    """y range [lo, hi] on the bin's side (side=+1) or mirrored (side=-1)."""
    return (lo, hi) if side > 0 else (-hi, -lo)


# ---- scene checks -------------------------------------------------------------------------------
def _aabb3(center: tuple[float, ...], half: tuple[float, ...]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    return tuple(c - h for c, h in zip(center, half)), tuple(c + h for c, h in zip(center, half))


def _apart(a, b) -> bool:
    return any(a[1][i] <= b[0][i] or b[1][i] <= a[0][i] for i in range(3))


def scene_issues(scene: Scene) -> list[str]:
    """D29 generator checklist violations (empty when the scene is valid)."""
    issues = []
    b, z0 = scene.bin, scene.table_top_z
    if abs(b.floor_center[2] - (z0 + b.wall_thickness)) > 1e-9:
        issues.append("bin floor does not sit on the table")
    boxes = {"bin": _aabb3((b.floor_center[0], b.floor_center[1], z0 + (b.wall_thickness + b.wall_height) / 2),
                           (_bin_fp(b)[2], _bin_fp(b)[3], (b.wall_thickness + b.wall_height) / 2))}
    if not _on_table(_bin_fp(b)):
        issues.append("bin is off the table")
    for o in scene.objects:
        if abs(o.pose.position[2] - o.size[2] / 2 - z0) > 1e-9:
            issues.append(f"{o.id} does not rest on the table")
        if not GRASP_HEIGHT_M[0] <= o.size[2] <= min(GRASP_HEIGHT_M[1], b.wall_height):
            issues.append(f"{o.id} height {o.size[2]:.3f} m is outside the graspable window")
        if max(o.size[:2]) >= GRIPPER_OPENING_M:
            issues.append(f"{o.id} is wider than the gripper opening")
        r = _object_half(o)
        if not _on_table((o.pose.position[0], o.pose.position[1], r, r)):
            issues.append(f"{o.id} is off the table")
        boxes[o.id] = _aabb3(o.pose.position, (r, r, o.size[2] / 2))
    for ob in scene.obstacles:
        if not _on_table((ob.pose.position[0], ob.pose.position[1], ob.half_extents[0], ob.half_extents[1])):
            issues.append(f"{ob.id} is off the table")
        boxes[ob.id] = _aabb3(ob.pose.position, ob.half_extents)
    movable = {o.id for o in scene.objects}
    names = list(boxes)
    for i, a in enumerate(names):
        for c in names[i + 1:]:
            # Static bodies (obstacles, bin) may touch each other, e.g. enclosure panels; objects never start in contact.
            if (a in movable or c in movable or "bin" in (a, c)) and not _apart(boxes[a], boxes[c]):
                issues.append(f"initial contact between {a} and {c}")
    return issues


# ---- impossibility proofs ----------------------------------------------------------------------
def _cavity_faces(params: dict[str, float]):
    """(axis, plane value, outward sign, tangential bounds) for the 4 sides and the lid of the stored cavity."""
    x0, x1, y0, y1, z0, z1 = (params[k] for k in ("x0", "x1", "y0", "y1", "z0", "z1"))
    return [(0, x1, 1, {1: (y0, y1), 2: (z0, z1)}), (0, x0, -1, {1: (y0, y1), 2: (z0, z1)}),
            (1, y1, 1, {0: (x0, x1), 2: (z0, z1)}), (1, y0, -1, {0: (x0, x1), 2: (z0, z1)}),
            (2, z1, 1, {0: (x0, x1), 1: (y0, y1)})]


def _sealed(scene: Scene, target: SceneObject, proof: ImpossibilityProof) -> bool:
    """Every closed face of the stored cavity lies inside one solid panel and the floor is the table: no gap at all,
    so no finger of any positive thickness can reach the target without a collision."""
    prm, eps = proof.params, 1e-9
    if not all(k in prm for k in ("x0", "x1", "y0", "y1", "z0", "z1")):
        return False
    panels = proof.enclosure
    if (not set(panels) <= set(scene.obstacles) or any(o.pose.orientation != (0., 0., 0., 1.) for o in panels)
            or abs(prm["z0"] - scene.table_top_z) > eps
            or not _on_table(((prm["x0"] + prm["x1"]) / 2, (prm["y0"] + prm["y1"]) / 2,
                              (prm["x1"] - prm["x0"]) / 2, (prm["y1"] - prm["y0"]) / 2))):
        return False
    r = _object_half(target)
    lo = (target.pose.position[0] - r, target.pose.position[1] - r, target.pose.position[2] - target.size[2] / 2)
    hi = (target.pose.position[0] + r, target.pose.position[1] + r, target.pose.position[2] + target.size[2] / 2)
    cav_lo, cav_hi = (prm["x0"], prm["y0"], prm["z0"]), (prm["x1"], prm["y1"], prm["z1"])
    if not all(cav_lo[i] - eps <= lo[i] and hi[i] <= cav_hi[i] + eps for i in range(3)):
        return False
    for axis, v, sign, tangent in _cavity_faces(prm):
        def covers(o: Obstacle) -> bool:
            plo, phi = _aabb3(o.pose.position, o.half_extents)
            outward = phi[axis] - v if sign > 0 else v - plo[axis]
            return (plo[axis] - eps <= v <= phi[axis] + eps and outward > eps
                    and all(plo[k] - eps <= a and b <= phi[k] + eps for k, (a, b) in tangent.items()))
        if not any(covers(o) for o in panels):
            return False
    return True


def check_impossibility(task: TaskSpec, oracle: TaskOracle) -> bool:
    """Re-derive the impossibility proof from the scene alone; False when it does not hold."""
    proof = oracle.proof
    if oracle.feasibility != "impossible" or proof is None:
        return False
    try:
        referents = resolve_referent(task.instruction, task.scene)
    except ValueError:
        return False
    if proof.type == 4:
        return referents == []
    if referents != [oracle.target_id]:
        return False
    target = next(o for o in task.scene.objects if o.id == oracle.target_id)
    bound = reach_bound_m() + REACH_MARGIN_M
    if proof.type == 1:
        return target_clearance_m(target.pose.position, target.size) > bound
    if proof.type == 2:
        return bin_clearance_m(task.scene.bin.floor_center, task.scene.bin.inner_size_xy) > bound
    return _sealed(task.scene, target, proof)


# ---- construction -------------------------------------------------------------------------------
def _object(rng: np.random.Generator, color: str, shape: str, x: float, y: float, edge: float, height: float,
            z0: float) -> SceneObject:
    yaw = float(rng.uniform(-math.pi / 4, math.pi / 4)) if shape == "cube" else 0.0
    size = (edge, edge, edge) if shape == "cube" else (edge, edge, height)
    return SceneObject(id=_id(rng), shape=shape, color=color, size=size,
                       pose=Pose(position=(x, y, z0 + size[2] / 2),
                                 orientation=(0., 0., math.sin(yaw / 2), math.cos(yaw / 2))))


def _enclosure(rng: np.random.Generator, target: SceneObject, z0: float) -> tuple[tuple[Obstacle, ...], dict[str, float]]:
    """Four walls on the table and a lid over a cavity around the target; corners overlap by construction."""
    x, y, _ = target.pose.position
    h = _object_half(target) + CAVITY_CLEARANCE_M
    cav = {"x0": x - h, "x1": x + h, "y0": y - h, "y1": y + h, "z0": z0, "z1": z0 + target.size[2] + CAVITY_CLEARANCE_M}
    t, zh = PANEL_M, (cav["z1"] - z0) / 2

    def box(c: tuple[float, float, float], half: tuple[float, float, float]) -> Obstacle:
        return Obstacle(id=_id(rng), half_extents=half, pose=Pose(position=c))
    panels = (box((cav["x1"] + t / 2, y, z0 + zh), (t / 2, h + t, zh)),
              box((cav["x0"] - t / 2, y, z0 + zh), (t / 2, h + t, zh)),
              box((x, cav["y1"] + t / 2, z0 + zh), (h, t / 2, zh)),
              box((x, cav["y0"] - t / 2, z0 + zh), (h, t / 2, zh)),
              box((x, y, cav["z1"] + t / 2), (h + t, h + t, t / 2)))
    return panels, cav


def draft_task(seed: int, kind: TaskKind, impossible_type: int | None = None, *,
               distractors: int | None = None, obstacles: int | None = None) -> Draft:
    """Sample geometry, instruction and label without simulation. Raises GenerationRejected on a dead end.

    distractors/obstacles (0-3) pin the stratum; the counts are drawn regardless so the stream stays the same."""
    _check_args(seed, kind, impossible_type)
    t = impossible_type or 0
    rng = np.random.default_rng([seed, t])  # never the global RNG; kind/type select independent streams
    n_d, n_o = int(rng.integers(0, 4)), int(rng.integers(0, 4))
    n_d = n_d if distractors is None else distractors
    n_o = n_o if obstacles is None else obstacles
    if not (0 <= n_d <= 3 and 0 <= n_o <= 3):
        raise ValueError("distractors and obstacles must be in 0..3")
    z0, side = 0.0, int(rng.choice((-1, 1)))
    bound = reach_bound_m() + REACH_MARGIN_M
    half_bin = BIN_INNER_M / 2 + BIN_THICKNESS_M
    if t == 2:
        bx, by = _place(rng, (.75, 1.06), _span(side, .40, .71), half_bin, half_bin, [],
                        ok=lambda x, y: bin_clearance_m((x, y, 0.), (BIN_INNER_M,) * 2) > bound + .01, what="bin")
    else:
        bx, by = _place(rng, (.40, .55), _span(side, .26, .33), half_bin, half_bin, [], what="bin")
    bin_ = Bin(floor_center=(bx, by, z0 + BIN_THICKNESS_M), inner_size_xy=(BIN_INNER_M, BIN_INNER_M),
               wall_height=BIN_WALL_M, wall_thickness=BIN_THICKNESS_M)
    bin_fp = _bin_fp(bin_)

    color, shape = str(rng.choice(COLORS)), str(rng.choice(SHAPES))
    edge, height = (float(v) for v in rng.uniform(*OBJECT_EDGE_M, size=2))
    r = edge / 2 * (math.sqrt(2) if shape == "cube" else 1.0)
    reach_r = r + PANEL_M + CAVITY_CLEARANCE_M if t == 3 else r  # an enclosure widens the target's footprint
    if t == 1:
        tx, ty = _place(rng, (.70, 1.12), (-.76, .76), r, r, [(bin_fp, HAND_GAP_M)],
                        ok=lambda x, y: target_clearance_m((x, y, z0 + edge / 2), (edge,) * 3) > bound + .01,
                        what="target")
    else:
        tx, ty = _place(rng, (.38, .66), _span(side, -.30, -.05), reach_r, reach_r, [(bin_fp, HAND_GAP_M)],
                        what="target")
    target = _object(rng, color, shape, tx, ty, edge, height, z0)
    target_fp = (tx, ty, reach_r, reach_r)
    objects = [target]
    keep_objects: list[tuple[tuple[float, ...], float]] = [(bin_fp, OBJECT_GAP_M), (target_fp, OBJECT_GAP_M)]
    for _ in range(n_d):
        # Spec §6 distractors: the same shape in another color, or the same color in another shape.
        if rng.random() < .5:
            dc, ds = str(rng.choice([c for c in COLORS if c != color])), shape
        else:
            dc, ds = color, next(s for s in SHAPES if s != shape)
        de, dh = (float(v) for v in rng.uniform(*OBJECT_EDGE_M, size=2))
        dr = de / 2 * (math.sqrt(2) if ds == "cube" else 1.0)
        dx, dy = _place(rng, (.38, .66), _span(side, -.30, .02), dr, dr, keep_objects, what="distractor")
        objects.append(_object(rng, dc, ds, dx, dy, de, dh, z0))
        keep_objects.append(((dx, dy, dr, dr), OBJECT_GAP_M))

    obstacle_list: list[Obstacle] = []
    proof_params: dict[str, float] = {}
    enclosure: tuple[Obstacle, ...] = ()
    if t == 3:
        enclosure, proof_params = _enclosure(rng, target, z0)
        obstacle_list.extend(enclosure)
    keep_obstacles = [(bin_fp, HAND_GAP_M), (target_fp, HAND_GAP_M)]
    keep_obstacles += [((o.pose.position[0], o.pose.position[1], _object_half(o), _object_half(o)), OBSTACLE_GAP_M)
                       for o in objects[1:]]
    for k in range(n_o):
        if k == 0:  # a wall between the objects and the bin, as in the pilot's obstacle stratum
            half = (float(rng.uniform(.08, .15)), .01, float(rng.uniform(.04, .08)))
            xs, ys = (.42, .58), _span(side, .06, .09)
        else:
            half = (float(rng.uniform(.02, .04)), float(rng.uniform(.02, .04)), float(rng.uniform(.03, .07)))
            xs, ys = (.33, .72), _span(side, -.38, .10)
        ox, oy = _place(rng, xs, ys, half[0], half[1], keep_obstacles, what="obstacle")
        obstacle_list.append(Obstacle(id=_id(rng), half_extents=half, pose=Pose(position=(ox, oy, z0 + half[2]))))
        keep_obstacles.append(((ox, oy, half[0], half[1]), OBSTACLE_GAP_M))

    absent: tuple[str, str] | None = None
    if t == 4:  # name a color/shape pair that no object has; the target object is just one more object
        present = {(o.color, o.shape) for o in objects}
        absent_pairs = [(c, s) for c in COLORS for s in SHAPES if (c, s) not in present]
        absent = absent_pairs[int(rng.integers(len(absent_pairs)))]
    order = list(rng.permutation(len(objects)))  # target index is not a label shortcut
    scene = Scene(table_top_z=z0, objects=tuple(objects[i] for i in order), obstacles=tuple(obstacle_list), bin=bin_)
    issues = scene_issues(scene)
    if issues:
        raise GenerationRejected(f"scene_check: {issues[0]}")

    if absent is not None:
        template, text, target_id = "color", instruction("color", *absent), None
    else:
        # Every phrasing that resolves to exactly the target; the color one always does (distractors differ in one
        # attribute), so the list is never empty. Ambiguous or absent phrasings are discarded (spec §6).
        phrasings = [("color", (color, shape)), ("shape", (shape,)), ("nearest_bin", ()), ("leftmost", ())]
        phrasings += [("next_to", (o.color, o.shape)) for o in objects[1:]]
        options = [(tmpl, instruction(tmpl, *a)) for tmpl, a in dict.fromkeys(phrasings)
                   if resolve_referent(instruction(tmpl, *a), scene) == [target.id]]
        template, text = options[int(rng.integers(len(options)))]
        target_id = target.id
    proof = None if t == 0 else ImpossibilityProof(type=t, refusal_reason=IMPOSSIBLE_REFUSALS[t - 1],
                                                   params=_proof_params(t, scene, target, proof_params),
                                                   enclosure=enclosure)
    task = TaskSpec(instruction=text, scene=scene,
                    robot_facts=RobotFacts(base_pose=scene.robot_base, reach_bound_m=reach_bound_m(),
                                           gripper_opening_m=GRIPPER_OPENING_M, table_height_m=z0))
    return Draft(task=task, target_id=target_id, proof=proof, template=template)


def _proof_params(t: int, scene: Scene, target: SceneObject, cavity: dict[str, float]) -> dict[str, float]:
    if t == 1:
        return {"clearance_m": target_clearance_m(target.pose.position, target.size),
                "reach_bound_m": reach_bound_m(), "margin_m": REACH_MARGIN_M}
    if t == 2:
        return {"clearance_m": bin_clearance_m(scene.bin.floor_center, scene.bin.inner_size_xy),
                "reach_bound_m": reach_bound_m(), "margin_m": REACH_MARGIN_M}
    return dict(cavity)


def _check_args(seed: int, kind: str, impossible_type: int | None) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if kind == "feasible" and impossible_type is not None:
        raise ValueError("feasible tasks take impossible_type=None")
    if kind == "impossible" and impossible_type not in (1, 2, 3, 4):
        raise ValueError("impossible tasks need impossible_type in 1..4 (spec §6)")
    if kind not in ("feasible", "impossible"):
        raise ValueError(f"unknown kind {kind!r}")


def witness_attempt(task: TaskSpec, target_id: str, *, seed: int = 0,
                    config_snapshot: dict[str, dict] | None = None) -> AttemptResult:
    """The unchanged heuristic plan through run_attempt (verifier, physics, independent Judge), with its execution
    record (the sigma_thr calibration reads it)."""
    from vplan.pipeline import run_attempt
    from vplan.planners.heuristic import heuristic_geometry

    plan = heuristic_geometry(task.scene, target_id)
    return run_attempt(plan, task.scene, target_id, seed=seed % (2**32), config_snapshot=config_snapshot)


def witness_evaluation(result: AttemptResult) -> WitnessEvaluation:
    """The part of a witness attempt the generator keeps."""
    return WitnessEvaluation(plan=result.plan, outcome=result.outcome, findings=tuple(result.findings),
                             result=result.judge)


def run_witness(task: TaskSpec, target_id: str, *, seed: int = 0,
                config_snapshot: dict[str, dict] | None = None) -> WitnessEvaluation:
    """The heuristic witness as the generator stores it."""
    return witness_evaluation(witness_attempt(task, target_id, seed=seed, config_snapshot=config_snapshot))


def witness_rejection(witness: WitnessEvaluation, target_id: str) -> str | None:
    """Why a witness cannot label a task feasible, or None when it can (same rules as prepare_cohort)."""
    if witness.outcome != "executed" or witness.findings:
        return "witness_blocked"
    if witness.result is None:
        return "missing_judge_result"
    if not witness.result.success:
        return "judge_failure"
    if witness.plan.target_id != target_id:
        return "witness_target_mismatch"
    return None


def generate_task(seed: int, kind: TaskKind, impossible_type: int | None = None, *,
                  distractors: int | None = None, obstacles: int | None = None,
                  witness_evaluator: WitnessEvaluator | None = None,
                  config_snapshot: dict[str, dict] | None = None) -> tuple[TaskSpec, TaskOracle] | None:
    """One deterministic task and its private oracle, or None when the seed yields neither witness nor proof."""
    _check_args(seed, kind, impossible_type)
    try:
        draft = draft_task(seed, kind, impossible_type, distractors=distractors, obstacles=obstacles)
    except GenerationRejected:
        return None
    if draft.proof is not None:
        oracle = TaskOracle(target_id=draft.target_id, feasibility="impossible", proof=draft.proof)
        return (draft.task, oracle) if check_impossibility(draft.task, oracle) else None
    assert draft.target_id is not None
    evaluate = witness_evaluator or (lambda task, target: run_witness(task, target, seed=seed,
                                                                       config_snapshot=config_snapshot))
    witness = evaluate(draft.task, draft.target_id)
    if witness_rejection(witness, draft.target_id) is not None:
        return None
    return draft.task, TaskOracle(target_id=draft.target_id, reference_plan=witness.plan,
                                  reference_result=witness.result)
