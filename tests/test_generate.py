"""W2.2 task generator: feasible and impossible-by-construction tasks, referents, reach bound (spec §6)."""
import itertools
import math

import numpy as np
import pytest
from pydantic import ValidationError

from vplan import records
from vplan.records import Bin, Obstacle, Pose, Scene, SceneObject

IMPOSSIBLE = (1, 2, 3, 4)
KINDS = [("feasible", None)] + [("impossible", t) for t in IMPOSSIBLE]


def _fake_witness(task, target_id):
    from vplan.planners.heuristic import heuristic_geometry
    return records.WitnessEvaluation(
        plan=heuristic_geometry(task.scene, target_id), outcome="executed",
        result=records.JudgeResult(success=True, predicates={"S1": True}, violations=(), details={}))


def _gen(seed, kind, impossible_type):
    from vplan.tasks import generate_task
    return generate_task(seed, kind, impossible_type, witness_evaluator=_fake_witness)


# ---- reach bound ------------------------------------------------------------------------------
def test_reach_bound_is_derived_from_the_panda_model_and_bounds_sampled_tcp_distances():
    from vplan.robot import load_panda
    from vplan.tasks import SHOULDER, reach_bound_m

    bound = reach_bound_m()
    assert bound == reach_bound_m()  # computed once
    # Nominal Panda reach is 0.855 m to the flange; the grasp target adds ~0.1 m along the last axis.
    assert .90 < bound < 1.0
    m = load_panda()
    rng = np.random.default_rng(123)
    for _ in range(200):  # independent kinematics (ManipulaPy, not the PyBullet sampler)
        tcp = m.fk(rng.uniform(m.lower, m.upper))[:3, 3]
        assert np.linalg.norm(tcp - np.asarray(SHOULDER)) <= bound + 1e-6
    assert SHOULDER == pytest.approx((0., 0., .333))


def test_pilot_candidates_carry_the_derived_reach_bound():
    from vplan.tasks import generate_candidate, reach_bound_m
    assert generate_candidate(0, "clear").task.robot_facts.reach_bound_m == reach_bound_m()


# ---- referents ---------------------------------------------------------------------------------
def _obj(i, color, shape, x, y):
    return SceneObject(id=i, shape=shape, color=color, size=(.04, .04, .04), pose=Pose(position=(x, y, .02)))


def _scene(*objects):
    return Scene(table_top_z=0., objects=objects,
                 bin=Bin(floor_center=(.45, .28, .01), inner_size_xy=(.16, .16), wall_height=.08))


def test_resolve_referent_templates():
    from vplan.tasks import resolve_referent

    s = _scene(_obj("a", "red", "cube", .50, -.15), _obj("b", "red", "cylinder", .60, -.05),
               _obj("c", "green", "cube", .45, .05))
    say = "Pick the {} and place it in the bin."
    assert resolve_referent(say.format("red cube"), s) == ["a"]
    assert resolve_referent(say.format("cylinder"), s) == ["b"]
    assert sorted(resolve_referent(say.format("cube"), s)) == ["a", "c"]  # ambiguous
    assert resolve_referent(say.format("blue cube"), s) == []  # absent
    assert resolve_referent(say.format("object nearest the bin"), s) == ["c"]
    assert resolve_referent("Pick the leftmost object and place it in the bin.", s) == ["c"]  # +y is the robot's left
    assert resolve_referent(say.format("object next to the red cylinder"), s) == ["a"]  # 0.14 m vs 0.18 m
    assert resolve_referent(say.format("object next to the red cube"), s) == ["b"]
    assert resolve_referent(say.format("object next to the blue cube"), s) == []  # absent anchor
    with pytest.raises(ValueError, match="template"):
        resolve_referent("Throw the red cube.", s)


def test_resolve_referent_spatial_ties_are_ambiguous():
    from vplan.tasks import resolve_referent
    s = _scene(_obj("a", "red", "cube", .50, .02), _obj("b", "blue", "cube", .60, .03))
    assert sorted(resolve_referent("Pick the leftmost object and place it in the bin.", s)) == ["a", "b"]


def test_resolve_instruction_still_serves_the_pilot_template():
    from vplan.tasks import resolve_instruction
    s = _scene(_obj("a", "red", "cube", .50, -.15))
    assert resolve_instruction("Pick the red cube and place it in the bin.", s) == "a"
    with pytest.raises(ValueError, match="exactly one"):
        resolve_instruction("Pick the blue cube and place it in the bin.", s)


# ---- generation -------------------------------------------------------------------------------
@pytest.mark.parametrize("kind,impossible_type", [("feasible", 1), ("impossible", None), ("impossible", 5),
                                                  ("other", None)])
def test_generate_task_rejects_invalid_kinds(kind, impossible_type):
    with pytest.raises(ValueError):
        _gen(0, kind, impossible_type)


def test_generation_is_deterministic_and_never_uses_the_global_rng():
    for seed, (kind, t) in itertools.product(range(4), KINDS):
        np.random.seed(seed)
        first = _gen(seed, kind, t)
        np.random.seed(seed + 99)
        assert _gen(seed, kind, t) == first


def test_generated_tasks_follow_the_referent_rule_and_their_labels():
    from vplan.tasks import resolve_referent
    produced = {k: 0 for k in KINDS}
    for seed, (kind, t) in itertools.product(range(12), KINDS):
        out = _gen(seed, kind, t)
        if out is None:
            continue
        produced[(kind, t)] += 1
        spec, oracle = out
        assert isinstance(spec, records.TaskSpec) and isinstance(oracle, records.TaskOracle)
        referents = resolve_referent(spec.instruction, spec.scene)
        assert oracle.feasibility == kind
        if t == 4:
            assert referents == [] and oracle.target_id is None
        else:
            assert referents == [oracle.target_id]
        if kind == "feasible":
            assert oracle.proof is None and oracle.reference_plan.target_id == oracle.target_id
        else:
            assert oracle.reference_plan is None and oracle.proof.type == t
            assert oracle.proof.refusal_reason == ("target_unreachable", "destination_unreachable",
                                                   "target_enclosed", "no_matching_object")[t - 1]
    assert all(n >= 9 for n in produced.values()), produced


def test_every_impossibility_proof_check_passes_and_rejects_feasible_scenes():
    from vplan.tasks import check_impossibility
    feasible = [o for o in (_gen(s, "feasible", None) for s in range(6)) if o is not None]
    for seed, t in itertools.product(range(12), IMPOSSIBLE):
        out = _gen(seed, "impossible", t)
        if out is None:
            continue
        spec, oracle = out
        assert check_impossibility(spec, oracle)
        for fspec, foracle in feasible:  # the same proof applied to a feasible scene must fail
            target = None if t == 4 else foracle.target_id
            fake = records.TaskOracle(target_id=target, feasibility="impossible", proof=oracle.proof)
            assert not check_impossibility(fspec, fake)


def test_reach_proofs_hold_against_the_derived_bound():
    from vplan.tasks import REACH_MARGIN_M, SHOULDER, reach_bound_m
    for seed in range(12):
        out = _gen(seed, "impossible", 1)
        if out is not None:
            spec, oracle = out
            target = next(o for o in spec.scene.objects if o.id == oracle.target_id)
            near = math.dist(target.pose.position, SHOULDER) - math.dist((0, 0, 0), target.size) / 2
            assert near > reach_bound_m() + REACH_MARGIN_M
        out = _gen(seed, "impossible", 2)
        if out is not None:
            b = out[0].scene.bin
            near = math.hypot(*b.floor_center[:2]) - math.hypot(*b.inner_size_xy) / 2
            assert near > reach_bound_m() + REACH_MARGIN_M


def test_enclosure_is_sealed_with_no_gaps():
    for seed in range(6):
        out = _gen(seed, "impossible", 3)
        if out is None:
            continue
        spec, oracle = out
        enclosure = oracle.proof.enclosure
        assert len(enclosure) == 5 and set(enclosure) <= set(spec.scene.obstacles)
        target = next(o for o in spec.scene.objects if o.id == oracle.target_id)
        lo = [min(o.pose.position[i] - o.half_extents[i] for o in enclosure) for i in range(3)]
        hi = [max(o.pose.position[i] + o.half_extents[i] for o in enclosure) for i in range(3)]
        assert all(lo[i] < target.pose.position[i] < hi[i] for i in range(3))
        assert lo[2] == pytest.approx(spec.scene.table_top_z)
        # Removing any one panel must break the seal: the proof checks every face, not just a count.
        from vplan.tasks import check_impossibility
        for drop in enclosure:
            scene = Scene.model_validate({**spec.scene.model_dump(),
                                          "obstacles": [o.model_dump() for o in spec.scene.obstacles if o != drop]})
            assert not check_impossibility(spec.model_copy(update={"scene": scene}), oracle)


def _footprint(scene, item):
    if isinstance(item, SceneObject):
        r = item.size[0] / 2 * (math.sqrt(2) if item.shape == "cube" else 1)
        return item.pose.position, (r, r, item.size[2] / 2)
    if isinstance(item, Obstacle):
        return item.pose.position, item.half_extents
    b = scene.bin
    return (b.floor_center, (b.inner_size_xy[0] / 2 + b.wall_thickness, b.inner_size_xy[1] / 2 + b.wall_thickness,
                             b.wall_height))


def test_generated_scenes_satisfy_the_d29_checklist():
    from vplan.tasks import scene_issues
    for seed, (kind, t) in itertools.product(range(12), KINDS):
        out = _gen(seed, kind, t)
        if out is None:
            continue
        spec = out[0]
        scene = spec.scene
        assert scene_issues(scene) == []
        assert scene.bin.floor_center[2] == pytest.approx(scene.table_top_z + scene.bin.wall_thickness)
        assert 0 <= len(scene.objects) - 1 <= 3
        assert len(scene.obstacles) <= (8 if t == 3 else 3)
        for o in scene.objects:
            assert .036 <= o.size[2] <= min(.059, scene.bin.wall_height)
            assert max(o.size[:2]) < spec.robot_facts.gripper_opening_m
            assert o.pose.position[2] - o.size[2] / 2 == pytest.approx(scene.table_top_z)
        enclosure = set(out[1].proof.enclosure) if t == 3 else set()
        items = [*scene.objects, *[o for o in scene.obstacles if o not in enclosure], "bin"]
        for a, b in itertools.combinations(items, 2):
            (pa, ha), (pb, hb) = _footprint(scene, a), _footprint(scene, b)
            assert any(abs(pa[i] - pb[i]) > ha[i] + hb[i] for i in (0, 1)), (seed, kind, t, a, b)
        for item in items:
            p, h = _footprint(scene, item)
            assert -.25 <= p[0] - h[0] and p[0] + h[0] <= 1.15 and -.8 <= p[1] - h[1] and p[1] + h[1] <= .8


def test_scene_issues_reports_violations():
    from vplan.tasks import scene_issues
    s = _scene(_obj("a", "red", "cube", .50, -.15), _obj("b", "red", "cube", .52, -.15))
    assert any("contact" in i for i in scene_issues(s))
    floating = Scene.model_validate({**s.model_dump(), "objects": [s.objects[0].model_dump()],
                                     "bin": {**s.bin.model_dump(), "floor_center": (.45, .28, .05)}})
    assert any("bin" in i for i in scene_issues(floating))


def test_oracle_label_contract():
    from vplan.planners.heuristic import heuristic_geometry
    s = _scene(_obj("a", "red", "cube", .50, -.15))
    plan = heuristic_geometry(s, "a")
    ok = records.JudgeResult(success=True, predicates={"S1": True}, violations=(), details={})
    proof = records.ImpossibilityProof(type=4, refusal_reason="no_matching_object")
    records.TaskOracle(target_id="a", reference_plan=plan, reference_result=ok)  # pilot form still valid
    records.TaskOracle(target_id=None, feasibility="impossible", proof=proof)
    with pytest.raises(ValidationError):
        records.TaskOracle(target_id="a", feasibility="impossible", proof=proof, reference_plan=plan,
                           reference_result=ok)
    with pytest.raises(ValidationError):
        records.TaskOracle(target_id="a", feasibility="impossible")
    with pytest.raises(ValidationError):
        records.TaskOracle(target_id=None, reference_plan=plan, reference_result=ok)
    with pytest.raises(ValidationError):
        records.ImpossibilityProof(type=1, refusal_reason="no_matching_object")


def test_generate_task_returns_none_when_the_witness_fails():
    from vplan.tasks import generate_task
    def failing(task, target_id):
        w = _fake_witness(task, target_id)
        return w.model_copy(update={"result": w.result.model_copy(update={"success": False})})
    assert all(generate_task(s, "feasible", None, witness_evaluator=failing) is None for s in range(3))


# a cube, a cylinder, and a cylinder rejected before D43 (S1 at_rest: the built-in cylinder contact artifact)
@pytest.mark.parametrize("seed", [1, 5, 8])
def test_feasible_witness_passes_run_attempt_with_judge_success(seed):
    """Real simulation: the stored witness replays to a Judge success under the same snapshot."""
    from vplan.pipeline import run_attempt
    from vplan.tasks import generate_task
    out = generate_task(seed, "feasible", None)
    assert out is not None, "accepted in the R6 spike (seed 8 since D43)"
    spec, oracle = out
    assert oracle.reference_result.success
    replay = run_attempt(oracle.reference_plan, spec.scene, oracle.target_id, seed=seed)
    assert replay.outcome == "executed" and replay.judge is not None and replay.judge.success
