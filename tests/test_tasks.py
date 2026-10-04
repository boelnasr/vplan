"""Development-cohort contracts; oracle truth never crosses the public task boundary."""
import itertools

import pytest
from pydantic import ValidationError

from vplan import records


def test_task_api_is_available():
    assert hasattr(records, "TaskSpec"), "public task contract is missing"
    assert hasattr(records, "TaskOracle"), "private oracle contract is missing"


def test_candidates_are_deterministic_public_and_unambiguous():
    from vplan.tasks import generate_candidate, resolve_instruction

    colors, positions, bins, target_slots = set(), set(), set(), set()
    for seed, stratum in itertools.product(range(12), ("clear", "distractor", "obstacle")):
        candidate = generate_candidate(seed, stratum)
        assert candidate == generate_candidate(seed, stratum)
        task = candidate.task
        assert set(task.model_dump()) == {"instruction", "scene", "robot_facts"}
        assert resolve_instruction(task.instruction, task.scene) == candidate.target_id
        target = next(o for o in task.scene.objects if o.id == candidate.target_id)
        assert all(o.id.startswith("o_") and o.color not in o.id and o.shape not in o.id
                   for o in task.scene.objects)
        colors.add(target.color)
        positions.add(target.pose.position)
        bins.add(task.scene.bin.floor_center)
        target_slots.add(task.scene.objects.index(target))
        with pytest.raises(ValidationError):
            records.TaskSpec.model_validate({**task.model_dump(), "target_id": candidate.target_id})
        with pytest.raises(ValidationError):
            task.instruction = "changed"
        if stratum == "clear":
            assert len(task.scene.objects) == 1 and not task.scene.obstacles
        if stratum == "distractor":
            assert any(o.color == target.color and o.shape != target.shape for o in task.scene.objects)
        if stratum == "obstacle":
            assert task.scene.obstacles
    assert len(colors) >= 4 and len(positions) > 10 and len(bins) > 10
    assert len(target_slots) > 1


def test_candidate_geometry_has_supported_nonoverlapping_objects_and_bins():
    from vplan.tasks import generate_candidate

    for seed, stratum in itertools.product(range(30), ("clear", "distractor", "obstacle")):
        scene = generate_candidate(seed, stratum).task.scene
        assert scene.bin.floor_center[2] - scene.bin.wall_thickness == pytest.approx(scene.table_top_z)
        boxes = []
        for obj in scene.objects:
            assert obj.pose.position[2] - obj.size[2] / 2 == pytest.approx(scene.table_top_z)
            assert .036 <= obj.size[2] <= .059  # documented center top-grasp height window (D29 I-6)
            boxes.append((obj.pose.position, tuple(s / 2 for s in obj.size)))
        boxes.extend((o.pose.position, o.half_extents) for o in scene.obstacles)
        b = scene.bin
        boxes.append((b.floor_center, (b.inner_size_xy[0] / 2 + b.wall_thickness,
                                       b.inner_size_xy[1] / 2 + b.wall_thickness, b.wall_height)))
        for center, half in boxes:
            assert -.25 <= center[0] - half[0] < center[0] + half[0] <= 1.15
            assert -.8 <= center[1] - half[1] < center[1] + half[1] <= .8
        for (a, ah), (b, bh) in itertools.combinations(boxes, 2):
            assert any(abs(a[i] - b[i]) > ah[i] + bh[i] for i in (0, 1))


def test_resolver_rejects_absent_or_ambiguous_referents():
    from vplan.tasks import make_basic_scene, resolve_instruction

    scene = make_basic_scene()
    with pytest.raises(ValueError, match="exactly one"):
        resolve_instruction("Pick the blue cube and place it in the bin.", scene)
    duplicate = scene.model_dump()
    duplicate["objects"][1]["color"] = "red"
    with pytest.raises(ValueError, match="exactly one"):
        resolve_instruction("Pick the red cube and place it in the bin.", records.Scene.model_validate(duplicate))


def _witness(candidate, *, success=True, outcome="executed"):
    from vplan.planners.heuristic import heuristic_geometry

    return records.WitnessEvaluation(
        plan=heuristic_geometry(candidate.task.scene, candidate.target_id), outcome=outcome,
        result=records.JudgeResult(success=success, predicates={"S1": success},
                                   violations=() if success else ("S1",), details={}))


def test_prepare_accounts_for_rejected_seeds_and_balances_strata():
    from vplan.tasks import prepare_cohort

    def evaluate(candidate):
        if candidate.metadata.seed == 11:
            raise RuntimeError("physics unavailable")
        return _witness(candidate, success=candidate.metadata.seed != 10)

    cohort = prepare_cohort(n=3, seed=10, max_candidates=6, witness_evaluator=evaluate)
    assert cohort.attempted == 5
    assert [t.metadata.seed for t in cohort.tasks] == [12, 13, 14]
    assert [t.metadata.stratum for t in cohort.tasks] == ["clear", "distractor", "obstacle"]
    assert [e.metadata.seed for e in cohort.exclusions] == [10, 11]
    assert cohort.exclusions[0].reason == "judge_failure"
    assert cohort.exclusions[1].reason.startswith("simulation_error: RuntimeError")
    assert all(t.oracle.reference_result.success for t in cohort.tasks)
    assert all(t.oracle.reference_plan.target_id == t.oracle.target_id for t in cohort.tasks)


def test_prepare_never_accepts_a_blocked_witness_and_failure_retains_exclusions():
    from vplan.tasks import CohortPreparationError, prepare_cohort

    with pytest.raises(CohortPreparationError, match="0/2") as caught:
        prepare_cohort(n=2, seed=0, max_candidates=3,
                       witness_evaluator=lambda c: _witness(c, outcome="blocked"))
    partial = caught.value.cohort
    assert not partial.tasks and partial.attempted == 3
    assert [e.metadata.seed for e in partial.exclusions] == [0, 1, 2]
    assert all(e.reason == "witness_blocked" for e in partial.exclusions)


def test_prepare_binds_config_hashes_and_rejects_mid_run_change(monkeypatch):
    from vplan.config import config_hash, config_snapshot
    from vplan.tasks import prepare_cohort
    from vplan.tasks import generator

    initial = config_snapshot()
    cohort = prepare_cohort(n=1, witness_evaluator=_witness)
    assert cohort.config_hashes == {key: config_hash(value) for key, value in initial.items()}

    changed = {**initial, "physics": {**initial["physics"], "changed": True}}
    snapshots = iter((initial, changed))
    monkeypatch.setattr(generator, "config_snapshot", lambda: next(snapshots))
    with pytest.raises(RuntimeError, match="configuration changed"):
        prepare_cohort(n=1, witness_evaluator=_witness)


@pytest.mark.parametrize("kwargs", [{"n": 0}, {"n": -1}, {"max_candidates": 0}, {"seed": -1}])
def test_prepare_rejects_invalid_bounds_without_evaluation(kwargs):
    from vplan.tasks import prepare_cohort

    with pytest.raises(ValueError):
        prepare_cohort(**kwargs, witness_evaluator=lambda c: pytest.fail("must validate before evaluation"))


@pytest.mark.parametrize("stratum", ["clear", "distractor", "obstacle"])
def test_real_reference_witness_succeeds_in_each_stratum(stratum):
    from vplan.tasks import evaluate_witness, generate_candidate

    result = evaluate_witness(generate_candidate(0, stratum))
    assert result.outcome == "executed" and not result.findings
    assert result.result is not None and result.result.success


def test_default_cohort_evaluator_receives_exact_captured_snapshot(monkeypatch):
    from vplan.config import config_hash, config_snapshot
    from vplan.tasks import generator
    captured = config_snapshot()
    seen = []
    def evaluate(candidate, *, config_snapshot=None):
        seen.append(config_snapshot)
        return _witness(candidate)
    monkeypatch.setattr(generator, "evaluate_witness", evaluate)
    monkeypatch.setattr(generator, "config_snapshot", lambda: captured)
    result = generator.prepare_cohort(n=2)
    assert seen == [captured, captured]
    assert result.config_hashes == {k: config_hash(v) for k, v in seen[0].items()}
