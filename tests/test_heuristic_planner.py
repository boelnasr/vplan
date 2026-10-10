"""W2.4 template-aware heuristic planner: grounding, refusals and the Planner interface. No simulation."""
import itertools

import pytest

from vplan.planners.heuristic import HeuristicPlanner, heuristic_geometry
from vplan.records import PickPlacePlan, PlannerCall, Refusal
from vplan.tasks import REACH_MARGIN_M, STRATA, TEMPLATES, generate_candidate
from vplan.tasks.generate import GenerationRejected, draft_task
from vplan.tasks.reach import target_clearance_m


def _drafts(kind, impossible_type=None, seeds=range(40)):
    out = []
    for seed in seeds:
        try:
            out.append(draft_task(seed, kind, impossible_type))
        except GenerationRejected:
            pass
    return out


def test_grounds_every_generated_dev_instruction():
    planner = HeuristicPlanner()
    drafts = _drafts("feasible", seeds=range(60))
    assert {d.template for d in drafts} == set(TEMPLATES)  # every template is exercised
    for d in drafts:
        call = planner.plan(d.task)
        assert call.outcome == "proposed" and isinstance(call.plan, PickPlacePlan), d.task.instruction
        assert call.plan == heuristic_geometry(d.task.scene, d.target_id)


def test_grounds_pilot_candidates():
    planner = HeuristicPlanner()
    for seed, stratum in itertools.product(range(5), STRATA):
        candidate = generate_candidate(seed, stratum)
        assert planner.plan(candidate.task).plan.target_id == candidate.target_id


def test_absent_referent_is_refused():
    drafts = _drafts("impossible", 4)
    assert drafts
    for d in drafts:
        plan = HeuristicPlanner().plan(d.task).plan
        assert isinstance(plan, Refusal) and plan.reason == "no_matching_object"


@pytest.mark.parametrize("impossible_type, reason", [(1, "target_unreachable"), (2, "destination_unreachable")])
def test_reach_bound_refusals(impossible_type, reason):
    drafts = _drafts("impossible", impossible_type)
    assert drafts
    for d in drafts:
        plan = HeuristicPlanner().plan(d.task).plan
        assert isinstance(plan, Refusal) and plan.reason == reason


def test_reach_refusal_uses_the_task_spec_bound_and_margin():
    d = _drafts("feasible", seeds=range(5))[0]
    target = next(o for o in d.task.scene.objects if o.id == d.target_id)
    clearance = target_clearance_m(target.pose.position, target.size)

    def with_bound(bound):
        facts = d.task.robot_facts.model_copy(update={"reach_bound_m": bound})
        return d.task.model_copy(update={"robot_facts": facts})
    inside = HeuristicPlanner().plan(with_bound(clearance - REACH_MARGIN_M + 1e-3)).plan
    outside = HeuristicPlanner().plan(with_bound(clearance - REACH_MARGIN_M - 1e-3)).plan
    assert isinstance(inside, PickPlacePlan)
    assert isinstance(outside, Refusal) and outside.reason == "target_unreachable"


def test_ambiguous_and_unparseable_instructions_are_refused_as_other():
    d = _drafts("feasible", seeds=range(5))[0]
    unknown = d.task.model_copy(update={"instruction": "Hand me something nice."})
    plan = HeuristicPlanner().plan(unknown).plan
    assert isinstance(plan, Refusal) and plan.reason == "other"
    objs = d.task.scene.objects
    twin = objs[0].model_copy(update={"id": "twin", "pose": objs[0].pose.model_copy(
        update={"position": (objs[0].pose.position[0], objs[0].pose.position[1] - 0.3, objs[0].pose.position[2])})})
    scene = d.task.scene.model_copy(update={"objects": objs + (twin,)})
    ambiguous = d.task.model_copy(update={
        "scene": scene, "instruction": f"Pick the {objs[0].color} {objs[0].shape} and place it in the bin."})
    plan = HeuristicPlanner().plan(ambiguous).plan
    assert isinstance(plan, Refusal) and plan.reason == "other"


def test_planner_interface_matches_nemotron():
    d = _drafts("feasible", seeds=range(5))[0]
    events = []
    call = HeuristicPlanner().plan(d.task, feedback=None, protocol="content", event_sink=events.append)
    assert isinstance(call, PlannerCall) and call.protocol == call.next_protocol == "content"
    assert call.provider == "heuristic" and call.temperature == 0.0
    assert [e["event"] for e in events] == ["request_started", "request_completed"]
    with pytest.raises(TypeError):
        HeuristicPlanner().plan(generate_candidate(0, "clear"))  # the private envelope, not its .task
    with pytest.raises(ValueError):
        HeuristicPlanner().plan(d.task, protocol="bogus")
