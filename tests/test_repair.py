"""Paired request budgets and feedback isolation with offline provider responses."""
import json

import pytest

from tests.test_planner import FakeClient, planner, response
from vplan import records
from vplan.planners.heuristic import heuristic_geometry
from vplan.tasks import generate_candidate


def generated():
    candidate = generate_candidate(2, "obstacle")
    plan = heuristic_geometry(candidate.task.scene, candidate.target_id)
    return records.GeneratedTask(metadata=candidate.metadata, task=candidate.task,
        oracle=records.TaskOracle(target_id=candidate.target_id, reference_plan=plan,
            reference_result=records.JudgeResult(success=True, predicates={"S1": True}, violations=(), details={})))


def blocked():
    return records.EvaluationResult(outcome="blocked", findings=(records.Finding(
        stage="transport", check="collision", plan_field="transport_via", measured=0,
        threshold=.01, direction="below margin"),))


def executed(success=True):
    return records.EvaluationResult(outcome="executed", judge=records.JudgeResult(
        success=success, predicates={"S1": success}, violations=() if success else ("S1",), details={}))


def test_shared_initial_then_repair_and_blind_resampling():
    from vplan.repair import evaluate_pair
    task = generated()
    initial = task.oracle.reference_plan.model_copy(update={"rationale": "initial answer"})
    fixed = initial.model_copy(update={"transport_via": ((.5, .1, .4),)})
    client = FakeClient([response(initial), response(fixed), response(initial)])
    evaluations, events = [], []
    def evaluate(plan):
        evaluations.append(plan)
        return blocked() if len(evaluations) == 1 else executed()
    result = evaluate_pair(task, planner(client), evaluator=evaluate, event_sink=events.append)
    assert result.unique_call_count == 3 and len(evaluations) == 3
    assert result.repair.call_count == result.resample.call_count == 2
    assert result.repair.rounds[0] == result.resample.rounds[0] == result.initial
    assert result.initial.evaluation.outcome == "blocked"
    assert result.repair.outcome == result.resample.outcome == "executed"
    assert result.unique_total_tokens == 360
    assert json.loads(client.requests[1]["messages"][1]["content"])["feedback"]["findings"]
    assert client.requests[2]["messages"] == client.requests[0]["messages"]
    assert "reference_plan" not in json.dumps(client.requests) and "oracle" not in json.dumps(client.requests)
    assert [e["event"] for e in events].count("request_completed") == 3
    assert json.loads(result.model_dump_json())["unique_call_count"] == 3


@pytest.mark.parametrize("mode", ["executed", "judge_failure", "refused", "service_failure", "sim_error", "stale_approval"])
def test_terminal_initial_is_shared_without_further_requests(mode):
    from vplan.execute.executor import StaleApproval
    from vplan.repair import evaluate_pair
    task = generated()
    answer = records.Refusal(reason="other", explanation="cannot plan") if mode == "refused" else task.oracle.reference_plan
    client = FakeClient([RuntimeError("transport") if mode == "service_failure" else response(answer)])
    def evaluate(plan):
        if mode == "sim_error":
            raise RuntimeError("NaN in solver")
        if mode == "stale_approval":
            raise StaleApproval("changed")
        return executed(success=mode != "judge_failure")
    result = evaluate_pair(task, planner(client), evaluator=evaluate)
    want = "executed" if mode == "judge_failure" else mode
    assert result.repair.outcome == result.resample.outcome == want
    assert result.unique_call_count == 1 and len(client.requests) == 1
    if mode == "judge_failure":
        assert not result.initial.evaluation.judge.success


def test_repeated_rounded_plan_consumes_budget_and_exhausts_both_arms():
    from vplan.repair import evaluate_pair
    task = generated()
    plan = task.oracle.reference_plan
    near = plan.model_copy(update={"rationale": "different rationale", "place": plan.place.model_copy(
        update={"position": tuple(v + .00001 for v in plan.place.position), "yaw_deg": .2})})
    client = FakeClient([response(plan)] + [response(near)] * 6)
    result = evaluate_pair(task, planner(client), evaluator=lambda p: blocked())
    assert result.unique_call_count == 7
    assert result.repair.call_count == result.resample.call_count == 4
    assert result.repair.outcome == result.resample.outcome == "budget_exhausted"
    assert all(r.repeat_plan for r in result.repair.rounds[1:] + result.resample.rounds[1:])
    assert result.repair.rounds[-1].evaluation.findings[0].check == "collision"


def test_protocol_fallback_budget_is_separate_per_arm():
    from vplan.repair import evaluate_pair
    task = generated()
    def unsupported():
        error = RuntimeError("unsupported tools")
        error.status_code = 400
        return error
    client = FakeClient([unsupported(), unsupported(), response(task.oracle.reference_plan),
                         response(task.oracle.reference_plan)])
    result = evaluate_pair(task, planner(client), evaluator=lambda p: executed())
    assert result.unique_call_count == 4 and result.repair.call_count == 3 and result.resample.call_count == 2
    assert [r.call.protocol for r in result.repair.rounds] == ["forced", "required", "content"]
    assert [r.call.protocol for r in result.resample.rounds] == ["forced", "required"]


def test_malformed_feedback_only_repair_and_last_round_retained():
    from vplan.repair import evaluate_pair
    task = generated()
    valid = response(task.oracle.reference_plan)
    client = FakeClient([valid, valid, valid, response(arguments='{"plan":{}}'), valid, valid, response(arguments='{}')])
    result = evaluate_pair(task, planner(client), evaluator=lambda p: blocked())
    assert result.repair.outcome == result.resample.outcome == "malformed_final"
    assert result.unique_call_count == 7
    assert result.repair.rounds[-1].call.errors
    client = FakeClient([response(arguments='{"plan":{}}'), valid, valid])
    result = evaluate_pair(task, planner(client), evaluator=lambda p: executed())
    payload = json.loads(client.requests[1]["messages"][1]["content"])
    assert payload["feedback"]["schema_errors"]
    assert "feedback" not in json.loads(client.requests[2]["messages"][1]["content"])


def test_events_preserve_call_before_evaluator_crash_and_no_oracle_feedback():
    from vplan.repair import evaluate_pair
    task, events = generated(), []
    client = FakeClient([response(generated().oracle.reference_plan)])
    def crash(plan):
        assert events[-1]["event"] == "request_completed"
        raise RuntimeError("solver diagnostic")
    result = evaluate_pair(task, planner(client), evaluator=crash, event_sink=events.append)
    assert result.repair.outcome == "sim_error"
    assert "solver diagnostic" in result.initial.evaluation.error_message
    assert "solver diagnostic" not in json.dumps(client.requests)


def test_only_one_schema_retry_per_consecutive_malformed_sequence():
    from vplan.repair import evaluate_pair
    client = FakeClient([response(arguments="{}")]*3)
    result = evaluate_pair(generated(), planner(client), evaluator=lambda p: pytest.fail("no invalid simulation"))
    assert result.repair.outcome == result.resample.outcome == "malformed_final"
    assert result.repair.call_count == result.resample.call_count == 2
    assert result.unique_call_count == 3


def test_pair_event_sink_failure_propagates_instead_of_becoming_sim_error():
    from vplan.repair import evaluate_pair
    task = generated()
    client = FakeClient([response(task.oracle.reference_plan)])
    def broken(event):
        if event["event"] == "round_completed":
            raise OSError("cannot persist round")
    with pytest.raises(OSError, match="persist"):
        evaluate_pair(task, planner(client), evaluator=lambda p: executed(), event_sink=broken)


def test_real_paired_execution_uses_supplied_snapshot(monkeypatch):
    import vplan.pipeline as pipeline
    from vplan.config import config_hash, config_snapshot
    from vplan.repair import evaluate_pair
    task, cfgs = generated(), config_snapshot()
    monkeypatch.setattr(pipeline, "read_config_snapshot", lambda: pytest.fail("must not reread configs"))
    client = FakeClient([response(task.oracle.reference_plan)])
    result = evaluate_pair(task, planner(client), config_snapshot=cfgs)
    assert result.repair.outcome == result.resample.outcome == "executed"
    assert result.initial.evaluation.judge.success
    assert result.config_hashes == {k: config_hash(v) for k, v in cfgs.items()}


def test_failed_judge_predicate_is_observable_even_without_contact_violations():
    from vplan.repair import evaluate_pair
    task = generated()
    client = FakeClient([response(task.oracle.reference_plan)])
    result = evaluate_pair(task, planner(client), evaluator=lambda p: records.EvaluationResult(
        outcome="executed", judge=records.JudgeResult(success=False, predicates={"S1": False}, violations=(), details={})))
    assert "judge:S1" in result.initial.observable_failures


def test_repeat_plan_treats_rounded_negative_and_positive_zero_as_equal():
    from vplan.repair import evaluate_pair
    task = generated()
    plan = task.oracle.reference_plan.model_copy(update={"transport_via": ((.5, -.0001, .4),)})
    near = plan.model_copy(update={"transport_via": ((.5, .0001, .4),)})
    client = FakeClient([response(plan), response(near), response(near)])
    calls = []
    def evaluate(proposal):
        calls.append(proposal)
        return blocked() if len(calls) == 1 else executed()
    result = evaluate_pair(task, planner(client), evaluator=evaluate)
    assert result.repair.rounds[1].repeat_plan and result.resample.rounds[1].repeat_plan


def test_malformed_provider_envelopes_keep_paired_budget_usage_and_completion_events():
    from vplan.repair import evaluate_pair
    task, events = generated(), []
    invalid_usage = {**response(task.oracle.reference_plan), "usage": {"total_tokens": -1}}
    client = FakeClient([{"choices": [None], "usage": {"total_tokens": 120}}, invalid_usage,
                         {"choices": [{"message": None}], "usage": {"total_tokens": 120}}])
    result = evaluate_pair(task, planner(client), event_sink=events.append,
                           evaluator=lambda p: pytest.fail("malformed response must never simulate"))
    assert result.unique_call_count == 3
    assert result.repair.call_count == result.resample.call_count == 2
    assert result.repair.outcome == result.resample.outcome == "malformed_final"
    assert result.unique_total_tokens is None and result.unique_known_total_tokens == 240
    assert result.unique_missing_usage_calls == 1
    assert sum(e["event"] == "request_completed" for e in events) == 3
    assert sum(e["event"] == "round_completed" for e in events) == 3
    assert json.loads(result.model_dump_json())["unique_call_count"] == 3


def test_repeat_detection_wraps_yaw_at_180():  # 179.7 and -179.8 are the same gripper heading
    from vplan.repair import _geometry
    from vplan.tasks.scenes import make_basic_scene
    base = heuristic_geometry(make_basic_scene(), "cube_red")
    a = base.model_copy(update={"grasp": base.grasp.model_copy(update={"yaw_deg": 179.7}),
                                "place": base.place.model_copy(update={"yaw_deg": -179.8})})
    b = base.model_copy(update={"grasp": base.grasp.model_copy(update={"yaw_deg": -179.8}),
                                "place": base.place.model_copy(update={"yaw_deg": 179.7})})
    assert _geometry(a) == _geometry(b)


def test_arm_order_runs_resample_first_without_changing_accounting():
    from vplan.repair import evaluate_pair
    task = generated()
    initial = task.oracle.reference_plan.model_copy(update={"rationale": "initial answer"})
    fixed = initial.model_copy(update={"transport_via": ((.5, .1, .4),)})
    client = FakeClient([response(initial), response(fixed), response(fixed)])
    events = []
    result = evaluate_pair(task, planner(client), evaluator=lambda p: blocked() if p == initial else executed(),
                           arm_order=("resample", "repair"), event_sink=events.append)
    assert [e["arm"] for e in events if e["event"] == "request_started"] == ["initial", "resample", "repair"]
    assert result.arm_order == ("resample", "repair")
    assert result.initial.call_id.endswith(":1") and result.resample.rounds[1].call_id.endswith(":2")
    assert result.repair.rounds[1].call_id.endswith(":3")
    assert result.repair.call_count == result.resample.call_count == 2 and result.unique_call_count == 3
    assert client.requests[1]["messages"] == client.requests[0]["messages"]  # resampling stays blind
    assert json.loads(client.requests[2]["messages"][1]["content"])["feedback"]["findings"]
    with pytest.raises(ValueError):
        evaluate_pair(task, planner(FakeClient([])), arm_order=("repair", "repair"))


def test_default_arm_order_is_recorded_and_old_results_still_validate():
    from vplan.repair import evaluate_pair
    task = generated()
    result = evaluate_pair(task, planner(FakeClient([response(task.oracle.reference_plan)])),
                           evaluator=lambda p: executed())
    assert result.arm_order == ("repair", "resample")
    old = result.model_dump(mode="json")
    del old["arm_order"]
    assert records.PairedResult.model_validate(old).arm_order == ("repair", "resample")
