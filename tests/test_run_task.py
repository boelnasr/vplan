"""W2.5 run_task: one attempt per mode with every spec §10 outcome, budget accounting and oracle isolation."""
import json

import pytest

from tests.test_planner import FakeClient, planner, response
from tests.test_repair import blocked, executed, generated
from vplan import records


def call(plan=None, outcome=None, **extra):
    """A PlannerCall as a provider-free scripted planner would return it."""
    outcome = outcome or ("proposed" if plan is not None else "malformed")
    return records.PlannerCall(outcome=outcome, plan=plan, provider="scripted", model="scripted", protocol="forced",
                               next_protocol="forced", temperature=0.0, latency_s=0.0, raw_request={},
                               total_tokens=10, **extra)


class Scripted:
    def __init__(self, calls):
        self.calls, self.seen = list(calls), []

    def plan(self, task, *, feedback=None, protocol="forced", event_sink=None):
        self.seen.append({"task": task, "feedback": feedback, "protocol": protocol})
        return self.calls.pop(0)


def refusal():
    return records.Refusal(reason="target_unreachable", explanation="too far")


def impossible():
    task = generated()
    return task.task, records.TaskOracle(target_id=task.oracle.target_id, feasibility="impossible",
        proof=records.ImpossibilityProof(type=1, refusal_reason="target_unreachable", params={"distance_m": 2.0}))


def absent_referent():
    task = generated()
    return task.task, records.TaskOracle(target_id=None, feasibility="impossible", proof=records.ImpossibilityProof(
        type=4, refusal_reason="no_matching_object"))


def run(scripted, evaluate, mode="repair", **kwargs):
    from vplan.repair import run_task
    task = generated()
    return run_task(task.task, task.oracle, scripted, mode=mode, evaluator=evaluate, **kwargs)


def test_executed_success_on_first_round():
    plan = generated().oracle.reference_plan
    result = run(Scripted([call(plan)]), lambda p: executed())
    assert isinstance(result, records.AttemptRecord)
    assert result.outcome == "executed" and result.success and result.call_count == 1
    assert result.mode == "repair" and result.k == 3 and result.judge.success


def test_executed_judge_failure_is_terminal_and_unsuccessful():
    plan = generated().oracle.reference_plan
    scripted = Scripted([call(plan), call(plan)])
    result = run(scripted, lambda p: executed(success=False))
    assert result.outcome == "executed" and not result.success and result.call_count == 1
    assert "judge:S1" in result.rounds[-1].observable_failures


def test_refused_on_feasible_task_is_a_false_refusal():
    result = run(Scripted([call(refusal())]), lambda p: pytest.fail("refusals never simulate"))
    assert result.outcome == "refused" and not result.success
    assert result.judge.predicates == {"R": False}


def test_refused_on_impossible_task_is_correct():
    from vplan.repair import run_task
    spec, oracle = impossible()
    result = run_task(spec, oracle, Scripted([call(refusal())]), evaluator=lambda p: pytest.fail("no simulation"))
    assert result.outcome == "refused" and result.success and result.judge.predicates == {"R": True}


def test_refusal_is_allowed_in_a_later_round():
    plan = generated().oracle.reference_plan
    result = run(Scripted([call(plan), call(refusal())]), lambda p: blocked())
    assert result.outcome == "refused" and result.call_count == 2


def test_budget_exhausted_after_k_repairs_uses_four_calls():
    plan = generated().oracle.reference_plan
    scripted = Scripted([call(plan)] * 4)
    result = run(scripted, lambda p: blocked())
    assert result.outcome == "budget_exhausted" and result.call_count == 4 and not scripted.calls
    assert [r.round_index for r in result.rounds] == [0, 1, 2, 3]
    assert not result.rounds[0].repeat_plan and all(r.repeat_plan for r in result.rounds[1:])


@pytest.mark.parametrize("k", [0, 1, 2])
def test_smaller_k_limits_planner_calls(k):
    plan = generated().oracle.reference_plan
    result = run(Scripted([call(plan)] * 4), lambda p: blocked(), k=k)
    assert result.outcome == "budget_exhausted" and result.call_count == k + 1 and result.k == k


@pytest.mark.parametrize("k", [-1, 4])
def test_k_outside_the_spec_budget_is_rejected(k):
    with pytest.raises(ValueError):
        run(Scripted([]), lambda p: blocked(), k=k)


def test_malformed_retry_uses_up_a_round():
    plan = generated().oracle.reference_plan
    scripted = Scripted([call(plan), call(), call(plan), call(plan)])
    result = run(scripted, lambda p: blocked())
    assert result.outcome == "budget_exhausted" and result.call_count == 4
    assert result.rounds[1].call.outcome == "malformed" and result.rounds[1].evaluation is None


def test_malformed_final_on_the_last_allowed_round():
    plan = generated().oracle.reference_plan
    result = run(Scripted([call(plan), call(plan), call(plan), call()]), lambda p: blocked())
    assert result.outcome == "malformed_final" and result.call_count == 4


def test_two_consecutive_malformed_outputs_end_the_attempt_like_the_pilot():
    result = run(Scripted([call(), call(), call(), call()]), lambda p: pytest.fail("no simulation"))
    assert result.outcome == "malformed_final" and result.call_count == 2


def test_service_failure_is_terminal():
    scripted = Scripted([call(outcome="service_failure", status_code=503), call()])
    result = run(scripted, lambda p: pytest.fail("no simulation"))
    assert result.outcome == "service_failure" and result.call_count == 1 and len(scripted.calls) == 1


def test_protocol_error_uses_a_round_and_switches_protocol():
    plan = generated().oracle.reference_plan
    fallback = call(outcome="protocol_error").model_copy(update={"next_protocol": "required"})
    scripted = Scripted([fallback, call(plan)])
    result = run(scripted, lambda p: executed())
    assert result.outcome == "executed" and result.call_count == 2
    assert [s["protocol"] for s in scripted.seen] == ["forced", "required"]


@pytest.mark.parametrize("error, outcome", [("stale", "stale_approval"), ("crash", "sim_error"),
                                            ("type", "sim_error")])
def test_evaluator_exceptions_become_terminal_outcomes(error, outcome):
    from vplan.execute.executor import StaleApproval
    plan = generated().oracle.reference_plan

    def evaluate(p):
        if error == "stale":
            raise StaleApproval("start state changed")
        if error == "crash":
            raise FloatingPointError("NaN in IK")
        return "not an EvaluationResult"
    result = run(Scripted([call(plan), call(plan)]), evaluate)
    assert result.outcome == outcome and result.call_count == 1 and not result.success
    assert result.rounds[0].evaluation.error_type is not None


def test_repair_mode_sends_findings_and_summaries_resample_mode_sends_nothing():
    plan = generated().oracle.reference_plan
    repair = Scripted([call(plan), call(plan), call(plan)])
    run(repair, lambda p: blocked(), k=2)
    assert repair.seen[0]["feedback"] is None
    feedback = repair.seen[2]["feedback"]
    assert feedback.previous_plan == plan and feedback.findings[0].check == "collision"
    assert feedback.summaries == ("round 0: blocked",)
    resample = Scripted([call(plan), call(plan), call(plan)])
    result = run(resample, lambda p: blocked(), mode="resample", k=2)
    assert result.mode == "resample" and all(s["feedback"] is None for s in resample.seen)


def test_planner_receives_only_the_task_spec_and_no_oracle_fields():
    from vplan.repair import run_task
    task = generated()
    initial = task.oracle.reference_plan.model_copy(update={"rationale": "first"})
    client = FakeClient([response(initial), response(initial)])
    result = run_task(task.task, task.oracle, planner(client), k=1, evaluator=lambda p: blocked())
    assert result.call_count == 2
    sent = json.dumps(client.requests)
    for field in ("reference_plan", "reference_result", "oracle", "feasibility", "proof"):
        assert field not in sent
    payload = json.loads(client.requests[1]["messages"][1]["content"])
    assert set(payload) == {"task", "feedback"} and payload["task"] == task.task.model_dump(mode="json")
    with pytest.raises(TypeError):
        run_task(task, task.oracle, Scripted([]), evaluator=lambda p: blocked())


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        run(Scripted([]), lambda p: blocked(), mode="visual")


def test_events_and_record_carry_attempt_id_and_config_hashes():
    from vplan.config import config_hash, config_snapshot
    plan, events = generated().oracle.reference_plan, []
    cfgs = config_snapshot()
    result = run(Scripted([call(plan)]), lambda p: executed(), attempt_id="c3:dev:7", event_sink=events.append,
                 config_snapshot=cfgs)
    assert result.attempt_id == "c3:dev:7" and result.rounds[0].call_id == "c3:dev:7:1"
    assert result.config_hashes == {k: config_hash(v) for k, v in cfgs.items()}
    assert [e["event"] for e in events] == ["round_completed", "attempt_completed"]
    assert all(e["attempt_id"] == "c3:dev:7" for e in events)
    assert records.AttemptRecord.model_validate_json(result.model_dump_json()) == result


def test_usage_totals_and_missing_usage():
    plan = generated().oracle.reference_plan
    unknown = call(plan).model_copy(update={"total_tokens": None})
    result = run(Scripted([call(plan), unknown]), lambda p: blocked(), k=1)
    assert result.total_tokens is None and result.known_total_tokens == 10 and result.missing_usage_calls == 1


# ---- real pipeline path (default evaluator) ---------------------------------------------------

def test_default_evaluator_executes_and_judges_the_reference_plan():
    from vplan.repair import run_task
    task = generated()
    result = run_task(task.task, task.oracle, Scripted([call(task.oracle.reference_plan)]))
    assert result.outcome == "executed" and result.success


def test_unverified_mode_executes_out_of_bin_place_and_fails_s1():
    from vplan.repair import run_task
    task = generated()
    good = task.oracle.reference_plan
    off_bin = good.model_copy(update={"place": good.place.model_copy(update={"position": (0.45, 0.10, 0.02)})})
    result = run_task(task.task, task.oracle, Scripted([call(off_bin)]), mode="unverified")
    assert result.outcome == "executed" and not result.success
    assert result.judge.predicates["S1"] is False


def test_unverified_mode_still_blocks_an_unknown_target_id():
    from vplan.repair import run_task
    task = generated()
    ghost = task.oracle.reference_plan.model_copy(update={"target_id": "no_such_object"})
    result = run_task(task.task, task.oracle, Scripted([call(ghost)]), mode="unverified", k=0)
    assert result.outcome == "budget_exhausted"
    assert result.rounds[0].evaluation.outcome == "blocked"
    assert result.rounds[0].evaluation.findings[0].plan_field == "target_id"


@pytest.mark.parametrize("stage", ["motion", "verify", "execute"])
def test_pipeline_exceptions_become_sim_error(monkeypatch, stage):
    import vplan.pipeline as pipeline
    from vplan.repair import run_task

    def boom(*args, **kwargs):
        raise ValueError(f"{stage} exploded")
    monkeypatch.setattr(pipeline, {"motion": "build_trajectory", "verify": "verify", "execute": "execute"}[stage], boom)
    task = generated()
    result = run_task(task.task, task.oracle, Scripted([call(task.oracle.reference_plan)]))
    assert result.outcome == "sim_error" and f"{stage} exploded" in result.rounds[0].evaluation.error_message


def test_stale_approval_from_the_executor(monkeypatch):
    import vplan.pipeline as pipeline
    from vplan.execute.executor import StaleApproval
    from vplan.repair import run_task

    def stale(*args, **kwargs):
        raise StaleApproval("approval does not match")
    monkeypatch.setattr(pipeline, "execute", stale)
    task = generated()
    result = run_task(task.task, task.oracle, Scripted([call(task.oracle.reference_plan)]))
    assert result.outcome == "stale_approval"


def test_executed_plan_on_an_impossible_task_is_never_a_success():
    from vplan.repair import run_task
    spec, oracle = impossible()
    plan = generated().oracle.reference_plan  # physically succeeds; the label says it should have been refused
    result = run_task(spec, oracle, Scripted([call(plan)]))
    assert result.outcome == "executed" and not result.success
    assert result.judge.predicates["R"] is False and "R" in result.judge.violations


def test_executed_plan_on_an_absent_referent_task_is_judged_without_crashing():
    from vplan.repair import run_task
    task = generated()
    oracle = records.TaskOracle(target_id=None, feasibility="impossible", proof=records.ImpossibilityProof(
        type=4, refusal_reason="no_matching_object"))
    result = run_task(task.task, oracle, Scripted([call(task.oracle.reference_plan)]))
    assert result.outcome == "executed" and not result.success and result.judge.predicates["R"] is False


@pytest.mark.parametrize("kind", ["feasible", "impossible", "absent_referent"])
def test_default_evaluator_records_the_old_scoring_of_the_same_execution(kind):  # D48
    """The stored execution record, scored the pre-D48 way (physics Judge, then the old wrapper on an impossible task),
    equals what run_task now records from the Judge alone."""
    from tests.test_judge import legacy_attempted_impossible
    from vplan.config import config_snapshot
    from vplan.judge.judge import judge
    from vplan.repair import run_task
    task = generated()
    spec, oracle = {"feasible": (task.task, task.oracle), "impossible": impossible(),
                    "absent_referent": absent_referent()}[kind]
    plan, cfgs = task.oracle.reference_plan, config_snapshot()
    result = run_task(spec, oracle, Scripted([call(plan)]), config_snapshot=cfgs)
    evaluation = result.rounds[0].evaluation
    assert result.outcome == "executed" and evaluation.record is not None
    physics = judge(evaluation.record, spec.scene, oracle.target_id or plan.target_id, "executed", True,
                    planned_target=plan.target_id, cfg=cfgs["judge"])
    assert physics.success  # the reference plan succeeds physically; only the label can fail it
    expected = physics if kind == "feasible" else legacy_attempted_impossible(physics)
    assert result.judge == evaluation.judge == expected
    assert result.judge.model_dump_json() == expected.model_dump_json()
    assert result.success == (kind == "feasible")


# ---- custom evaluators on impossible tasks (D48: the Judge scores R) --------------------------------------------

@pytest.mark.parametrize("judged", [True, False], ids=["judge_told_impossible", "no_judge_result"])
def test_custom_evaluator_owns_its_judge_call_and_run_task_does_not_rescore(judged):
    from tests.test_judge import S, rec
    from vplan.judge.judge import judge
    from vplan.repair import run_task
    spec, oracle = impossible()
    evaluation = records.EvaluationResult(
        outcome="executed", judge=judge(rec(), S, "cube_red", feasible=False) if judged else None)
    result = run_task(spec, oracle, Scripted([call(generated().oracle.reference_plan)]), evaluator=lambda p: evaluation)
    assert result.outcome == "executed" and not result.success
    assert result.rounds[0].evaluation == evaluation and result.judge == evaluation.judge


@pytest.mark.parametrize("claimed", [
    records.JudgeResult(success=True, predicates={"S1": True}, violations=(), details={}),
    records.JudgeResult(success=False, predicates={"S1": False}, violations=("S1",), details={}),
    records.JudgeResult(success=True, predicates={"S1": True, "R": True}, violations=(), details={}),
], ids=["claims_success", "r_never_scored", "r_scored_as_correct"])
def test_custom_evaluator_contradicting_the_impossible_label_ends_as_sim_error(claimed):
    from vplan.repair import run_task
    spec, oracle = impossible()
    plan = generated().oracle.reference_plan
    result = run_task(spec, oracle, Scripted([call(plan), call(plan)]),
                      evaluator=lambda p: records.EvaluationResult(outcome="executed", judge=claimed))
    assert result.outcome == "sim_error" and not result.success and result.call_count == 1
    error = result.rounds[0].evaluation
    assert error.error_type == "ValueError" and "feasible=False" in error.error_message
