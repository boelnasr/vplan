"""Repair loop (spec §10): run_task for one attempt, and the paired pilot sharing the first request and attempt."""
from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
import re
import traceback
from typing import Literal

from vplan.config import config_hash, config_snapshot as read_config_snapshot
from vplan.records import (ArmResult, AttemptRecord, EvaluationResult, GeneratedTask, JudgeResult, PairedResult,
                           PickPlacePlan, PlannerFeedback, Refusal, RoundRecord, TaskOracle, TaskSpec)


def _simulation_error(exc: Exception, outcome: str, secret: str | None) -> EvaluationResult:
    """Persist useful exception evidence without credentials or captured frame locals."""
    from vplan.planners.nemotron import _sanitize

    def sanitize(text):
        text = _sanitize(text, secret)
        text = re.sub(r"(?i)\bbasic\s+[A-Za-z0-9+/=_-]+", "Basic [REDACTED]", text)
        text = re.sub(r"\b(?:sk-|nvapi-)[A-Za-z0-9_-]{8,}", "[REDACTED]", text)
        return re.sub(r"(?i)\b(api[_-]?key|access[_-]?token|authorization|password|secret)"
                      r"([\"']?\s*[:=]\s*)"
                      r"(?:\"(?:\\[^\r\n]|[^\"\\\r\n])*\"|'(?:\\[^\r\n]|[^'\\\r\n])*'"
                      r"|[\"'][^\r\n]*|[^\s\"'<>;,]+)",
                      r"\1\2[REDACTED]", text)

    trace = "".join(traceback.TracebackException.from_exception(exc, capture_locals=False).format())
    return EvaluationResult(outcome=outcome, error_type=type(exc).__name__,
                            error_message=sanitize(str(exc)), error_traceback=sanitize(trace))


def _wrap_deg(yaw: float) -> int:
    """Round to 1 degree and wrap to [-180, 180) so 179.7 and -179.8 count as the same heading."""
    return (round(yaw) + 180) % 360 - 180


def _geometry(plan):
    if plan is None:
        return None
    data = plan.model_dump(mode="json")
    data.pop("rationale", None)
    if isinstance(plan, PickPlacePlan):
        data["grasp"]["yaw_deg"] = _wrap_deg(plan.grasp.yaw_deg)
        data["grasp"]["pregrasp_offset_m"] = round(plan.grasp.pregrasp_offset_m, 3)
        data["transport_via"] = [[round(v, 3) for v in xyz] for xyz in plan.transport_via]
        data["place"]["position"] = [round(v, 3) for v in plan.place.position]
        data["place"]["yaw_deg"] = _wrap_deg(plan.place.yaw_deg)
        data["place"]["release_height_m"] = round(plan.place.release_height_m, 3)
    return data  # Numeric equality identifies rounded -0.0 and +0.0 as the same geometry.


def _terminal(record):
    if record.call.outcome == "service_failure":
        return "service_failure"
    if record.evaluation is not None and record.evaluation.outcome != "blocked":
        return record.evaluation.outcome
    return None


def _usage(rounds):
    values = [r.call.total_tokens for r in rounds]
    known = sum(v for v in values if v is not None)
    missing = sum(v is None for v in values)
    return (known if not missing else None), known, missing, sum(r.call.latency_s for r in rounds)


def _observable_failures(call, evaluation) -> tuple[str, ...]:
    if evaluation is None:
        return () if call.outcome == "proposed" else (call.outcome,)
    failures = tuple(f"{f.check}:{f.stage}" for f in evaluation.findings)
    if evaluation.outcome in ("sim_error", "stale_approval", "refused", "visual_uncertain"):
        failures += (evaluation.outcome,)
    if evaluation.outcome == "executed" and evaluation.judge is not None and not evaluation.judge.success:
        failures += tuple(dict.fromkeys(
            [f"judge:{name}" for name, passed in evaluation.judge.predicates.items() if not passed]
            + [f"judge:{v}" for v in evaluation.judge.violations]))
    return failures


def _round(planner, spec: TaskSpec, *, call_id: str, index: int, protocol, feedback, previous,
           evaluator: Callable[[PickPlacePlan], EvaluationResult],
           refused: Callable[[Refusal], EvaluationResult], call_event) -> RoundRecord:
    """One planner call and, for a plan, one evaluation; motion/verify/execute exceptions become outcomes."""
    from vplan.execute.executor import StaleApproval

    # Sink errors deliberately stay outside the simulation exception handler.
    call = planner.plan(spec, protocol=protocol, feedback=feedback, event_sink=call_event)
    evaluation = None
    if isinstance(call.plan, Refusal):
        evaluation = refused(call.plan)
    elif isinstance(call.plan, PickPlacePlan):
        try:
            evaluation = evaluator(call.plan)
            if not isinstance(evaluation, EvaluationResult):
                raise TypeError("evaluator must return EvaluationResult")
        except StaleApproval as exc:
            evaluation = _simulation_error(exc, "stale_approval", getattr(planner, "_secret", None))
        except Exception as exc:
            evaluation = _simulation_error(exc, "sim_error", getattr(planner, "_secret", None))
    return RoundRecord(call_id=call_id, round_index=index, call=call, evaluation=evaluation,
                       repeat_plan=call.plan is not None and previous is not None and
                                   _geometry(call.plan) == _geometry(previous),
                       observable_failures=_observable_failures(call, evaluation))


def _continue(first: RoundRecord, run_round, *, k: int, feedback: bool) -> tuple[str, list[RoundRecord]]:
    """Up to k further rounds after `first` until a terminal outcome; returns (outcome, rounds).

    run_round(index, protocol, feedback, previous) -> RoundRecord. A malformed retry uses up a round, and a
    second consecutive malformed reply ends the attempt (only one schema retry per malformed sequence).
    """
    rounds = [first]
    terminal = _terminal(first)
    protocol = first.call.next_protocol
    previous = first.call.plan
    malformed_count = int(first.call.outcome == "malformed")
    for index in range(1, k + 1):
        if terminal is not None:
            break
        last = rounds[-1]
        prompt = None
        if feedback:
            # Verifier already limits findings; preserve order and one finding per check.
            findings, checks = [], set()
            for finding in (() if last.evaluation is None else last.evaluation.findings):
                if finding.check not in checks and len(findings) < 5:
                    findings.append(finding)
                    checks.add(finding.check)
            prompt = PlannerFeedback(previous_plan=previous, findings=tuple(findings),
                schema_errors=last.call.errors,
                summaries=tuple(f"round {r.round_index}: " +
                                (r.evaluation.outcome if r.evaluation is not None else r.call.outcome)
                                for r in rounds[:-1]))
        record = run_round(index, protocol, prompt, previous)
        rounds.append(record)
        protocol = record.call.next_protocol
        if record.call.plan is not None:
            previous = record.call.plan
        terminal = _terminal(record)
        if record.call.outcome == "malformed":
            malformed_count += 1
        elif record.call.outcome == "proposed":
            malformed_count = 0
        if malformed_count >= 2:
            terminal = "malformed_final"
    if terminal is None:
        terminal = "malformed_final" if rounds[-1].call.outcome == "malformed" else "budget_exhausted"
    return terminal, rounds


def evaluate_pair(task: GeneratedTask, planner, *,
                  evaluator: Callable[[PickPlacePlan], EvaluationResult] | None = None,
                  config_snapshot: dict[str, dict] | None = None,
                  event_sink: Callable[[dict], None] | None = None,
                  arm_order: tuple[str, str] = ("repair", "resample")) -> PairedResult:
    """Run at most seven unique requests (four logical calls per arm).

    The initial proposal and its evaluation are shared. Subsequent resampling requests
    receive no previous answers, findings, errors or summaries. event_sink is invoked
    before/after each request and after each evaluation; sink exceptions propagate.
    The optional evaluator is a one-plan offline seam; production closes over private
    oracle truth only at the Judge boundary and uses one copied config snapshot.
    arm_order lets callers alternate which arm runs first, so a mid-task outage or
    rate cap does not systematically land on the same arm.
    """
    if sorted(arm_order) != ["repair", "resample"]:
        raise ValueError("arm_order must contain 'repair' and 'resample' once each")

    cfgs = deepcopy(read_config_snapshot() if config_snapshot is None else config_snapshot)
    hashes = {key: config_hash(value) for key, value in cfgs.items()}
    if evaluator is None:
        from vplan.pipeline import run_attempt

        def evaluator(plan):
            attempt = run_attempt(plan, task.task.scene, task.oracle.target_id, feasible=True,
                                  seed=task.metadata.seed % (2**32), config_snapshot=cfgs)
            return EvaluationResult(outcome=attempt.outcome, findings=tuple(attempt.findings),
                                    judge=attempt.judge, record=attempt.record)

    def refused(plan):
        # Pilot tasks are feasible; refusal is terminal and unsuccessful.
        return EvaluationResult(outcome="refused", judge=JudgeResult(
            success=False, predicates={"R": False}, violations=("R",), details={"reason": "feasible task refused"}))

    counter = 0

    def emit(event):
        if event_sink is not None:
            event_sink({"task_id": task.metadata.task_id, **event})

    def run_round(arm, index, protocol, feedback, previous):
        nonlocal counter
        counter += 1
        call_id = f"{task.metadata.task_id}:{counter}"
        def call_event(event):
            emit({"arm": arm, "round_index": index, "call_id": call_id, **event})
        record = _round(planner, task.task, call_id=call_id, index=index, protocol=protocol, feedback=feedback,
                        previous=previous, evaluator=evaluator, refused=refused, call_event=call_event)
        emit({"event": "round_completed", "arm": arm, "round": record.model_dump(mode="json")})
        return record

    initial = run_round("initial", 0, "forced", None, None)

    def run_arm(name):
        terminal, rounds = _continue(initial, lambda *args: run_round(name, *args), k=3, feedback=name == "repair")
        total, known, missing, latency = _usage(rounds)
        return ArmResult(arm=name, outcome=terminal, rounds=tuple(rounds), call_count=len(rounds),
                         total_tokens=total, known_total_tokens=known, missing_usage_calls=missing, latency_s=latency)

    arms = {name: run_arm(name) for name in arm_order}
    repair, resample = arms["repair"], arms["resample"]
    unique = (initial,) + repair.rounds[1:] + resample.rounds[1:]
    total, known, missing, latency = _usage(unique)
    result = PairedResult(task_id=task.metadata.task_id, initial=initial, repair=repair, resample=resample,
                          unique_call_count=len(unique), unique_total_tokens=total, unique_known_total_tokens=known,
                          unique_missing_usage_calls=missing, unique_latency_s=latency, config_hashes=hashes,
                          arm_order=arm_order)
    emit({"event": "pair_completed", "result": result.model_dump(mode="json")})
    return result


def _attempted_impossible(evaluation):
    """Spec §13 R: an impossible task is correct only if refused, so any executed plan fails it."""
    if not isinstance(evaluation, EvaluationResult) or evaluation.outcome != "executed" or evaluation.judge is None:
        return evaluation
    j = evaluation.judge
    return evaluation.model_copy(update={"judge": JudgeResult(
        success=False, predicates={**j.predicates, "R": False}, violations=j.violations + ("R",),
        details={**j.details, "refusal": "impossible task attempted"})})


def run_task(spec: TaskSpec, oracle: TaskOracle, planner, k: int = 3,
             mode: Literal["repair", "resample", "unverified"] = "repair", *,
             evaluator: Callable[[PickPlacePlan], EvaluationResult] | None = None,
             config_snapshot: dict[str, dict] | None = None,
             event_sink: Callable[[dict], None] | None = None,
             attempt_id: str = "attempt", seed: int = 0, visual_gate: Callable | None = None) -> AttemptRecord:
    """One attempt (spec §10): at most k + 1 planner calls and exactly one outcome.

    repair: stateless feedback (latest plan, its findings, one summary per earlier round).
    resample: identical blind requests (config 7). unverified: blind requests with the verify gate off
    except target grounding (config 1, D29). The planner sees only `spec`; `oracle` is used only at the
    Judge boundary.

    visual_gate is the demo-only live Cosmos gate (D47), called as visual_gate(plan, traj, scene, subject=...) after
    the Verifier passes. Its visual Finding is repair feedback like any other; a final uncertain ends the attempt as
    visual_uncertain. Evaluation runs never pass it (spec §11 shadow study instead).
    """
    if type(spec) is not TaskSpec:
        raise TypeError("run_task takes the public TaskSpec, never a private task envelope")
    if type(oracle) is not TaskOracle:
        raise TypeError("oracle must be a TaskOracle")
    if mode not in ("repair", "resample", "unverified"):
        raise ValueError(f"unknown mode {mode!r}")
    if not 0 <= k <= 3:
        raise ValueError("k must be 0..3 (spec §10 budget: at most four planner calls)")
    if visual_gate is not None and (mode == "unverified" or evaluator is not None):
        raise ValueError("the visual gate runs only in the verified default pipeline")

    cfgs = deepcopy(read_config_snapshot() if config_snapshot is None else config_snapshot)
    hashes = {key: config_hash(value) for key, value in cfgs.items()}
    feasible = oracle.feasibility == "feasible"
    if evaluator is None:
        from vplan.pipeline import run_attempt
        gate = None
        if visual_gate is not None:
            from vplan.records import VisualGateReport
            from vplan.visual.gate import subject
            obj = subject(spec)
            gate = ((lambda plan, traj, scene: VisualGateReport(status="skipped", note="no single instructed object"))
                    if obj is None else (lambda plan, traj, scene: visual_gate(plan, traj, scene, subject=obj)))

        def evaluator(plan):
            # An absent referent (type 4) has no oracle object: the planned target stands in for the physics
            # evidence, and R (below) already makes the attempt a failure.
            attempt = run_attempt(plan, spec.scene, oracle.target_id or plan.target_id, feasible=feasible,
                                  verify_gate=mode != "unverified", seed=seed % (2**32), config_snapshot=cfgs,
                                  visual_gate=gate)
            return EvaluationResult(outcome=attempt.outcome, findings=tuple(attempt.findings),
                                    judge=attempt.judge, record=attempt.record, visual=attempt.visual)
    evaluate = evaluator if feasible else (lambda plan: _attempted_impossible(evaluator(plan)))

    def refused(plan):
        from vplan.judge.judge import judge
        # The Judge's refusal branch returns before it reads the execution record.
        return EvaluationResult(outcome="refused", judge=judge(None, spec.scene, oracle.target_id or "",
                                                               "refused", feasible, cfg=cfgs["judge"]))

    counter = 0

    def emit(event):
        if event_sink is not None:
            event_sink({"attempt_id": attempt_id, "mode": mode, **event})

    def run_round(index, protocol, feedback, previous):
        nonlocal counter
        counter += 1
        call_id = f"{attempt_id}:{counter}"
        def call_event(event):
            emit({"round_index": index, "call_id": call_id, **event})
        record = _round(planner, spec, call_id=call_id, index=index, protocol=protocol, feedback=feedback,
                        previous=previous, evaluator=evaluate, refused=refused, call_event=call_event)
        emit({"event": "round_completed", "round": record.model_dump(mode="json")})
        return record

    terminal, rounds = _continue(run_round(0, "forced", None, None), run_round, k=k, feedback=mode == "repair")
    last = rounds[-1].evaluation
    judged = last is not None and terminal in ("executed", "refused")
    total, known, missing, latency = _usage(rounds)
    result = AttemptRecord(attempt_id=attempt_id, mode=mode, k=k, outcome=terminal, rounds=tuple(rounds),
                           call_count=len(rounds), judge=last.judge if judged else None, total_tokens=total,
                           known_total_tokens=known, missing_usage_calls=missing, latency_s=latency,
                           config_hashes=hashes)
    emit({"event": "attempt_completed", "result": result.model_dump(mode="json")})
    return result

