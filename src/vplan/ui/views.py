"""View models for the W3.5 UI (spec §15): plain, typed data built from the shared records (records.py), without
Gradio or PyBullet, so every panel's content is tested offline. ui.markup turns them into HTML and ui.gradio_app lays it out.

Check statuses (spec §9 V-1..V-8, one row per check and round):
- pass: the check ran over the whole trajectory and reported nothing;
- fail: the Verifier reported a Finding for it (at most one per check);
- partial: nothing reported, but the segment checks stopped at an IK or continuity failure, so later segments were
  not checked (verifier.verify);
- unknown: it may have failed without being reported (verify keeps at most five Findings), or the round ended in a
  simulation error before or during verification;
- off: it did not run (the unverified ablation, or an unknown target_id, which stops everything after grounding).
"""
from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from vplan.records import (AttemptRecord, EvaluationResult, Finding, JudgeResult, PlannerCall, RoundRecord,
                           VisualGateReport)

# spec §9 table order: (code, Finding.check, title)
CHECKS: tuple[tuple[str, str, str], ...] = (
    ("V-1", "grounding", "Grounding"), ("V-2", "ik_accuracy", "IK accuracy"), ("V-3", "joint_limits", "Joint limits"),
    ("V-4", "singularity", "Singularity"), ("V-5", "continuity", "Continuity"), ("V-6", "collision", "Collision"),
    ("V-7", "grasp_preconditions", "Grasp preconditions"), ("V-8", "release", "Release"))
SEGMENT_CHECKS = frozenset({"joint_limits", "singularity", "collision"})  # skipped after an IK/continuity stop
STOP_CHECKS = frozenset({"ik_accuracy", "continuity"})
MAX_FINDINGS = 5  # verifier.verify reports at most five Findings, at most one per check
# spec §13 predicate meanings, in report order
PREDICATES: tuple[tuple[str, str], ...] = (
    ("S1", "the target is inside the bin, at rest"), ("S2", "the object in the bin is the instructed one"),
    ("V1", "no unsafe contact (penetration or force)"), ("V2", "every joint within its URDF limits"),
    ("V3", "sigma_min(J_s) above the Judge threshold"), ("V4", "no self-collision"),
    ("R", "refusal judged against the task's feasibility"))

# Cosmos banners (spec §11, D22, D47, D58). The outage text is the CLI's (cli._cosmos_gate), so demo and UI agree.
COSMOS_OFF = ("Cosmos visual check off (no --cosmos-url): approved plans execute on the geometric checks only.")
COSMOS_DOWN = "COSMOS UNAVAILABLE - proceeding on the geometric checks only"
COSMOS_ON = "Cosmos: learned visual check (not verification; D34 pilot balanced accuracy 0.70)"
COSMOS_SKIPPED = ("Cosmos gate skipped (the instruction names no single object to ask about): approved plans execute on "
                  "the geometric checks only.")
COSMOS_NONE = ("This attempt ran without the Cosmos gate: on the geometric checks only (the evaluation path never "
               "uses Cosmos, spec §5.1).")

CheckStatus = Literal["pass", "fail", "partial", "unknown", "off"]


@dataclass(frozen=True)
class CheckRow:
    code: str  # V-1..V-8
    check: str  # Finding.check
    title: str
    status: CheckStatus
    evidence: str = ""
    finding: Finding | None = None


@dataclass(frozen=True)
class VisualRow:
    """One Cosmos request (a learned check, never verification)."""
    stage: str
    views: tuple[str, ...]
    verdict: str
    reason: str
    latency_s: float


@dataclass(frozen=True)
class Predicate:
    id: str
    meaning: str
    passed: bool


@dataclass(frozen=True)
class JudgeView:
    success: bool
    verdict: str  # SUCCESS or FAILURE
    predicates: tuple[Predicate, ...]
    violations: tuple[str, ...]
    details: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class RoundView:
    index: int
    call_outcome: str  # PlannerCall.outcome
    planner: str  # provider/model
    latency_s: float
    plan_kind: Literal["pick_place", "refusal"] | None
    plan_json: str | None  # pretty JSON of the plan or refusal
    outcome: str  # the evaluation outcome, else the call outcome
    summary: str  # one line for the round header
    checks: tuple[CheckRow, ...]
    checks_note: str
    visual_status: str | None  # VisualGateReport.status; None when no gate ran
    visual: tuple[VisualRow, ...]
    visual_note: str | None
    judge: JudgeView | None
    repeat_plan: bool
    problems: tuple[str, ...]  # schema, protocol, provider or simulation errors, as safe one-liners


@dataclass(frozen=True)
class AttemptView:
    attempt_id: str
    mode: str
    k: int
    outcome: str
    success: bool
    call_count: int
    rounds: tuple[RoundView, ...]
    judge: JudgeView | None
    headline: str


@dataclass(frozen=True)
class Banner:
    level: Literal["warning", "info"]
    text: str


def _num(x: float) -> str:
    return f"{x:.4g}"


def _unit(f: Finding) -> str:
    """The unit of measured/threshold where the check fixes it; '' where the Finding does not say (ik_accuracy and
    grasp_preconditions mix metres and degrees, and their direction text carries the units)."""
    if f.check == "joint_limits":
        return " rad/s" if "velocity" in f.direction else " rad"
    return {"grounding": " m", "release": " m", "continuity": " rad", "collision": " m"}.get(f.check, "")


def finding_evidence(f: Finding) -> str:
    """The offending bodies and the measured value against its threshold, where along the trajectory, and the plan
    field (spec §9 Finding fields; no computed fixes, D17)."""
    where = f.stage if f.sample_index is None else f"{f.stage}, sample {f.sample_index}"
    if (f.check == "collision" and len(f.bodies) == 2 and f.measured is not None and f.threshold is not None):
        a, b = f.bodies
        gap = (f"penetration {-f.measured * 100:.2f} cm" if f.measured < 0
               else f"clearance {f.measured * 100:.2f} cm")
        return f"{a} ↔ {b}: {gap} (minimum {f.threshold * 100:.2f} cm) in {where}; plan field {f.plan_field}"
    parts = [f.direction]
    if f.measured is not None:
        unit = _unit(f)
        parts.append(f"measured {_num(f.measured)}{unit}"
                     + ("" if f.threshold is None else f" vs threshold {_num(f.threshold)}{unit}"))
    parts.append(f"in {where}; plan field {f.plan_field}")
    if f.bodies:
        parts.append("bodies " + ", ".join(f.bodies))
    return "; ".join(parts)


def _rows(status: dict[str, CheckStatus], found: dict[str, Finding], notes: dict[str, str]) -> tuple[CheckRow, ...]:
    return tuple(CheckRow(code=code, check=check, title=title, status=status[check],
                          evidence=finding_evidence(found[check]) if check in found else notes.get(check, ""),
                          finding=found.get(check))
                 for code, check, title in CHECKS)


def check_timeline(evaluation: EvaluationResult | None, mode: str) -> tuple[tuple[CheckRow, ...], str]:
    """(rows, note) for one round's verification; no rows when no plan reached the Verifier."""
    if evaluation is None:
        return (), "no plan to verify"
    if evaluation.outcome == "refused":
        return (), "the planner refused: there is no motion to verify"
    found = {f.check: f for f in evaluation.findings if f.check != "visual"}
    names = [check for _, check, _ in CHECKS]
    if mode == "unverified":
        # Config 1 (D29): only the target_id grounding check runs before motion.
        status = {c: "off" for c in names}
        status["grounding"] = "fail" if "grounding" in found else "partial"
        notes = {"grounding": "target_id only (unverified ablation)"}
        return _rows(status, found, notes), "Verifier off: the unverified ablation (config 1) checks only target_id"
    if evaluation.outcome == "sim_error":
        status = {c: "fail" if c in found else "unknown" for c in names}
        return _rows(status, found, {}), "the round ended in a simulation error, so unreported checks are unknown"
    if "grounding" in found and found["grounding"].plan_field == "target_id":
        status = {c: "off" for c in names}
        status["grounding"] = "fail"
        return _rows(status, found, {}), "the target is unknown, so nothing after grounding ran"
    status: dict[str, CheckStatus] = {c: "fail" if c in found else "pass" for c in names}
    notes: dict[str, str] = {}
    note = "every check ran" if not found else f"{len(found)} finding(s)"
    stop = next((found[c] for c in ("ik_accuracy", "continuity") if c in found), None)
    if stop is not None:
        for c in SEGMENT_CHECKS - set(found):
            status[c] = "partial"
            notes[c] = f"nothing up to the {stop.stage} segment; later segments not checked"
        note += f"; segment checks stopped at the {stop.stage} segment ({stop.check})"
    if len(found) >= MAX_FINDINGS:
        for c in names:
            if c not in found:
                status[c] = "unknown"
                notes[c] = "not reported: the Verifier lists at most five findings"
        note += "; the five-finding cap may hide further failures"
    return _rows(status, found, notes), note


def judge_view(result: JudgeResult) -> JudgeView:
    meaning = dict(PREDICATES)
    order = [pid for pid, _ in PREDICATES if pid in result.predicates]
    order += sorted(set(result.predicates) - set(order))
    return JudgeView(success=result.success, verdict="SUCCESS" if result.success else "FAILURE",
                     predicates=tuple(Predicate(pid, meaning.get(pid, ""), result.predicates[pid]) for pid in order),
                     violations=tuple(result.violations), details=tuple(sorted(result.details.items())))


def plan_json(plan) -> str | None:
    return None if plan is None else json.dumps(plan.model_dump(mode="json"), indent=2)


def _call_problems(call: PlannerCall) -> list[str]:
    out = []
    if call.outcome == "service_failure":
        out.append(f"provider failure: {call.error_type or 'unknown error'}"
                   + (f", HTTP {call.status_code}" if call.status_code is not None else ""))
    for error in call.errors[:3]:  # pydantic errors carry no input values (include_input=False in the planner)
        loc = ".".join(str(x) for x in error.get("loc", ())) if isinstance(error, dict) else ""
        msg = error.get("msg") or error.get("message") or error.get("type") if isinstance(error, dict) else str(error)
        out.append(f"{call.outcome}: {loc + ': ' if loc else ''}{msg}")
    if len(call.errors) > 3:
        out.append(f"... and {len(call.errors) - 3} more errors")
    return out


def _summary(rnd: RoundRecord) -> str:
    call, ev = rnd.call, rnd.evaluation
    plan = call.plan
    if ev is None:
        text = {"service_failure": "the planner call failed",
                "malformed": f"malformed reply ({len(call.errors)} schema errors)",
                "protocol_error": f"protocol error; the next request uses '{call.next_protocol}'"}.get(
            call.outcome, call.outcome)
    elif ev.outcome == "refused":
        text = f"refused: {plan.reason} - {plan.explanation}" if plan is not None else "refused"
    elif ev.outcome == "blocked" and ev.visual is not None and ev.visual.status == "fail":
        text = "rejected by Cosmos (a learned visual check, not verification)"
    elif ev.outcome == "blocked":
        text = f"rejected by the Verifier: {len(ev.findings)} finding(s)"
    elif ev.outcome == "executed":
        text = ("executed; Judge: " + ("success" if ev.judge.success else "failure")) if ev.judge else "executed"
    elif ev.outcome == "visual_uncertain":
        text = "Cosmos stayed uncertain after the third view: not executed"
    elif ev.outcome == "stale_approval":
        text = "stale approval: the start state no longer matched the approval; not executed"
    else:
        text = f"simulation error: {ev.error_type or 'unknown'}"
    return text + (" (repeats the previous plan)" if rnd.repeat_plan else "")


def round_view(rnd: RoundRecord, *, mode: str) -> RoundView:
    call, ev = rnd.call, rnd.evaluation
    rows, note = check_timeline(ev, mode)
    visual = ev.visual if ev is not None else None
    problems = _call_problems(call)
    if ev is not None and ev.error_type:
        problems.append(f"{ev.outcome}: {ev.error_type}" + (f": {ev.error_message}" if ev.error_message else ""))
    return RoundView(
        index=rnd.round_index, call_outcome=call.outcome, planner=f"{call.provider}/{call.model}",
        latency_s=call.latency_s, plan_kind=None if call.plan is None else call.plan.kind,
        plan_json=plan_json(call.plan), outcome=ev.outcome if ev is not None else call.outcome,
        summary=_summary(rnd), checks=rows, checks_note=note,
        visual_status=None if visual is None else visual.status,
        visual=() if visual is None else tuple(
            VisualRow(c.stage, tuple(c.views), c.verdict.verdict, c.verdict.reason, c.verdict.latency_s)
            for c in visual.checks),
        visual_note=None if visual is None else visual.note,
        judge=judge_view(ev.judge) if ev is not None and ev.judge is not None else None,
        repeat_plan=rnd.repeat_plan, problems=tuple(problems))


def pending_round_view(index: int, call: PlannerCall | None) -> RoundView:
    """A round whose planner call is in flight (call None) or whose plan is being verified and executed."""
    if call is None:
        summary, outcome = "waiting for the planner", "planning"
    elif call.plan is None:
        summary, outcome = f"planner reply: {call.outcome}", call.outcome
    else:
        summary, outcome = "verifying (and, if approved, executing)", "verifying"
    return RoundView(index=index, call_outcome="pending" if call is None else call.outcome,
                     planner="" if call is None else f"{call.provider}/{call.model}",
                     latency_s=0.0 if call is None else call.latency_s,
                     plan_kind=None if call is None or call.plan is None else call.plan.kind,
                     plan_json=None if call is None else plan_json(call.plan), outcome=outcome, summary=summary,
                     checks=(), checks_note="", visual_status=None, visual=(), visual_note=None, judge=None,
                     repeat_plan=False, problems=() if call is None else tuple(_call_problems(call)))


def headline(record: AttemptRecord) -> str:
    j = record.judge
    o = record.outcome
    if o == "executed":
        return ("Executed: the Judge reports success" if j and j.success else
                "Executed: the Judge reports failure" + (f" ({', '.join(j.violations)})" if j and j.violations
                                                          else ""))
    if o == "refused":
        return ("Correct refusal: the Judge accepts it" if j and j.success
                else "False refusal: the Judge rejects it")
    return {"budget_exhausted": f"Budget exhausted: no plan passed the gate in {record.call_count} planner calls",
            "malformed_final": "Malformed final reply: no valid plan",
            "service_failure": "Provider failure: the planner call did not complete",
            "stale_approval": "Stale approval: the start state changed after approval; nothing executed",
            "sim_error": "Simulation error in the motion layer, Verifier or simulator",
            "visual_uncertain": "Cosmos stayed uncertain after the third view; nothing executed"}.get(o, o)


def attempt_view(record: AttemptRecord) -> AttemptView:
    return AttemptView(attempt_id=record.attempt_id, mode=record.mode, k=record.k, outcome=record.outcome,
                       success=record.success, call_count=record.call_count,
                       rounds=tuple(round_view(r, mode=record.mode) for r in record.rounds),
                       judge=judge_view(record.judge) if record.judge is not None else None,
                       headline=headline(record))


def cosmos_banner(cosmos_url: str | None, reports: Iterable[VisualGateReport | None]) -> Banner:
    """The live demo's Cosmos banner (spec §11, D22, D58): without a URL, or when the endpoint failed, plans execute on
    the geometric checks only, and the banner must say so; a working gate is labelled a learned check."""
    if not cosmos_url:
        return Banner("warning", COSMOS_OFF)
    reports = [r for r in reports if r is not None]
    down = [r for r in reports if r.status == "unavailable"]
    if down:
        note = down[-1].note
        return Banner("warning", f"{COSMOS_DOWN} ({note})" if note else COSMOS_DOWN)
    if reports and all(r.status == "skipped" for r in reports):  # no stage was asked: same as no gate
        return Banner("warning", COSMOS_SKIPPED)
    return Banner("info", COSMOS_ON)


def replay_cosmos_banner(record: AttemptRecord) -> Banner:
    """A recorded attempt carries its gate reports (EvaluationResult.visual) only when it ran with the live gate."""
    reports = [r.evaluation.visual for r in record.rounds if r.evaluation is not None and r.evaluation.visual]
    if not reports:
        return Banner("info", COSMOS_NONE)
    return cosmos_banner("recorded", reports)


def latency_text(seconds: float) -> str:
    return "" if not math.isfinite(seconds) or seconds <= 0 else f"{seconds:.1f} s"
