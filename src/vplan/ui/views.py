"""View models for the W3.5 UI (spec §15): plain, typed data built from the shared records (records.py), without
Gradio or PyBullet, so every panel's content is tested offline. ui.markup turns them into HTML and ui.gradio_app lays it out.

On-screen text uses plain names (W4.1): checks and Judge predicates by name (their spec codes only in tooltips), scene
bodies as "red cube", "wall" or "arm link 5" instead of their ids, and no spec or decision references.

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
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

from vplan.records import (AttemptRecord, EvaluationResult, Finding, JudgeResult, Obstacle, PlannerCall, RoundRecord,
                           Scene, VisualGateReport)

# spec §9 table order: (code, Finding.check, plain name shown on screen)
CHECKS: tuple[tuple[str, str, str], ...] = (
    ("V-1", "grounding", "Target and drop-off"), ("V-2", "ik_accuracy", "Reach"), ("V-3", "joint_limits", "Joint limits"),
    ("V-4", "singularity", "Singularity margin"), ("V-5", "continuity", "Smooth joint motion"),
    ("V-6", "collision", "Collision"), ("V-7", "grasp_preconditions", "Grasp"), ("V-8", "release", "Release in bin"))
# What each check asks, for its tooltip (spec §9 table, in words)
CHECK_HELP = {
    "grounding": "the target object exists and the drop-off point is over the bin",
    "ik_accuracy": "the arm can reach every planned pose accurately",
    "joint_limits": "every joint stays clear of its position and velocity limits",
    "singularity": "the arm stays away from singular (locked-up) poses",
    "continuity": "no sudden joint jumps between neighbouring samples",
    "collision": "at least 1 cm between the arm or carried object and everything else",
    "grasp_preconditions": "the object fits between the fingers and the approach is square to a face",
    "release": "the object is released fully inside the bin at its resting height"}
SEGMENT_CHECKS = frozenset({"joint_limits", "singularity", "collision"})  # skipped after an IK/continuity stop
STOP_CHECKS = frozenset({"ik_accuracy", "continuity"})
MAX_FINDINGS = 5  # verifier.verify reports at most five Findings, at most one per check
# spec §13 predicates in report order: (id, plain name, meaning)
PREDICATES: tuple[tuple[str, str, str], ...] = (
    ("S1", "Object resting in the bin", "the target is inside the bin, at rest"),
    ("S2", "Correct object", "the object in the bin is the instructed one"),
    ("V1", "No unsafe contact", "no penetration or excessive contact force"),
    ("V2", "Joint limits kept", "every joint within its URDF limits"),
    ("V3", "Singularity margin kept", "sigma_min(J_s) above the Judge threshold"),
    ("V4", "No self-collision", "no link of the arm touches another"),
    ("R", "Refusal matches the task", "refusing is correct only for an impossible task"))
REFUSAL_WORDS = {"target_unreachable": "object out of reach", "destination_unreachable": "bin out of reach",
                 "target_enclosed": "object enclosed", "no_matching_object": "object not in scene",
                 "other": "other reason"}
# Round and call outcomes as a viewer reads them (badges and summaries)
OUTCOME_WORDS = {"executed": "Executed", "blocked": "Rejected", "refused": "Refused", "planning": "Planning",
                 "verifying": "Checking", "pending": "Pending", "malformed": "Invalid reply",
                 "service_failure": "Provider error", "protocol_error": "Protocol error", "sim_error": "Simulation error",
                 "stale_approval": "Start state changed", "visual_uncertain": "Visual check unsure",
                 "proposed": "Plan received", "budget_exhausted": "No plan passed", "malformed_final": "Invalid reply"}

# Cosmos banners (spec §11, D22, D47, D58). The outage text is the CLI's (cli._cosmos_gate), so demo and UI agree.
COSMOS_OFF = "Cosmos visual check off: approved plans execute on the geometric checks only."
COSMOS_DOWN = "COSMOS UNAVAILABLE - proceeding on the geometric checks only"
COSMOS_ON = ("Cosmos visual check on: a learned check after the geometric checks pass, not verification (balanced "
             "accuracy 0.70 in our pilot).")
COSMOS_SKIPPED = ("Cosmos visual check skipped (the instruction names no single object to ask about): approved plans "
                  "execute on the geometric checks only.")
COSMOS_NONE = ("This attempt ran without the Cosmos visual check, on the geometric checks only (evaluation runs never "
               "use Cosmos).")

# ---- plain names for scene bodies ----------------------------------------------------------------------------------

ROBOT_LINKS = {"panda_hand": "gripper", "panda_leftfinger": "left finger", "panda_rightfinger": "right finger",
               "panda_link0": "arm base", "panda_link8": "wrist flange", "panda_grasptarget": "gripper"}
_LINK = re.compile(r"panda_link(\d)")
_THIN_M = 0.015  # a box at most this thick (half-extent) is a wall or a lid


def link_name(link: str) -> str:
    if link in ROBOT_LINKS:
        return ROBOT_LINKS[link]
    m = _LINK.fullmatch(link)
    return f"arm link {m.group(1)}" if m else link.replace("_", " ")


def obstacle_kind(o: Obstacle) -> str:
    hx, hy, hz = o.half_extents
    if o.id.startswith("wall") or (min(hx, hy) <= _THIN_M and hz > min(hx, hy)):
        return "wall"
    return "lid" if hz <= _THIN_M else "block"


def scene_names(scene: Scene | None, enclosure: Iterable[str] = ()) -> dict[str, str]:
    """id -> plain name for the scene's objects ("red cube") and obstacles ("wall", "block", or "enclosure" for the
    panels of an impossible type-3 task's sealed box)."""
    if scene is None:
        return {}
    enclosure = set(enclosure)
    names = {o.id: f"{o.color} {o.shape}" for o in scene.objects}
    names.update({o.id: "enclosure" if o.id in enclosure else obstacle_kind(o) for o in scene.obstacles})
    return names


def body_name(body: str, names: Mapping[str, str]) -> str:
    """A Finding body ('robot:<link>', a scene id, or a built-in body such as bin_wall_0) in plain words."""
    if body.startswith("robot:"):
        return link_name(body[len("robot:"):])
    if body in names:
        return names[body]
    if body.startswith("bin_wall"):
        return "bin wall"
    return body.replace("_", " ")


def obstacle_summary(scene: Scene, enclosure: Iterable[str] = ()) -> str:
    enclosure = set(enclosure)
    parts = ["a closed box around the object"] if any(o.id in enclosure for o in scene.obstacles) else []
    counts = Counter(obstacle_kind(o) for o in scene.obstacles if o.id not in enclosure)
    parts += [f"{n} {kind}{'s' if n > 1 else ''}" for kind, n in sorted(counts.items(), key=lambda kv: kv[0] != "wall")]
    return ", ".join(parts) or "none"


def object_summary(scene: Scene) -> str:
    return ", ".join(f"{o.color} {o.shape}" for o in scene.objects)


def humanize(text: str, names: Mapping[str, str]) -> str:
    """Verifier and Judge text with ids replaced by plain names: robot links, joints, and quoted or bare scene ids."""
    text = re.sub(r"robot link (panda_\w+)", lambda m: "the " + link_name(m.group(1)) if m.group(1) in ROBOT_LINKS
                  else link_name(m.group(1)), text)
    text = re.sub(r"panda_joint(\d)", r"joint \1", text)
    text = re.sub(r"'([^']+)'", lambda m: (("the " if names[m.group(1)] in ("wall", "block", "lid", "enclosure")
                                            else "") + names[m.group(1)]) if m.group(1) in names else m.group(0), text)
    for body, name in names.items():
        text = re.sub(rf"{re.escape(name)} \({re.escape(body)}\)", name, text, flags=re.IGNORECASE)
        text = re.sub(rf"\b{re.escape(body)}\b", name, text)
    return text


def stage_name(stage: str) -> str:
    return stage.replace("_", " ")


def round_title(index: int, mode: str = "repair") -> str:
    """Rounds count from 1 on screen (round_index 0 is the first plan)."""
    if index == 0:
        return "Round 1: first plan"
    return f"Round {index + 1}: " + (f"new sample {index}" if mode == "resample" else f"repair {index}")


def planner_name(provider: str, model: str) -> str:
    if provider == "heuristic":
        return "Rule-based baseline"
    return f"{model} via {provider}" if provider else model


def predicate_name(pid: str) -> str:
    return next((name for p, name, _ in PREDICATES if p == pid), pid)


def contact_text(violation: str, names: Mapping[str, str]) -> str:
    """A Judge violation ('<body>:<link><-><body>:<link>@<stage>', judge.judge) in plain words."""
    pair, _, stage = violation.partition("@")
    if "<->" not in pair:
        return predicate_name(violation)

    def side(text: str) -> str:
        body, _, link = text.partition(":")
        return link_name(link) if body == "robot" else body_name(body, names)
    a, b = pair.split("<->", 1)
    return f"{side(a)} ↔ {side(b)}" + (f" during {stage_name(stage)}" if stage else "")

CheckStatus = Literal["pass", "fail", "partial", "unknown", "off"]


@dataclass(frozen=True)
class CheckRow:
    code: str  # V-1..V-8, shown only as a tooltip
    check: str  # Finding.check
    title: str  # the plain name shown on screen
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
    id: str  # S1, S2, V1..V4, R: shown only as a tooltip
    meaning: str
    passed: bool
    name: str = ""  # the plain name shown on screen


@dataclass(frozen=True)
class JudgeView:
    success: bool
    verdict: str  # Success or Failure
    predicates: tuple[Predicate, ...]
    violations: tuple[str, ...]  # unsafe contacts in plain words (contact_text); predicate ids are left out
    details: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class RoundView:
    index: int
    call_outcome: str  # PlannerCall.outcome
    planner: str  # planner_name(provider, model)
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
    title: str = ""  # round_title: "Round 1: first plan"


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


def finding_evidence(f: Finding, names: Mapping[str, str] | None = None) -> str:
    """The offending bodies (by plain name) and the measured value against its threshold, where along the trajectory,
    and the plan field (spec §9 Finding fields; no computed fixes, D17)."""
    names = names or {}
    stage = stage_name(f.stage)
    where = stage if f.sample_index is None else f"{stage}, sample {f.sample_index}"
    if (f.check == "collision" and len(f.bodies) == 2 and f.measured is not None and f.threshold is not None):
        a, b = (body_name(x, names) for x in f.bodies)
        gap = (f"penetration {-f.measured * 100:.2f} cm" if f.measured < 0
               else f"clearance {f.measured * 100:.2f} cm")
        return f"{a} ↔ {b}: {gap} (minimum {f.threshold * 100:.2f} cm) during {where}; plan field {f.plan_field}"
    parts = [humanize(f.direction, names)]
    if f.measured is not None:
        unit = _unit(f)
        parts.append(f"measured {_num(f.measured)}{unit}"
                     + ("" if f.threshold is None else f" vs threshold {_num(f.threshold)}{unit}"))
    parts.append(f"during {where}; plan field {f.plan_field}")
    if f.bodies:
        parts.append("parts: " + ", ".join(body_name(b, names) for b in f.bodies))
    return "; ".join(parts)


def _rows(status: dict[str, CheckStatus], found: dict[str, Finding], notes: dict[str, str],
          names: Mapping[str, str]) -> tuple[CheckRow, ...]:
    return tuple(CheckRow(code=code, check=check, title=title, status=status[check],
                          evidence=finding_evidence(found[check], names) if check in found else notes.get(check, ""),
                          finding=found.get(check))
                 for code, check, title in CHECKS)


def check_timeline(evaluation: EvaluationResult | None, mode: str,
                   names: Mapping[str, str] | None = None) -> tuple[tuple[CheckRow, ...], str]:
    """(rows, note) for one round's verification; no rows when no plan reached the Verifier."""
    names = names or {}
    if evaluation is None:
        return (), "no plan to check"
    if evaluation.outcome == "refused":
        return (), "the planner refused, so there is no motion to check"
    found = {f.check: f for f in evaluation.findings if f.check != "visual"}
    checks = [check for _, check, _ in CHECKS]
    if mode == "unverified":
        # Config 1 (D29): only the target_id grounding check runs before motion.
        status = {c: "off" for c in checks}
        status["grounding"] = "fail" if "grounding" in found else "partial"
        notes = {"grounding": "target name only (run without verification)"}
        return (_rows(status, found, notes, names),
                "Checks off: this run executes without verification (only the target name is checked)")
    if evaluation.outcome == "sim_error":
        status = {c: "fail" if c in found else "unknown" for c in checks}
        return _rows(status, found, {}, names), "the round ended in a simulation error, so unreported checks are unknown"
    if "grounding" in found and found["grounding"].plan_field == "target_id":
        status = {c: "off" for c in checks}
        status["grounding"] = "fail"
        return _rows(status, found, {}, names), "the target object does not exist, so no later check ran"
    status: dict[str, CheckStatus] = {c: "fail" if c in found else "pass" for c in checks}
    notes: dict[str, str] = {}
    note = f"all {len(checks)} checks ran" if not found else f"{len(found)} of {len(checks)} checks failed"
    stop = next((found[c] for c in ("ik_accuracy", "continuity") if c in found), None)
    if stop is not None:
        stage = stage_name(stop.stage)
        for c in SEGMENT_CHECKS - set(found):
            status[c] = "partial"
            notes[c] = f"nothing up to the {stage} segment; later segments not checked"
        note += f"; the motion checks stopped at the {stage} segment"
    if len(found) >= MAX_FINDINGS:
        for c in checks:
            if c not in found:
                status[c] = "unknown"
                notes[c] = "not reported: at most five failures are listed"
        note += "; at most five failures are listed, so others may be hidden"
    return _rows(status, found, notes, names), note


def judge_view(result: JudgeResult, names: Mapping[str, str] | None = None) -> JudgeView:
    names = names or {}
    info = {pid: (name, meaning) for pid, name, meaning in PREDICATES}
    order = [pid for pid, _, _ in PREDICATES if pid in result.predicates]
    order += sorted(set(result.predicates) - set(order))
    return JudgeView(success=result.success, verdict="Success" if result.success else "Failure",
                     predicates=tuple(Predicate(pid, info.get(pid, ("", ""))[1], result.predicates[pid],
                                                info.get(pid, (pid, ""))[0]) for pid in order),
                     violations=tuple(contact_text(v, names) for v in result.violations if "<->" in v),
                     details=tuple((k, humanize(v, names)) for k, v in sorted(result.details.items())))


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


def _summary(rnd: RoundRecord, names: Mapping[str, str]) -> str:
    call, ev = rnd.call, rnd.evaluation
    plan = call.plan
    if ev is None:
        text = {"service_failure": "the planner call failed",
                "malformed": f"the reply was not a valid plan ({len(call.errors)} errors)",
                "protocol_error": f"protocol error; the next request uses '{call.next_protocol}'"}.get(
            call.outcome, call.outcome)
    elif ev.outcome == "refused":  # the explanation is the model's; ids in it become names (the JSON keeps them)
        text = (f"refused ({REFUSAL_WORDS.get(plan.reason, plan.reason)}): {humanize(plan.explanation, names)}"
                if plan is not None else "refused")
    elif ev.outcome == "blocked" and ev.visual is not None and ev.visual.status == "fail":
        text = "rejected by the Cosmos visual check (learned, not verification)"
    elif ev.outcome == "blocked":
        failed = len([f for f in ev.findings if f.check != "visual"])
        text = f"rejected: {failed} check{'s' if failed != 1 else ''} failed"
    elif ev.outcome == "executed":
        text = (("executed; the judge scores it a " + ("success" if ev.judge.success else "failure")) if ev.judge
                else "executed")
    elif ev.outcome == "visual_uncertain":
        text = "the Cosmos visual check stayed unsure after a third view; not executed"
    elif ev.outcome == "stale_approval":
        text = "the start state changed after approval; not executed"
    else:
        text = f"simulation error: {ev.error_type or 'unknown'}"
    return text + (" (repeats the previous plan)" if rnd.repeat_plan else "")


def round_view(rnd: RoundRecord, *, mode: str, names: Mapping[str, str] | None = None) -> RoundView:
    call, ev = rnd.call, rnd.evaluation
    rows, note = check_timeline(ev, mode, names)
    visual = ev.visual if ev is not None else None
    problems = _call_problems(call)
    if ev is not None and ev.error_type:
        problems.append(f"{ev.outcome}: {ev.error_type}" + (f": {ev.error_message}" if ev.error_message else ""))
    return RoundView(
        index=rnd.round_index, call_outcome=call.outcome, planner=planner_name(call.provider, call.model),
        latency_s=call.latency_s, plan_kind=None if call.plan is None else call.plan.kind,
        plan_json=plan_json(call.plan), outcome=ev.outcome if ev is not None else call.outcome,
        summary=_summary(rnd, names or {}), checks=rows, checks_note=note,
        visual_status=None if visual is None else visual.status,
        visual=() if visual is None else tuple(
            VisualRow(c.stage, tuple(c.views), c.verdict.verdict, c.verdict.reason, c.verdict.latency_s)
            for c in visual.checks),
        visual_note=None if visual is None else visual.note,
        judge=judge_view(ev.judge, names) if ev is not None and ev.judge is not None else None,
        repeat_plan=rnd.repeat_plan, problems=tuple(problems), title=round_title(rnd.round_index, mode))


def pending_round_view(index: int, call: PlannerCall | None, mode: str = "repair") -> RoundView:
    """A round whose planner call is in flight (call None) or whose plan is being verified and executed."""
    if call is None:
        summary, outcome = "waiting for the planner", "planning"
    elif call.plan is None:
        summary, outcome = f"planner reply: {OUTCOME_WORDS.get(call.outcome, call.outcome).lower()}", call.outcome
    else:
        summary, outcome = "checking the plan (and executing it if it passes)", "verifying"
    return RoundView(index=index, call_outcome="pending" if call is None else call.outcome,
                     planner="" if call is None else planner_name(call.provider, call.model),
                     latency_s=0.0 if call is None else call.latency_s,
                     plan_kind=None if call is None or call.plan is None else call.plan.kind,
                     plan_json=None if call is None else plan_json(call.plan), outcome=outcome, summary=summary,
                     checks=(), checks_note="", visual_status=None, visual=(), visual_note=None, judge=None,
                     repeat_plan=False, problems=() if call is None else tuple(_call_problems(call)),
                     title=round_title(index, mode))


def headline(record: AttemptRecord) -> str:
    j = record.judge
    o = record.outcome
    # "Word: sentence": the status card shows the word as a chip and the sentence beside it
    if o == "executed":
        if j and j.success:
            return "Success: the plan was executed and the judge confirms it"
        failed = ", ".join(predicate_name(p) for p, ok in j.predicates.items() if not ok) if j else ""
        return "Failure: the plan was executed, but the judge found a problem" + (f" (failed: {failed})" if failed
                                                                                 else "")
    if o == "refused":
        return ("Correct refusal: the task is impossible and the judge accepts the refusal" if j and j.success
                else "False refusal: the task was solvable, so the judge rejects the refusal")
    calls = f"{record.call_count} planner call{'s' if record.call_count != 1 else ''}"
    return {"budget_exhausted": f"Blocked: no plan passed the checks in {calls}, so nothing moved",
            "malformed_final": "Invalid reply: the last reply was not a valid plan, so nothing moved",
            "service_failure": "Provider error: the planner call failed, so nothing moved",
            "stale_approval": "Not executed: the start state changed after approval",
            "sim_error": "Simulation error: in the motion planning, the checks or the simulator",
            "visual_uncertain": "Not executed: the Cosmos visual check stayed unsure after a third view"}.get(o, o)


def attempt_view(record: AttemptRecord, names: Mapping[str, str] | None = None) -> AttemptView:
    return AttemptView(attempt_id=record.attempt_id, mode=record.mode, k=record.k, outcome=record.outcome,
                       success=record.success, call_count=record.call_count,
                       rounds=tuple(round_view(r, mode=record.mode, names=names) for r in record.rounds),
                       judge=judge_view(record.judge, names) if record.judge is not None else None,
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
