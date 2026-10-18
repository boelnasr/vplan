"""Panel contents for both UI modes (spec §15), without Gradio: ui.app maps a Display onto its components in
Display.FIELDS order. Live progress (LiveProgress, fed with ui.live events), a finished live attempt and a replayed
recording all end up as the same panels: banner, status, plan JSON, check timeline, highlighted findings, Cosmos frames,
execution and Judge."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from vplan.records import AttemptRecord, PickPlacePlan, RoundRecord, VisualGateReport
from vplan.ui import markup, render
from vplan.ui.live import CosmosFrame, LiveEvent, LiveResult, LiveTask
from vplan.ui.sources import ReplayCase
from vplan.ui.views import (Banner, attempt_view, cosmos_banner, pending_round_view, plan_json, replay_cosmos_banner,
                            round_view)

Gallery = list[tuple[np.ndarray, str]]


@dataclass
class Display:
    banner_html: str = ""
    status_html: str = ""
    plan_json: str = ""
    timeline_html: str = ""
    findings: Gallery = field(default_factory=list)
    cosmos: Gallery = field(default_factory=list)
    execution_html: str = ""
    judge_html: str = ""

    FIELDS = ("banner_html", "status_html", "plan_json", "timeline_html", "findings", "cosmos", "execution_html",
              "judge_html")

    def values(self) -> tuple:
        return tuple(getattr(self, name) for name in self.FIELDS)


def error_display(message: str) -> Display:
    return Display(banner_html=markup.banner_html(Banner("warning", message)))


def _shots(shots: Sequence[render.Shot]) -> Gallery:
    """Square images only (overview, then close-up), so the gallery's square cells show them uncropped."""
    out: Gallery = []
    for s in shots:
        out.append((s.image, s.caption + (" (overview)" if s.close_up is not None else "")))
        if s.close_up is not None:
            out.append((s.close_up, s.caption + " (close-up)"))
    return out


def _verdict(report: VisualGateReport | None, stage: str, view: str) -> str:
    if report is None:
        return ""
    checks = [c for c in report.checks if c.stage == stage and view in c.views]
    return f": {checks[-1].verdict.verdict} (learned check)" if checks else ""


def cosmos_gallery(frames: Sequence[CosmosFrame], rounds: Sequence[RoundRecord], note: str = "") -> Gallery:
    reports = {r.round_index: r.evaluation.visual for r in rounds if r.evaluation is not None}
    return [(render.png_rgb(f.png), f"round {f.round_index} {f.stage} {f.view}"
             + _verdict(reports.get(f.round_index), f.stage, f.view) + note) for f in frames]


def _latest_plan(rounds: Sequence[RoundRecord], pending_plan=None) -> str:
    plan = pending_plan
    if plan is None:
        plan = next((r.call.plan for r in reversed(rounds) if r.call.plan is not None), None)
    return plan_json(plan) or ""


def _task_lines(task_label: str, truth: str | None, planner: str) -> list[tuple[str, str]]:
    lines = [("Task", task_label), ("Planner", planner)]
    if truth:
        lines.append(("Ground truth (oracle; never shown to the planner)", truth))
    return lines


def _truth(oracle) -> str | None:
    if oracle is None:
        return None
    if oracle.feasibility == "feasible":
        return "feasible"
    return f"impossible, type {oracle.proof.type}: the correct answer is a refusal ({oracle.proof.refusal_reason})"


def attempt_display(*, record: AttemptRecord, task_label: str, truth: str | None, planner: str, banner: Banner,
                    shots: Sequence[render.Shot], cosmos: Gallery, execution_gif: bytes | None, execution_note: str,
                    extra: Sequence[tuple[str, str]] = (), notes: Sequence[str] = ()) -> Display:
    view = attempt_view(record)
    headline, kind = markup.attempt_status(view)
    lines = _task_lines(task_label, truth, planner) + list(extra) + [
        ("Attempt", f"{view.attempt_id} ({view.mode}, k={view.k}): {view.outcome} after {view.call_count} "
                    f"planner call(s)")]
    lines += [("Note", n) for n in notes]
    judge_note = "" if view.judge is not None else f"No Judge verdict: the attempt ended as {view.outcome}."
    return Display(banner_html=markup.banner_html(banner), status_html=markup.status_html(lines, headline, kind),
                   plan_json=_latest_plan(record.rounds), timeline_html=markup.timeline_html(view.rounds),
                   findings=_shots(shots), cosmos=cosmos, execution_html=markup.gif_html(execution_gif, execution_note),
                   judge_html=markup.judge_html(view.judge, judge_note))


def live_result_display(result: LiveResult, outage: Banner | None = None) -> Display:
    reports = [r.evaluation.visual for r in result.record.rounds if r.evaluation is not None]
    banner = outage or cosmos_banner(result.cosmos_url, reports)
    extra = [("Recorded for keyless replay", str(result.bundle))] if result.bundle else []
    return attempt_display(record=result.record, task_label=result.task.label, truth=result.task.truth,
                           planner=result.planner, banner=banner, shots=result.shots,
                           cosmos=cosmos_gallery(result.cosmos, result.record.rounds),
                           execution_gif=result.execution_gif, execution_note=result.execution_note, extra=extra)


class LiveProgress:
    """Live panels while run_live is still going: completed rounds, the round in flight, and its status."""

    def __init__(self, task: LiveTask, planner: str, cosmos_url: str | None, k: int):
        self.task, self.planner, self.cosmos_url, self.k = task, planner, cosmos_url, k
        self.rounds: list[RoundRecord] = []
        self.pending: tuple[int, object] | None = None  # (round index, PlannerCall or None)
        self.shots: list[render.Shot] = []
        self.frames: list[CosmosFrame] = []
        self.outage: Banner | None = None
        self.status = "starting"

    def apply(self, event: LiveEvent) -> None:
        if event.kind == "request":
            self.pending = (event.round_index, None)
            self.status = f"round {event.round_index}: waiting for the planner"
        elif event.kind == "plan":
            self.pending = (event.round_index, event.call)
            self.status = (f"round {event.round_index}: plan received; verifying" if event.call.plan is not None
                           else f"round {event.round_index}: {event.call.outcome}")
        elif event.kind == "round":
            self.rounds.append(event.round)
            self.pending = None
            self.shots.extend(event.shots)
            self.frames.extend(event.cosmos)
            self.status = f"round {event.round_index} done: {round_view(event.round, mode='repair').summary}"
        elif event.kind == "banner":
            self.outage = event.banner
        elif event.kind == "status":
            self.status = event.text

    def display(self) -> Display:
        views = [round_view(r, mode="repair") for r in self.rounds]
        pending_plan = None
        if self.pending is not None:
            index, call = self.pending
            views.append(pending_round_view(index, call))
            pending_plan = getattr(call, "plan", None)
        reports = [r.evaluation.visual for r in self.rounds if r.evaluation is not None]
        banner = self.outage or cosmos_banner(self.cosmos_url, reports)
        lines = _task_lines(self.task.label, self.task.truth, self.planner) + [
            ("Repair budget", f"k = {self.k} (at most {self.k + 1} planner calls)"), ("Status", self.status)]
        return Display(banner_html=markup.banner_html(banner), status_html=markup.status_html(lines, "running",
                                                                                                "info"),
                       plan_json=_latest_plan(self.rounds, pending_plan), timeline_html=markup.timeline_html(views),
                       findings=_shots(self.shots), cosmos=cosmos_gallery(self.frames, self.rounds),
                       execution_html=markup.gif_html(None, "the execution appears here once a plan is approved"),
                       judge_html=markup.judge_html(None, "waiting for the attempt to finish"))


def replay_display(case: ReplayCase, *, cfgs: dict[str, dict] | None = None, render_execution: bool = True) -> Display:
    """A recorded attempt: findings and Cosmos key frames re-posed from the recorded plans, and the executed round
    played back from its recorded streams. No planner, key, GPU or network is involved."""
    from vplan.config import config_hash, config_snapshot
    from vplan.visual.gate import render_stage

    cfgs = cfgs if cfgs is not None else config_snapshot()
    record, scene = case.record, case.task.scene
    notes = []
    current = {k: config_hash(v) for k, v in cfgs.items()}
    changed = sorted(k for k, v in record.config_hashes.items() if current.get(k) != v)
    if changed:
        notes.append(f"configuration changed since this attempt ({', '.join(changed)}): the recorded verdicts stand, "
                     "but highlight and key frames are re-posed with the current configuration")
    shots: list[render.Shot] = []
    frames: list[CosmosFrame] = []
    for rnd in record.rounds:
        ev, plan = rnd.evaluation, rnd.call.plan
        if ev is None or not isinstance(plan, PickPlacePlan):
            continue
        if ev.findings:
            shots += render.finding_shots(plan, scene, ev.findings, seed=case.seed, verifier_cfg=cfgs["verifier"],
                                          label=f"round {rnd.round_index}: ")
        if ev.visual is not None and ev.visual.checks:
            try:
                traj = render.rebuild(plan, scene, seed=case.seed, verifier_cfg=cfgs["verifier"])
                for check in ev.visual.checks:
                    pngs = render_stage(plan, traj, scene, check.stage, tuple(check.views))
                    frames += [CosmosFrame(rnd.round_index, check.stage, v, png) for v, png in pngs.items()]
            except Exception as exc:  # display only
                notes.append(f"Cosmos key frames of round {rnd.round_index} could not be re-rendered: "
                             f"{type(exc).__name__}")
    gif, note = None, "nothing was executed in this attempt"
    executed = [r for r in record.rounds if r.evaluation is not None and r.evaluation.record is not None]
    if executed and render_execution:
        rnd = executed[-1]
        try:
            gif = render.gif_bytes(render.execution_frames(rnd.evaluation.record, scene))
            note = (f"round {rnd.round_index}, played back from its recorded 60 Hz joint and object states "
                    f"({len(rnd.evaluation.record.q_replay)} frames; no physics, no keys)")
        except ValueError as exc:
            note = f"cannot play back round {rnd.round_index}: {exc}"
    label = case.label if case.task.instruction in case.label else f"{case.label}: {case.task.instruction}"
    return attempt_display(record=record, task_label=label,
                           truth=_truth(case.oracle), planner=dict(case.provenance).get("planner", ""),
                           banner=replay_cosmos_banner(record), shots=shots,
                           cosmos=cosmos_gallery(frames, record.rounds, " (re-rendered)"), execution_gif=gif,
                           execution_note=note, extra=[(k, v) for k, v in case.provenance if k != "planner"],
                           notes=notes)
