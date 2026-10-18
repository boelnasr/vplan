"""Live mode of the W3.5 UI (spec §3 storyboard steps 1-8, §15): pick a task, plan with the template heuristic (no
key) or Nemotron, and run one repair attempt (repair.run_task, k <= 3) with the demo-only Cosmos gate when a URL is
given (D47). Nothing here changes the planner, Verifier, executor or Judge: run_live only listens to run_task's events.

Tasks: the basic demo scene (tasks.scenes, with or without its wall) with a typed template instruction, or a
generated task by seed and kind (tasks.generate_task, spec §6). Every task has a TaskOracle, so the Judge can score it:
a basic-scene instruction must name exactly one object, and the heuristic witness must solve it (the generator's rule
for feasible tasks); otherwise it is refused before any planner call.

Credentials (D33): an LLM planner is built once, from an explicitly named env file, behind the free-model guard; the
key stays inside the planner. Recordings (UI bundles, read back by ui.sources for keyless replay) are refused if their
text contains the key.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from vplan.config import config_snapshot
from vplan.records import (AttemptRecord, PickPlacePlan, PlannerCall, RobotFacts, RoundRecord,
                           TaskOracle, TaskSpec)
from vplan.ui import render
from vplan.ui.views import Banner, cosmos_banner

BASIC_INSTRUCTION = "Pick the red cube and place it in the bin."
DEFAULT_TEMPERATURE = 0.7  # every evaluation call uses it (runner.DEFAULT_TEMPERATURE); the demo matches
TASK_KINDS: dict[str, tuple[Literal["feasible", "impossible"], int | None]] = {
    "feasible": ("feasible", None),
    "impossible 1: target out of reach": ("impossible", 1),
    "impossible 2: bin out of reach": ("impossible", 2),
    "impossible 3: target enclosed": ("impossible", 3),
    "impossible 4: absent referent": ("impossible", 4),
}
BUNDLE_KIND = "vplan_ui_attempt"
BUNDLE_SCHEMA = 1


class LiveError(RuntimeError):
    """A refused task or recording; the message is safe to show (never contains credentials)."""


@dataclass(frozen=True)
class LiveTask:
    """A planner-visible TaskSpec with its private oracle (the UI shows the truth to the reviewer, never to a planner)
    and the seed the motion layer uses."""
    label: str
    task: TaskSpec
    oracle: TaskOracle
    seed: int
    origin: dict[str, Any] = field(default_factory=dict)

    @property
    def truth(self) -> str:
        o = self.oracle
        if o.feasibility == "feasible":
            return "feasible (the heuristic witness solved it)"
        return f"impossible, type {o.proof.type}: the correct answer is a refusal ({o.proof.refusal_reason})"


def basic_task(instruction: str, *, obstacle: bool, cfgs: dict[str, dict] | None = None) -> LiveTask:
    """The CLI demo scene with a typed instruction (a spec §6 template naming exactly one object)."""
    from vplan.tasks import generate as gen
    from vplan.tasks import make_basic_scene, reach_bound_m, resolve_instruction

    scene = make_basic_scene(with_obstacle=obstacle)
    instruction = instruction.strip()
    task = TaskSpec(instruction=instruction, scene=scene,
                    robot_facts=RobotFacts(base_pose=scene.robot_base, reach_bound_m=reach_bound_m(),
                                           gripper_opening_m=gen.GRIPPER_OPENING_M, table_height_m=scene.table_top_z))
    try:
        target = resolve_instruction(instruction, scene)
    except ValueError:
        names = ", ".join(f"the {o.color} {o.shape}" for o in scene.objects)
        raise LiveError(f"the instruction must name exactly one object as 'Pick <object> and place it in the bin.' "
                        f"(objects here: {names})") from None
    witness = gen.run_witness(task, target, seed=0, config_snapshot=cfgs)
    if gen.witness_rejection(witness, target) is not None:
        raise LiveError("the heuristic witness could not solve this instruction here, so the task's feasibility is "
                        "undetermined (spec §6); choose another object or scene")
    oracle = TaskOracle(target_id=target, reference_plan=witness.plan, reference_result=witness.result)
    label = f"basic scene{' with wall' if obstacle else ''}: {instruction}"
    return LiveTask(label, task, oracle, 0, {"kind": "basic", "obstacle": obstacle})


def generated_task(seed: int, kind: str, *, cfgs: dict[str, dict] | None = None) -> LiveTask:
    """tasks.generate_task(seed, kind, type): feasible tasks carry a witness, impossible ones a checked proof."""
    from vplan.tasks import generate_task

    if kind not in TASK_KINDS:
        raise LiveError(f"unknown task kind {kind!r}")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise LiveError("the seed must be a non-negative integer")
    task_kind, impossible_type = TASK_KINDS[kind]
    pair = generate_task(seed, task_kind, impossible_type, config_snapshot=cfgs)
    if pair is None:
        raise LiveError(f"seed {seed} yields no {kind} task (neither a witness nor a proof); try another seed")
    spec, oracle = pair
    return LiveTask(f"generated seed {seed} ({kind}): {spec.instruction}", spec, oracle, seed,
                    {"kind": "generated", "seed": seed, "task_kind": task_kind, "impossible_type": impossible_type})


@dataclass(frozen=True)
class PlannerOption:
    """A planner the UI offers. secret is the API key of an LLM planner, kept only to keep it out of recordings."""
    label: str
    planner: Any
    llm: bool = False
    secret: str | None = field(default=None, repr=False, compare=False)


def heuristic_option() -> PlannerOption:
    from vplan.planners.heuristic import HeuristicPlanner
    return PlannerOption("Template heuristic (no LLM, no key)", HeuristicPlanner())


def llm_option(env_file: str | Path, *, model: str | None = None, allow_paid: bool = False,
               temperature: float = DEFAULT_TEMPERATURE) -> PlannerOption:
    """Nemotron from an explicitly named env file (D33): without allow_paid only free OpenRouter nvidia models.
    Raises PilotError with a printable message; the per-call deadline is the spec's 60 s (D63)."""
    from vplan.evaluate.pilot import llm_settings, provider_guard
    from vplan.planners.nemotron import NemotronPlanner

    base_url, api_key, model = llm_settings(Path(env_file), model)
    host, provider = provider_guard(base_url, model, allow_paid=allow_paid)
    planner = NemotronPlanner(model=model, provider=provider, base_url=base_url, api_key=api_key,
                              temperature=temperature)
    return PlannerOption(f"Nemotron: {model} via {host}", planner, llm=True, secret=api_key)


@dataclass(frozen=True)
class CosmosFrame:
    """One key frame the live gate sent to Cosmos (PNG bytes, as rendered by visual.gate.render_stage)."""
    round_index: int
    stage: str
    view: str
    png: bytes


@dataclass(frozen=True)
class LiveEvent:
    """Progress from run_live, in order: request (planner call started), plan (reply received), banner (Cosmos
    outage, sent before motion starts), round (verified, and executed or judged), status (free text)."""
    kind: Literal["request", "plan", "round", "banner", "status"]
    round_index: int | None = None
    call: PlannerCall | None = None
    round: RoundRecord | None = None
    shots: tuple[render.Shot, ...] = ()
    cosmos: tuple[CosmosFrame, ...] = ()
    banner: Banner | None = None
    text: str = ""


@dataclass(frozen=True)
class LiveResult:
    task: LiveTask
    planner: str
    record: AttemptRecord
    shots: tuple[render.Shot, ...]
    cosmos: tuple[CosmosFrame, ...]
    cosmos_url: str | None
    execution_gif: bytes | None
    execution_note: str
    bundle: Path | None


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def run_live(task: LiveTask, option: PlannerOption, *, k: int = 3, cosmos_url: str | None = None,
             cfgs: dict[str, dict] | None = None, record_dir: str | Path | None = None,
             on_event: Callable[[LiveEvent], None] | None = None, render_execution: bool = True) -> LiveResult:
    """One repair attempt (spec §10) with events for the UI. Findings are rendered with their offending bodies
    highlighted as each round completes, and an executed round is played back from its record as a GIF
    (render_execution=False skips the GIF)."""
    from vplan.repair import run_task

    cfgs = deepcopy(cfgs if cfgs is not None else config_snapshot())
    emit = on_event or (lambda event: None)
    pending: list[tuple[str, dict[str, bytes]]] = []  # the frames Cosmos saw in the round being evaluated
    gate = None
    if cosmos_url:
        from vplan.visual import gate as visual_gate

        def recording_render(plan, traj, scene, stage, views):
            pngs = visual_gate.render_stage(plan, traj, scene, stage, views)
            pending.append((stage, pngs))
            return pngs

        cosmos = visual_gate.CosmosGate(cosmos_url, render=recording_render)

        def gate(plan, traj, scene, *, subject):
            report = cosmos(plan, traj, scene, subject=subject)
            if report.status == "unavailable":  # the executor runs as soon as this returns (spec §11, D58)
                emit(LiveEvent("banner", banner=cosmos_banner(cosmos_url, [report])))
            return report

    shots: list[render.Shot] = []
    frames: list[CosmosFrame] = []

    def sink(event: dict) -> None:
        name = event.get("event")
        if name == "request_started":
            emit(LiveEvent("request", round_index=event.get("round_index")))
        elif name == "request_completed":
            emit(LiveEvent("plan", round_index=event.get("round_index"),
                           call=PlannerCall.model_validate(event["call"])))
        elif name == "round_completed":
            rnd = RoundRecord.model_validate(event["round"])
            ev, plan = rnd.evaluation, rnd.call.plan
            new_shots: tuple[render.Shot, ...] = ()
            if ev is not None and ev.findings and isinstance(plan, PickPlacePlan):
                new_shots = tuple(render.finding_shots(plan, task.task.scene, ev.findings, seed=task.seed,
                                                       verifier_cfg=cfgs["verifier"],
                                                       label=f"round {rnd.round_index}: "))
            new_frames = tuple(CosmosFrame(rnd.round_index, stage, view, png)
                               for stage, pngs in pending for view, png in pngs.items())
            pending.clear()
            shots.extend(new_shots)
            frames.extend(new_frames)
            emit(LiveEvent("round", round_index=rnd.round_index, round=rnd, shots=new_shots, cosmos=new_frames))

    record = run_task(task.task, task.oracle, option.planner, k=k, mode="repair", config_snapshot=cfgs,
                      event_sink=sink, attempt_id=f"ui:{_utc_stamp()}", seed=task.seed, visual_gate=gate)
    gif, note = None, "nothing was executed"
    last = record.rounds[-1].evaluation
    if last is not None and last.record is not None and not render_execution:
        note = "executed (the execution render was skipped)"
    elif last is not None and last.record is not None:
        emit(LiveEvent("status", text="rendering the execution from its recorded joint states"))
        try:
            gif = render.gif_bytes(render.execution_frames(last.record, task.task.scene))
            note = (f"round {record.rounds[-1].round_index} as executed in PyBullet, played back from the recorded "
                    f"60 Hz joint and object states ({len(last.record.q_replay)} frames, shown at "
                    f"{render.GIF_FPS} fps)")
        except Exception as exc:  # display only; the Judge's verdict stands
            note = f"execution render failed: {type(exc).__name__}: {exc}"[:300]
    bundle = None
    if record_dir is not None:
        bundle = save_bundle(record_dir, task, record, planner=option.label, cosmos_url=cosmos_url,
                             secret=option.secret)
    return LiveResult(task, option.label, record, tuple(shots), tuple(frames), cosmos_url, gif, note, bundle)


def bundle_text(task: LiveTask, record: AttemptRecord, *, planner: str, cosmos_url: str | None,
                secret: str | None = None) -> str:
    """A UI recording: everything keyless replay needs (task, oracle, seed, the AttemptRecord with its replay
    streams). Refused if it would contain the API key."""
    data = {"kind": BUNDLE_KIND, "schema_version": BUNDLE_SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "label": task.label, "origin": task.origin, "seed": task.seed, "planner": planner,
            "cosmos_host": urlparse(cosmos_url).hostname if cosmos_url else None,
            "task": task.task.model_dump(mode="json"), "oracle": task.oracle.model_dump(mode="json"),
            "record": record.model_dump(mode="json")}
    text = json.dumps(data)
    if secret and secret in text:
        raise LiveError("refusing to record this attempt: its text contains the API key")
    return text


def save_bundle(record_dir: str | Path, task: LiveTask, record: AttemptRecord, *, planner: str,
                cosmos_url: str | None, secret: str | None = None) -> Path:
    text = bundle_text(task, record, planner=planner, cosmos_url=cosmos_url, secret=secret)
    out = Path(record_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = _utc_stamp()
    for n in range(100):  # Windows clocks tick coarsely: two quick attempts can share a stamp
        path = out / f"ui-{stamp}-{n:02d}.json"  # fixed width: name order is creation order (newest-first listing)
        try:
            # Exclusive create never overwrites a recording; a listing skips one that is still being written.
            with open(path, "x", encoding="utf-8") as f:
                f.write(text + "\n")
        except FileExistsError:
            continue
        return path
    raise LiveError(f"no free recording name for {stamp} in {out}")


__all__ = ["BASIC_INSTRUCTION", "BUNDLE_KIND", "CosmosFrame", "LiveError", "LiveEvent",
           "LiveResult", "LiveTask", "PlannerOption", "TASK_KINDS", "basic_task", "bundle_text", "generated_task",
           "heuristic_option", "llm_option", "run_live", "save_bundle"]
