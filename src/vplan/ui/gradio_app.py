"""The Gradio layer of the W3.5 UI (spec §15, D22): two tabs, Live and Replay, laid out from ui.display panels.

Handlers are plain module functions (load_task, run_attempt, replay_open, replay_show) so tests drive them without a
browser; build_app only wires them to components. Simulator work is serialized by SIM_LOCK, a process-wide lock: a
live attempt's worker thread holds it until run_live returns, even after its page closed and Gradio freed the event's
concurrency slot, so a second attempt, a scene load or a replay waits briefly and is then refused with a message
instead of running beside it (motion/trajectory.py reseeds numpy's global RNG; the Panda model is shared). Telemetry and Gradio's version check are off
(GRADIO_ANALYTICS_ENABLED), and the theme uses system fonts, so the page loads nothing from the internet: replay works
offline, with no keys and no GPU.

On-screen text uses plain words (W4.1); the theme (theme()) and markup.CSS give one restrained style, and the panels run
in the story's order: task, plan, checks, execution, judge."""
from __future__ import annotations

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # before gradio is imported: no telemetry, no version check

import queue
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field

import gradio as gr

from vplan.ui import display, live, markup, render, sources, views
from vplan.ui.display import Display

TITLE = "vplan: LLMs plan, physics verifies"
BRAND, TAGLINE = "vplan", "LLMs plan, physics verifies"
SUBTITLE = ("An LLM proposes a pick-and-place plan for a Panda arm. Kinematics and physics checks verify it before "
            "anything moves, and an independent judge scores the result.")
FLOW = ("Plan", "Verify", "Execute", "Judge")  # the header's pipeline; the result sections carry the same numbers
SIM = "sim"  # the Gradio concurrency group of every handler that drives PyBullet (one at a time per event queue)
SIM_LOCK = threading.Lock()  # the simulator itself; released by the live worker thread, not by the Gradio event
BUSY_WAIT_S = 2.0
BUSY = ("The simulator is still running an earlier attempt (it finishes its planner calls and execution even after "
        "its page closes). Try again when it is done.")
SOURCES = ("Basic scene", "Basic scene with wall", "Generated task")
DEFAULT_SEED = 1  # seed 0 has no solvable generated task, so the default would open on an error
BASIC_INFO = "Name one object in the scene, e.g. 'Pick the green cube and place it in the bin.'"
GENERATED_INFO = "A generated task comes with its own instruction. To type your own, choose a basic scene."
# System fonts only: the page loads nothing from the internet (Segoe UI on Windows, the platform UI font elsewhere)
FONTS = ["Segoe UI", "system-ui", "-apple-system", "ui-sans-serif", "Roboto", "Helvetica Neue", "Arial", "sans-serif"]
MONO = ["Cascadia Mono", "Consolas", "ui-monospace", "Menlo", "monospace"]


@dataclass(frozen=True)
class UISettings:
    """planners: the heuristic first, then an LLM planner when --env-file was given. replay: a path to open in the
    Replay tab at start. record_dir: where live attempts are saved for keyless replay (None: not saved)."""
    planners: tuple[live.PlannerOption, ...] = field(default_factory=lambda: (live.heuristic_option(),))
    cosmos_url: str | None = None
    replay: str | None = None
    seed_dir: str | None = None
    record_dir: str | None = "runs/ui"
    host: str = "127.0.0.1"
    port: int = 7860

    def replay_roots(self) -> tuple[str, ...]:
        """Where the Replay tab may read (sources.check_path): the working directory and the paths the operator
        named on the command line."""
        return tuple(str(r) for r in (os.getcwd(), self.record_dir, self.replay, self.seed_dir) if r)


def _path_error(value: str) -> str | None:
    from vplan.evaluate.pilot import PilotError, check_path_arg
    try:
        check_path_arg(value)
    except PilotError as exc:
        return str(exc)
    return None


# ---- live ---------------------------------------------------------------------------------------------------------

def task_key(source: str, seed, kind: str, instruction: str) -> tuple:
    if source == SOURCES[2]:
        return ("generated", int(seed or 0), kind)
    return ("basic", source == SOURCES[1], instruction.strip())


def make_task(source: str, seed, kind: str, instruction: str) -> live.LiveTask:
    key = task_key(source, seed, kind, instruction)
    if key[0] == "generated":
        return live.generated_task(key[1], kind)
    return live.basic_task(instruction, obstacle=key[1])


def load_task(source: str, seed, kind: str, instruction: str):
    """(scene image, task panel, instruction, task state). A generated task fills in its own instruction; on a refused
    task the panel holds the warning and the instruction comes back unchanged."""
    if not SIM_LOCK.acquire(timeout=BUSY_WAIT_S):
        return None, markup.banner_html(display.Banner("warning", BUSY)), instruction, None
    try:
        return _load_task(source, seed, kind, instruction)
    finally:
        SIM_LOCK.release()


def _load_task(source: str, seed, kind: str, instruction: str):
    try:
        task = make_task(source, seed, kind, instruction)
    except live.LiveError as exc:
        return None, markup.banner_html(display.Banner("warning", str(exc))), instruction, None
    scene = task.task.scene
    lines = [("Task", task.task.instruction), (display.TRUTH, task.truth),
             ("Objects", views.object_summary(scene)), ("Obstacles", views.obstacle_summary(scene, task.enclosure))]
    key = task_key(source, seed, kind, instruction if source != SOURCES[2] else "")
    return render.scene_image(scene), markup.status_html(lines), task.task.instruction, (key, task)


# Outputs of scene_change and task_change, in order (build_app wires them to these components)
SCENE_CHANGE_OUTPUTS = ("generated_row", "instruction", "scene", "info", "state", "memory", "banner", "status",
                        "plan", "timeline", "findings", "cosmos", "execution", "judge")
TASK_CHANGE_OUTPUTS = ("instruction", "scene", "info", "state", "banner", "status", "plan", "timeline", "findings",
                       "cosmos", "execution", "judge")


def _cleared() -> tuple:
    """The live result panels as on a fresh page: no warning, no stale result of another task."""
    return shown(display.idle_display("live").values())


def scene_change(source: str, seed, kind: str, instruction: str, memory: tuple[str, str]) -> tuple:
    """The Scene choice changed (SCENE_CHANGE_OUTPUTS). memory is (the previous scene choice, the last text typed for a
    basic scene). The new task is loaded at once; a generated task fills in its own instruction and locks the box, a
    basic scene unlocks it with the text last typed there. Any warning and the previous task's results are cleared."""
    last_source, typed = memory
    if last_source != SOURCES[2]:  # the box was editable, so it holds the user's text (loaded or not)
        typed = instruction
    generated = source == SOURCES[2]
    image, info, filled, state = load_task(source, seed, kind, "" if generated else typed)
    box = (gr.update(value=filled if state is not None else "", interactive=False, info=GENERATED_INFO)
           if generated else gr.update(value=typed, interactive=True, info=BASIC_INFO))
    return (gr.update(visible=generated), box, image, info, state, (source, typed), *_cleared())


def task_change(source: str, seed, kind: str) -> tuple:
    """The seed or task type of a generated task changed (TASK_CHANGE_OUTPUTS): reload it, as scene_change does."""
    if source != SOURCES[2]:
        return tuple(gr.skip() for _ in TASK_CHANGE_OUTPUTS)
    image, info, instruction, state = load_task(source, seed, kind, "")
    return (instruction if state is not None else "", image, info, state, *_cleared())


def run_attempt(settings: UISettings, state, source: str, seed, kind: str, instruction: str, planner_label: str,
                k) -> Iterator[tuple]:
    """Stream the panels while run_live works in a background thread; the last yield is the finished attempt."""
    option = next((o for o in settings.planners if o.label == planner_label), None)
    if option is None:
        yield error_values(f"unknown planner {planner_label!r}")
        return
    if not SIM_LOCK.acquire(timeout=BUSY_WAIT_S):
        yield error_values(BUSY)
        return
    handed_over = False  # once the worker starts, it alone releases SIM_LOCK
    try:
        key = task_key(source, seed, kind, instruction if source != SOURCES[2] else "")
        task = state[1] if state is not None and state[0] == key else None
        if task is None:
            try:
                task = make_task(source, seed, kind, instruction)
            except live.LiveError as exc:
                yield error_values(str(exc))
                return
        # A generated task always runs with its own instruction (the box is locked and may lag behind a seed change)
        k = int(k)
        progress = display.LiveProgress(task, option.label, settings.cosmos_url, k)
        events: queue.Queue = queue.Queue()
        outcome: dict = {}

        def work() -> None:
            try:
                outcome["result"] = live.run_live(task, option, k=k, cosmos_url=settings.cosmos_url,
                                                  record_dir=settings.record_dir, on_event=events.put)
            except Exception as exc:  # shown in the UI; the attempt's own failures are outcomes, not exceptions
                outcome["error"] = exc
            finally:
                SIM_LOCK.release()  # here, not in the event: a closed page must not free the simulator early
                events.put(None)

        threading.Thread(target=work, name="vplan-live", daemon=True).start()
        handed_over = True
    finally:
        if not handed_over:
            SIM_LOCK.release()
    yield progress.display().values()
    while (event := events.get()) is not None:
        progress.apply(event)
        yield progress.display().values()
    if "error" in outcome:
        exc = outcome["error"]
        text = f"the attempt stopped: {type(exc).__name__}: {exc}"  # tests and logs match this prefix
        if option.secret:  # an unexpected exception must not carry the key onto the page
            text = text.replace(option.secret, "[REDACTED]")
        yield error_values(text[:500])
        return
    yield display.live_result_display(outcome["result"], progress.outage).values()


def error_values(message: str) -> tuple:
    return display.error_display(message).values()


# ---- replay -------------------------------------------------------------------------------------------------------

def replay_open(settings: UISettings, path: str):
    """(case dropdown update, info panel) for a run directory, pilot run, recording or folder of recordings under
    the replay roots."""
    path = (path or "").strip()
    problem = _path_error(path) if path else "enter a path to a run folder or a recording"
    if problem is None:
        try:
            cases = sources.list_cases(path, roots=settings.replay_roots())
        except sources.ReplayError as exc:
            problem = str(exc)
    if problem is not None:
        return gr.update(choices=[], value=None), markup.banner_html(display.Banner("warning", problem))
    choices = [(c.label, c.case_id) for c in cases]
    n = f"{len(cases)} recorded attempt{'s' if len(cases) != 1 else ''}"
    return (gr.update(choices=choices, value=choices[0][1]), markup.status_html([("Opened", f"{path} ({n})")]))


def replay_show(settings: UISettings, path: str, case_id: str | None, seed_dir: str | None = None) -> tuple:
    path = (path or "").strip()
    seed_dir = (seed_dir or "").strip() or None
    for value in (path, seed_dir):
        if value and (problem := _path_error(value)):
            return error_values(problem)
    if not SIM_LOCK.acquire(timeout=BUSY_WAIT_S):
        return error_values(BUSY)
    try:
        try:
            case = sources.load_case(path, case_id or None, seed_dir=seed_dir, roots=settings.replay_roots())
        except sources.ReplayError as exc:
            return error_values(str(exc))
        return display.replay_display(case).values()
    finally:
        SIM_LOCK.release()


# ---- layout -------------------------------------------------------------------------------------------------------

@dataclass
class Panels:
    """The output components of one tab, in Display.FIELDS order."""
    banner: gr.HTML
    status: gr.HTML
    plan: gr.Code
    timeline: gr.HTML
    findings: gr.Gallery
    cosmos: gr.Gallery
    execution: gr.HTML
    judge: gr.HTML

    def outputs(self) -> list:
        return [self.banner, self.status, self.plan, self.timeline, self.findings, self.cosmos, self.execution,
                self.judge]


def _panels(mode: str) -> Panels:
    """Storyboard order (spec §3), numbered as the header's pipeline: the outcome, then 1 plan, 2 checks (timeline,
    highlighted failures, Cosmos frames), 3 execution, 4 judge."""
    idle = display.idle_display(mode)
    banner = gr.HTML()
    status = gr.HTML()
    with gr.Row(equal_height=False):
        with gr.Column(scale=2, min_width=360):
            gr.HTML(markup.step_html(1, "Plan", "the planner's proposal, latest round"))
            plan = gr.Code(language="json", label="Plan (JSON)", interactive=False, max_lines=30, buttons=["copy"])
        with gr.Column(scale=3, min_width=480):
            gr.HTML(markup.step_html(2, "Verify", "kinematics and physics checks before anything moves"))
            timeline = gr.HTML(idle.timeline_html, label="Checks, round by round")
            findings = gr.Gallery(label="What failed: problem parts in magenta", columns=2, height="auto",
                                  object_fit="contain", visible=False, buttons=["fullscreen"])
            cosmos = gr.Gallery(label="What Cosmos saw: key frames and verdicts (a learned visual check, not "
                                "verification)", columns=3, height="auto", object_fit="contain", visible=False,
                                buttons=["fullscreen"])
    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=360):
            gr.HTML(markup.step_html(3, "Execute", "the approved plan in the PyBullet physics simulation"))
            execution = gr.HTML(idle.execution_html, label="Execution")
        with gr.Column(scale=1, min_width=360):
            gr.HTML(markup.step_html(4, "Judge", "independent scoring of the result"))
            judge = gr.HTML(idle.judge_html, label="Judge")
    return Panels(banner, status, plan, timeline, findings, cosmos, execution, judge)


def shown(values: tuple) -> tuple:
    """Display values for Panels.outputs(); a gallery is hidden while it has no images."""
    return tuple(gr.update(value=v, visible=bool(v)) if name in ("findings", "cosmos") else v
                 for name, v in zip(Display.FIELDS, values))


def build_app(settings: UISettings) -> gr.Blocks:
    labels = [o.label for o in settings.planners]
    with gr.Blocks(title=TITLE, analytics_enabled=False) as demo:
        gr.HTML(markup.header_html(BRAND, TAGLINE, SUBTITLE, FLOW))
        with gr.Tabs(selected="replay" if settings.replay else "live"):
            with gr.Tab("Live", id="live"):
                gr.HTML(markup.step_html(None, "Task", "choose a scene and a planner, then run the attempt"))
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, min_width=420):
                        source = gr.Radio(list(SOURCES), value=SOURCES[0], label="Scene")
                        with gr.Row(visible=False) as generated_row:
                            seed = gr.Number(value=DEFAULT_SEED, precision=0, minimum=0, label="Task seed")
                            kind = gr.Dropdown([(live.KIND_LABELS[key], key) for key in live.TASK_KINDS],
                                               value="feasible", label="Task type")
                        instruction = gr.Textbox(value=live.BASIC_INSTRUCTION, label="Instruction", info=BASIC_INFO)
                        planner = gr.Radio(labels, value=labels[-1], label="Planner")
                        k = gr.Slider(0, 3, value=3, step=1, label="Repair rounds (up to 3)",
                                      info="After a rejected plan, the planner sees which checks failed and tries "
                                           "again.")
                        with gr.Row():
                            load = gr.Button("Load scene", variant="secondary")
                            run = gr.Button("Plan, verify, execute", variant="primary")
                    with gr.Column(scale=4, min_width=360):
                        scene = gr.Image(label="Scene", type="numpy", interactive=False, height=420,
                                         buttons=["fullscreen"])
                        info = gr.HTML()
                live_panels = _panels("live")
                state = gr.State(None)
                memory = gr.State((SOURCES[0], live.BASIC_INSTRUCTION))  # see scene_change
                task_outputs = [scene, info, instruction, state]
                load.click(load_task, [source, seed, kind, instruction], task_outputs, concurrency_id=SIM,
                           concurrency_limit=1)
                instruction.submit(load_task, [source, seed, kind, instruction], task_outputs, concurrency_id=SIM,
                                   concurrency_limit=1)
                source.input(scene_change, [source, seed, kind, instruction, memory],
                             [generated_row, instruction, scene, info, state, memory, *live_panels.outputs()],
                             concurrency_id=SIM, concurrency_limit=1, trigger_mode="always_last")
                for control in (seed, kind):
                    control.input(task_change, [source, seed, kind],
                                  [instruction, scene, info, state, *live_panels.outputs()], concurrency_id=SIM,
                                  concurrency_limit=1, trigger_mode="always_last")
                demo.load(load_task, [source, seed, kind, instruction], task_outputs, concurrency_id=SIM,
                          concurrency_limit=1)  # the page opens with the default scene shown

                def run_live(*args):
                    for values in run_attempt(settings, *args):
                        yield shown(values)

                run.click(run_live, [state, source, seed, kind, instruction, planner, k], live_panels.outputs(),
                          concurrency_id=SIM, concurrency_limit=1)
            with gr.Tab("Replay", id="replay"):
                gr.HTML(markup.step_html(None, "Recorded attempt",
                                         "replay a saved run with no API key, no GPU and no network"))
                with gr.Row(equal_height=False, elem_classes=["vp-align-end"]):
                    path = gr.Textbox(value=settings.replay or settings.record_dir or "",
                                      label="Run folder or recording", scale=3, max_lines=1,
                                      info="An evaluation run, a pilot run, or saved live attempts (runs/ui).")
                    seed_dir = gr.Textbox(value=settings.seed_dir or "", label="Seed-set folder (optional)", scale=2,
                                          max_lines=1, info="Only for an evaluation run whose seed set has moved.")
                    open_ = gr.Button("Open", scale=1, min_width=140)
                with gr.Row(equal_height=False, elem_classes=["vp-align-end"]):
                    case = gr.Dropdown(choices=[], label="Attempt", allow_custom_value=False, scale=5)
                    show = gr.Button("Replay", variant="primary", scale=1, min_width=140)
                replay_info = gr.HTML()
                replay_panels = _panels("replay")

                def open_replay(value):
                    return replay_open(settings, value)

                open_.click(open_replay, [path], [case, replay_info])
                show.click(lambda *args: shown(replay_show(settings, *args)), [path, case, seed_dir],
                           replay_panels.outputs(), concurrency_id=SIM, concurrency_limit=1)
                if settings.replay:
                    demo.load(open_replay, [path], [case, replay_info])
    return demo


def theme() -> gr.themes.Base:
    """Gradio's Default theme in slate and blue with system fonts. Primary buttons use blue 600 (white text 5.2:1);
    secondary text is slate 600 (7.2:1 on white), not the default slate 400, which fails contrast."""
    return gr.themes.Default(primary_hue="blue", secondary_hue="blue", neutral_hue="slate", text_size="lg",
                             font=FONTS, font_mono=MONO).set(
        body_background_fill="*neutral_50", body_background_fill_dark="*neutral_950",
        body_text_color_subdued="*neutral_600", body_text_color_subdued_dark="*neutral_400",
        block_info_text_color="*neutral_600", block_info_text_color_dark="*neutral_400",
        block_shadow="0 1px 2px rgba(15, 23, 42, 0.06)", block_label_text_weight="600",
        block_title_text_weight="600", button_large_text_weight="600",
        button_primary_background_fill="*primary_600", button_primary_background_fill_hover="*primary_700",
        button_primary_background_fill_dark="*primary_600", button_primary_background_fill_hover_dark="*primary_500",
        slider_color="*primary_600", color_accent="*primary_600", loader_color="*primary_600")


def launch(settings: UISettings) -> None:
    """Serve until interrupted. 127.0.0.1 by default: anyone who can reach the port can drive the planner. share and
    mcp_server are pinned off, because None would let GRADIO_SHARE open a public tunnel or GRADIO_MCP_SERVER expose the
    handlers as tools."""
    build_app(settings).queue(default_concurrency_limit=1).launch(
        server_name=settings.host, server_port=settings.port, share=False, mcp_server=False, theme=theme(),
        css=markup.CSS, show_error=True, inbrowser=False)
