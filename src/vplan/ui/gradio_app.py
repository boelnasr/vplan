"""The Gradio layer of the W3.5 UI (spec §15, D22): two tabs, Live and Replay, laid out from ui.display panels.

Handlers are plain module functions (load_task, run_attempt, replay_open, replay_show) so tests drive them without a
browser; build_app only wires them to components. Simulator work is serialized by SIM_LOCK, a process-wide lock: a
live attempt's worker thread holds it until run_live returns, even after its page closed and Gradio freed the event's
concurrency slot, so a second attempt, a scene load or a replay waits briefly and is then refused with a message
instead of running beside it (motion/trajectory.py reseeds numpy's global RNG; the Panda model is shared). Telemetry and Gradio's version check are off
(GRADIO_ANALYTICS_ENABLED), and the theme uses system fonts, so the page loads nothing from the internet: replay works
offline, with no keys and no GPU."""
from __future__ import annotations

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # before gradio is imported: no telemetry, no version check

import queue
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field

import gradio as gr

from vplan.ui import display, live, markup, render, sources
from vplan.ui.display import Display

TITLE = "vplan: LLMs plan, physics verifies"
SIM = "sim"  # the Gradio concurrency group of every handler that drives PyBullet (one at a time per event queue)
SIM_LOCK = threading.Lock()  # the simulator itself; released by the live worker thread, not by the Gradio event
BUSY_WAIT_S = 2.0
BUSY = ("the simulator is still running an earlier attempt (it continues after its page closes, until its planner "
        "calls and execution finish); try again when it is done")
SOURCES = ("basic scene", "basic scene with wall", "generated task (seed)")
FONTS = ["ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "sans-serif"]
MONO = ["ui-monospace", "Consolas", "Menlo", "monospace"]


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
    """(scene image, task panel, instruction, task state). A generated task fills in its own instruction."""
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
    lines = [("Task", task.label), ("Ground truth (oracle; never shown to the planner)", task.truth),
             ("Objects", ", ".join(f"{o.id}: {o.color} {o.shape}" for o in task.task.scene.objects)),
             ("Obstacles", str(len(task.task.scene.obstacles)))]
    key = task_key(source, seed, kind, instruction if source != SOURCES[2] else "")
    return render.scene_image(task.task.scene), markup.status_html(lines), task.task.instruction, (key, task)


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
        if source == SOURCES[2] and instruction.strip() and instruction.strip() != task.task.instruction:
            yield error_values("a generated task's instruction is fixed by its oracle; choose a basic scene to type "
                               "your own instruction")
            return
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
        text = f"the attempt stopped: {type(exc).__name__}: {exc}"
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
    problem = _path_error(path) if path else "enter a path"
    if problem is None:
        try:
            cases = sources.list_cases(path, roots=settings.replay_roots())
        except sources.ReplayError as exc:
            problem = str(exc)
    if problem is not None:
        return gr.update(choices=[], value=None), markup.banner_html(display.Banner("warning", problem))
    choices = [(c.label, c.case_id) for c in cases]
    return (gr.update(choices=choices, value=choices[0][1]),
            markup.status_html([("Source", f"{path}: {len(cases)} recorded attempt(s)")]))


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


def _panels() -> Panels:
    """Storyboard order (spec §3): plan, check timeline, highlighted findings, Cosmos frames, execution, Judge."""
    banner = gr.HTML()
    status = gr.HTML()
    with gr.Row():
        plan = gr.Code(language="json", label="Plan (latest round, JSON)", interactive=False, scale=1)
        timeline = gr.HTML(label="Check timeline (per repair round)", scale=2)
    findings = gr.Gallery(label="Verifier findings: offending bodies in magenta (overview, close-up)", columns=2,
                          height="auto", object_fit="contain", visible=False)
    cosmos = gr.Gallery(label="Cosmos key frames and verdicts (learned visual check, not verification)", columns=3,
                        height="auto", object_fit="contain", visible=False)
    with gr.Row():
        execution = gr.HTML(label="Execution", scale=1)
        judge = gr.HTML(label="Judge", scale=1)
    return Panels(banner, status, plan, timeline, findings, cosmos, execution, judge)


def shown(values: tuple) -> tuple:
    """Display values for Panels.outputs(); a gallery is hidden while it has no images."""
    return tuple(gr.update(value=v, visible=bool(v)) if name in ("findings", "cosmos") else v
                 for name, v in zip(Display.FIELDS, values))


def build_app(settings: UISettings) -> gr.Blocks:
    labels = [o.label for o in settings.planners]
    with gr.Blocks(title=TITLE, analytics_enabled=False) as demo:
        gr.Markdown(f"## {TITLE}\nAn LLM proposes a Panda pick-and-place plan; deterministic kinematics and physics "
                    "check it before anything moves (spec §9); an independent Judge scores the executed run (§13).")
        with gr.Tabs(selected="replay" if settings.replay else "live"):
            with gr.Tab("Live", id="live"):
                with gr.Row():
                    with gr.Column(scale=1):
                        source = gr.Radio(list(SOURCES), value=SOURCES[0], label="Scene")
                        with gr.Row():
                            seed = gr.Number(value=0, precision=0, minimum=0, label="Seed (generated task)")
                            kind = gr.Dropdown(list(live.TASK_KINDS), value="feasible", label="Kind (generated task)")
                        instruction = gr.Textbox(value=live.BASIC_INSTRUCTION, label="Instruction (basic scene)")
                        planner = gr.Radio(labels, value=labels[-1], label="Planner")
                        k = gr.Slider(0, 3, value=3, step=1, label="Repair rounds k (spec §10: at most 3)")
                        with gr.Row():
                            load = gr.Button("Load scene")
                            run = gr.Button("Plan, verify, execute", variant="primary")
                    with gr.Column(scale=1):
                        scene = gr.Image(label="Scene", type="numpy", interactive=False)
                        info = gr.HTML()
                live_panels = _panels()
                state = gr.State(None)
                load.click(load_task, [source, seed, kind, instruction], [scene, info, instruction, state],
                           concurrency_id=SIM, concurrency_limit=1)

                def run_live(*args):
                    for values in run_attempt(settings, *args):
                        yield shown(values)

                run.click(run_live, [state, source, seed, kind, instruction, planner, k], live_panels.outputs(),
                          concurrency_id=SIM, concurrency_limit=1)
            with gr.Tab("Replay", id="replay"):
                gr.Markdown("Replay a recorded attempt: a W3.1 run directory, a pilot run, or a UI recording (live "
                            "attempts are saved under `runs/ui`). No keys, no GPU and no network are needed.")
                with gr.Row():
                    path = gr.Textbox(value=settings.replay or settings.record_dir or "", label="Path", scale=3)
                    seed_dir = gr.Textbox(value=settings.seed_dir or "", label="Seed-set directory (W3.1 runs; "
                                          "optional)", scale=2)
                    open_ = gr.Button("Open", scale=1)
                case = gr.Dropdown(choices=[], label="Attempt", allow_custom_value=False)
                show = gr.Button("Replay", variant="primary")
                replay_info = gr.HTML()
                replay_panels = _panels()
                def open_replay(value):
                    return replay_open(settings, value)

                open_.click(open_replay, [path], [case, replay_info])
                show.click(lambda *args: shown(replay_show(settings, *args)), [path, case, seed_dir],
                           replay_panels.outputs(), concurrency_id=SIM, concurrency_limit=1)
                if settings.replay:
                    demo.load(open_replay, [path], [case, replay_info])
    return demo


def theme() -> gr.themes.Base:
    return gr.themes.Default(font=FONTS, font_mono=MONO)


def launch(settings: UISettings) -> None:
    """Serve until interrupted. 127.0.0.1 by default: anyone who can reach the port can drive the planner. share and
    mcp_server are pinned off, because None would let GRADIO_SHARE open a public tunnel or GRADIO_MCP_SERVER expose the
    handlers as tools."""
    build_app(settings).queue(default_concurrency_limit=1).launch(
        server_name=settings.host, server_port=settings.port, share=False, mcp_server=False, theme=theme(),
        css=markup.CSS, show_error=True, inbrowser=False)
