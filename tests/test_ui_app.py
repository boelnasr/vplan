"""W3.5 Gradio wiring and `vplan ui`: the app builds without a browser, its handlers drive both modes, the replay
starts with no .env, and LLM credentials come only from --env-file behind the free-model guard (D33)."""
import os
import threading

import numpy as np
import pytest

from tests.ui_fixtures import JARGON, SECRET, basic_result, visible
from vplan import cli
from vplan.evaluate import pilot
from vplan.ui import gradio_app as app, live


@pytest.fixture
def launched(monkeypatch):
    """Capture the settings `vplan ui` would serve, instead of starting a server."""
    seen = []
    monkeypatch.setattr(app, "launch", seen.append)
    return seen


def _env(tmp_path, url="https://openrouter.ai/api/v1", model="nvidia/nemotron-3.5-lightning:free"):
    path = tmp_path / "llm.env"
    path.write_text(f"LLM_BASE_URL={url}\nLLM_API_KEY={SECRET}\nLLM_MODEL={model}\n", encoding="utf-8")
    return path


def test_help_lists_the_ui_options(capsys):
    with pytest.raises(SystemExit) as done:
        cli.main(["ui", "--help"])
    assert done.value.code == 0
    out = capsys.readouterr().out
    for flag in ("--replay", "--env-file", "--model", "--cosmos-url", "--allow-paid", "--host", "--port",
                 "--seed-dir", "--record-dir"):
        assert flag in out


def test_replay_starts_without_any_env_file_and_never_reads_one(tmp_path, monkeypatch, launched, capsys):
    result, _ = basic_result()
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(f"LLM_API_KEY={SECRET}\n", encoding="utf-8")  # must stay unread

    def no_env(*args, **kwargs):
        raise AssertionError("replay must not read an env file")

    monkeypatch.setattr(pilot, "read_env_file", no_env)
    for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    saved = live.save_bundle("runs/ui", result.task, result.record, planner="heuristic", cosmos_url=None)
    relative = os.path.join("runs", "ui", saved.name)
    assert cli.main(["ui", "--replay", relative]) == 0
    (settings,) = launched
    assert settings.replay == relative and [o.llm for o in settings.planners] == [False]
    app.build_app(settings)  # the layout builds without a browser
    case, _ = app.replay_open(settings, settings.replay)
    values = app.replay_show(settings, settings.replay, case["value"])
    shown = dict(zip(("banner", "status", "plan", "timeline", "findings", "cosmos", "execution", "judge"), values))
    assert "data:image/gif;base64," in shown["execution"] and "Success" in shown["judge"]
    assert SECRET not in capsys.readouterr().out
    assert os.environ.get("GRADIO_ANALYTICS_ENABLED") == "False"  # no telemetry or version check (offline replay)


def test_env_file_adds_the_llm_planner_and_the_key_is_never_printed(tmp_path, launched, capsys):
    assert cli.main(["ui", "--env-file", str(_env(tmp_path))]) == 0
    (settings,) = launched
    assert [o.llm for o in settings.planners] == [False, True]
    llm = settings.planners[1]
    assert llm.label == "Nemotron: nvidia/nemotron-3.5-lightning:free via openrouter.ai"
    assert llm.planner.call_deadline_s == 60.0 and llm.planner.temperature == 0.7
    captured = capsys.readouterr()
    assert "nemotron-3.5-lightning:free" in captured.out
    assert SECRET not in captured.out + captured.err and SECRET not in repr(settings)


def test_paid_models_need_allow_paid(tmp_path, launched, capsys):
    env = str(_env(tmp_path, "https://api.tokenfactory.example/v1", "nvidia/Nemotron-3-Ultra"))
    assert cli.main(["ui", "--env-file", env]) == 2 and not launched
    captured = capsys.readouterr()
    assert "--allow-paid" in captured.err and SECRET not in captured.out + captured.err
    assert cli.main(["ui", "--env-file", env, "--allow-paid"]) == 0
    assert launched[0].planners[1].label == "Nemotron: nvidia/Nemotron-3-Ultra via api.tokenfactory.example"


@pytest.mark.parametrize("argv", [["--model", "nvidia/x:free"], ["--allow-paid"], ["--temperature", "0.2"]])
def test_llm_flags_without_an_env_file_are_refused(argv, launched, capsys):
    assert cli.main(["ui", *argv]) == 2 and not launched
    assert "--env-file" in capsys.readouterr().err


def test_a_bad_replay_path_is_refused_before_the_server_starts(tmp_path, launched, capsys):
    assert cli.main(["ui", "--replay", str(tmp_path / "missing")]) == 2 and not launched
    assert "does not exist" in capsys.readouterr().err


@pytest.mark.skipif(os.name != "nt", reason="the /mnt/ refusal protects Windows Python called from WSL")
def test_wsl_paths_are_refused_on_windows(launched, capsys):
    assert cli.main(["ui", "--replay", "/mnt/c/runs/x"]) == 2 and not launched
    assert "WSL path" in capsys.readouterr().err
    assert "WSL path" in app.replay_open(app.UISettings(), "/mnt/c/runs/x")[1]


def test_cli_settings_carry_cosmos_host_port_and_recording(tmp_path, launched):
    assert cli.main(["ui", "--cosmos-url", "http://127.0.0.1:8000/v1", "--host", "0.0.0.0", "--port", "7999",
                     "--no-record"]) == 0
    (s,) = launched
    assert (s.cosmos_url, s.host, s.port, s.record_dir) == ("http://127.0.0.1:8000/v1", "0.0.0.0", 7999, None)
    assert cli.main(["ui"]) == 0 and launched[1].record_dir == "runs/ui"


# ---- handlers --------------------------------------------------------------------------------------------------------

def test_live_handlers_load_a_scene_and_stream_an_attempt_to_the_judge():
    image, info, instruction, state = app.load_task(app.SOURCES[1], 0, "feasible", live.BASIC_INSTRUCTION)
    assert isinstance(image, np.ndarray) and image.shape[2] == 3
    text = visible(info)
    assert "Task Pick the red cube and place it in the bin." in text and "Objects red cube, green cube" in text
    assert "Ground truth (hidden from the planner) solvable" in text and "Obstacles 1 wall" in text
    settings = app.UISettings(record_dir=None)
    steps = list(app.run_attempt(settings, state, app.SOURCES[1], 0, "feasible", instruction,
                                 settings.planners[0].label, 3))
    assert len(steps) >= 4  # start, request, plan, round, status ... final
    final = dict(zip(app.display.Display.FIELDS, steps[-1]))
    assert "Success" in final["judge_html"] and "data:image/gif;base64," in final["execution_html"]
    assert "geometric checks only" in final["banner_html"]


def test_live_handlers_report_bad_input_in_the_banner():
    _, info, _, state = app.load_task(app.SOURCES[0], 0, "feasible", "Pick the blue cube and place it in the bin.")
    assert state is None and "exactly one object" in info
    settings = app.UISettings(record_dir=None)
    (values,) = app.run_attempt(settings, None, app.SOURCES[0], 0, "feasible", live.BASIC_INSTRUCTION, "nope", 1)
    assert "unknown planner" in values[0]


def test_a_generated_task_runs_with_its_own_instruction_whatever_the_box_holds(monkeypatch):
    """W4.1: pressing run on a generated task without loading it first used to trip the 'instruction is fixed'
    warning, because the box still held the basic scene's text. The box is locked for generated tasks, and the
    attempt always uses the task's own instruction."""
    seen = []

    def fake_run_live(task, option, **kwargs):
        seen.append(task)
        raise RuntimeError("stopped by the test")

    monkeypatch.setattr(live, "run_live", fake_run_live)
    settings = app.UISettings(record_dir=None)
    steps = list(app.run_attempt(settings, None, app.SOURCES[2], 1, KIND3, live.BASIC_INSTRUCTION,
                                 settings.planners[0].label, 1))
    assert "the attempt stopped: RuntimeError" in steps[-1][0]  # it ran; no instruction warning
    assert seen and seen[0].task.instruction == live.generated_task(1, KIND3).task.instruction


def test_choosing_a_generated_task_fills_and_locks_the_instruction_and_clears_warnings():
    out = dict(zip(app.SCENE_CHANGE_OUTPUTS, app.scene_change(app.SOURCES[2], 1, KIND3, live.BASIC_INSTRUCTION)))
    task = live.generated_task(1, KIND3)
    assert out["generated_row"]["visible"] is True
    assert out["instruction"]["value"] == task.task.instruction and out["instruction"]["interactive"] is False
    assert out["instruction"]["info"] == app.GENERATED_INFO
    assert out["state"][1].task.instruction == task.task.instruction
    assert isinstance(out["scene"], np.ndarray) and "a closed box around the object" in visible(out["info"])
    assert out["banner"] == "" and "appear here" in out["timeline"]  # a stale warning or result is gone
    assert out["findings"]["visible"] is False and out["cosmos"]["visible"] is False


def test_choosing_a_basic_scene_unlocks_the_instruction_and_restores_the_typed_text():
    typed = "Pick the green cube and place it in the bin."
    out = dict(zip(app.SCENE_CHANGE_OUTPUTS, app.scene_change(app.SOURCES[0], 1, KIND3, typed)))
    assert out["generated_row"]["visible"] is False
    assert out["instruction"]["value"] == typed and out["instruction"]["interactive"] is True
    assert out["instruction"]["info"] == app.BASIC_INFO
    assert out["state"][1].task.instruction == typed and out["banner"] == ""
    bad = dict(zip(app.SCENE_CHANGE_OUTPUTS, app.scene_change(app.SOURCES[0], 1, KIND3, "Pick the blue cube.")))
    assert bad["state"] is None and "exactly one object" in bad["info"] and bad["banner"] == ""


def test_changing_the_seed_or_type_reloads_the_generated_task():
    out = dict(zip(app.TASK_CHANGE_OUTPUTS, app.task_change(app.SOURCES[2], 2, KIND3)))
    assert out["instruction"] == live.generated_task(2, KIND3).task.instruction and out["banner"] == ""
    out = dict(zip(app.TASK_CHANGE_OUTPUTS, app.task_change(app.SOURCES[2], 0, "feasible")))
    assert out["state"] is None and out["instruction"] == "" and "Try another seed" in out["info"]
    assert app.task_change(app.SOURCES[0], 2, KIND3) == tuple(app.gr.skip() for _ in app.TASK_CHANGE_OUTPUTS)


def test_the_default_generated_task_exists():
    assert live.generated_task(app.DEFAULT_SEED, "feasible").oracle.feasibility == "feasible"


def test_replay_handlers_report_problems_in_the_panels(tmp_path):
    settings = app.UISettings(replay=str(tmp_path))
    case, info = app.replay_open(settings, "")
    assert case["choices"] == [] and "enter a path" in info
    case, info = app.replay_open(settings, str(tmp_path / "missing"))
    assert "does not exist" in info
    assert "does not exist" in app.replay_show(settings, str(tmp_path / "missing"), None)[0]


def test_the_app_builds_with_both_tabs_and_the_llm_planner_preselected(tmp_path):
    import gradio as gr
    llm = live.llm_option(_env(tmp_path))
    demo = app.build_app(app.UISettings(planners=(live.heuristic_option(), llm), replay=str(tmp_path)))
    assert isinstance(demo, gr.Blocks)
    config = demo.get_config_file()
    labels = {c.get("props", {}).get("label") for c in config["components"]}
    assert {"Live", "Replay", "Scene", "Task seed", "Task type", "Instruction", "Planner", "Repair rounds (up to 3)",
            "What failed: problem parts in magenta"} <= labels
    radio = next(c for c in config["components"] if c.get("props", {}).get("label") == "Planner")
    assert radio["props"]["value"] == llm.label
    scene = next(c for c in config["components"] if c.get("props", {}).get("label") == "Scene"
                 and c["type"] == "radio")
    assert [choice[0] for choice in scene["props"]["choices"]] == ["Basic scene", "Basic scene with wall",
                                                                   "Generated task"]
    kinds = next(c for c in config["components"] if c.get("props", {}).get("label") == "Task type")
    assert [choice[0] for choice in kinds["props"]["choices"]][0] == "Solvable"
    assert SECRET not in str(config)


def test_no_internal_terms_on_the_page():
    """W4.1: labels, help texts, choices, headings and placeholders a viewer sees carry no spec or decision
    references, check codes, oracle/witness/heuristic jargon or raw ids."""
    config = app.build_app(app.UISettings()).get_config_file()
    texts = []
    for c in config["components"]:
        props = c.get("props", {})
        texts += [str(props[key]) for key in ("label", "info", "placeholder") if props.get(key)]
        texts += [str(choice[0]) for choice in props.get("choices") or []]
        if c["type"] in ("html", "markdown") and isinstance(props.get("value"), str):
            texts.append(visible(props["value"]))
    assert "An LLM proposes a pick-and-place plan for a Panda arm." in " ".join(texts)
    found = [t for t in texts if JARGON.search(t)]
    assert not found, found


def test_empty_galleries_are_hidden_and_filled_ones_shown():
    values = app.display.Display(findings=[], cosmos=[(np.zeros((2, 2, 3), np.uint8), "x")]).values()
    out = dict(zip(app.display.Display.FIELDS, app.shown(values)))
    assert out["findings"]["visible"] is False and out["cosmos"]["visible"] is True
    assert out["judge_html"] == "" and out["plan_json"] == ""


def test_an_unexpected_failure_is_shown_without_the_key(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError(f"transport failed with Authorization: Bearer {SECRET}")

    monkeypatch.setattr(live, "run_live", boom)
    option = live.PlannerOption("Nemotron: test", object(), llm=True, secret=SECRET)
    settings = app.UISettings(planners=(live.heuristic_option(), option), record_dir=None)
    state = (app.task_key(app.SOURCES[2], 1, "impossible 3: target enclosed", ""),
             live.generated_task(1, "impossible 3: target enclosed"))
    steps = list(app.run_attempt(settings, state, app.SOURCES[2], 1, "impossible 3: target enclosed", "",
                                 option.label, 1))
    banner = steps[-1][0]
    assert "the attempt stopped: RuntimeError" in banner and SECRET not in banner and "[REDACTED]" in banner



# ---- review fixes: the simulator lock, launch pins, replay roots, host warning ----------------------------------

KIND3 = "impossible 3: target enclosed"


def test_an_attempt_that_outlives_its_page_keeps_the_simulator_until_it_ends(monkeypatch, tmp_path):
    """Gradio frees its concurrency slot when the browser goes away, but the worker thread keeps simulating; new
    simulator work must wait for it, not run beside it (global numpy RNG, shared Panda model)."""
    result, _ = basic_result()
    recording = live.save_bundle(tmp_path, result.task, result.record, planner="h", cosmos_url=None)
    started, release = threading.Event(), threading.Event()

    def slow_run_live(task, option, **kwargs):
        started.set()
        release.wait(30)
        raise RuntimeError("ended by the test")

    monkeypatch.setattr(live, "run_live", slow_run_live)
    monkeypatch.setattr(app, "BUSY_WAIT_S", 0.1)
    settings = app.UISettings(record_dir=None, replay=str(tmp_path))
    state = (app.task_key(app.SOURCES[2], 1, KIND3, ""), live.generated_task(1, KIND3))
    first = app.run_attempt(settings, state, app.SOURCES[2], 1, KIND3, "", settings.planners[0].label, 1)
    next(first)
    assert started.wait(10)
    first.close()  # the page closed: the event ends, the worker does not
    try:
        (busy,) = app.run_attempt(settings, state, app.SOURCES[2], 1, KIND3, "", settings.planners[0].label, 1)
        assert "still running" in busy[0]
        assert "still running" in app.load_task(app.SOURCES[0], 0, "feasible", live.BASIC_INSTRUCTION)[1]
        assert "still running" in app.replay_show(settings, str(recording), None)[0]
    finally:
        release.set()
    assert app.SIM_LOCK.acquire(timeout=10)  # the worker released it when run_live ended
    app.SIM_LOCK.release()


def test_launch_pins_loopback_no_share_link_and_no_mcp_server(monkeypatch):
    seen = {}

    class FakeBlocks:
        def queue(self, **kwargs):
            return self

        def launch(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(app, "build_app", lambda settings: FakeBlocks())
    monkeypatch.setenv("GRADIO_SHARE", "true")  # an explicit False must win over the environment
    monkeypatch.setenv("GRADIO_MCP_SERVER", "true")
    app.launch(app.UISettings())
    assert seen["server_name"] == "127.0.0.1" and seen["share"] is False and seen["mcp_server"] is False


def test_replay_paths_outside_the_roots_are_refused(tmp_path, monkeypatch):
    result, _ = basic_result()
    outside = live.save_bundle(tmp_path / "elsewhere", result.task, result.record, planner="h", cosmos_url=None)
    inside = tmp_path / "work"
    inside.mkdir()
    monkeypatch.chdir(inside)
    settings = app.UISettings(record_dir="runs/ui")
    case, info = app.replay_open(settings, str(outside))
    assert case["choices"] == [] and "outside" in info
    assert "outside" in app.replay_show(settings, str(outside), outside.name)[0]
    assert "outside" in app.replay_show(settings, "runs/ui", None, str(tmp_path / "elsewhere"))[0]  # seed dir too
    allowed = app.UISettings(record_dir="runs/ui", replay=str(outside.parent))
    case, _ = app.replay_open(allowed, str(outside))
    assert case["value"] == outside.name


@pytest.mark.parametrize("path", ["\\\\server\\share\\runs", "//server/share/runs"])
def test_network_paths_are_refused(path):
    settings = app.UISettings(replay=path)
    assert "network" in app.replay_open(settings, path)[1]
    assert "network" in app.replay_show(settings, path, None)[0]


def test_cli_refuses_a_network_replay_path(launched, capsys):
    assert cli.main(["ui", "--replay", "//server/share/runs"]) == 2 and not launched
    assert "network" in capsys.readouterr().err


@pytest.mark.parametrize("host, warned", [("127.0.0.1", False), ("localhost", False), ("::1", False),
                                          ("0.0.0.0", True), ("172.18.80.1", True)])
def test_a_non_loopback_host_prints_a_warning(host, warned, launched, capsys):
    assert cli.main(["ui", "--host", host]) == 0
    assert ("not a loopback address" in capsys.readouterr().err) is warned
