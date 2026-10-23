"""W3.5 Gradio wiring and `vplan ui`: the app builds without a browser, its handlers drive both modes, the replay
starts with no .env, and LLM credentials come only from --env-file behind the free-model guard (D33)."""
import os
import threading
from pathlib import Path

import numpy as np
import pytest

from tests.ui_fixtures import JARGON, SECRET, basic_result, basic_task, visible, write_eval_run
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
    assert "kinematic and collision checks only" in final["banner_html"]


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
    typed = "Pick the green cube and place it in the bin."
    out = dict(zip(app.SCENE_CHANGE_OUTPUTS, app.scene_change(app.SOURCES[2], 1, KIND3, typed,
                                                              (app.SOURCES[0], live.BASIC_INSTRUCTION))))
    task = live.generated_task(1, KIND3)
    assert out["memory"] == (app.SOURCES[2], typed)  # the text typed for the basic scene is kept
    assert out["generated_row"]["visible"] is True
    assert out["instruction"]["value"] == task.task.instruction and out["instruction"]["interactive"] is False
    assert out["instruction"]["info"] == app.GENERATED_INFO
    assert out["state"][1].task.instruction == task.task.instruction
    assert isinstance(out["scene"], np.ndarray) and "a closed box around the object" in visible(out["info"])
    assert out["banner"] == "" and "appear here" in out["timeline"]  # a stale warning or result is gone
    assert out["findings"]["visible"] is False and out["cosmos"]["visible"] is False


def test_choosing_a_basic_scene_unlocks_the_instruction_and_restores_the_typed_text():
    typed = "Pick the green cube and place it in the bin."
    generated = live.generated_task(1, KIND3).task.instruction
    out = dict(zip(app.SCENE_CHANGE_OUTPUTS, app.scene_change(app.SOURCES[0], 1, KIND3, generated,
                                                              (app.SOURCES[2], typed))))
    assert out["generated_row"]["visible"] is False
    assert out["instruction"]["value"] == typed and out["instruction"]["interactive"] is True
    assert out["instruction"]["info"] == app.BASIC_INFO
    assert out["state"][1].task.instruction == typed and out["banner"] == ""
    assert out["memory"] == (app.SOURCES[0], typed)
    # between the two basic scenes the box keeps whatever was typed, even if it was never loaded
    out = dict(zip(app.SCENE_CHANGE_OUTPUTS, app.scene_change(app.SOURCES[1], 1, KIND3, "Pick the blue cube.",
                                                              (app.SOURCES[0], typed))))
    assert out["instruction"]["value"] == "Pick the blue cube." and out["memory"] == (app.SOURCES[1],
                                                                                      "Pick the blue cube.")
    assert out["state"] is None and "exactly one object" in out["info"] and out["banner"] == ""


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
    assert seen["footer_links"] == ["gradio", "settings"]  # no API-docs link on a page recorded for the video


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


@pytest.mark.parametrize("path", ["\\\\server\\share\\runs", "//server/share/runs",
                                  r"\/server/share/runs", r"/\server/share/runs"])
def test_network_paths_are_refused(path):
    settings = app.UISettings(replay=path)
    assert "network" in app.replay_open(settings, path)[1]
    assert "network" in app.replay_show(settings, path, None)[0]


@pytest.mark.parametrize("path", [r"\/server/share/runs", r"/\server/share/runs"])
def test_mixed_separator_unc_paths_are_refused_before_resolving(path, monkeypatch):
    from vplan.ui import sources

    def resolve_would_access_network(self, *args, **kwargs):
        raise AssertionError("UNC paths must be rejected before Path.resolve()")

    monkeypatch.setattr(Path, "resolve", resolve_would_access_network)
    with pytest.raises(sources.ReplayError, match=r"network \(UNC\)"):
        sources.check_path(path, roots=(r"C:\local\runs",))


@pytest.mark.parametrize("path", ["/local/runs", r"C:\local\runs", r"\rooted\local"])
def test_local_absolute_paths_are_not_classified_as_network_paths(path):
    from vplan.ui import sources

    assert not sources.is_network_path(path)


def test_cli_refuses_a_network_replay_path(launched, capsys):
    assert cli.main(["ui", "--replay", "//server/share/runs"]) == 2 and not launched
    assert "network" in capsys.readouterr().err


@pytest.mark.parametrize("host, warned", [("127.0.0.1", False), ("localhost", False), ("::1", False),
                                          ("0.0.0.0", True), ("172.18.80.1", True)])
def test_a_non_loopback_host_prints_a_warning(host, warned, launched, capsys):
    assert cli.main(["ui", "--host", host]) == 0
    assert ("not a loopback address" in capsys.readouterr().err) is warned


def test_progress_shows_on_one_panel_not_over_every_output():
    """Gradio covers every output with a spinner by default; on video that hides the panels being filled in."""
    config = app.build_app(app.UISettings(replay="runs/ui")).get_config_file()
    ids = {(c["type"], c.get("props", {}).get("label")): c["id"] for c in config["components"]}
    timelines = [c["id"] for c in config["components"] if c.get("props", {}).get("label") == "Checks, round by round"]
    deps = {d["api_name"]: d for d in config["dependencies"]}
    scene = ids[("image", "Scene")]
    for name in ("load_scene", "load_scene_1", "load_scene_2", "scene_change", "task_change", "task_change_1"):
        assert deps[name]["show_progress_on"] == [scene], name
    assert deps["run_live"]["show_progress"] == "minimal"  # the status card itself says "Running"
    assert deps["show_replay"]["show_progress_on"] == [timelines[1]]



# ---- review round (W4.1): stale results, click-time races, fixed texts --------------------------------------------

def _done_state():
    """The state after loading the basic scene with its wall."""
    return app.load_task(app.SOURCES[1], 0, "feasible", live.BASIC_INSTRUCTION)[3]


def test_load_and_enter_clear_old_results_when_the_task_changes():
    state = _done_state()
    out = dict(zip(app.LOAD_OUTPUTS, app.load_scene(app.SOURCES[0], 0, "feasible", live.BASIC_INSTRUCTION, state)))
    assert out["state"][0] != state[0] and out["banner"] == "" and "appear here" in out["timeline"]
    bad = dict(zip(app.LOAD_OUTPUTS, app.load_scene(app.SOURCES[0], 0, "feasible", "Pick the blue cube.", state)))
    assert bad["state"] is None and "exactly one object" in bad["info"] and bad["banner"] == ""


def test_reloading_the_same_task_keeps_its_results_and_reuses_it(monkeypatch):
    """Gradio reads inputs when an event is triggered but State when it runs: a queued load, scene or seed change
    whose task equals the current one must not wipe a finished result (or generate the task again)."""
    state, generated = _done_state(), app.load_task(app.SOURCES[2], 1, KIND3, "")[3]
    monkeypatch.setattr(app, "make_task", lambda *a: (_ for _ in ()).throw(AssertionError("task rebuilt")))
    skipped = tuple(app.gr.skip() for _ in app.PANEL_NAMES)
    out = app.load_scene(app.SOURCES[1], 0, "feasible", live.BASIC_INSTRUCTION, state)
    assert out[4:] == skipped and out[3] is state
    out = app.scene_change(app.SOURCES[1], 0, "feasible", live.BASIC_INSTRUCTION, (app.SOURCES[1], ""), state)
    assert out[-len(skipped):] == skipped and dict(zip(app.SCENE_CHANGE_OUTPUTS, out))["state"] is state
    out = app.task_change(app.SOURCES[2], 1, KIND3, generated)
    assert out[-len(skipped):] == skipped and dict(zip(app.TASK_CHANGE_OUTPUTS, out))["state"] is generated


def test_controls_are_locked_while_a_load_or_run_is_in_progress():
    assert all(u == {"__type__": "update", "interactive": False} for u in app.lock_controls())
    assert all(u == {"__type__": "update", "interactive": True} for u in app.unlock_controls())
    config = app.build_app(app.UISettings()).get_config_file()
    ids = {c.get("props", {}).get("label") or c.get("props", {}).get("value"): c["id"] for c in config["components"]
           if c["type"] in ("button", "radio", "number", "dropdown")}
    controls = sorted(ids[n] for n in ("Plan, verify, execute", "Load scene", "Scene", "Task seed", "Task type"))
    deps = config["dependencies"]
    by_id = {d["id"]: d for d in deps}
    guarded = [d for d in deps if d["api_name"] in ("run_live", "load_scene", "load_scene_1", "load_scene_2",
                                                    "scene_change", "task_change", "task_change_1")]
    assert len(guarded) == 7
    for d in guarded:
        lock = by_id[d["trigger_after"]]
        assert sorted(lock["outputs"]) == controls and lock["queue"] is False, d["api_name"]
        unlock = [u for u in deps if u["trigger_after"] == d["id"]]
        assert len(unlock) == 1 and sorted(unlock[0]["outputs"]) == controls and unlock[0]["queue"] is False


def test_fixed_texts_and_task_cards_carry_no_internal_terms():
    texts = [app.BASIC_INFO, app.GENERATED_INFO, app.BUSY, app.SUBTITLE]
    for source, seed, kind in ((app.SOURCES[0], 0, "feasible"), (app.SOURCES[1], 0, "feasible"),
                               (app.SOURCES[2], 1, KIND3), (app.SOURCES[2], 1, "impossible 4: absent referent")):
        texts.append(visible(app.load_task(source, seed, kind, live.BASIC_INSTRUCTION)[1]))
    found = [t for t in texts if JARGON.search(t)]
    assert not found, found
    assert "Kinematic and collision checks verify it before anything moves" in app.SUBTITLE


def test_the_replay_path_is_shown_relative_to_the_working_directory(tmp_path, monkeypatch):
    result, _ = basic_result()
    monkeypatch.chdir(tmp_path)
    saved = live.save_bundle(tmp_path / "runs" / "ui", result.task, result.record, planner="h", cosmos_url=None)
    _, info = app.replay_open(app.UISettings(record_dir="runs/ui"), str(saved.parent.resolve()))
    assert os.path.join("runs", "ui") in info and str(tmp_path) not in info


def test_the_footer_has_no_leading_divider():
    assert "footer > .divider:first-child{display:none}" in app.markup.CSS


# ---- the Hugging Face Space: bring your own key (byok) ---------------------------------------------------------------

ENDPOINT = "https://api.example.com/v1"


@pytest.fixture
def space(tmp_path):
    """Space settings with one bundled evaluation run (a real one-attempt run and its seed set)."""
    result, _ = basic_result()
    run, seeds = write_eval_run(tmp_path, basic_task(), result.record)
    return app.UISettings(byok=True, record_dir=None, replay_runs=(("Bundled run", str(run), str(seeds)),))


def _components(settings):
    config = app.build_app(settings).get_config_file()
    return config, config["components"]


def _labelled(components, label, kind=None):
    return next(c for c in components if c.get("props", {}).get("label") == label
                and (kind is None or c["type"] == kind))


@pytest.mark.parametrize("kwargs", [{}, {"record_dir": "runs/ui"},
                                    {"record_dir": None, "cosmos_url": "http://127.0.0.1:8000/v1"}])
def test_the_space_settings_refuse_recording_and_cosmos(kwargs):
    """Strangers' attempts are not kept, and the visitor's key must not reach a Cosmos endpoint: both stay off."""
    with pytest.raises(ValueError, match="byok"):
        app.UISettings(byok=True, **kwargs)  # no kwargs: record_dir keeps its default, "runs/ui"
    assert app.UISettings(byok=True, record_dir=None).byok is True
    assert app.UISettings().byok is False and app.UISettings().replay_runs == ()


def test_the_space_replay_roots_are_only_the_bundled_runs_and_seed_sets(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    settings = app.UISettings(byok=True, record_dir=None, replay="elsewhere", seed_dir="elsewhere",
                              replay_runs=(("A", "data/runs/a", "data/seeds/x"), ("B", "data/runs/b", "data/seeds/x")))
    assert settings.replay_roots() == ("data/runs/a", "data/seeds/x", "data/runs/b", "data/seeds/x")
    assert str(tmp_path) not in settings.replay_roots()  # the working directory is not a root there


def test_the_space_page_has_a_key_box_a_provider_form_and_no_server_paths(space):
    config, components = _components(space)
    key = _labelled(components, "API key", "textbox")
    assert key["props"]["type"] == "password" and key["props"].get("value") in (None, "")
    assert "never stored or logged" in key["props"]["info"] and "billed to your account" in key["props"]["info"]
    assert "tool_choice" in key["props"]["info"] and key["props"]["max_lines"] == 1
    assert [c["props"]["label"] for c in components if c["type"] == "textbox"
            and c["props"].get("type") == "password"] == ["API key"]
    provider = _labelled(components, "Provider", "dropdown")
    assert [c[0] for c in provider["props"]["choices"]] == list(live.PROVIDER_PRESETS)
    assert provider["props"]["value"] == "Nebius Token Factory"
    endpoint = _labelled(components, "Endpoint URL", "textbox")
    assert endpoint["props"]["value"] == live.PROVIDER_PRESETS["Nebius Token Factory"][0]
    assert endpoint["props"]["interactive"] is False  # only a custom provider takes a typed URL
    model = _labelled(components, "Model", "dropdown")
    assert model["props"]["allow_custom_value"] is True and model["props"]["value"] == "nvidia/Nemotron-3_5-Lightning"
    planner = _labelled(components, "Planner", "radio")
    assert [c[0] for c in planner["props"]["choices"]] == [live.heuristic_option().label, app.BYOK_LABEL]
    assert planner["props"]["value"] == live.heuristic_option().label  # runs at once, no key needed
    labels = {c.get("props", {}).get("label") for c in components}
    assert not labels & {"Run folder or recording", "Seed-set folder (optional)"}
    runs = _labelled(components, "Recorded run", "dropdown")
    assert [c[0] for c in runs["props"]["choices"]] == ["Bundled run"] and runs["props"]["value"] == "Bundled run"
    assert SECRET not in str(config)


def test_the_normal_page_has_no_key_form():
    components = _components(app.UISettings())[1]
    labels = {c.get("props", {}).get("label") for c in components}
    assert not labels & {"API key", "Provider", "Endpoint URL", "Model", "Recorded run"}
    assert not [c for c in components if c.get("props", {}).get("type") == "password"]
    assert {"Run folder or recording", "Seed-set folder (optional)"} <= labels


def test_the_space_page_carries_no_internal_terms(space):
    texts = []
    for c in _components(space)[1]:
        props = c.get("props", {})
        texts += [str(props[key]) for key in ("label", "info", "placeholder") if props.get(key)]
        texts += [str(choice[0]) for choice in props.get("choices") or []]
    found = [t for t in texts if JARGON.search(t)]
    assert not found, found


def test_the_key_reaches_only_the_run_handler_and_is_never_an_output(space):
    config, components = _components(space)
    key = _labelled(components, "API key")["id"]
    deps = config["dependencies"]
    (run,) = [d for d in deps if d["api_name"] == "run_live"]
    ids = [_labelled(components, label)["id"] for label in ("Endpoint URL", "Model", "API key")]
    assert run["inputs"][-3:] == ids and run["inputs"].count(key) == 1
    assert [d["api_name"] for d in deps if key in d["inputs"]] == ["run_live"]
    assert not [d["api_name"] for d in deps if key in d["outputs"]]
    states = {c["id"] for c in components if c["type"] == "state"}
    assert key not in states and not set(run["inputs"][-3:]) & states


def test_choosing_a_provider_fills_the_endpoint_and_the_model_choices():
    url, model = app.provider_change("OpenRouter")
    assert (url["value"], url["interactive"]) == ("https://openrouter.ai/api/v1", False)
    assert model["choices"] == list(live.PROVIDER_PRESETS["OpenRouter"][1])
    assert model["value"] == "nvidia/nemotron-3.5-lightning:free"
    url, model = app.provider_change("Custom (OpenAI-compatible)")
    assert (url["value"], url["interactive"]) == ("", True) and model["choices"] == [] and model["value"] is None


def test_the_key_form_shows_only_for_the_visitors_own_model():
    assert app.show_key_form(app.BYOK_LABEL)["visible"] is True
    assert app.show_key_form(live.heuristic_option().label)["visible"] is False


def _byok_steps(settings, monkeypatch, option, **given):
    seen = []

    def key_option(*args, **kwargs):
        seen.append((args, kwargs))
        return option

    monkeypatch.setattr(live, "key_option", key_option)
    state = app.load_task(app.SOURCES[1], 0, "feasible", live.BASIC_INSTRUCTION)[3]
    args = {"url": ENDPOINT, "model": "some/model", "api_key": SECRET, **given}
    steps = list(app.run_attempt(settings, state, app.SOURCES[1], 0, "feasible", live.BASIC_INSTRUCTION,
                                 app.BYOK_LABEL, 3, **args))
    return steps, seen


def test_a_space_run_builds_the_planner_from_the_visitors_input_and_shows_no_key(space, monkeypatch):
    planner = live.heuristic_option().planner  # stands in for the model; the key is only the option's secret
    option = live.PlannerOption("Your key: some/model via api.example.com", planner, llm=True, secret=SECRET)
    steps, seen = _byok_steps(space, monkeypatch, option)
    assert seen == [((ENDPOINT, SECRET, "some/model"), {"public": True})]  # built for this run, public checks on
    final = dict(zip(app.display.Display.FIELDS, steps[-1]))
    assert "Success" in final["judge_html"] and "Your key: some/model via api.example.com" in final["status_html"]
    assert SECRET not in repr(steps) and SECRET not in repr(space)


def test_a_failure_inside_a_space_run_is_shown_without_the_key(space, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError(f"transport failed with Authorization: Bearer {SECRET}")

    monkeypatch.setattr(live, "run_live", boom)
    option = live.PlannerOption("Your key: some/model via api.example.com", object(), llm=True, secret=SECRET)
    steps, _ = _byok_steps(space, monkeypatch, option)
    assert "the attempt stopped: RuntimeError" in steps[-1][0] and SECRET not in repr(steps)


@pytest.mark.parametrize("given, message", [({"api_key": ""}, "Enter your API key"), ({"model": " "}, "model ID"),
                                            ({"url": "http://api.example.com/v1"}, "https URL")])
def test_a_refused_space_endpoint_is_reported_in_the_banner(space, given, message):
    args = {"url": ENDPOINT, "model": "some/model", "api_key": SECRET, **given}
    (values,) = app.run_attempt(space, None, app.SOURCES[0], 0, "feasible", live.BASIC_INSTRUCTION, app.BYOK_LABEL,
                                1, **args)
    assert message in values[0] and SECRET not in repr(values)


def test_the_own_model_choice_exists_only_on_the_space():
    settings = app.UISettings(record_dir=None)
    (values,) = app.run_attempt(settings, None, app.SOURCES[0], 0, "feasible", live.BASIC_INSTRUCTION, app.BYOK_LABEL,
                                1, ENDPOINT, "some/model", SECRET)
    assert "unknown planner" in values[0] and SECRET not in repr(values)


def test_space_replay_names_bundled_runs_by_label_and_nothing_else(space, tmp_path):
    (label, run_dir, seeds), = space.replay_runs
    assert app.replay_target(space, label) == (run_dir, seeds)
    assert app.replay_target(space, "/etc") == ("", None) and app.replay_target(space, run_dir) == ("", None)
    assert app.replay_target(app.UISettings(), "runs/x", "seeds") == ("runs/x", "seeds")  # typed paths: as before
    case, info = app.replay_open(space, run_dir)
    assert case["value"] == "5:dev:0" and "1 recorded attempt" in visible(info)
    values = app.replay_show(space, run_dir, case["value"], seeds)
    assert "Success" in dict(zip(app.display.Display.FIELDS, values))["judge_html"]
    outside = write_eval_run(tmp_path / "outside", basic_task(), basic_result()[0].record)
    assert "outside" in app.replay_open(space, str(outside[0]))[1]  # not a bundled run: refused by the roots
    assert "outside" in app.replay_show(space, str(outside[0]), None, str(outside[1]))[0]
    assert "outside" in app.replay_show(space, "", None, None)[0]  # what an unknown label becomes
