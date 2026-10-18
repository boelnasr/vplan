"""W3.5 Gradio wiring and `vplan ui`: the app builds without a browser, its handlers drive both modes, the replay
starts with no .env, and LLM credentials come only from --env-file behind the free-model guard (D33)."""
import os

import numpy as np
import pytest

from tests.ui_fixtures import SECRET, basic_result
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
    case, _ = app.replay_open(settings.replay)
    values = app.replay_show(settings.replay, case["value"])
    shown = dict(zip(("banner", "status", "plan", "timeline", "findings", "cosmos", "execution", "judge"), values))
    assert "data:image/gif;base64," in shown["execution"] and "SUCCESS" in shown["judge"]
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
    assert "WSL path" in app.replay_open("/mnt/c/runs/x")[1]


def test_cli_settings_carry_cosmos_host_port_and_recording(tmp_path, launched):
    assert cli.main(["ui", "--cosmos-url", "http://127.0.0.1:8000/v1", "--host", "0.0.0.0", "--port", "7999",
                     "--no-record"]) == 0
    (s,) = launched
    assert (s.cosmos_url, s.host, s.port, s.record_dir) == ("http://127.0.0.1:8000/v1", "0.0.0.0", 7999, None)
    assert cli.main(["ui"]) == 0 and launched[1].record_dir == "runs/ui"


# ---- handlers --------------------------------------------------------------------------------------------------------

def test_live_handlers_load_a_scene_and_stream_an_attempt_to_the_judge():
    image, info, instruction, state = app.load_task(app.SOURCES[1], 0, "feasible", live.BASIC_INSTRUCTION)
    assert isinstance(image, np.ndarray) and image.shape[2] == 3 and "feasible" in info
    settings = app.UISettings(record_dir=None)
    steps = list(app.run_attempt(settings, state, app.SOURCES[1], 0, "feasible", instruction,
                                 settings.planners[0].label, 3))
    assert len(steps) >= 4  # start, request, plan, round, status ... final
    final = dict(zip(app.display.Display.FIELDS, steps[-1]))
    assert "SUCCESS" in final["judge_html"] and "data:image/gif;base64," in final["execution_html"]
    assert "geometric checks only" in final["banner_html"]


def test_live_handlers_report_bad_input_in_the_banner():
    _, info, _, state = app.load_task(app.SOURCES[0], 0, "feasible", "Pick the blue cube and place it in the bin.")
    assert state is None and "exactly one object" in info
    settings = app.UISettings(record_dir=None)
    (values,) = app.run_attempt(settings, None, app.SOURCES[2], 1, "impossible 3: target enclosed",
                                "Pick the red cube and place it in the bin.", settings.planners[0].label, 1)
    assert "fixed by its oracle" in values[0]
    (values,) = app.run_attempt(settings, None, app.SOURCES[0], 0, "feasible", live.BASIC_INSTRUCTION, "nope", 1)
    assert "unknown planner" in values[0]


def test_replay_handlers_report_problems_in_the_panels(tmp_path):
    case, info = app.replay_open("")
    assert case["choices"] == [] and "enter a path" in info
    case, info = app.replay_open(str(tmp_path / "missing"))
    assert "does not exist" in info
    assert "does not exist" in app.replay_show(str(tmp_path / "missing"), None)[0]


def test_the_app_builds_with_both_tabs_and_the_llm_planner_preselected(tmp_path):
    import gradio as gr
    llm = live.llm_option(_env(tmp_path))
    demo = app.build_app(app.UISettings(planners=(live.heuristic_option(), llm), replay=str(tmp_path)))
    assert isinstance(demo, gr.Blocks)
    config = demo.get_config_file()
    labels = {c.get("props", {}).get("label") for c in config["components"]}
    assert {"Live", "Replay", "Planner", "Check timeline (per repair round)"} <= labels
    radio = next(c for c in config["components"] if c.get("props", {}).get("label") == "Planner")
    assert radio["props"]["value"] == llm.label
    assert SECRET not in str(config)


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
