"""W3.5 replay mode (spec §15, D41): recorded attempts from W3.1 runs, pilot runs and UI recordings, replayed with no
keys, no GPU and no network."""
import json

import pytest

from tests.ui_fixtures import JARGON, basic_result, basic_task, enclosed_result, magenta_pixels, visible, write_eval_run
from vplan import records
from vplan.ui import display, live, render, sources


@pytest.fixture
def recording(tmp_path):
    result, _ = basic_result()
    return live.save_bundle(tmp_path / "ui", result.task, result.record, planner=live.heuristic_option().label,
                            cosmos_url=None)


def test_a_recording_round_trips(recording):
    result, _ = basic_result()
    assert sources.source_kind(recording) == "recording"
    (ref,) = sources.list_cases(recording)
    case = sources.load_case(recording)
    assert ref.case_id == recording.name and case.record == result.record
    assert case.task == result.task.task and case.oracle == result.task.oracle and case.seed == 0
    assert dict(case.provenance)["Cosmos visual check"] == "off"
    assert dict(case.provenance)["Planner"] == "Rule-based baseline (no LLM)"
    assert ref.label.endswith(" · Basic scene with wall: Pick the red cube and place it in the bin. · Executed · success")


def test_a_folder_of_recordings_lists_newest_first_and_skips_other_json(recording, tmp_path):
    result, _ = enclosed_result()
    second = live.save_bundle(recording.parent, result.task, result.record, planner="heuristic", cosmos_url=None)
    (recording.parent / "notes.json").write_text("{}", encoding="utf-8")
    refs = sources.list_cases(recording.parent)
    assert [r.case_id for r in refs] == [second.name, recording.name]
    assert sources.load_case(recording.parent, second.name).record.outcome == "budget_exhausted"
    with pytest.raises(sources.ReplayError):
        sources.load_case(recording.parent, "../escape.json")


def test_replay_display_plays_back_the_recorded_execution(recording):
    shown = display.replay_display(sources.load_case(recording))
    assert "data:image/gif;base64," in shown.execution_html and "no physics rerun" in shown.execution_html
    assert "Success" in shown.judge_html and "without the Cosmos visual check" in shown.banner_html
    assert "saved live attempt" in shown.status_html and "Ground truth (hidden from the planner)" in shown.status_html
    assert shown.status_html.count("Pick the red cube and place it in the bin.") == 1  # the Task line only
    for name in ("banner_html", "status_html", "timeline_html", "execution_html", "judge_html"):
        assert not JARGON.search(visible(getattr(shown, name))), name


def test_recordings_made_before_the_rename_show_the_new_planner_label(tmp_path):
    result, _ = basic_result()
    path = live.save_bundle(tmp_path, result.task, result.record, planner="Template heuristic (no LLM, no key)",
                            cosmos_url=None)
    assert dict(sources.load_case(path).provenance)["Planner"] == "Rule-based baseline (no LLM)"


def test_replay_rerenders_highlights_for_blocked_rounds(tmp_path):
    result, _ = enclosed_result()
    path = live.save_bundle(tmp_path, result.task, result.record, planner="heuristic", cosmos_url=None)
    shown = display.replay_display(sources.load_case(path))
    assert shown.findings and any(magenta_pixels(img) > 50 for img, _ in shown.findings)
    assert all(c.startswith("Round ") and not JARGON.search(c) for _, c in shown.findings)
    assert "Nothing was executed" in shown.execution_html and "no plan passed the checks" in shown.status_html
    text = visible(shown.timeline_html)
    assert "enclosure" in text and not JARGON.search(text), JARGON.search(text)


def test_replay_shows_a_cosmos_report_and_rerenders_its_frames(tmp_path):
    result, _ = basic_result()
    check = records.VisualCheck(stage="grasp", views=("front_oblique", "top"), verdict=records.VisualVerdict(
        verdict="pass", reason="between the fingers", latency_s=1.2, raw=""))
    report = records.VisualGateReport(status="pass", checks=(check,))
    rnd = result.record.rounds[0]
    with_gate = result.record.model_copy(update={"rounds": (rnd.model_copy(update={
        "evaluation": rnd.evaluation.model_copy(update={"visual": report})}),)})
    path = live.save_bundle(tmp_path, result.task, with_gate, planner="heuristic", cosmos_url="http://127.0.0.1:8000")
    shown = display.replay_display(sources.load_case(path), render_execution=False)
    assert [c for _, c in shown.cosmos] == ["Pass (learned check) · Round 1 · grasp, front oblique view (redrawn)",
                                           "Pass (learned check) · Round 1 · grasp, top view (redrawn)"]
    assert "not verification" in shown.banner_html and "between the fingers" in shown.timeline_html


def test_a_w31_run_directory_replays_with_its_seed_set(tmp_path):
    result, _ = basic_result()
    run, seeds = write_eval_run(tmp_path, basic_task(), result.record)
    (ref,) = sources.list_cases(run)
    assert ref.case_id == "5:dev:0" and ref.label == "Task 0 (dev set) · Executed · success"
    case = sources.load_case(run, "5:dev:0")
    assert case.task == basic_task().task and case.record.outcome == "executed" and case.seed == 0
    assert dict(case.provenance)["Source"] == "evaluation run run, rule-based baseline (config 5)"
    assert dict(case.provenance)["Planner"] == "Rule-based baseline"
    shown = display.replay_display(case, render_execution=False)
    assert "Success" in shown.judge_html and "Attempt id: 5:dev:0" in shown.status_html


def test_a_w31_run_refuses_a_seed_set_that_does_not_hash_to_the_run(tmp_path):
    result, _ = basic_result()
    run, seeds = write_eval_run(tmp_path, basic_task(), result.record)
    (seeds / "oracle.jsonl").write_text((seeds / "oracle.jsonl").read_text(encoding="utf-8") + "\n",
                                        encoding="utf-8")
    with pytest.raises(sources.ReplayError, match="hash"):
        sources.load_case(run)


def test_a_w31_run_takes_an_explicit_seed_dir(tmp_path):
    result, _ = basic_result()
    run, seeds = write_eval_run(tmp_path, basic_task(), result.record)
    moved = seeds.rename(tmp_path / "elsewhere")
    with pytest.raises(sources.ReplayError, match="seed set"):
        sources.load_case(run)
    assert sources.load_case(run, seed_dir=moved).record.outcome == "executed"


def test_a_w31_row_without_a_record_cannot_be_replayed(tmp_path):
    result, _ = basic_result()
    run, _ = write_eval_run(tmp_path, basic_task(), result.record)
    crash = records.EvalAttempt(attempt_id="5:dev:0", config=5, split="dev", task_index=0, task_id="dev_00000000",
                                session=0, outcome="sim_error", tries=1, wall_s=600.0, error_type="worker_timeout")
    (run / "attempts.jsonl").write_text(crash.model_dump_json() + "\n", encoding="utf-8")
    with pytest.raises(sources.ReplayError, match="worker_timeout"):
        sources.load_case(run)


def test_a_pilot_run_replays_each_arm(tmp_path):
    result, _ = basic_result()
    task = basic_task()
    cohort = tmp_path / "cohort"
    run = cohort / "runs" / "r1"
    run.mkdir(parents=True)
    (cohort / "tasks.jsonl").write_text(json.dumps({"task_id": "t1", "seed": 7, "stratum": "obstacle",
                                                    "task": task.task.model_dump(mode="json")}) + "\n",
                                        encoding="utf-8")
    (cohort / "oracle.jsonl").write_text(json.dumps({"task_id": "t1", "oracle": task.oracle.model_dump(mode="json")})
                                         + "\n", encoding="utf-8")
    import hashlib
    files = {n: hashlib.sha256((cohort / n).read_bytes()).hexdigest() for n in ("tasks.jsonl", "oracle.jsonl")}
    (cohort / "manifest.json").write_text(json.dumps({"files": files}), encoding="utf-8")
    (run / "run.json").write_text(json.dumps({"provider": "openrouter", "model": "nvidia/ultra:free"}),
                                  encoding="utf-8")
    first = result.record.rounds[0]
    arm = {"outcome": "executed", "rounds": (first,), "call_count": 1, "total_tokens": None, "known_total_tokens": 0,
           "missing_usage_calls": 1, "latency_s": 0.0}
    paired = records.PairedResult(task_id="t1", initial=first, repair=records.ArmResult(arm="repair", **arm),
                                  resample=records.ArmResult(arm="resample", **arm), unique_call_count=1,
                                  unique_total_tokens=None, unique_known_total_tokens=0, unique_missing_usage_calls=1,
                                  unique_latency_s=0.0, config_hashes=result.record.config_hashes)
    (run / "results.jsonl").write_text(paired.model_dump_json() + "\n", encoding="utf-8")
    assert [r.case_id for r in sources.list_cases(run)] == ["t1:repair", "t1:resample"]
    case = sources.load_case(run, "t1:resample")
    assert case.record.mode == "resample" and case.record.success and case.seed == 7
    assert dict(case.provenance)["Planner"] == "nvidia/ultra:free via openrouter"
    assert "fresh samples" in display.replay_display(case, render_execution=False).status_html


@pytest.mark.parametrize("make", ["missing", "seed_set", "empty_dir"])
def test_unreplayable_paths_are_refused_with_a_message(tmp_path, make):
    path = tmp_path / "x"
    if make == "seed_set":
        path.mkdir()
        (path / "manifest.json").write_text(json.dumps({"kind": "seed_set"}), encoding="utf-8")
    elif make == "empty_dir":
        path.mkdir()
    with pytest.raises(sources.ReplayError):
        sources.list_cases(path)


def test_replay_of_a_record_from_another_scene_reports_instead_of_crashing(recording):
    case = sources.load_case(recording)
    other = live.generated_task(1, "impossible 3: target enclosed")
    moved = sources.ReplayCase(case.case_id, case.label, other.task, None, case.record, 0, "test")
    shown = display.replay_display(moved)
    assert "Cannot play back" in shown.execution_html and "different scene" in shown.execution_html


def test_execution_frames_sample_the_recording_at_the_gif_rate():
    result, _ = basic_result()
    rec = result.record.rounds[-1].evaluation.record
    frames = render.execution_frames(rec, basic_task().task.scene, fps=10)
    n = len(rec.q_replay)
    assert len(frames) == len(range(0, n, 6)) + ((n - 1) % 6 != 0)
    assert frames[0].shape == (render.UI_SIZE_PX, render.UI_SIZE_PX, 3)
    assert render.gif_bytes(frames[:3]).startswith(b"GIF89a")


# ---- damaged files: always a ReplayError with a message, never a crash -------------------------------------------

def _damage_attempts(tmp_path, line):
    result, _ = basic_result()
    run, _ = write_eval_run(tmp_path, basic_task(), result.record)
    (run / "attempts.jsonl").write_text(line + "\n", encoding="utf-8")
    return run


@pytest.mark.parametrize("line", ["[1]", '{"outcome": "executed"}', '"text"', '{"attempt_id": 5, "outcome": 1}'])
def test_damaged_attempt_rows_are_refused_with_a_message(tmp_path, line):
    run = _damage_attempts(tmp_path, line)
    with pytest.raises(sources.ReplayError):
        sources.list_cases(run)
    with pytest.raises(sources.ReplayError):
        sources.load_case(run)


def test_a_damaged_run_manifest_is_refused(tmp_path):
    result, _ = basic_result()
    run, _ = write_eval_run(tmp_path, basic_task(), result.record)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    (run / "manifest.json").write_text(json.dumps({**manifest, "seed_set": [1]}), encoding="utf-8")
    with pytest.raises(sources.ReplayError):
        sources.load_case(run)


def test_a_pilot_row_without_a_task_id_is_refused(tmp_path):
    run = tmp_path / "cohort" / "runs" / "r1"
    run.mkdir(parents=True)
    (run / "run.json").write_text("{}", encoding="utf-8")
    (run / "results.jsonl").write_text('{"repair": {"outcome": "executed"}}\n', encoding="utf-8")
    with pytest.raises(sources.ReplayError):
        sources.list_cases(run)
    with pytest.raises(sources.ReplayError):
        sources.load_case(run, "t1:repair")


def _recording_with(tmp_path, **changes):
    result, _ = basic_result()
    path = live.save_bundle(tmp_path / "ui", result.task, result.record, planner="h", cosmos_url=None)
    data = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**data, **changes}), encoding="utf-8")
    return path


@pytest.mark.parametrize("changes", [{"record": [1]}, {"seed": "x"}, {"task": "scene"}, {"label": ["x"]}])
def test_damaged_recordings_are_refused_with_a_message(tmp_path, changes):
    path = _recording_with(tmp_path, **changes)
    sources.list_cases(path.parent)  # a listing still names the file
    with pytest.raises(sources.ReplayError):
        sources.load_case(path)
    with pytest.raises(sources.ReplayError):
        sources.load_case(path.parent, path.name)


def test_the_ui_and_the_cli_report_a_damaged_source_cleanly(tmp_path, capsys, monkeypatch):
    from vplan import cli
    from vplan.ui import gradio_app as app
    run = _damage_attempts(tmp_path, "[1]")
    case, info = app.replay_open(app.UISettings(replay=str(run)), str(run))
    assert case["choices"] == [] and "line 1" in info
    monkeypatch.setattr(app, "launch", lambda settings: None)
    assert cli.main(["ui", "--replay", str(run)]) == 2
    assert "error:" in capsys.readouterr().err


def test_case_lists_and_provenance_carry_no_internal_terms(tmp_path, recording):
    result, _ = basic_result()
    run, _ = write_eval_run(tmp_path / "eval", basic_task(), result.record, config=3)
    texts = []
    for source in (run, recording, recording.parent):
        texts += [ref.label for ref in sources.list_cases(source)]
        texts += [f"{k} {v}" for k, v in sources.load_case(source).provenance]
    assert "evaluation run run, verify + repair (config 3)" in texts[1]
    found = [t for t in texts if JARGON.search(t)]
    assert not found, found


def test_a_type4_replay_never_claims_a_correct_object(tmp_path):
    """The Judge scores S1/S2 on the planned object when the instruction names no object (type 4)."""
    task = live.generated_task(1, "impossible 4: absent referent")
    result, _ = basic_result()
    plan = result.record.rounds[0].call.plan
    obj = task.task.scene.objects[0].id
    judged = records.JudgeResult(success=False, predicates={"S1": True, "S2": True, "V1": True, "V2": True,
                                                           "V3": True, "V4": True, "R": False},
                                 violations=("R",), details={"refusal": "impossible task attempted"})
    rnd = result.record.rounds[0].model_copy(update={
        "call": result.record.rounds[0].call.model_copy(update={"plan": plan.model_copy(update={"target_id": obj})}),
        "evaluation": records.EvaluationResult(outcome="executed", judge=judged)})
    record = result.record.model_copy(update={"rounds": (rnd,), "judge": judged})
    path = live.save_bundle(tmp_path, task, record, planner="h", cosmos_url=None)
    shown = display.replay_display(sources.load_case(path), render_execution=False)
    text = visible(shown.judge_html)
    assert "Only the planned object in the bin" in text and "Correct object" not in text and "instructed" not in text
    assert "the planner should have refused" in visible(shown.status_html)
