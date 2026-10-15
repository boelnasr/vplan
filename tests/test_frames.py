"""Key frames for the config-4 Cosmos shadow study (spec §11, §14 frames/): rendered by the worker after an attempt's
result is final, from the round's plan and its rebuilt trajectory, in separate DIRECT clients. One real executed
attempt (heuristic plan on the basic scene with its obstacle) is simulated once per module; nothing calls Cosmos."""
from __future__ import annotations

import hashlib
import io
import json

import pytest
from PIL import Image
from pydantic import ValidationError

from tests import eval_rows as ev
from vplan import records
from vplan.config import config_snapshot

SEED = 7
ATTEMPT = "3:dev:4"
SPEC = ev.SPEC.model_copy(update={"instruction": "Pick the red cube and place it in the bin."})  # a template phrase


@pytest.fixture(scope="module")
def executed():
    from vplan.planners.heuristic import HeuristicPlanner
    from vplan.repair import run_task
    record = run_task(SPEC, ev.FEASIBLE, HeuristicPlanner(), k=3, mode="repair", attempt_id=ATTEMPT, seed=SEED,
                      config_snapshot=config_snapshot())
    assert record.outcome == "executed" and record.success
    return record


def export(record, tmp_path, **kw):
    from vplan.evaluate.frames import export_frames
    return export_frames(record, SPEC, seed=SEED, config_snapshot=config_snapshot(), out_dir=tmp_path / "0004",
                         rel_dir="frames/0004", **kw)


@pytest.fixture(scope="module")
def written(executed, tmp_path_factory):
    out = tmp_path_factory.mktemp("frames")
    return export(executed, out), out / "0004"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_every_gate_stage_and_view_of_the_executed_round_is_written_with_an_index(written, executed):
    from vplan.evaluate.frames import FRAME_VIEWS, plan_hash
    result, folder = written
    final = executed.rounds[-1]
    assert result.status == "written" and result.error_type is None
    assert result.rounds == (final.round_index,) and result.files == 9 and FRAME_VIEWS == ("front_oblique", "top",
                                                                                           "side")
    assert result.index == "frames/0004/index.json"
    index_bytes = (folder / "index.json").read_bytes()
    assert result.index_sha256 == sha(index_bytes)
    index = records.FrameIndex.model_validate_json(index_bytes)
    assert (index.attempt_id, index.subject) == (ATTEMPT, "red cube")
    assert index.gate_views == ("front_oblique", "top") and index.retry_views == FRAME_VIEWS
    assert [(f.stage, f.view) for f in index.frames] == [(s, v) for s in ("pregrasp", "grasp", "prerelease")
                                                         for v in FRAME_VIEWS]
    for f in index.frames:
        data = (folder / f.file).read_bytes()
        assert f.file == f"r{final.round_index}-{f.stage}-{f.view}.png" and ":" not in f.file
        assert f.sha256 == sha(data) and f.attempt_id == ATTEMPT and f.round_index == final.round_index
        assert f.scene_hash == records.scene_hash(SPEC.scene) and f.plan_hash == plan_hash(final.call.plan)
        assert f.approval == final.evaluation.record.approval
        assert Image.open(io.BytesIO(data)).size == (512, 512)
    assert sorted(p.name for p in folder.iterdir()) == sorted([f.file for f in index.frames] + ["index.json"])
    assert result.bytes == sum(p.stat().st_size for p in folder.iterdir()) and result.render_s > 0


def test_frames_are_deterministic(written, executed, tmp_path):
    again = export(executed, tmp_path)
    assert again.index_sha256 == written[0].index_sha256


def test_the_frames_show_the_executed_trajectory_or_nothing(executed, tmp_path):
    # The rebuilt trajectory must reproduce the executed approval token; a record that does not match is refused.
    final = executed.rounds[-1]
    forged = final.evaluation.record.model_copy(update={"approval": "0" * 64})
    tampered = executed.model_copy(update={"rounds": executed.rounds[:-1] + (final.model_copy(update={
        "evaluation": final.evaluation.model_copy(update={"record": forged})}),)})
    result = export(tampered, tmp_path)
    assert result.status == "failed" and result.error_type == "FrameMismatch" and "approval" in result.error_message
    assert not (tmp_path / "0004").exists() and result.files == 0 and result.index is None


def test_a_rendering_failure_is_reported_not_raised(executed, tmp_path):
    def broken(*args, **kwargs):
        raise RuntimeError("TinyRenderer gave up")
    result = export(executed, tmp_path, render=broken)
    assert result.status == "failed" and result.error_type == "RuntimeError"
    assert "TinyRenderer gave up" in result.error_message and not (tmp_path / "0004").exists()


def test_an_attempt_without_an_executed_round_has_no_frames(tmp_path):
    refused = ev.attempt(ev.impossible(1), [ev.call(ev.refusal())])
    blocked = ev.attempt(ev.FEASIBLE, [ev.call(ev.GOOD)], [ev.blocked(ev.COLLISION)], k=0)
    for record in (refused, blocked):
        result = export(record, tmp_path)
        assert (result.status, result.files, result.index, result.rounds) == ("none", 0, None, ())
    assert not (tmp_path / "0004").exists()


@pytest.mark.parametrize("fields, message", [
    ({"status": "written"}, "index"), ({"status": "failed"}, "error_type"),
    ({"status": "none", "files": 3}, "none"), ({"status": "written", "index": "frames/0001/index.json",
                                                "index_sha256": "ab" * 32, "files": 0, "rounds": (0,)}, "files")])
def test_frame_export_rows_are_consistent(fields, message):
    with pytest.raises(ValidationError, match=message):
        records.FrameExport(**fields)


def test_frame_file_names_are_plain_png_names():
    entry = dict(attempt_id=ATTEMPT, round_index=0, stage="grasp", view="top", file="r0-grasp-top.png",
                 sha256="ab" * 32, scene_hash="cd" * 32, plan_hash="ef" * 32, approval="01" * 32)
    records.FrameEntry(**entry)
    for bad in ("r0:grasp:top.png", "../r0.png", "sub/r0.png", "r0.jpg"):
        with pytest.raises(ValidationError):
            records.FrameEntry(**{**entry, "file": bad})


def job_params(tmp_path) -> dict:
    from vplan.evaluate import runner
    return {"attempt_id": ATTEMPT, "task": SPEC.model_dump(mode="json"),
            "oracle": ev.FEASIBLE.model_dump(mode="json"), "seed": SEED, "config_snapshot": config_snapshot(),
            "mode": "repair", "k": 3, "planner": runner.planner_settings(runner.CONFIGS[5]).params(),
            "pacing": None, "frames": {"dir": str(tmp_path / "frames" / "0004"), "rel": "frames/0004"}}


def test_the_worker_job_renders_after_its_result_and_never_changes_it(executed, tmp_path, monkeypatch):
    # run_task_job returns the finished record first; the frame export is its after step (worker.AFTER).
    from vplan.evaluate import runner, worker
    from vplan import repair
    monkeypatch.setattr(repair, "run_task", lambda *a, **kw: executed)
    params = job_params(tmp_path)
    out = runner.run_task_job(params)
    before = json.dumps(out["payload"]["record"], sort_keys=True)
    assert callable(out[worker.AFTER]) and not (tmp_path / "frames").exists()
    extra = out[worker.AFTER]()
    assert json.dumps(out["payload"]["record"], sort_keys=True) == before == json.dumps(
        executed.model_dump(mode="json"), sort_keys=True)
    assert records.FrameExport.model_validate(extra["frames"]).status == "written"
    assert (tmp_path / "frames" / "0004" / "index.json").is_file()
    assert worker.AFTER not in runner.run_task_job({**params, "frames": None})


def test_the_after_step_reports_its_whole_time_including_the_rendering_imports(executed, tmp_path, monkeypatch):
    # Importing the rendering stack takes about 0.4 s in a fresh worker. It belongs to the export, whose time report
    # rows subtract from the attempt's wall time (evaluate.collect.report_wall_s).
    from types import SimpleNamespace
    from vplan import repair
    from vplan.evaluate import frames, runner, worker
    monkeypatch.setattr(repair, "run_task", lambda *a, **kw: executed)
    monkeypatch.setattr(frames, "export_frames", lambda *a, **kw: records.FrameExport(status="none", render_s=0.5))
    monkeypatch.setattr(runner, "time", SimpleNamespace(perf_counter=iter([10.0, 12.5]).__next__))
    after = runner.run_task_job(job_params(tmp_path))[worker.AFTER]
    assert after()["frames"] == records.FrameExport(status="none", render_s=2.5).model_dump(mode="json")
