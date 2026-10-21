"""W3.5 live mode, offline: the heuristic planner (no key), a fake provider client for the LLM path, and a faked
Cosmos endpoint. Covers task loading, streamed events, highlighted findings, the Cosmos banner before motion, the
execution GIF and the recording for keyless replay."""
import json

import pytest

from tests.test_planner import FakeClient, response
from tests.test_visual_gate import FakeAsk
from tests.ui_fixtures import (JARGON, SECRET, basic_result, basic_task, enclosed_result, low_transport,
                               magenta_pixels, visible)
from vplan import pipeline
from vplan.evaluate.pilot import PilotError
from vplan.planners.heuristic import heuristic_geometry
from vplan.ui import display, live, render, views
from vplan.visual import gate as visual_gate
from vplan.visual.cosmos import CosmosUnavailable


# ---- tasks ---------------------------------------------------------------------------------------------------------

def test_basic_task_resolves_the_instruction_and_carries_a_feasible_oracle():
    task = basic_task()
    assert task.oracle.feasibility == "feasible" and task.oracle.target_id == "cube_red"
    assert task.task.instruction == live.BASIC_INSTRUCTION and len(task.task.scene.obstacles) == 1
    assert task.truth == "solvable" and task.label == f"Basic scene with wall: {live.BASIC_INSTRUCTION}"


@pytest.mark.parametrize("instruction", ["Pick the blue cube and place it in the bin.", "put the red cube away",
                                         "Pick the cube and place it in the bin."])  # absent, off-template, ambiguous
def test_basic_task_refuses_instructions_without_exactly_one_object(instruction):
    with pytest.raises(live.LiveError, match="exactly one object"):
        live.basic_task(instruction, obstacle=False)


def test_generated_impossible_task_carries_its_proof():
    task = live.generated_task(1, "impossible 3: target enclosed")
    assert task.oracle.feasibility == "impossible" and task.oracle.proof.type == 3
    assert task.truth == "impossible: object enclosed, so the correct answer is to refuse" and task.seed == 1
    assert task.label.startswith("Generated task, seed 1 (Impossible: object enclosed): Pick ")
    assert task.enclosure == frozenset(o.id for o in task.oracle.proof.enclosure) and len(task.enclosure) == 5


@pytest.mark.parametrize("seed, kind", [(-1, "feasible"), (0, "impossible 9"), (True, "feasible"), (0, "feasible")])
def test_generated_task_refuses_bad_input(seed, kind):
    with pytest.raises(live.LiveError) as refused:
        live.generated_task(seed, kind)
    assert not JARGON.search(str(refused.value)), refused.value  # shown in the page's banner


def test_task_types_and_the_baseline_have_plain_labels():
    assert list(live.KIND_LABELS.values()) == ["Solvable", "Impossible: object out of reach",
                                               "Impossible: bin out of reach", "Impossible: object enclosed",
                                               "Impossible: object not in scene"]
    assert list(live.KIND_LABELS) == list(live.TASK_KINDS)  # the dropdown's values stay the generator's kinds
    assert live.heuristic_option().label == "Rule-based baseline (no LLM)"


def test_an_unsolvable_basic_instruction_is_refused_in_plain_words(monkeypatch):
    from vplan.tasks import generate as gen
    monkeypatch.setattr(gen, "witness_rejection", lambda witness, target: "blocked")
    with pytest.raises(live.LiveError) as refused:
        live.basic_task(live.BASIC_INSTRUCTION, obstacle=False)
    assert "rule-based baseline could not solve" in str(refused.value)
    assert not JARGON.search(str(refused.value)), refused.value


# ---- the attempt ---------------------------------------------------------------------------------------------------

def test_heuristic_attempt_streams_events_and_ends_executed_with_a_gif():
    result, events = basic_result()
    assert result.record.outcome == "executed" and result.record.success
    assert [e.kind for e in events][:3] == ["request", "plan", "round"]
    assert events[1].call.plan.target_id == "cube_red"  # the plan is shown before verification finishes
    assert result.execution_gif.startswith(b"GIF89a") and "60 Hz" in result.execution_note
    assert result.shots == () and result.cosmos == () and result.bundle is None


def test_rejected_rounds_render_the_offending_bodies_highlighted():
    result, events = enclosed_result()
    assert result.record.outcome == "budget_exhausted" and result.record.call_count == 2
    rounds = [e for e in events if e.kind == "round"]
    assert len(rounds) == 2 and all(e.round.evaluation.findings for e in rounds)
    assert result.shots and all(s.caption.startswith("Round 1 · Collision: ") for s in result.shots)
    assert [r.repeat_plan for r in result.record.rounds] == [False, True]  # round 2 repeats it: no second picture
    collision = [s for s in result.shots if "Collision" in s.caption]
    assert all("enclosure" in s.caption for s in collision)
    assert not any(JARGON.search(s.caption) for s in result.shots), [s.caption for s in result.shots]
    assert collision and all(magenta_pixels(s.image) > 50 for s in collision)
    assert all(s.close_up is not None and magenta_pixels(s.close_up) > magenta_pixels(s.image) for s in collision)
    assert result.execution_gif is None and result.execution_note == "Nothing was executed."


def _outside_frame(h, camera, margin: float = 0.03) -> list[str]:
    """Names of bodies (and robot links) with a bounding-box corner outside the camera's image, with a margin."""
    import itertools

    import numpy as np
    import pybullet as p
    eye, target, up = camera
    view = np.reshape(p.computeViewMatrix(eye, target, up, physicsClientId=h.client), (4, 4), order="F")
    proj = np.reshape(p.computeProjectionMatrixFOV(render.FOV_DEG, 1.0, 0.02, 6.0, physicsClientId=h.client), (4, 4),
                      order="F")
    boxes = [(n, p.getAABB(b, physicsClientId=h.client)) for n, b in h.bodies.items() if n != "table"]
    boxes += [(f"link {j}", p.getAABB(h.robot, j, physicsClientId=h.client)) for j in range(-1, 12)]
    out = []
    for name, (lo, hi) in boxes:
        for corner in itertools.product(*zip(lo, hi)):
            clip = proj @ view @ np.append(corner, 1.0)
            if np.any(np.abs(clip[:2] / clip[3]) > 1 - margin):
                out.append(name)
                break
    return out


@pytest.mark.parametrize("plan_field, leg", [("transport_via", -1), ("place", -1), ("transport_via[0]", 0)])
def test_a_transport_finding_is_drawn_on_its_own_leg(plan_field, leg):
    """A collision after the last waypoint names transport_via (D74); recordings made before name place. Both are the
    leg into the drop-off approach, not the leg into waypoint 1."""
    from vplan import records
    from vplan.config import load_config
    from vplan.motion.trajectory import build_trajectory
    from vplan.robot import HOME_Q
    task = basic_task()
    plan = task.oracle.reference_plan
    assert len(plan.transport_via) >= 1
    traj = build_trajectory(plan, task.task.scene, HOME_Q, cfg=load_config("verifier"))
    legs = [i for i, s in enumerate(traj.segments) if s.label == "transport"]
    f = records.Finding(stage="transport", check="collision", plan_field=plan_field, measured=0.0, threshold=0.01,
                        sample_index=0, bodies=("cube_red", "wall_1"), direction="x")
    assert render._segment(traj, f) == legs[leg]


@pytest.mark.parametrize("seed", [1, 2, 3, 5, 8])
def test_the_overview_keeps_the_arm_and_every_scene_body_in_frame(seed):
    import pybullet as p

    from vplan.robot import HOME_Q
    from vplan.sim.scene import build_scene
    for task in (basic_task(), live.generated_task(seed, "feasible")):
        cid = render._client()
        try:
            assert _outside_frame(build_scene(task.task.scene, cid, HOME_Q), render.OVERVIEW) == []
        finally:
            p.disconnect(cid)


def test_the_overview_frames_the_workspace_closely():
    """The table's work area fills the picture: objects are not specks (the camera is no farther than 2 m)."""
    eye, target, _ = render.OVERVIEW
    assert sum((a - b) ** 2 for a, b in zip(eye, target)) ** 0.5 < 2.0


def test_progress_display_follows_the_events():
    result, events = enclosed_result()
    progress = display.LiveProgress(result.task, "heuristic", None, 1)
    progress.apply(events[0])
    shown = progress.display()
    assert "waiting for the planner" in shown.timeline_html
    for event in events:
        progress.apply(event)
    shown = progress.display()
    assert shown.timeline_html.count("Round ") == 2 and "Collision" in visible(shown.timeline_html)
    assert len(shown.findings) == len(result.shots) + sum(s.close_up is not None for s in result.shots)
    assert [c.endswith(("(overview)", "(close-up)")) for _, c in shown.findings] == [True] * len(shown.findings)
    assert "kinematic and collision checks only" in shown.banner_html  # no Cosmos URL: the D22 banner


def test_live_result_display_has_every_panel():
    result, _ = basic_result()
    shown = display.live_result_display(result)
    assert "Success" in shown.judge_html and "data:image/gif;base64," in shown.execution_html
    assert '"target_id": "cube_red"' in shown.plan_json and "V-6" in shown.timeline_html
    assert "Cosmos visual check off" in shown.banner_html
    assert len(shown.values()) == len(display.Display.FIELDS)


# ---- Cosmos ---------------------------------------------------------------------------------------------------------

def _fake_cosmos(monkeypatch, replies):
    real = visual_gate.CosmosGate
    ask = FakeAsk(replies)
    monkeypatch.setattr(visual_gate, "CosmosGate", lambda url, **kw: real(url, ask=ask, **kw))
    return ask


def test_cosmos_outage_banner_is_sent_before_motion_and_execution_continues(monkeypatch):
    order = []
    real_execute = pipeline.execute

    def execute(*args, **kwargs):
        order.append("execute")
        return real_execute(*args, **kwargs)

    monkeypatch.setattr(pipeline, "execute", execute)
    _fake_cosmos(monkeypatch, [CosmosUnavailable("APIConnectionError: offline")])
    events = []

    def on_event(event):
        events.append(event)
        order.append(event.kind)

    result = live.run_live(basic_task(), live.heuristic_option(), cosmos_url="http://127.0.0.1:9/v1",
                           on_event=on_event, render_execution=False)
    assert order.index("banner") < order.index("execute")  # D58: the user sees it before the arm moves
    banner = next(e.banner for e in events if e.kind == "banner")
    assert banner.text.startswith(views.COSMOS_DOWN) and "offline" in banner.text
    assert result.record.outcome == "executed" and result.record.success  # geometric checks only (D22)
    shown = display.live_result_display(result)
    assert "COSMOS UNAVAILABLE" in shown.banner_html


def test_cosmos_frames_and_verdicts_are_shown_with_the_round(monkeypatch):
    ask = _fake_cosmos(monkeypatch, ["pass", "pass", "pass"])
    result = live.run_live(basic_task(), live.heuristic_option(), cosmos_url="http://127.0.0.1:9/v1",
                           render_execution=False)
    assert len(ask.sent) == 3 and result.record.outcome == "executed"
    assert [(f.stage, f.view) for f in result.cosmos] == [
        (s, v) for s in ("pregrasp", "grasp", "prerelease") for v in ("front_oblique", "top")]
    assert all(f.png.startswith(b"\x89PNG") and f.round_index == 0 for f in result.cosmos)
    shown = display.live_result_display(result)
    assert len(shown.cosmos) == 6 and all(caption.startswith("Pass (learned check) · ") for _, caption in shown.cosmos)
    assert shown.cosmos[0][1] == "Pass (learned check) · Round 1 · pregrasp, front oblique view"  # verdict first
    assert "not verification" in shown.banner_html and "Cosmos visual check" in visible(shown.timeline_html)


# ---- the LLM path and the key ---------------------------------------------------------------------------------------

def _llm_option(client):
    from vplan.planners.nemotron import NemotronPlanner
    planner = NemotronPlanner(model="nvidia/fake:free", provider="openrouter", api_key=SECRET, client=client)
    return live.PlannerOption("Nemotron: nvidia/fake:free via openrouter.ai", planner, llm=True, secret=SECRET)


def test_llm_repair_round_trip_never_exposes_the_key(tmp_path):
    task = basic_task()
    good = heuristic_geometry(task.task.scene, "cube_red")
    client = FakeClient([response(low_transport(good)), response(good)])
    option = _llm_option(client)
    events = []
    result = live.run_live(task, option, k=1, on_event=events.append, record_dir=tmp_path,
                           render_execution=False)
    assert [r.evaluation.outcome for r in result.record.rounds] == ["blocked", "executed"]
    assert result.record.rounds[0].evaluation.findings[0].check == "collision"
    feedback = json.loads(client.requests[1]["messages"][1]["content"])["feedback"]
    assert feedback["findings"][0]["bodies"] == ["cube_red", "wall_1"]  # repair sees the evidence
    progress = display.LiveProgress(task, option.label, None, 1)
    texts = [result.bundle.read_text(encoding="utf-8"), repr(option), repr(result)]
    for event in events:
        progress.apply(event)
        texts += [repr(event), *(v for v in progress.display().values() if isinstance(v, str))]
    texts += [v for v in display.live_result_display(result).values() if isinstance(v, str)]
    assert all(SECRET not in text for text in texts)


def test_a_recording_containing_the_key_is_refused(tmp_path):
    result, _ = basic_result()
    leaked = result.record.model_copy(update={"attempt_id": f"leak {SECRET}"})
    with pytest.raises(live.LiveError, match="API key"):
        live.save_bundle(tmp_path, result.task, leaked, planner="x", cosmos_url=None, secret=SECRET)
    assert list(tmp_path.iterdir()) == []


def _env(tmp_path, url, model):
    path = tmp_path / "llm.env"
    path.write_text(f"LLM_BASE_URL={url}\nLLM_API_KEY={SECRET}\nLLM_MODEL={model}\n", encoding="utf-8")
    return path


def test_llm_option_keeps_the_free_model_guard_and_hides_the_key(tmp_path):
    with pytest.raises(PilotError, match="allow-paid") as refused:
        live.llm_option(_env(tmp_path, "https://api.example.com/v1", "nvidia/ultra"))
    assert SECRET not in str(refused.value)
    paid = live.llm_option(_env(tmp_path, "https://api.example.com/v1", "nvidia/ultra"), allow_paid=True)
    assert paid.llm and paid.label == "Nemotron: nvidia/ultra via api.example.com"
    free = live.llm_option(_env(tmp_path, "https://openrouter.ai/api/v1", "x"), model="nvidia/nemotron:free")
    assert free.planner.model == "nvidia/nemotron:free" and free.planner.temperature == live.DEFAULT_TEMPERATURE
    assert free.planner.call_deadline_s == 60.0  # D63; a longer demo deadline is not decided
    assert SECRET not in repr(paid) and SECRET not in repr(free)


def test_recordings_never_overwrite_each_other(tmp_path, monkeypatch):
    result, _ = basic_result()
    monkeypatch.setattr(live, "_utc_stamp", lambda: "20261016T000000000000Z")  # a coarse Windows clock tick
    paths = [live.save_bundle(tmp_path, result.task, result.record, planner="h", cosmos_url=None) for _ in range(3)]
    assert len(set(paths)) == 3 and sorted(p.name for p in tmp_path.iterdir()) == [p.name for p in paths]


def test_live_panels_carry_no_internal_terms():
    """W4.1: what a viewer reads in every panel (the plan JSON keeps its recorded field names and values)."""
    for result, events in (basic_result(), enclosed_result()):
        progress = display.LiveProgress(result.task, result.planner, None, 1)
        shown = []
        for event in events:
            progress.apply(event)
            shown.append(progress.display())
        shown.append(display.live_result_display(result))
        for d in shown:
            for name in ("banner_html", "status_html", "timeline_html", "execution_html", "judge_html"):
                text = visible(getattr(d, name))
                assert not JARGON.search(text), (name, JARGON.search(text), text)
            for _, caption in d.findings + d.cosmos:
                assert not JARGON.search(caption), caption
    status = visible(display.live_result_display(basic_result()[0]).status_html)
    assert "Task Pick the red cube and place it in the bin." in status
    assert "Ground truth (hidden from the planner) solvable" in status and "Planner Rule-based baseline" in status


def test_only_collision_captions_pair_parts_with_an_arrow():
    """"Release in bin: red cube ↔ bin" would read as a collision; other checks list their parts."""
    from vplan.records import Finding
    from vplan.ui import render
    task = basic_task()
    release = Finding(stage="plan", check="release", plan_field="place.position", measured=0.01, threshold=0.0,
                      direction="carried object's footprint extends 0.010 m outside the bin interior")
    (shot,) = render.finding_shots(task.oracle.reference_plan, task.task.scene, [release], seed=0,
                                   verifier_cfg=__import__("vplan.config", fromlist=["x"]).config_snapshot()["verifier"],
                                   label="Round 1 · ")
    assert shot.caption.startswith("Round 1 · Release in bin: red cube, bin floor, bin wall") and "↔" not in shot.caption
    _, events = enclosed_result()
    assert all("↔" in s.caption for e in events if e.kind == "round" for s in e.shots if "Collision" in s.caption)


def test_no_repair_rounds_read_as_at_most_one_call():
    result, events = basic_result()
    progress = display.LiveProgress(result.task, result.planner, None, 0)
    text = visible(progress.display().status_html)
    assert "Repair rounds none (at most 1 planner call)" in text and "up to 0" not in text
    record = result.record.model_copy(update={"k": 0})
    assert display._attempt_line(record) == "1 planner call; checked, no repair rounds"
