"""D47 live Cosmos gate (demo only): stage order, uncertain re-ask with the side view, outage fallback, and its
place in run_attempt / run_task. Cosmos itself is faked; render_stage runs against the real preview scene."""
import numpy as np
import pytest

from tests.test_repair import generated
from tests.test_run_task import Scripted, call
from vplan import records
from vplan.visual.cosmos import CosmosUnavailable


def verdict(v, reason="r"):
    return records.VisualVerdict(verdict=v, reason=reason, latency_s=0.1, raw=f"VERDICT: {v}\nREASON: {reason}")


class FakeAsk:
    """Scripted ask_cosmos: pops one verdict (or exception) per request and records what was sent."""
    def __init__(self, replies):
        self.replies, self.sent = list(replies), []

    def __call__(self, images, question, base_url, **kwargs):
        self.sent.append({"views": tuple(images), "question": question, "kwargs": kwargs})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return verdict(*reply) if isinstance(reply, tuple) else verdict(reply)


def fake_render(plan, traj, scene, stage, views):
    return {v: b"png" for v in views}


def gate(replies, **kwargs):
    from vplan.visual.gate import CosmosGate
    ask = FakeAsk(replies)
    return CosmosGate("http://cosmos.invalid/v1", ask=ask, render=fake_render, **kwargs), ask


def report(status, **kw):
    return records.VisualGateReport(status=status, **kw)


# ---- the gate itself ------------------------------------------------------------------------------

def test_every_stage_passing_passes_with_two_views_each():
    g, ask = gate(["pass", "pass", "pass"])
    r = g(None, None, None, subject="red cube")
    assert r.status == "pass" and not r.findings
    assert [c.stage for c in r.checks] == ["pregrasp", "grasp", "prerelease"]
    assert all(s["views"] == ("front_oblique", "top") for s in ask.sent)
    assert all("red cube" in s["question"] for s in ask.sent)


def test_thinking_is_on_by_default_and_passed_through():
    g, ask = gate(["pass", "pass", "pass"])
    g(None, None, None, subject="red cube")
    assert ask.sent[0]["kwargs"]["enable_thinking"] is True


def test_a_fail_stops_the_gate_with_one_visual_finding_for_repair():
    g, ask = gate(["pass", ("fail", "the fingers straddle empty table")])
    r = g(None, None, None, subject="red cube")
    assert r.status == "fail" and len(ask.sent) == 2
    (f,) = r.findings
    assert (f.check, f.stage, f.plan_field) == ("visual", "grasp", "grasp")
    assert f.direction.startswith("learned visual check") and f.measured is None
    assert "straddle" not in f.direction  # free text stays in the report, never in planner feedback (D17)
    assert r.checks[-1].verdict.reason == "the fingers straddle empty table"


def test_uncertain_is_re_asked_once_with_the_side_view_added():
    g, ask = gate(["pass", "uncertain", "pass", "pass"])
    r = g(None, None, None, subject="red cube")
    assert r.status == "pass"
    assert ask.sent[2]["views"] == ("front_oblique", "top", "side")
    assert [(c.stage, c.verdict.verdict) for c in r.checks] == [
        ("pregrasp", "pass"), ("grasp", "uncertain"), ("grasp", "pass"), ("prerelease", "pass")]


def test_still_uncertain_after_the_side_view_blocks_execution():
    g, _ = gate(["uncertain", "uncertain"])
    r = g(None, None, None, subject="red cube")
    assert r.status == "uncertain" and not r.findings and len(r.checks) == 2


def test_a_fail_after_the_re_ask_is_a_fail():
    g, _ = gate(["uncertain", "fail"])
    assert g(None, None, None, subject="red cube").status == "fail"


def test_an_outage_means_unavailable_and_keeps_the_verdicts_so_far():
    g, _ = gate(["pass", CosmosUnavailable("APIConnectionError: refused")])
    r = g(None, None, None, subject="red cube")
    assert r.status == "unavailable" and len(r.checks) == 1 and "refused" in r.note


def test_an_outage_on_the_first_request_is_unavailable_with_no_checks():
    g, _ = gate([CosmosUnavailable("down")])
    r = g(None, None, None, subject="red cube")
    assert r.status == "unavailable" and r.checks == ()


@pytest.mark.parametrize("where", ["render", "on_check"])
def test_any_gate_side_error_degrades_to_unavailable(where):
    from vplan.visual.gate import CosmosGate

    def boom(*a, **k):
        raise RuntimeError("no GL")
    kw = {"render": boom} if where == "render" else {"render": fake_render, "on_check": boom}
    g = CosmosGate("http://cosmos.invalid/v1", ask=FakeAsk(["pass"]), **kw)
    r = g(None, None, None, subject="red cube")
    assert r.status == "unavailable" and "RuntimeError" in r.note


def test_on_check_reports_each_verdict_as_it_arrives():
    seen = []
    g, _ = gate(["pass", "pass", "pass"], on_check=seen.append)
    g(None, None, None, subject="red cube")
    assert [c.stage for c in seen] == ["pregrasp", "grasp", "prerelease"]


def test_report_requires_a_finding_exactly_when_failed():
    f = records.Finding(stage="grasp", check="visual", plan_field="grasp", measured=None, threshold=None,
                        direction="x")
    with pytest.raises(ValueError):
        report("fail")
    with pytest.raises(ValueError):
        report("pass", findings=(f,))


def test_subject_names_the_instructed_object_by_color_and_shape():
    from vplan.visual.gate import subject
    task = generated().task
    obj = next(o for o in task.scene.objects if o.id == generated().oracle.target_id)
    assert subject(task) == f"{obj.color} {obj.shape}"


def test_subject_is_none_when_the_instruction_names_no_single_object():
    from vplan.visual.gate import subject
    task = generated().task
    assert subject(task.model_copy(update={"instruction": "put the gold sphere in the bin"})) is None


# ---- key-frame rendering --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def verified():
    from vplan.config import config_snapshot
    from vplan.motion.trajectory import build_trajectory
    from vplan.robot import HOME_Q
    task = generated()
    plan = task.oracle.reference_plan
    return plan, build_trajectory(plan, task.task.scene, HOME_Q, cfg=config_snapshot()["verifier"]), task.task.scene


@pytest.mark.parametrize("stage", ["pregrasp", "grasp", "prerelease"])
def test_render_stage_gives_one_png_per_view(verified, stage):
    from vplan.visual.gate import render_stage
    images = render_stage(*verified, stage, ("front_oblique", "top"))
    assert list(images) == ["front_oblique", "top"]
    assert all(b.startswith(b"\x89PNG") for b in images.values())


def test_prerelease_frame_carries_the_planned_object_at_the_release_pose(verified):
    from vplan.visual.gate import key_frame_state
    plan, traj, scene = verified
    tcp, obj = key_frame_state(plan, traj, scene, "prerelease")
    # held at the release pose: the object sits under the TCP, well above the table, near the planned place
    assert obj[2] > scene.table_top_z + 0.02
    assert np.linalg.norm(np.array(obj[:2]) - np.array(plan.place.position[:2])) < 0.03
    assert np.linalg.norm(np.array(tcp[:2]) - np.array(obj[:2])) < 0.03


@pytest.mark.parametrize("approach,yaw,size,half_width", [
    ("top", 45.0, (0.04, 0.04, 0.04), 0.028284271247),
    ("top", 0.0, (0.06, 0.04, 0.04), 0.03),
    ("side_+x", 0.0, (0.04, 0.04, 0.06), 0.03),
])
def test_prerelease_fingers_fit_the_object_in_the_grasp_tcp_frame(monkeypatch, approach, yaw, size, half_width):
    import pybullet as p
    from vplan import config
    from vplan.geometry import capture_check, pose_to_T
    from vplan.motion.trajectory import build_trajectory
    from vplan.planners.heuristic import heuristic_geometry
    from vplan.robot import HOME_Q
    from vplan.tasks.scenes import make_basic_scene
    from vplan.visual.gate import _pose, _tcp, _with_client

    scene = make_basic_scene()
    obj = scene.objects[0].model_copy(update={"size": size,
        "pose": records.Pose(position=(0.50, -0.15, 0.20))})
    scene = scene.model_copy(update={"objects": (obj, scene.objects[1])})
    plan = heuristic_geometry(scene, obj.id)
    plan = plan.model_copy(update={"grasp": plan.grasp.model_copy(update={"approach": approach, "yaw_deg": yaw})})
    cfg = config.config_snapshot()["verifier"]
    traj = build_trajectory(plan, scene, HOME_Q, cfg=cfg)

    def check_grasp(client):
        h = _pose(client, plan, traj, scene, "grasp")
        ok, _, why = capture_check(_tcp(h), pose_to_T(obj.pose.position, obj.pose.orientation), obj.size, cfg)
        assert ok, why  # the requested width belongs to a valid capture geometry
    _with_client(check_grasp)

    monkeypatch.setattr(config, "load_config", lambda *a: pytest.fail("renderer reread attempt configuration"))

    def read_fingers(client):
        h = _pose(client, plan, traj, scene, "prerelease")
        return [p.getJointState(h.robot, j, physicsClientId=client)[0] for j in h.finger_joints]
    assert _with_client(read_fingers) == pytest.approx([half_width] * 2, abs=2e-5)


# ---- run_attempt and run_task ---------------------------------------------------------------------

def _attempt(gate_fn, monkeypatch=None, **kw):
    from vplan.pipeline import run_attempt
    task = generated()
    return run_attempt(task.oracle.reference_plan, task.task.scene, task.oracle.target_id, visual_gate=gate_fn, **kw)


def test_a_passing_gate_executes_the_same_verified_trajectory(monkeypatch):
    from vplan import pipeline
    seen = {}
    real_execute = pipeline.execute

    def spy(h, traj, *a, **k):
        seen["executed"] = traj
        return real_execute(h, traj, *a, **k)
    monkeypatch.setattr(pipeline, "execute", spy)

    def g(plan, traj, scene):
        seen["gated"] = traj
        return report("pass")
    res = _attempt(g)
    assert res.outcome == "executed" and res.visual.status == "pass"
    assert seen["gated"] is seen["executed"]  # the approval token covers exactly what Cosmos looked at


@pytest.mark.parametrize("status,outcome", [("fail", "blocked"), ("uncertain", "visual_uncertain")])
def test_a_failing_or_uncertain_gate_never_executes(monkeypatch, status, outcome):
    from vplan import pipeline
    monkeypatch.setattr(pipeline, "execute", lambda *a, **k: pytest.fail("executed past the gate"))
    f = records.Finding(stage="grasp", check="visual", plan_field="grasp", measured=None, threshold=None,
                        direction="learned visual check: no")
    res = _attempt(lambda *a: report(status, findings=(f,) if status == "fail" else ()))
    assert res.outcome == outcome and res.record is None
    assert res.findings == ([f] if status == "fail" else [])


def test_an_unavailable_gate_executes_on_the_geometric_checks_alone():
    res = _attempt(lambda *a: report("unavailable", note="down"))
    assert res.outcome == "executed" and res.judge.success and res.visual.status == "unavailable"


def test_the_gate_never_runs_when_the_verifier_blocks(monkeypatch):
    from vplan import pipeline
    f = records.Finding(stage="transport", check="collision", plan_field="transport_via", measured=0.0,
                        threshold=0.01, direction="below margin")
    monkeypatch.setattr(pipeline, "verify", lambda *a, **k: [f])
    res = _attempt(lambda *a: pytest.fail("gate ran on a blocked plan"))
    assert res.outcome == "blocked" and res.visual is None


def test_no_gate_leaves_the_attempt_unchanged():
    res = _attempt(None)
    assert res.outcome == "executed" and res.visual is None


def test_run_task_feeds_a_visual_fail_back_and_repairs():
    from vplan.repair import run_task
    task = generated()
    plan = task.oracle.reference_plan
    statuses, subjects = ["fail", "pass"], []
    f = records.Finding(stage="grasp", check="visual", plan_field="grasp", measured=None, threshold=None,
                        direction="learned visual check: no")

    def g(plan, traj, scene, *, subject):
        subjects.append(subject)
        s = statuses.pop(0)
        return report(s, findings=(f,) if s == "fail" else ())
    planner = Scripted([call(plan), call(plan)])
    result = run_task(task.task, task.oracle, planner, visual_gate=g)
    assert result.outcome == "executed" and result.success and result.call_count == 2
    assert result.rounds[0].evaluation.visual.status == "fail"
    assert result.rounds[0].observable_failures == ("visual:grasp",)
    assert planner.seen[1]["feedback"].findings == (f,)
    obj = next(o for o in task.task.scene.objects if o.id == task.oracle.target_id)
    assert subjects == [f"{obj.color} {obj.shape}"] * 2


def test_run_task_visual_uncertain_ends_the_attempt():
    from vplan.repair import run_task
    task = generated()
    result = run_task(task.task, task.oracle, Scripted([call(task.oracle.reference_plan)]),
                      visual_gate=lambda *a, subject: report("uncertain"))
    assert result.outcome == "visual_uncertain" and not result.success and result.call_count == 1
    assert result.rounds[0].observable_failures == ("visual_uncertain",)


def test_run_task_skips_the_gate_without_a_single_instructed_object():
    from vplan.repair import run_task
    task = generated()
    spec = task.task.model_copy(update={"instruction": "put the gold sphere in the bin"})
    result = run_task(spec, task.oracle, Scripted([call(task.oracle.reference_plan)]),
                      visual_gate=lambda *a, subject: pytest.fail("asked about no object"))
    assert result.rounds[0].evaluation.visual.status == "skipped"


@pytest.mark.parametrize("kwargs", [{"mode": "unverified"}, {"evaluator": lambda plan: None}])
def test_run_task_rejects_a_gate_outside_the_verified_default_pipeline(kwargs):
    from vplan.repair import run_task
    task = generated()
    with pytest.raises(ValueError):
        run_task(task.task, task.oracle, Scripted([]), visual_gate=lambda *a, subject: report("pass"), **kwargs)
