"""W2.1 Cosmos visual gate: reply parsing, request shape, rendering, state-predicate truth and pilot scoring."""
import base64
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx2 as httpx  # openai 3.x transport
import numpy as np
import openai
import pytest
from pydantic import ValidationError

from tests.conftest import make_basic_scene
from vplan.config import load_config
from vplan.geometry import pose_to_T
from vplan.motion.poses import approach_rotation
from vplan.records import RobotFacts, TaskSpec, VisualVerdict
from vplan.sim.scene import build_scene
from vplan.visual import cosmos, predicates
from vplan.visual.render import render_views, to_png

TOP = approach_rotation("top", 0.0)  # TCP z = -world z


def _T(R, pos) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


# ---- record ----------------------------------------------------------------------------------
def test_visual_verdict_rejects_unknown_verdict_and_negative_latency():
    VisualVerdict(verdict="pass", reason="ok", latency_s=0.0, raw="VERDICT: pass")
    with pytest.raises(ValidationError):
        VisualVerdict(verdict="yes", reason="", latency_s=0.1, raw="")
    with pytest.raises(ValidationError):
        VisualVerdict(verdict="fail", reason="", latency_s=-1.0, raw="")


# ---- parse_verdict -----------------------------------------------------------------------------
@pytest.mark.parametrize("raw, expected", [
    ("VERDICT: pass\nREASON: the red cube is between the fingers", ("pass", "the red cube is between the fingers")),
    ("<think>maybe VERDICT: fail</think>\nVERDICT: pass\nREASON: aligned", ("pass", "aligned")),
    ("I looked.</think>VERDICT: fail\nREASON: wrong object", ("fail", "wrong object")),
    ("VERDICT: Uncertain\nREASON: occluded", ("uncertain", "occluded")),
    ("VERDICT: FAIL\nREASON: x", ("fail", "x")),
])
def test_parse_verdict_accepts_well_formed_replies(raw, expected):
    assert cosmos.parse_verdict(raw) == expected


@pytest.mark.parametrize("raw", [
    "",
    "REASON: no verdict line",
    "VERDICT: pass\nVERDICT: fail\nREASON: two verdicts",
    "VERDICT: yes\nREASON: junk value",
    "VERDICT: pass because aligned",
    "<think>VERDICT: pass\nREASON: still thinking",  # truncated reasoning: the verdict is inside the think block
])
def test_parse_verdict_falls_back_to_uncertain(raw):
    verdict, _ = cosmos.parse_verdict(raw)
    assert verdict == "uncertain"


# ---- questions --------------------------------------------------------------------------------
def _task() -> TaskSpec:
    scene = make_basic_scene()
    return TaskSpec(instruction="Pick the red cube and place it in the bin.", scene=scene,
                    robot_facts=RobotFacts(base_pose=scene.robot_base, reach_bound_m=1.6, gripper_opening_m=0.08,
                                           table_height_m=0.0))


@pytest.mark.parametrize("stage", ["pregrasp", "grasp", "prerelease"])
def test_stage_question_names_the_object_by_attributes_and_asks_for_the_reply_format(stage):
    q = cosmos.stage_question(stage, _task())
    assert "red cube" in q and "cube_red" not in q
    assert "VERDICT: pass|fail|uncertain" in q and "REASON:" in q


# ---- ask_cosmos -------------------------------------------------------------------------------
class _FakeClient:
    def __init__(self, content=None, exc: Exception | None = None):
        self.calls: list[dict] = []
        self._content, self._exc = content, exc
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        if self._exc is not None:
            raise self._exc
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))])


IMAGES = {"front_oblique": b"front-png", "top": b"top-png"}


def test_ask_cosmos_sends_images_in_view_order_then_text_at_temperature_zero():
    fake = _FakeClient("VERDICT: pass\nREASON: ok")
    v = cosmos.ask_cosmos(IMAGES, "Q?", "http://x/v1", client=fake)
    assert (v.verdict, v.reason, v.raw) == ("pass", "ok", "VERDICT: pass\nREASON: ok") and v.latency_s >= 0
    (call,) = fake.calls
    (msg,) = call["messages"]
    urls = [c["image_url"]["url"] for c in msg["content"] if c["type"] == "image_url"]
    assert urls == ["data:image/png;base64," + base64.b64encode(b).decode() for b in IMAGES.values()]
    assert msg["content"][-1] == {"type": "text", "text": "Q?"}
    assert call["temperature"] == 0 and call["model"] == "nvidia/Cosmos3-Edge"
    assert call["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}} and call["max_tokens"] == 128


def test_ask_cosmos_thinking_flag_raises_the_token_budget():
    fake = _FakeClient("<think>hm</think>VERDICT: fail\nREASON: no")
    assert cosmos.ask_cosmos(IMAGES, "Q?", "http://x/v1", enable_thinking=True, client=fake).verdict == "fail"
    assert fake.calls[0]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}
    assert fake.calls[0]["max_tokens"] == 1024


def test_ask_cosmos_builds_a_client_without_sdk_retries(monkeypatch):
    made = {}
    fake = _FakeClient("VERDICT: pass\nREASON: ok")

    def factory(**kw):
        made.update(kw)
        return fake

    monkeypatch.setattr(cosmos.openai, "OpenAI", factory)
    cosmos.ask_cosmos(IMAGES, "Q?", "http://127.0.0.1:8000/v1", timeout=7.0)
    assert made["base_url"] == "http://127.0.0.1:8000/v1" and made["max_retries"] == 0 and made["timeout"] == 7.0


@pytest.mark.parametrize("exc", [
    openai.APIConnectionError(request=httpx.Request("POST", "http://x/v1/chat/completions")),
    openai.InternalServerError("boom", response=httpx.Response(500, request=httpx.Request("POST", "http://x")),
                               body=None),
])
def test_ask_cosmos_transport_and_http_errors_mean_unavailable(exc):
    with pytest.raises(cosmos.CosmosUnavailable):
        cosmos.ask_cosmos(IMAGES, "Q?", "http://x/v1", client=_FakeClient(exc=exc))


@pytest.mark.parametrize("content", [None, "", "the cube looks fine"])
def test_ask_cosmos_empty_or_garbled_reply_is_uncertain(content):
    assert cosmos.ask_cosmos(IMAGES, "Q?", "http://x/v1", client=_FakeClient(content)).verdict == "uncertain"


# ---- rendering ---------------------------------------------------------------------------------
def test_render_views_is_512_rgb_uint8_and_deterministic(client):
    h = build_scene(make_basic_scene(), client)
    a = render_views(h)
    b = render_views(h, views=("front_oblique", "top", "side"))
    assert list(a) == ["front_oblique", "top"] and list(b) == ["front_oblique", "top", "side"]
    for img in b.values():
        assert img.shape == (512, 512, 3) and img.dtype == np.uint8 and img.std() > 0
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert to_png(a["top"]).startswith(b"\x89PNG")


def test_render_views_rejects_unknown_view(client):
    with pytest.raises(KeyError):
        render_views(build_scene(make_basic_scene(), client), views=("nadir",))


# ---- state-predicate truth -------------------------------------------------------------------------
OBJECTS = {"t": (0.50, -0.15, 0.02), "d": (0.60, -0.22, 0.02)}


def test_pregrasp_truth_needs_alignment_and_the_target_nearest():
    above_t = _T(TOP, (0.50, -0.15, 0.12))
    assert predicates.pregrasp_truth(above_t, OBJECTS, "t")
    assert not predicates.pregrasp_truth(_T(TOP, (0.60, -0.22, 0.12)), OBJECTS, "t")  # approaching the distractor
    tilted = _T(approach_rotation("side_-y", 0.0), (0.50, -0.15, 0.12))  # approach axis +y: 90 deg off
    assert not predicates.pregrasp_truth(tilted, OBJECTS, "t")


def test_pregrasp_truth_angle_threshold_is_ten_degrees():
    for deg, ok in ((9.0, True), (11.0, False)):
        a = np.deg2rad(deg)
        tcp = np.array([0.50, -0.15, 0.02]) + 0.10 * np.array([np.sin(a), 0.0, np.cos(a)])
        assert predicates.pregrasp_truth(_T(TOP, tcp), OBJECTS, "t") is ok


def test_grasp_truth_uses_the_capture_volume_closing_band():  # 3 cm off the finger plane, band is 1 cm
    cfg = load_config("verifier")
    T_obj = pose_to_T((0.5, -0.15, 0.02), (0, 0, 0, 1))
    assert predicates.grasp_truth(_T(TOP, (0.5, -0.15, 0.02)), T_obj, (0.04, 0.04, 0.04), cfg)
    assert not predicates.grasp_truth(_T(TOP, (0.5, -0.12, 0.02)), T_obj, (0.04, 0.04, 0.04), cfg)


def test_prerelease_truth_needs_the_target_with_its_footprint_inside_the_bin():
    bn = make_basic_scene().bin  # center (0.45, 0.28), inner half-size 0.08
    size = (0.04, 0.04, 0.04)
    at = lambda x, y: pose_to_T((x, y, 0.08), (0, 0, 0, 1))  # noqa: E731
    assert predicates.prerelease_truth("t", at(0.45, 0.28), size, "t", bn)
    assert not predicates.prerelease_truth("d", at(0.45, 0.28), size, "t", bn)  # wrong object carried
    assert not predicates.prerelease_truth("t", at(0.63, 0.28), size, "t", bn)  # clear of the bin
    assert not predicates.prerelease_truth("t", at(0.45 + 0.07, 0.28), size, "t", bn)  # one face over the wall
    assert not predicates.prerelease_truth(None, None, size, "t", bn)  # nothing carried


# ---- scoring -------------------------------------------------------------------------------------
def test_score_counts_a_final_uncertain_as_wrong_and_reports_abstention_separately():
    s = cosmos.score([True, True, False, False], ["pass", "uncertain", "fail", "pass"])
    assert s == {"n": 4, "tpr": 0.5, "tnr": 0.5, "balanced_accuracy": 0.5, "abstention_rate": 0.25}
    assert cosmos.score([True, False], ["pass", "fail"])["balanced_accuracy"] == 1.0
    assert cosmos.score([True, False], ["uncertain", "uncertain"])["balanced_accuracy"] == 0.0


def test_score_needs_both_classes():
    with pytest.raises(ValueError):
        cosmos.score([True, True], ["pass", "pass"])


def test_cosmos_client_imports_neither_pybullet_nor_manipulapy():  # the ask step runs in a venv without them
    code = ("import sys, vplan.visual.cosmos; "
            "bad = sorted(m for m in sys.modules if m.startswith(('pybullet', 'ManipulaPy'))); "
            "print(bad); sys.exit(1 if bad else 0)")
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(root / "src"), str(root)])}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=root)
    assert r.returncode == 0, r.stdout + r.stderr
