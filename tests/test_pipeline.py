import numpy as np
import pybullet as p
import pytest

import vplan.pipeline as pipeline
from tests.conftest import make_basic_scene
from vplan.config import config_snapshot
from vplan.execute.executor import init_physics
from vplan.motion.trajectory import build_trajectory
from vplan.pipeline import run_attempt
from vplan.planners.heuristic import heuristic_geometry
from vplan.records import Grasp, PickPlacePlan, Place
from vplan.robot import HOME_Q
from vplan.verify.verifier import verify


def test_heuristic_plan_succeeds_end_to_end() -> None:
    s = make_basic_scene()
    res = run_attempt(heuristic_geometry(s, "cube_red"), s, "cube_red")
    assert res.findings == [] and res.outcome == "executed" and res.judge.success


def test_heuristic_routes_over_obstacle() -> None:
    s = make_basic_scene(with_obstacle=True)
    res = run_attempt(heuristic_geometry(s, "cube_red"), s, "cube_red")
    assert res.judge is not None and res.judge.success


def test_bad_plan_is_blocked_and_never_executed() -> None:
    s = make_basic_scene(with_obstacle=True)
    bad = PickPlacePlan(target_id="cube_red", grasp=Grasp(approach="top", yaw_deg=0.0, pregrasp_offset_m=0.10),
                        transport_via=((0.48, 0.08, 0.10),),
                        place=Place(position=(0.45, 0.28, 0.02), yaw_deg=0.0, release_height_m=0.05))
    res = run_attempt(bad, s, "cube_red")
    assert res.outcome == "blocked" and res.record is None and res.findings


def test_unverified_mode_executes_the_bad_plan_and_judge_catches_it() -> None:
    s = make_basic_scene(with_obstacle=True)
    bad = PickPlacePlan(target_id="cube_red", grasp=Grasp(approach="top", yaw_deg=0.0, pregrasp_offset_m=0.10),
                        transport_via=((0.48, 0.08, 0.10),),
                        place=Place(position=(0.45, 0.28, 0.02), yaw_deg=0.0, release_height_m=0.05))
    res = run_attempt(bad, s, "cube_red", verify_gate=False)
    assert res.outcome == "executed" and not res.judge.success  # config 1 (evaluation-only, simulation)


def test_preview_verification_does_not_touch_physics_client() -> None:  # client isolation
    s, cfgs = make_basic_scene(), config_snapshot()
    pl = heuristic_geometry(s, "cube_red")
    phys, prev = p.connect(p.DIRECT), p.connect(p.DIRECT)
    try:
        h = init_physics(s, phys, HOME_Q, cfgs["physics"])
        before = p.getBasePositionAndOrientation(h.bodies["cube_red"], physicsClientId=phys)
        verify(pl, build_trajectory(pl, s, HOME_Q, cfg=cfgs["verifier"]), s, prev, cfgs["verifier"])
        assert p.getBasePositionAndOrientation(h.bodies["cube_red"], physicsClientId=phys) == before
    finally:
        p.disconnect(phys)
        p.disconnect(prev)


def test_pipeline_passes_planned_target_and_config_snapshot_to_the_judge(monkeypatch) -> None:  # review I-3, M-2
    seen = {}
    real = pipeline.judge

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(pipeline, "judge", spy)
    s = make_basic_scene()
    res = run_attempt(heuristic_geometry(s, "cube_red"), s, "cube_red")
    assert res.judge.success
    assert seen["planned_target"] == "cube_red" and seen["cfg"] == config_snapshot()["judge"]


def test_unverified_mode_still_grounds_the_target_id() -> None:  # review M-11: a hallucinated id is an LLM error
    s = make_basic_scene()
    bad = heuristic_geometry(s, "cube_red").model_copy(update={"target_id": "cube_purple"})
    res = run_attempt(bad, s, "cube_red", verify_gate=False)
    assert res.outcome == "blocked" and res.record is None
    assert [(f.check, f.plan_field) for f in res.findings] == [("grounding", "target_id")]


def test_failed_connect_raises_and_disconnects_only_opened_clients(monkeypatch) -> None:  # review M-16
    real_connect, real_disconnect = p.connect, p.disconnect
    opened, closed = [], []

    def connect(*args, **kwargs):
        if opened:
            return -1  # second client fails
        opened.append(real_connect(p.DIRECT))
        return opened[-1]

    def disconnect(cid, *args, **kwargs):
        closed.append(cid)
        return real_disconnect(cid, *args, **kwargs)

    monkeypatch.setattr(pipeline.p, "connect", connect)
    monkeypatch.setattr(pipeline.p, "disconnect", disconnect)
    s = make_basic_scene()
    with pytest.raises(RuntimeError):
        run_attempt(heuristic_geometry(s, "cube_red"), s, "cube_red")
    assert closed == opened and not p.isConnected(opened[0])


def test_realtime_flag_reaches_the_executor(monkeypatch) -> None:  # demo-once --gui paces execution at wall-clock speed
    seen = {}
    real_execute = pipeline.execute

    def spy(*args, **kwargs):
        seen["realtime"] = kwargs.get("realtime")
        return real_execute(*args, **{**kwargs, "realtime": False})  # keep the test fast

    monkeypatch.setattr(pipeline, "execute", spy)
    s = make_basic_scene()
    run_attempt(heuristic_geometry(s, "cube_red"), s, "cube_red", realtime=True)
    assert seen["realtime"] is True
