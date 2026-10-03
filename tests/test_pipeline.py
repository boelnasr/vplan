import numpy as np
import pybullet as p

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
