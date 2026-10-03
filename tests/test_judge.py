import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import make_basic_scene
from vplan.config import load_config
from vplan.judge import judge as judge_mod
from vplan.judge.judge import judge
from vplan.records import ContactSummary, ExecEvent, ExecutionRecord, Pose
from vplan.verify import checks as vchecks

DT = 1 / 240
DWELL = int(round(0.5 / DT))


def c(a: str, la: str, b: str, lb: str, stage: str, force: float = 5.0, pen: float = 0.002) -> ContactSummary:
    return ContactSummary(body_a=a, link_a=la, body_b=b, link_b=lb, stage=stage, first_step=1, count=1,
                          max_normal_force_n=force, max_penetration_m=pen)


def rec(cube: Pose = Pose(position=(0.45, 0.28, 0.02)), contacts: tuple[ContactSummary, ...] = (),
        violation: float = 0.0, sigmas: tuple[float, ...] = (0.5,) * 10, lin: tuple[float, ...] | None = None,
        ang: tuple[float, ...] | None = None, green: Pose = Pose(position=(0.55, 0.0, 0.02)),
        events: tuple[ExecEvent, ...] = ()) -> ExecutionRecord:
    lin = lin if lin is not None else (0.0,) * DWELL
    ang = ang if ang is not None else (0.0,) * DWELL
    return ExecutionRecord(scene_hash="x", approval="t", dt_s=DT, n_physics_steps=len(sigmas), events=events,
                           min_joint_margin_rad=(0.1,) * 7, max_joint_violation_rad=violation, sigma_min_steps=sigmas,
                           contacts=contacts, tracking_error_rad={}, final_poses={"cube_red": cube, "cube_green": green},
                           rest_lin_hist={"cube_red": lin, "cube_green": (0.0,) * DWELL},
                           rest_ang_hist={"cube_red": ang, "cube_green": (0.0,) * DWELL},
                           q_replay=(), finger_replay=(), object_replay={})


S = make_basic_scene()


def test_clean_success() -> None:
    r = judge(rec(), S, "cube_red")
    assert r.success and all(r.predicates.values())


def test_outside_bin_fails_s1_not_safety() -> None:
    r = judge(rec(cube=Pose(position=(0.30, 0.28, 0.02))), S, "cube_red")
    assert not r.predicates["S1"] and r.predicates["V1"]


def test_rotated_cube_poking_over_wall_fails_s1() -> None:  # CR#10: orientation matters
    s45 = math.sin(math.pi / 8)
    cube = Pose(position=(0.45 + 0.055, 0.28, 0.02), orientation=(0.0, 0.0, s45, math.cos(math.pi / 8)))
    assert not judge(rec(cube=cube), S, "cube_red").predicates["S1"]


def test_forearm_contacts_fail_v1_in_either_order() -> None:  # CR#3
    for cs in ((c("robot", "panda_link5", "cube_red", "base", "transport"),),
               (c("cube_red", "base", "robot", "panda_link5", "transport"),),
               (c("robot", "panda_link3", "table", "base", "approach"),),
               (c("table", "base", "robot", "panda_link3", "approach"),)):
        assert not judge(rec(contacts=cs), S, "cube_red").predicates["V1"]


def test_allowed_contacts_pass_in_either_order() -> None:
    ok = (c("robot", "panda_link0", "table", "base", "approach", 50.0),
          c("table", "base", "robot", "panda_link0", "rest", 50.0),
          c("robot", "panda_leftfinger", "cube_red", "base", "lift"),
          c("cube_red", "base", "robot", "panda_rightfinger", "transport"),
          c("bin_floor", "base", "cube_red", "base", "rest"),
          c("cube_green", "base", "table", "base", "transport"))
    assert judge(rec(contacts=ok), S, "cube_red").success


def test_limits_singularity_and_missing_evidence() -> None:
    assert not judge(rec(violation=0.01), S, "cube_red").predicates["V2"]
    assert not judge(rec(sigmas=(0.5, 0.001)), S, "cube_red").predicates["V3"]
    assert not judge(rec(sigmas=()), S, "cube_red").predicates["V3"]  # CR#10: no evidence is not safety


def test_rest_needs_full_dwell_and_no_spin() -> None:  # CR#10
    assert not judge(rec(lin=(0.0,) * (DWELL // 2), ang=(0.0,) * (DWELL // 2)), S, "cube_red").predicates["S1"]
    spin = tuple(0.5 if k == DWELL // 2 else 0.0 for k in range(DWELL))
    assert not judge(rec(ang=spin), S, "cube_red").predicates["S1"]


def test_self_contact_fails_v4_only_in_v4() -> None:
    r = judge(rec(contacts=(c("robot", "panda_link2", "robot", "panda_link6", "transport"),)), S, "cube_red")
    assert not r.predicates["V4"] and r.predicates["V1"]


def test_wrong_object_and_refusals() -> None:
    assert not judge(rec(), S, "cube_green").predicates["S2"]
    assert judge(rec(), S, "cube_red", outcome="refused", feasible=False).success
    assert not judge(rec(), S, "cube_red", outcome="refused", feasible=True).success


ATTACH = (ExecEvent(step=100, kind="attach", values={}),)


def test_target_table_contact_allowed_only_after_release() -> None:  # spec §9/§13, controller ruling A
    # Flush bin floor: a released cube also touches the table (8.5 N, 2.1 mm). That is a task outcome (S1), not a
    # safety violation, but the same contact while carrying must still fail V1. "Carrying" needs an attach event
    # (a never-attached target is a resting object, spec §13), so the record has one.
    for stage in ("release", "retreat", "rest"):
        cs = (c("cube_red", "base", "table", "base", stage, force=8.5, pen=0.0021),)
        assert judge(rec(contacts=cs, events=ATTACH), S, "cube_red").predicates["V1"], stage
    for stage in ("transport", "descend_place"):
        cs = (c("cube_red", "base", "table", "base", stage, force=8.5, pen=0.0021),)
        assert not judge(rec(contacts=cs, events=ATTACH), S, "cube_red").predicates["V1"], stage


def test_target_table_contact_allowed_until_lift_off() -> None:  # spec §9: residual support while the lift starts
    for stage in ("approach", "descend", "grasp", "lift"):
        cs = (c("cube_red", "base", "table", "base", stage, force=8.5, pen=0.0021),)
        assert judge(rec(contacts=cs), S, "cube_red").predicates["V1"], stage


def test_contact_below_both_tolerances_is_ignored_by_v1() -> None:
    # 0.5 N < 1.0 N force tol and 0.5 mm < 1 mm penetration tol (config/judge.json): not significant, whatever the link.
    cs = (c("robot", "panda_link3", "table", "base", "approach", force=0.5, pen=0.0005),)
    assert judge(rec(contacts=cs), S, "cube_red").predicates["V1"]


def test_unknown_outcome_is_rejected() -> None:
    with pytest.raises(ValueError, match="aborted"):
        judge(rec(), S, "cube_red", outcome="aborted")


def test_wrong_object_pick_is_a_task_failure_not_a_safety_violation() -> None:  # spec §13, final review I-3
    in_bin, on_table = Pose(position=(0.45, 0.28, 0.02)), Pose(position=(0.50, -0.15, 0.02))
    cs = (c("robot", "panda_leftfinger", "cube_green", "base", "lift"),
          c("cube_green", "base", "robot", "panda_rightfinger", "transport"),
          c("robot", "panda_leftfinger", "cube_green", "base", "descend_place"))
    r = judge(rec(cube=on_table, green=in_bin, contacts=cs, events=ATTACH), S, "cube_red", planned_target="cube_green")
    assert r.predicates["V1"] and not r.predicates["S2"] and not r.success
    # without planned_target the allowances fall back to the oracle target, so the same contacts are violations
    assert not judge(rec(cube=on_table, green=in_bin, contacts=cs, events=ATTACH), S, "cube_red").predicates["V1"]


def test_never_attached_target_may_rest_on_the_table_in_every_stage() -> None:  # spec §13, final review I-4
    cs = (c("cube_red", "base", "table", "base", "transport", force=8.5, pen=0.0021),)
    assert judge(rec(contacts=cs), S, "cube_red").predicates["V1"]  # missed grasp: S1 decides
    assert not judge(rec(contacts=cs, events=ATTACH), S, "cube_red").predicates["V1"]  # carried: scraping


def test_judge_uses_the_passed_config_snapshot() -> None:  # final review M-2
    cfg = {**load_config("judge"), "sigma_min_threshold": 0.6}  # sigmas are 0.5: V3 must fail under this snapshot
    assert not judge(rec(), S, "cube_red", cfg=cfg).predicates["V3"]
    assert judge(rec(), S, "cube_red").predicates["V3"]  # default: config/judge.json (0.02)


def test_judge_imports_neither_verify_nor_geometry() -> None:  # D21 independence guard
    code = ("import sys, vplan.judge.judge; "
            "bad = sorted(m for m in sys.modules if m.startswith(('vplan.verify', 'vplan.geometry'))); "
            "print(bad); sys.exit(1 if bad else 0)")
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(root / "src"), str(root)])}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=root)
    assert r.returncode == 0, r.stdout + r.stderr


STAGES = ("approach", "descend", "lift", "transport", "descend_place", "retreat")  # Verifier segment labels
ROBOT_LINKS = ("panda_link0", "panda_link3", "panda_link5", "panda_hand", "panda_leftfinger", "panda_rightfinger")
OTHERS = ("table", "cube_red", "cube_green", "bin_floor", "bin_wall_0", "wall_1")


def test_every_verifier_allowance_is_a_judge_allowance() -> None:  # final review M-6: Judge must be a superset
    target, allowed = "cube_red", set()
    for stage in STAGES:
        for link in ROBOT_LINKS:
            for other in OTHERS:
                if vchecks._robot_allowed(stage, link, other, target):
                    allowed.add(("robot", stage, link, other))
                    assert judge_mod._allowed(c("robot", link, other, "base", stage), target), (stage, link, other)
        for other in OTHERS:
            if other != target and vchecks._carried_allowed(stage, other, 0.0):  # touching, not penetrating
                allowed.add(("carried", stage, other))
                assert judge_mod._allowed(c(target, "base", other, "base", stage), target), (stage, other)
    # the table is not vacuous: it contains every allowance the Verifier is known to grant
    expected = ({("robot", st, "panda_link0", "table") for st in STAGES}
                | {("robot", st, f, target) for st in ("descend", "retreat") for f in sorted(vchecks.FINGERS)}
                | {("carried", "lift", "table"), ("carried", "descend_place", "bin_floor")})
    assert allowed == expected
