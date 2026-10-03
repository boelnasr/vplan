import math

from tests.conftest import make_basic_scene
from vplan.judge.judge import judge
from vplan.records import ContactSummary, ExecutionRecord, Pose

DT = 1 / 240
DWELL = int(round(0.5 / DT))


def c(a, la, b, lb, stage, force=5.0, pen=0.002) -> ContactSummary:
    return ContactSummary(body_a=a, link_a=la, body_b=b, link_b=lb, stage=stage, first_step=1, count=1,
                          max_normal_force_n=force, max_penetration_m=pen)


def rec(cube=Pose(position=(0.45, 0.28, 0.02)), contacts=(), violation=0.0, sigmas=(0.5,) * 10, lin=None, ang=None):
    lin = lin if lin is not None else (0.0,) * DWELL
    ang = ang if ang is not None else (0.0,) * DWELL
    green = Pose(position=(0.55, 0.0, 0.02))
    return ExecutionRecord(scene_hash="x", approval="t", dt_s=DT, n_physics_steps=len(sigmas), events=(),
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


def test_target_table_contact_allowed_only_after_release() -> None:  # spec §9/§13, controller ruling A
    # Flush bin floor: a released cube also touches the table (8.5 N, 2.1 mm). That is a task outcome (S1), not a
    # safety violation, but the same contact while carrying must still fail V1.
    for stage in ("release", "retreat", "rest"):
        cs = (c("cube_red", "base", "table", "base", stage, force=8.5, pen=0.0021),)
        assert judge(rec(contacts=cs), S, "cube_red").predicates["V1"], stage
    for stage in ("transport", "descend_place"):
        cs = (c("cube_red", "base", "table", "base", stage, force=8.5, pen=0.0021),)
        assert not judge(rec(contacts=cs), S, "cube_red").predicates["V1"], stage
