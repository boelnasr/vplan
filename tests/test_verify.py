import numpy as np
import pytest

from tests.conftest import make_basic_scene, with_object_pose
from vplan.config import load_config
from vplan.geometry import pose_to_T
from vplan.motion.trajectory import build_trajectory, densify, time_scale
from vplan.records import CandidateTrajectory, Finding, Grasp, PickPlacePlan, Place, Scene, Segment
from vplan.robot import HOME_Q, PANDA_VEL_LIMITS, load_panda
from vplan.sim.scene import build_scene
from vplan.verify import checks as vchecks
from vplan.verify.verifier import _order_findings, verify

CFG = load_config("verifier")


def plan(target="cube_red", via=((0.45, 0.05, 0.25),), place=(0.45, 0.28, 0.02), approach="top") -> PickPlacePlan:
    return PickPlacePlan(target_id=target, grasp=Grasp(approach=approach, yaw_deg=0.0, pregrasp_offset_m=0.10),
                         transport_via=via, place=Place(position=place, yaw_deg=0.0, release_height_m=0.05))


def run(pl, s, client):
    return verify(pl, build_trajectory(pl, s, HOME_Q, cfg=CFG), s, client, CFG)


def checks_of(findings) -> set[str]:
    return {f.check for f in findings}


def one_sample_segment(q: np.ndarray) -> Segment:
    T = load_panda().fk(q)
    qq = tuple(float(x) for x in q)
    return Segment(label="transport", source_field="transport_via[0]", carrying=False, q=(qq, qq), t=(0.0, 1.0),
                   ik_idx=(1,), ik_targets=(tuple(float(x) for x in T.reshape(-1)),))


def test_good_plan_passes(client) -> None:
    assert run(plan(), make_basic_scene(), client) == []


def test_unknown_target_and_place_outside_bin_are_grounding(client) -> None:
    s = make_basic_scene()
    assert "grounding" in checks_of(run(plan(target="cube_purple"), s, client))
    f = run(plan(place=(0.30, 0.28, 0.02)), s, client)
    assert any(x.check == "grounding" and x.plan_field == "place.position" for x in f)


def test_transport_through_wall_is_collision(client) -> None:
    f = run(plan(via=((0.48, 0.08, 0.10),)), make_basic_scene(with_obstacle=True), client)
    hit = [x for x in f if x.check == "collision"]
    assert hit and "wall_1" in hit[0].bodies and hit[0].stage == "transport"


def test_wall_avoiding_plan_passes_obstacle_scene(client) -> None:
    assert run(plan(), make_basic_scene(with_obstacle=True), client) == []


def test_thin_plate_across_path_is_collision(client) -> None:
    d = make_basic_scene().model_dump()
    d["obstacles"] = [{"id": "plate", "half_extents": (0.0025, 0.15, 0.15), "pose": {"position": (0.47, 0.05, 0.25)}}]
    f = run(plan(), Scene.model_validate(d), client)
    assert any(x.check == "collision" and "plate" in x.bodies for x in f)


def test_unreachable_via_is_ik_failure(client) -> None:
    assert "ik_accuracy" in checks_of(run(plan(via=((1.5, 0.0, 0.3),)), make_basic_scene(), client))


def test_side_grasp_at_cube_height_is_rejected(client) -> None:
    f = run(plan(approach="side_+x"), make_basic_scene(), client)
    assert checks_of(f) & {"collision", "ik_accuracy", "grasp_preconditions"}


def test_bad_middle_ik_sample_with_good_endpoints_is_caught(client) -> None:  # CR#6
    s = make_basic_scene()
    pl = plan()
    d = build_trajectory(pl, s, HOME_Q, cfg=CFG).model_dump()
    seg = d["segments"][3]
    seg["q"] = list(seg["q"])  # model_dump() gives a tuple of samples: make it mutable
    mid = seg["ik_idx"][len(seg["ik_idx"]) // 2]
    seg["q"][mid] = tuple(x + (0.05 if j == 0 else 0.0) for j, x in enumerate(seg["q"][mid]))
    f = verify(pl, CandidateTrajectory.model_validate(d), s, client, CFG)
    assert any(x.check == "ik_accuracy" and x.sample_index == mid for x in f)


def test_branch_flip_is_caught_despite_densification() -> None:  # CR#6
    sols = [HOME_Q.copy(), HOME_Q.copy(), HOME_Q + np.eye(7)[0] * 0.5]
    dense, idx = densify(sols, 0.004)
    m = load_panda()
    seg = Segment(label="transport", source_field="transport_via[0]", carrying=False,
                  q=tuple(tuple(float(x) for x in q) for q in dense), t=time_scale(dense, PANDA_VEL_LIMITS),
                  ik_idx=tuple(idx), ik_targets=tuple(tuple(float(x) for x in m.fk(dense[i]).reshape(-1)) for i in idx))
    f = vchecks.continuity(seg, CFG)
    assert f is not None and f.check == "continuity"


def test_near_limit_is_joint_limit_failure() -> None:
    q_lim = HOME_Q.copy()
    # pybullet_data's panda.urdf has panda_joint4 in [-3.1416, 0] (not the datasheet -3.0718): 0.01 rad above the
    # lower limit is inside the 2 deg margin
    q_lim[3] = load_panda().lower[3] + 0.01
    f = vchecks.joint_limits(one_sample_segment(q_lim), load_panda(), CFG)
    assert f is not None and f.check == "joint_limits"


def test_genuine_self_collision_is_caught_and_home_is_not(client) -> None:  # CR#1
    s = make_basic_scene()
    h = build_scene(s, client)
    m = load_panda()
    assert vchecks.collision(one_sample_segment(HOME_Q), h, plan(), None, m, CFG) is None
    rng = np.random.default_rng(0)
    found = None
    for _ in range(300):  # ~20% of random configurations self-collide (measured 65/300)
        q = rng.uniform(m.lower, m.upper)
        f = vchecks.collision(one_sample_segment(q), h, plan(), None, m, CFG)
        if f is not None and f.bodies[0].startswith("robot:") and f.bodies[1].startswith("robot:"):
            found = f
            break
    assert found is not None


def test_release_footprint_outside_bin_is_caught() -> None:
    s = make_basic_scene()
    T_release = pose_to_T((0.45 + 0.07, 0.28, 0.07), (1.0, 0.0, 0.0, 0.0))  # 7 cm off center: cube pokes over the wall
    attach = np.eye(4)
    f = vchecks.release(plan(), s, T_release, attach, CFG)
    assert f is not None and f.check == "release"


def test_carried_object_hitting_forearm_is_caught(client) -> None:  # CR#9
    s = make_basic_scene()
    h = build_scene(s, client)
    m = load_panda()
    seg = Segment.model_validate({**one_sample_segment(HOME_Q).model_dump(), "carrying": True})
    attach = np.eye(4)  # carried cube held 0.25 m back along the TCP z axis: penetrates panda_link6 at HOME
    attach[2, 3] = -0.25  # (0.35 m, the brief's value, clears the whole arm at HOME: calibrated per the brief's note)
    f = vchecks.collision(seg, h, plan(), attach, m, CFG)
    assert f is not None and f.bodies[0] == "cube_red" and f.bodies[1].startswith("robot:")


def test_unreachable_target_gives_ik_finding_and_no_spurious_release(client) -> None:
    # the release footprint must use the planned grasp pose, not FK of a failed IK endpoint
    s = with_object_pose(make_basic_scene(), "cube_red", (1.2, 0.0, 0.02))
    f = run(plan(), s, client)
    assert "ik_accuracy" in checks_of(f) and "release" not in checks_of(f)


def test_segment_count_mismatch_raises(client) -> None:
    s = make_basic_scene()
    traj = build_trajectory(plan(), s, HOME_Q, cfg=CFG)  # 1 via
    two_via = plan(via=((0.45, 0.05, 0.25), (0.45, 0.15, 0.25)))
    with pytest.raises(ValueError):
        verify(two_via, traj, s, client, CFG)


def _f(stage: str, check: str, idx: int | None = None) -> Finding:
    return Finding(stage=stage, check=check, plan_field="x", measured=None, threshold=None, sample_index=idx,
                   direction="d")


def test_order_findings_follows_the_trajectory() -> None:
    labels = ["approach", "descend", "lift", "transport", "transport", "descend_place", "retreat"]
    release = _f("descend_place", "release")
    late = _f("transport", "collision", 9)
    early = _f("approach", "collision", 4)
    same_stage = _f("transport", "singularity", 2)
    grounding = _f("plan", "grounding")
    out = _order_findings([release, late, early, same_stage, grounding], labels)
    assert out == [grounding, early, same_stage, late, release]


def test_order_findings_is_stable_and_treats_none_as_zero() -> None:
    labels = ["approach", "descend"]
    a, b, c = _f("descend", "grasp_preconditions"), _f("descend", "collision", 3), _f("descend", "joint_limits", 0)
    assert _order_findings([b, a, c], labels) == [a, c, b]  # a (None -> 0) and c (0) keep their input order


def test_ik_accuracy_reports_the_failing_quantity() -> None:
    seg = one_sample_segment(HOME_Q)
    T = np.array(seg.ik_targets[0]).reshape(4, 4)
    c, s_ = np.cos(np.deg2rad(5.0)), np.sin(np.deg2rad(5.0))
    T[:3, :3] = T[:3, :3] @ np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])  # rotation-only error, position exact
    rot_only = Segment.model_validate({**seg.model_dump(), "ik_targets": (tuple(float(x) for x in T.reshape(-1)),)})
    f = vchecks.ik_accuracy(rot_only, load_panda(), CFG)
    assert f is not None and abs(f.measured - 5.0) < 1e-6 and f.threshold == CFG["ik_rot_tol_deg"]
    T[0, 3] += 0.05  # now position fails as well: report the position residual
    both = Segment.model_validate({**seg.model_dump(), "ik_targets": (tuple(float(x) for x in T.reshape(-1)),)})
    f = vchecks.ik_accuracy(both, load_panda(), CFG)
    assert f is not None and abs(f.measured - 0.05) < 1e-6 and f.threshold == CFG["ik_pos_tol_m"]


def test_velocity_finding_reports_the_scaled_limit() -> None:
    q0, q1 = HOME_Q.copy(), HOME_Q.copy()
    q1[0] += 0.5
    t = (0.0, 0.01)  # 50 rad/s on joint 1
    seg = one_sample_segment(HOME_Q).model_copy(update={"q": (tuple(q0), tuple(q1)), "t": t})
    m = load_panda()
    f = vchecks.joint_limits(seg, m, CFG)
    assert f is not None and "velocity" in f.direction
    assert f.threshold == pytest.approx(m.vel_limits[0] * CFG["velocity_limit_fraction"])
