import numpy as np

from tests.conftest import with_object_pose
from vplan.geometry import quat_to_R
from vplan.motion.poses import approach_rotation, key_poses
from vplan.motion.trajectory import build_trajectory, densify, time_scale
from vplan.planners.heuristic import heuristic_geometry
from vplan.records import Grasp, PickPlacePlan, Place
from vplan.robot import HOME_Q, PANDA_VEL_LIMITS, load_panda


def top_plan(via=((0.45, 0.05, 0.25),)) -> PickPlacePlan:
    return PickPlacePlan(target_id="cube_red", grasp=Grasp(approach="top", yaw_deg=0.0, pregrasp_offset_m=0.10),
                         transport_via=via,
                         place=Place(position=(0.45, 0.28, 0.02), yaw_deg=0.0, release_height_m=0.05))


def test_approach_axes() -> None:
    assert np.allclose(approach_rotation("top", 0.0)[:, 2], [0, 0, -1])
    assert np.allclose(approach_rotation("side_+x", 0.0)[:, 2], [-1, 0, 0])


def test_key_poses_order_and_geometry(basic_scene) -> None:
    poses = key_poses(top_plan(), basic_scene, load_panda().fk(HOME_Q))
    assert [k.name for k in poses] == ["start", "pregrasp", "grasp", "lift", "via_0", "preplace", "release", "retreat"]
    assert np.allclose(poses[2].T[:3, 3], [0.50, -0.15, 0.02])
    assert np.allclose(poses[1].T[:3, 3], [0.50, -0.15, 0.12])


def test_trajectory_labels_carrying_and_ik_targets_met(basic_scene) -> None:
    m = load_panda()
    traj = build_trajectory(top_plan(), basic_scene, HOME_Q, seed=0)
    assert [s.label for s in traj.segments] == ["approach", "descend", "lift", "transport", "transport",
                                                "descend_place", "retreat"]
    assert [s.carrying for s in traj.segments] == [False, False, True, True, True, True, False]
    for s in traj.segments:  # every IK sample of a reachable plan meets its Cartesian target
        for idx, Tf in zip(s.ik_idx, s.ik_targets):
            assert np.linalg.norm(m.fk(np.array(s.q[idx]))[:3, 3] - np.array(Tf).reshape(4, 4)[:3, 3]) < 2e-3


def test_trajectory_is_deterministic(basic_scene) -> None:
    assert build_trajectory(top_plan(), basic_scene, HOME_Q, seed=3) == build_trajectory(top_plan(), basic_scene, HOME_Q, seed=3)


def test_densify_bounds_joint_step_and_keeps_ik_indices() -> None:
    sols = [np.zeros(7), np.full(7, 0.1), np.full(7, 0.1) + np.eye(7)[0] * 0.5]
    dense, idx = densify(sols, 0.004)
    steps = np.abs(np.diff(np.array(dense), axis=0)).sum(axis=1)
    assert np.all(steps <= 0.004 + 1e-12)
    assert idx[-1] == len(dense) - 1 and np.allclose(dense[idx[0]], sols[1])


def test_time_scale_respects_velocity_limits_for_sequential_joints() -> None:  # CR#8: old timing exceeded by 3.49x
    sols = [np.zeros(7)]
    for j in range(7):
        sols.append(sols[-1] + np.eye(7)[j] * 0.5)
    dense, _ = densify(sols, 0.004)
    t = np.array(time_scale(dense, PANDA_VEL_LIMITS))
    v = np.abs(np.diff(np.array(dense), axis=0)) / np.diff(t)[:, None]
    assert np.all(v <= PANDA_VEL_LIMITS * 1.0001)


def _yaw_plan(grasp_yaw: float, place_yaw: float) -> PickPlacePlan:
    return PickPlacePlan(target_id="cube_red", grasp=Grasp(approach="top", yaw_deg=grasp_yaw, pregrasp_offset_m=0.10),
                         transport_via=((0.45, 0.05, 0.25),),
                         place=Place(position=(0.45, 0.28, 0.02), yaw_deg=place_yaw, release_height_m=0.05))


def test_top_yaw_is_counterclockwise_about_world_z() -> None:  # spec §7 (corrected after the final week-1 review)
    R = approach_rotation("top", 90.0)
    assert np.allclose(R[:, 1], [0, 1, 0], atol=1e-12)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-12) and np.isclose(np.linalg.det(R), 1.0)
    assert np.allclose(np.cross(R[:, 0], R[:, 1]), R[:, 2], atol=1e-12)  # right-handed


def test_release_closing_axis_follows_place_yaw(basic_scene) -> None:
    T0 = load_panda().fk(HOME_Q)
    rel = {k.name: k.T for k in key_poses(_yaw_plan(0.0, 45.0), basic_scene, T0)}["release"]
    assert np.allclose(rel[:3, 1], [np.sqrt(0.5), np.sqrt(0.5), 0.0], atol=1e-12)
    kp = {k.name: k.T for k in key_poses(_yaw_plan(30.0, 30.0), basic_scene, T0)}
    assert np.allclose(kp["release"][:3, 1], kp["grasp"][:3, 1], atol=1e-12)


def test_heuristic_yaw_aligns_closing_axis_with_a_cube_face(basic_scene) -> None:
    a = np.deg2rad(20.0)
    q = (0.0, 0.0, float(np.sin(a / 2)), float(np.cos(a / 2)))  # cube rotated +20 deg about world z
    s = with_object_pose(basic_scene, "cube_red", (0.50, -0.15, 0.02), q)
    pl = heuristic_geometry(s, "cube_red")
    closing = approach_rotation(pl.grasp.approach, pl.grasp.yaw_deg)[:, 1]
    assert abs(abs(float(closing @ quat_to_R(q)[:, 0])) - 1.0) < 1e-6
