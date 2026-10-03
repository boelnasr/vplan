import numpy as np
import pybullet as p

from vplan.robot import HOME_Q
from vplan.sim.scene import build_scene, set_arm


def test_build_scene_creates_all_bodies(basic_scene, client) -> None:
    h = build_scene(basic_scene, client)
    for key in ("table", "cube_red", "cube_green", "bin_floor", "bin_wall_0", "bin_wall_3"):
        assert key in h.bodies
    assert len(h.arm_joints) == 7 and len(h.finger_joints) == 2
    assert h.names[h.robot] == "robot" and h.link_names[-1] == "panda_link0"


def test_set_arm_moves_joints_kinematically(basic_scene, client) -> None:
    h = build_scene(basic_scene, client)
    set_arm(h, HOME_Q + 0.1)
    q = [p.getJointState(h.robot, j, physicsClientId=client)[0] for j in h.arm_joints]
    assert np.allclose(q, HOME_Q + 0.1)


def test_exclusions_follow_urdf_topology(basic_scene, client) -> None:
    h = build_scene(basic_scene, client)
    name_to_idx = {v: k for k, v in h.link_names.items()}
    link7, hand = name_to_idx["panda_link7"], name_to_idx["panda_hand"]
    link1 = name_to_idx["panda_link1"]
    assert tuple(sorted((link7, hand))) in h.self_excluded  # rigid group via fixed panda_link8 (CR#1)
    assert tuple(sorted((link1, link7))) not in h.self_excluded


def test_home_has_no_unexcluded_self_proximity(basic_scene, client) -> None:
    h = build_scene(basic_scene, client)
    pairs = {tuple(sorted((c[3], c[4]))) for c in p.getClosestPoints(h.robot, h.robot, 0.01, physicsClientId=client)}
    assert not {pr for pr in pairs if pr[0] != pr[1] and pr not in h.self_excluded}
