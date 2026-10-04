import numpy as np
import pybullet as p
import pytest

from tests.conftest import make_basic_scene
from vplan.records import Pose, Scene, SceneObject

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
    left, right = name_to_idx["panda_leftfinger"], name_to_idx["panda_rightfinger"]
    assert tuple(sorted((left, right))) in h.self_excluded  # sibling prismatic fingers (Franka SRDF disables this pair)


def test_home_has_no_unexcluded_self_proximity(basic_scene, client) -> None:
    h = build_scene(basic_scene, client)
    pairs = {tuple(sorted((c[3], c[4]))) for c in p.getClosestPoints(h.robot, h.robot, 0.01, physicsClientId=client)}
    assert not {pr for pr in pairs if pr[0] != pr[1] and pr not in h.self_excluded}


def test_set_arm_rejects_wrong_length(basic_scene, client) -> None:
    h = build_scene(basic_scene, client)
    for bad in (HOME_Q[:6], np.append(HOME_Q, 0.0)):
        with pytest.raises(ValueError, match="7"):
            set_arm(h, bad)


def test_filter_pair_suppresses_excluded_contact_in_physics(basic_scene, client) -> None:
    h = build_scene(basic_scene, client)
    name_to_idx = {v: k for k, v in h.link_names.items()}
    pair = tuple(sorted((name_to_idx["panda_link7"], name_to_idx["panda_hand"])))
    assert pair in h.self_excluded
    # the pair penetrates at HOME (unfiltered getClosestPoints sees it), but physics must not report it
    p.stepSimulation(physicsClientId=client)
    contacts = p.getContactPoints(h.robot, h.robot, physicsClientId=client)
    assert not {tuple(sorted((c[3], c[4]))) for c in contacts} & {pair}


def test_cylinder_object_builds_and_registers(client) -> None:
    d = make_basic_scene().model_dump()
    d["objects"] = [*d["objects"], SceneObject(id="can", shape="cylinder", color="blue", size=(0.04, 0.04, 0.08),
                                                pose=Pose(position=(0.40, 0.10, 0.04))).model_dump()]
    h = build_scene(Scene.model_validate(d), client)
    assert "can" in h.bodies and h.names[h.bodies["can"]] == "can"
    assert p.getBodyInfo(h.bodies["can"], physicsClientId=client) is not None
