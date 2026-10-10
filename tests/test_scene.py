import numpy as np
import pybullet as p
import pytest

from tests.conftest import make_basic_scene
from vplan.records import Pose, Scene, SceneObject

from vplan.robot import HOME_Q
from vplan.sim.scene import CONVEX_MARGIN_M, CYLINDER_SIDES, _cylinder_prism, build_scene, set_arm


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


def test_cylinder_collides_as_a_prism_with_its_true_size(client) -> None:
    """D43: the built-in cylinder's rim contacts kick it during the Judge dwell; a convex prism rests like a box.
    The effective surface (vertices plus the collision margin) keeps the nominal height and radius."""
    d = make_basic_scene().model_dump()
    d["objects"] = [*d["objects"], SceneObject(id="can", shape="cylinder", color="blue", size=(0.04, 0.04, 0.08),
                                                pose=Pose(position=(0.40, 0.10, 0.045))).model_dump()]
    scene = Scene.model_validate(d)
    h = build_scene(scene, client)
    can = h.bodies["can"]
    assert p.getCollisionShapeData(can, -1, physicsClientId=client)[0][2] == p.GEOM_MESH
    assert p.getVisualShapeData(can, physicsClientId=client)[0][2] == p.GEOM_CYLINDER
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    for _ in range(480):
        p.stepSimulation(physicsClientId=client)
    z = p.getBasePositionAndOrientation(can, physicsClientId=client)[0][2]
    assert abs(z - (scene.table_top_z + 0.04)) < 1e-4  # resting height = half height, as for a box
    radii = [np.hypot(x, y) for x, y, _ in _cylinder_prism(0.02, 0.08)]
    assert np.allclose(radii, (0.02 - CONVEX_MARGIN_M) / np.cos(np.pi / CYLINDER_SIDES))  # flats on r - margin


@pytest.mark.parametrize(("diameter", "height"), ((0.001, 0.001), (0.001, 0.05), (0.04, 0.001)))
def test_small_cylinder_uses_nominal_collision_dimensions(client, diameter, height) -> None:
    """A valid object smaller than the convex margin must not acquire inverted prism dimensions."""
    d = make_basic_scene().model_dump()
    d["objects"] = [SceneObject(id="tiny", shape="cylinder", color="blue", size=(diameter, diameter, height),
                                pose=Pose(position=(0.40, 0.10, 0.02))).model_dump()]
    h = build_scene(Scene.model_validate(d), client)
    body = h.bodies["tiny"]
    shape = p.getCollisionShapeData(body, -1, physicsClientId=client)[0]
    assert shape[2] == p.GEOM_CYLINDER
    assert shape[3][0] == pytest.approx(height)
    assert shape[3][1] == pytest.approx(diameter / 2)


def test_cylinder_side_contact_stays_within_polygon_error(client) -> None:
    """The collision prism has nominal flats and at most 0.4 mm outward corner error."""
    radius, height, probe_radius = 0.02, 0.08, 0.005
    shape = p.createCollisionShape(p.GEOM_MESH, vertices=_cylinder_prism(radius, height), physicsClientId=client)
    body = p.createMultiBody(0, shape, physicsClientId=client)
    sphere = p.createCollisionShape(p.GEOM_SPHERE, radius=probe_radius, physicsClientId=client)
    probe = p.createMultiBody(0, sphere, physicsClientId=client)
    for angle, expected in ((0.0, 0.0), (np.pi / 16, -0.0003722)):
        p.resetBasePositionAndOrientation(probe, ((radius + probe_radius) * np.cos(angle),
                                               (radius + probe_radius) * np.sin(angle), 0),
                                          (0, 0, 0, 1), physicsClientId=client)
        contacts = p.getClosestPoints(body, probe, 0.01, physicsClientId=client)
        assert contacts
        assert contacts[0][8] == pytest.approx(expected, abs=0.0001)
