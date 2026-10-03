import copy

import numpy as np
import pybullet as p
import pytest

from tests.conftest import make_basic_scene, with_object_pose
from vplan.config import config_snapshot
from vplan.execute.executor import StaleApproval, approval_token, attach_object, execute, init_physics
from vplan.motion.trajectory import build_trajectory
from vplan.records import Grasp, PickPlacePlan, Place
from vplan.robot import HOME_Q


def good_plan() -> PickPlacePlan:
    return PickPlacePlan(target_id="cube_red", grasp=Grasp(approach="top", yaw_deg=0.0, pregrasp_offset_m=0.10),
                         transport_via=((0.45, 0.05, 0.25),),
                         place=Place(position=(0.45, 0.28, 0.02), yaw_deg=0.0, release_height_m=0.05))


def setup(scene=None):
    s = scene or make_basic_scene()
    cfgs = config_snapshot()
    traj = build_trajectory(good_plan(), s, HOME_Q, cfg=cfgs["verifier"])
    return s, cfgs, traj


def test_objects_rest_stably_on_table(client) -> None:
    s, cfgs, traj = setup()
    h = init_physics(s, client, traj.q_start, cfgs["physics"])
    p0 = np.array(p.getBasePositionAndOrientation(h.bodies["cube_red"], physicsClientId=client)[0])
    for _ in range(240):
        p.stepSimulation(physicsClientId=client)
    p1 = np.array(p.getBasePositionAndOrientation(h.bodies["cube_red"], physicsClientId=client)[0])
    assert np.linalg.norm(p1 - p0) < 1e-3


def test_attach_preserves_object_pose(client) -> None:  # CR#2: link-frame parent displaced the object by ~4 cm
    s, cfgs, traj = setup()
    h = init_physics(s, client, traj.q_start, cfgs["physics"])
    body = h.bodies["cube_green"]
    before = np.array(p.getBasePositionAndOrientation(body, physicsClientId=client)[0])
    attach_object(h, body, cfgs["physics"]["attach_max_force_n"])
    for j in h.arm_joints:  # hold the arm still with the URDF effort limit (a default force may not hold it)
        p.setJointMotorControl2(h.robot, j, p.POSITION_CONTROL,
                                targetPosition=p.getJointState(h.robot, j, physicsClientId=client)[0],
                                force=p.getJointInfo(h.robot, j, physicsClientId=client)[10], physicsClientId=client)
    for _ in range(50):
        p.stepSimulation(physicsClientId=client)
    after = np.array(p.getBasePositionAndOrientation(body, physicsClientId=client)[0])
    assert np.linalg.norm(after - before) < 1e-3


def test_good_transfer_attaches_and_lands_in_bin(client) -> None:
    s, cfgs, traj = setup()
    h = init_physics(s, client, traj.q_start, cfgs["physics"])
    rec = execute(h, traj, s, "cube_red", cfgs, approval_token(traj, s, "cube_red", cfgs))
    assert rec.attached and len(rec.sigma_min_steps) == rec.n_physics_steps
    x, y, z = rec.final_poses["cube_red"].position
    assert abs(x - 0.45) < 0.05 and abs(y - 0.28) < 0.05 and z < 0.05


def test_approval_rejects_each_mutation(client) -> None:  # CR#4
    s, cfgs, traj = setup()
    tok = approval_token(traj, s, "cube_red", cfgs)
    h = init_physics(s, client, traj.q_start, cfgs["physics"])
    with pytest.raises(StaleApproval):
        execute(h, traj, s, "cube_green", cfgs, tok)  # target changed
    cfg2 = copy.deepcopy(cfgs)
    cfg2["physics"]["position_gain"] = 0.5
    with pytest.raises(StaleApproval):
        execute(h, traj, s, "cube_red", cfg2, tok)  # config changed
    p.resetBasePositionAndOrientation(h.bodies["cube_red"], (0.5, -0.149, 0.02), (0, 0, 0, 1), physicsClientId=client)
    with pytest.raises(StaleApproval):
        execute(h, traj, s, "cube_red", cfgs, tok)  # world drifted after approval (no reset hides it)


def test_missed_grasps_do_not_attach(client) -> None:  # CR#5
    s, cfgs, traj = setup()
    for pos in ((0.50, -0.20, 0.02), (0.53, -0.15, 0.02)):  # off the finger plane / along the closing axis
        moved = with_object_pose(s, "cube_red", pos)
        h = init_physics(moved, client, traj.q_start, cfgs["physics"])
        rec = execute(h, traj, moved, "cube_red", cfgs, approval_token(traj, moved, "cube_red", cfgs))
        assert not rec.attached
        assert any(e.kind == "attach_rejected" for e in rec.events)
