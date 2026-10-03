import numpy as np
import pybullet as p

from vplan.judge.kinematics import jacobian_s_current, jacobian_s_pybullet
from vplan.robot import load_panda
from vplan.sim.scene import build_scene


def test_pybullet_and_manipulapy_jacobians_agree(basic_scene, client) -> None:
    h = build_scene(basic_scene, client)
    m = load_panda()
    rng = np.random.default_rng(2)
    for _ in range(20):
        q = rng.uniform(m.lower * 0.9, m.upper * 0.9)
        assert np.allclose(jacobian_s_pybullet(h, q, 0.5), m.jacobian_s(q, 0.5), atol=1e-6)


def test_current_jacobian_does_not_touch_state(basic_scene, client) -> None:
    h = build_scene(basic_scene, client)
    for j in h.arm_joints:
        p.resetJointState(h.robot, j, 0.1, targetVelocity=0.3, physicsClientId=client)
    before = [p.getJointState(h.robot, j, physicsClientId=client)[:2] for j in h.arm_joints]
    jacobian_s_current(h, 0.5)
    after = [p.getJointState(h.robot, j, physicsClientId=client)[:2] for j in h.arm_joints]
    assert before == after  # resetJointState would zero velocities (CR#20, D21)
