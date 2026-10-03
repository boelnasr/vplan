import numpy as np
import pybullet as p
import pytest

from vplan.robot import ARM_JOINTS, HOME_Q, PANDA_URDF, PANDA_VEL_LIMITS, TCP_LINK, load_panda


def _pb(cid: int) -> tuple[int, int, list[int]]:
    rid = p.loadURDF(PANDA_URDF, useFixedBase=True, physicsClientId=cid)
    infos = [p.getJointInfo(rid, i, physicsClientId=cid) for i in range(p.getNumJoints(rid, physicsClientId=cid))]
    joints = {i[1].decode(): i[0] for i in infos}
    links = {i[12].decode(): i[0] for i in infos}
    return rid, links[TCP_LINK], [joints[n] for n in ARM_JOINTS]


def test_model_is_seven_dof() -> None:
    m = load_panda()
    assert m.lower.shape == (7,) and m.upper.shape == (7,)
    assert m.jacobian_s(HOME_Q, 0.5).shape == (6, 7)
    assert np.asarray(m.manip.omega_list).shape == (3, 7)  # CR#15: not extract_omega_list's (6, 3)


def test_fk_matches_pybullet(client: int) -> None:
    m = load_panda()
    rid, tcp, arm = _pb(client)
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.uniform(m.lower, m.upper)
        for j, qi in zip(arm, q):
            p.resetJointState(rid, j, qi, physicsClientId=client)
        ls = p.getLinkState(rid, tcp, computeForwardKinematics=True, physicsClientId=client)
        T = m.fk(q)
        assert np.linalg.norm(T[:3, 3] - np.array(ls[4])) < 1e-6
        R_pb = np.array(p.getMatrixFromQuaternion(ls[5])).reshape(3, 3)
        assert np.allclose(T[:3, :3], R_pb, atol=1e-6)


def test_jacobian_matches_finite_difference() -> None:
    m = load_panda()
    L, h = 0.5, 1e-6
    rng = np.random.default_rng(1)
    for _ in range(20):
        q = rng.uniform(m.lower * 0.9, m.upper * 0.9)
        T0 = m.fk(q)
        J_fd = np.zeros((6, 7))
        for i in range(7):
            dq = np.zeros(7)
            dq[i] = h
            T1 = m.fk(q + dq)
            J_fd[:3, i] = (T1[:3, 3] - T0[:3, 3]) / h / L
            dR = T1[:3, :3] @ T0[:3, :3].T  # world-frame small rotation
            J_fd[3:, i] = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / (2 * h)
        assert np.allclose(m.jacobian_s(q, L), J_fd, atol=1e-5)


def test_ik_round_trip_is_deterministic() -> None:
    m = load_panda()
    T = m.fk(HOME_Q + 0.2)
    np.random.seed(0)
    q1, ok1 = m.ik(T, HOME_Q)
    np.random.seed(0)
    q2, ok2 = m.ik(T, HOME_Q)
    assert ok1 and ok2 and np.allclose(q1, q2)
    assert np.linalg.norm(m.fk(q1)[:3, 3] - T[:3, 3]) < 1e-3


def test_shared_arrays_are_read_only() -> None:  # a long-lived process must not corrupt them in place
    m = load_panda()
    for arr in (HOME_Q, PANDA_VEL_LIMITS, m.lower, m.upper, m.vel_limits):
        with pytest.raises(ValueError):
            arr[0] = 0.123


def test_joint_limits_match_pybullet(client: int) -> None:  # V-3 (ManipulaPy limits) vs Judge V2 (PyBullet limits)
    m = load_panda()
    rid, _, arm = _pb(client)
    info = [p.getJointInfo(rid, j, physicsClientId=client) for j in arm]
    np.testing.assert_allclose(m.lower, [i[8] for i in info], rtol=0, atol=1e-12)
    np.testing.assert_allclose(m.upper, [i[9] for i in info], rtol=0, atol=1e-12)
