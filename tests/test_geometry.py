import numpy as np

from vplan.config import load_config
from vplan.geometry import R_to_quat, capture_check, pose_to_T, quat_to_R, rot_angle, slerp_R
from vplan.motion.poses import approach_rotation

CFG = load_config("verifier")


def Rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def Rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def test_quat_round_trip() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        R = quat_to_R(q)
        assert np.allclose(quat_to_R(R_to_quat(R)), R, atol=1e-12)


def test_slerp_is_robust_at_180_degrees() -> None:
    R1 = Rz(np.pi)
    assert np.allclose(slerp_R(np.eye(3), R1, 1.0), R1, atol=1e-9)
    assert abs(rot_angle(np.eye(3), slerp_R(np.eye(3), R1, 0.5)) - np.pi / 2) < 1e-9


def _tcp_top(pos) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = approach_rotation("top", 0.0)  # closing axis = world x
    T[:3, 3] = pos
    return T


def test_capture_accepts_centered_cube() -> None:
    ok, _, why = capture_check(_tcp_top((0.5, 0, 0.02)), pose_to_T((0.5, 0, 0.02), (0, 0, 0, 1)), (0.04,) * 3, CFG)
    assert ok, why


def test_capture_rejects_displacement_along_closing_axis() -> None:  # CR#5: the old infinite-line test passed this
    ok, vals, _ = capture_check(_tcp_top((0.5, 0, 0.02)), pose_to_T((0.53, 0, 0.02), (0, 0, 0, 1)), (0.04,) * 3, CFG)
    assert not ok and abs(vals["off_closing_m"] - 0.03) < 1e-9


def test_capture_rejects_off_axis_and_too_wide_and_tilted() -> None:
    T = _tcp_top((0.5, 0, 0.02))
    assert not capture_check(T, pose_to_T((0.5, 0.02, 0.02), (0, 0, 0, 1)), (0.04,) * 3, CFG)[0]  # off band
    assert not capture_check(T, pose_to_T((0.5, 0, 0.02), (0, 0, 0, 1)), (0.09, 0.04, 0.04), CFG)[0]  # too wide
    q_tilt = R_to_quat(Rx(np.deg2rad(30)))
    assert not capture_check(T, pose_to_T((0.5, 0, 0.02), q_tilt), (0.04,) * 3, CFG)[0]  # approach 30 deg off a face
