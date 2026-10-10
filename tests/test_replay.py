"""W2.6: replay plays back the recorded 60 Hz streams of an ExecutionRecord; it never re-simulates [CR#11]."""
import numpy as np
import pybullet as p
import pytest

from tests.conftest import make_basic_scene
from tests.test_execute import good_plan
from vplan.config import config_snapshot
from vplan.execute.executor import approval_token, execute, init_physics
from vplan.motion.trajectory import build_trajectory
from vplan.records import Pose
from vplan.replay import replay
from vplan.robot import HOME_Q


@pytest.fixture(scope="module")
def executed():
    s, cfgs = make_basic_scene(), config_snapshot()
    traj = build_trajectory(good_plan(), s, HOME_Q, cfg=cfgs["verifier"])
    cid = p.connect(p.DIRECT)
    try:
        h = init_physics(s, cid, traj.q_start, cfgs["physics"])
        rec = execute(h, traj, s, "cube_red", cfgs, approval_token(traj, s, "cube_red", cfgs))
    finally:
        p.disconnect(cid)
    return s, rec


def test_replay_reproduces_recorded_frames_bit_for_bit(executed, client) -> None:
    s, rec = executed
    frames = replay(rec, s, client=client)
    assert len(rec.q_replay) > 100 and len(frames) == len(rec.q_replay)
    # exact equality, not allclose: playback reads back the very values that were recorded
    assert tuple(f.q for f in frames) == rec.q_replay
    assert tuple(f.fingers for f in frames) == rec.finger_replay
    for oid, poses in rec.object_replay.items():
        assert tuple(f.objects[oid] for f in frames) == poses


def test_replay_shows_the_recorded_state_in_the_client(executed, client) -> None:
    s, rec = executed
    seen = []

    def read_back(i, h):
        q = tuple(p.getJointState(h.robot, j, physicsClientId=client)[0] for j in h.arm_joints)
        f = tuple(p.getJointState(h.robot, j, physicsClientId=client)[0] for j in h.finger_joints)
        objs = {oid: p.getBasePositionAndOrientation(h.bodies[oid], physicsClientId=client) for oid in rec.object_replay}
        seen.append((i, q, f, objs))
    replay(rec, s, client=client, on_frame=read_back)
    assert [x[0] for x in seen] == list(range(len(rec.q_replay)))
    for i, q, f, objs in seen:
        assert q == rec.q_replay[i] and f == rec.finger_replay[i]  # joint resets read back bit-for-bit
        for oid, (pos, orn) in objs.items():
            assert pos == rec.object_replay[oid][i].position
            # PyBullet keeps a rotation matrix: quaternion -> matrix -> quaternion costs a few ulp, nothing more
            assert np.max(np.abs(np.subtract(orn, rec.object_replay[oid][i].orientation))) < 1e-15


def test_replay_does_not_step_physics(executed, client, monkeypatch) -> None:
    s, rec = executed

    def no_step(*a, **k):
        raise AssertionError("replay must not re-simulate")
    monkeypatch.setattr(p, "stepSimulation", no_step)
    assert len(replay(rec, s, client=client)) == len(rec.q_replay)


def test_replay_opens_and_closes_its_own_direct_client(executed, monkeypatch) -> None:
    s, rec = executed
    opened, closed = [], []
    connect, disconnect = p.connect, p.disconnect
    monkeypatch.setattr(p, "connect", lambda mode: opened.append((mode, connect(mode))) or opened[-1][1])
    monkeypatch.setattr(p, "disconnect", lambda c: closed.append(c) or disconnect(c))
    assert len(replay(rec, s)) == len(rec.q_replay)
    assert [m for m, _ in opened] == [p.DIRECT] and closed == [opened[0][1]]


def test_replay_rejects_record_from_another_scene(executed, client) -> None:
    _, rec = executed
    with pytest.raises(ValueError, match="scene"):
        replay(rec, make_basic_scene(with_obstacle=True), client=client)


def test_replay_rejects_misaligned_streams(executed, client) -> None:
    s, rec = executed
    short = rec.model_copy(update={"finger_replay": rec.finger_replay[:-1]})
    with pytest.raises(ValueError, match="length"):
        replay(short, s, client=client)
    missing = rec.model_copy(update={"object_replay": {k: v for k, v in rec.object_replay.items() if k != "cube_red"}})
    with pytest.raises(ValueError, match="object"):
        replay(missing, s, client=client)


def test_empty_record_replays_nothing(executed, client) -> None:
    s, rec = executed
    empty = rec.model_copy(update={"q_replay": (), "finger_replay": (),
                                   "object_replay": {k: () for k in rec.object_replay}})
    assert replay(empty, s, client=client) == ()


def test_frame_objects_are_poses(executed, client) -> None:
    s, rec = executed
    f = replay(rec, s, client=client)[0]
    assert isinstance(f.objects["cube_red"], Pose) and len(f.q) == 7 and len(f.fingers) == 2


# ---- CLI: pilot show --replay plays the stored record back instead of re-executing the plan -------------------
def _fake_round(monkeypatch, record):
    from types import SimpleNamespace

    from vplan import cli
    from vplan.evaluate import pilot
    from vplan.records import EvaluationResult, JudgeResult
    task = SimpleNamespace(task=SimpleNamespace(instruction="put the red cube in the bin", scene=make_basic_scene()))
    ev = EvaluationResult(outcome="executed", record=record,
                          judge=JudgeResult(success=True, predicates={"S1": True}, violations=(), details={}))
    rnd = SimpleNamespace(round_index=0, evaluation=ev, call=SimpleNamespace(plan=good_plan(), outcome="plan"))
    monkeypatch.setattr(pilot, "select_round", lambda *a: (task, rnd))
    monkeypatch.setattr(pilot, "check_path_arg", lambda *a: None)
    monkeypatch.setattr(cli, "_open_gui", lambda: 7)
    monkeypatch.setattr(cli, "_wait_and_close", lambda gui: None)

    def no_resim(*a, **k):
        raise AssertionError("--replay must not re-simulate")
    monkeypatch.setattr(cli, "run_attempt", no_resim)
    return task


def test_cli_show_replay_plays_back_the_stored_record(executed, monkeypatch, capsys) -> None:
    import vplan.replay
    from vplan.cli import main
    s, rec = executed
    task = _fake_round(monkeypatch, rec)
    calls = []
    monkeypatch.setattr(vplan.replay, "replay", lambda r, sc, **kw: calls.append((r, sc, kw)) or ())
    assert main(["pilot", "show", "--run", "r", "--task", "t", "--replay"]) == 0
    assert calls == [(rec, task.task.scene, {"gui": True, "client": 7})]
    assert "replaying" in capsys.readouterr().out


def test_cli_show_replay_without_record_exits_1(monkeypatch, capsys) -> None:
    from vplan.cli import main
    _fake_round(monkeypatch, None)
    assert main(["pilot", "show", "--run", "r", "--task", "t", "--replay"]) == 1
    assert "no execution record" in capsys.readouterr().out
