"""Keyless replay of an executed attempt (W2.6). Playback of the recorded 60 Hz streams by kinematic resets only:
no stepSimulation, so no physics, LLM, key or GPU is involved and the frames are the recording itself [CR#11].
Joint values and positions read back from PyBullet unchanged; orientations come back within ~1e-19 because PyBullet
stores a rotation matrix and converts back to a quaternion, so the returned frames are the applied recorded values."""
from __future__ import annotations

import time
from typing import Callable, NamedTuple

import numpy as np
import pybullet as p

from vplan.execute.executor import REPLAY_EVERY
from vplan.records import ExecutionRecord, Pose, Scene, scene_hash
from vplan.sim.scene import SceneHandles, build_scene, set_arm


class ReplayFrame(NamedTuple):
    """One recorded 60 Hz frame as applied to the replay client."""
    q: tuple[float, ...]  # arm joints, rad
    fingers: tuple[float, ...]  # finger joints, m
    objects: dict[str, Pose]  # object base (COM) poses, world frame


def _check(record: ExecutionRecord, scene: Scene) -> None:
    if record.scene_hash != scene_hash(scene):
        raise ValueError("record was executed in a different scene than the one given")
    if set(record.object_replay) != {o.id for o in scene.objects}:
        raise ValueError("record's object streams do not match the scene's objects")
    n = len(record.q_replay)
    if len(record.finger_replay) != n or any(len(v) != n for v in record.object_replay.values()):
        raise ValueError("replay streams differ in length")


def replay(record: ExecutionRecord, scene: Scene, gui: bool = False, client: int | None = None,
           on_frame: Callable[[int, SceneHandles], None] | None = None) -> tuple[ReplayFrame, ...]:
    """Play the record back in `client` (opened here, GUI or DIRECT, and closed afterwards if None).
    The scene is needed for geometry: a record stores only its hash. gui=True paces frames at the recorded 60 Hz
    and stops early if the window is closed. on_frame(i, handles) runs after frame i is applied (e.g. rendering)."""
    _check(record, scene)
    if not record.q_replay:
        return ()
    own = client is None
    if own:
        client = p.connect(p.GUI if gui else p.DIRECT)
    try:
        p.resetSimulation(physicsClientId=client)  # a reused client may hold an earlier world
        h = build_scene(scene, client, np.asarray(record.q_replay[0], dtype=float))
        period = record.dt_s * REPLAY_EVERY
        frames: list[ReplayFrame] = []
        for i, q in enumerate(record.q_replay):
            try:
                set_arm(h, q)
                for j, x in zip(h.finger_joints, record.finger_replay[i]):
                    p.resetJointState(h.robot, j, x, physicsClientId=client)
                for oid, poses in record.object_replay.items():
                    p.resetBasePositionAndOrientation(h.bodies[oid], poses[i].position, poses[i].orientation,
                                                      physicsClientId=client)
            except p.error:
                if gui and not p.isConnected(client):  # the viewer closed the window: stop, not an error
                    break
                raise
            frames.append(ReplayFrame(q=q, fingers=record.finger_replay[i],
                                      objects={oid: poses[i] for oid, poses in record.object_replay.items()}))
            if on_frame is not None:
                on_frame(i, h)
            if gui:
                time.sleep(period)
        return tuple(frames)
    finally:
        if own and p.isConnected(client):
            p.disconnect(client)
