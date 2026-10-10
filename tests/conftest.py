"""Shared fixtures. Basic scene: table top at z=0, Panda base at the origin (world = base frame)."""
from __future__ import annotations

import pybullet as p
import pytest

from vplan.records import Scene
from vplan.tasks.scenes import make_basic_scene


def with_object_pose(scene: Scene, obj_id: str, position, orientation=(0.0, 0.0, 0.0, 1.0)) -> Scene:
    """Validated scene edit (never model_copy(update=...), which skips validation)."""
    d = scene.model_dump()
    for o in d["objects"]:
        if o["id"] == obj_id:
            o["pose"] = {"position": tuple(position), "orientation": tuple(orientation)}
    return Scene.model_validate(d)


@pytest.fixture
def basic_scene() -> Scene:
    return make_basic_scene()


@pytest.fixture
def client():
    cid = p.connect(p.DIRECT)
    yield cid
    p.disconnect(cid)
