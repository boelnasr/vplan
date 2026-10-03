"""Shared fixtures. Basic scene: table top at z=0, Panda base at the origin (world = base frame)."""
from __future__ import annotations

import pybullet as p
import pytest

from vplan.records import Bin, Obstacle, Pose, Scene, SceneObject


def make_basic_scene(with_obstacle: bool = False) -> Scene:
    objects = (
        SceneObject(id="cube_red", shape="cube", color="red", size=(0.04, 0.04, 0.04),
                    pose=Pose(position=(0.50, -0.15, 0.02))),
        SceneObject(id="cube_green", shape="cube", color="green", size=(0.04, 0.04, 0.04),
                    pose=Pose(position=(0.55, 0.00, 0.02))),
    )
    obstacles = (
        (Obstacle(id="wall_1", half_extents=(0.01, 0.10, 0.10), pose=Pose(position=(0.48, 0.08, 0.10))),)
        if with_obstacle else ()
    )
    return Scene(table_top_z=0.0, objects=objects, obstacles=obstacles,
                 bin=Bin(floor_center=(0.45, 0.28, 0.0), inner_size_xy=(0.16, 0.16), wall_height=0.08))


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
