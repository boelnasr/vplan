"""Reusable demonstration scene; world frame is the fixed Panda base."""
from vplan.records import Bin, Obstacle, Pose, Scene, SceneObject


def make_basic_scene(with_obstacle: bool = False) -> Scene:
    objects = (
        SceneObject(id="cube_red", shape="cube", color="red", size=(0.04, 0.04, 0.04),
                    pose=Pose(position=(0.50, -0.15, 0.02))),
        SceneObject(id="cube_green", shape="cube", color="green", size=(0.04, 0.04, 0.04),
                    pose=Pose(position=(0.55, 0.00, 0.02))),
    )
    obstacles = (
        # thin wall between cube and bin, clear of the arm's descend path (the earlier 0.48 m wall sat within 1 cm of
        # panda_link5 while descending, so any plan was rejected for the wrong reason)
        (Obstacle(id="wall_1", half_extents=(0.15, 0.01, 0.075), pose=Pose(position=(0.50, 0.06, 0.075))),)
        if with_obstacle else ()
    )
    return Scene(table_top_z=0.0, objects=objects, obstacles=obstacles,
                 bin=Bin(floor_center=(0.45, 0.28, 0.0), inner_size_xy=(0.16, 0.16), wall_height=0.08))

