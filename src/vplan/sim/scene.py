"""Build an immutable Scene into a PyBullet client. Preview and physics clients each get their own copy (spec §5.3)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pybullet as p

from vplan.records import Scene
from vplan.robot import ARM_JOINTS, HOME_Q, PANDA_URDF, TCP_LINK

FINGER_JOINTS = ("panda_finger_joint1", "panda_finger_joint2")
HAND_LINK = "panda_hand"
FINGER_OPEN_M = 0.04  # per finger
COLORS = {"red": (0.85, 0.1, 0.1, 1), "green": (0.1, 0.7, 0.2, 1), "blue": (0.1, 0.3, 0.9, 1),
          "yellow": (0.95, 0.85, 0.1, 1), "purple": (0.6, 0.2, 0.7, 1), "orange": (0.95, 0.5, 0.1, 1)}


@dataclass
class SceneHandles:
    client: int
    robot: int
    arm_joints: list[int]
    finger_joints: list[int]
    tcp_link: int
    hand_link: int
    link_names: dict[int, str] = field(default_factory=dict)
    bodies: dict[str, int] = field(default_factory=dict)
    names: dict[int, str] = field(default_factory=dict)
    self_excluded: set[tuple[int, int]] = field(default_factory=set)
    physics_hash: str = ""  # config_hash of the physics config the world was initialised with (set by init_physics)


def _box(client: int, half: tuple[float, float, float], pos: tuple[float, float, float],
         orn: tuple[float, float, float, float] = (0, 0, 0, 1), mass: float = 0.0,
         rgba: tuple[float, float, float, float] = (0.6, 0.6, 0.6, 1)) -> int:
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=client)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=rgba, physicsClientId=client)
    return p.createMultiBody(mass, col, vis, basePosition=pos, baseOrientation=orn, physicsClientId=client)


def self_collision_exclusions(robot: int, client: int) -> set[tuple[int, int]]:
    """Exclude pairs inside a rigid group (links joined by fixed joints), between groups joined by a joint, and
    between sibling groups (movable joints sharing the same parent group).
    Index-distance rules are wrong: panda_link7 and panda_hand are rigidly joined through panda_link8 (CR#1).
    Siblings: the two prismatic fingers hang off the hand and touch at the closed/open ends; Franka's SRDF disables
    that pair, and in physics a finger-finger contact would surface as a false Judge V4 self-collision."""
    n = p.getNumJoints(robot, physicsClientId=client)
    info = [p.getJointInfo(robot, i, physicsClientId=client) for i in range(n)]
    parent = {i: info[i][16] for i in range(n)}
    group = {-1: -1}
    for i in range(n):  # URDF order: parents precede children
        group[i] = group[parent[i]] if info[i][2] == p.JOINT_FIXED else i
    joined = {frozenset((group[i], group[parent[i]])) for i in range(n) if info[i][2] != p.JOINT_FIXED}
    by_parent: dict[int, list[int]] = {}
    for i in range(n):
        if info[i][2] != p.JOINT_FIXED:
            by_parent.setdefault(group[parent[i]], []).append(i)  # i is its own group id (non-fixed joint)
    siblings = {frozenset((a, b)) for kids in by_parent.values() for a in kids for b in kids if a < b}
    links = [-1, *range(n)]
    out = set()
    for a in links:
        for b in links:
            pair = frozenset((group[a], group[b]))
            if a < b and (group[a] == group[b] or pair in joined or pair in siblings):
                out.add((a, b))
    return out


def build_scene(scene: Scene, client: int, q: np.ndarray = HOME_Q) -> SceneHandles:
    """Create table, robot (fixed base), objects, obstacles and bin walls. No simulation steps are taken.
    q: initial 7 arm joint angles, rad."""
    z0 = scene.table_top_z
    table = _box(client, (0.7, 0.8, 0.025), (0.45, 0.0, z0 - 0.025), rgba=(0.8, 0.75, 0.65, 1))
    robot = p.loadURDF(PANDA_URDF, basePosition=(0, 0, 0), useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION,
                       physicsClientId=client)
    infos = [p.getJointInfo(robot, i, physicsClientId=client) for i in range(p.getNumJoints(robot, physicsClientId=client))]
    joint_by_name = {i[1].decode(): i[0] for i in infos}
    link_names = {i[0]: i[12].decode() for i in infos}
    link_names[-1] = "panda_link0"
    link_by_name = {v: k for k, v in link_names.items()}
    h = SceneHandles(client=client, robot=robot, arm_joints=[joint_by_name[n] for n in ARM_JOINTS],
                     finger_joints=[joint_by_name[n] for n in FINGER_JOINTS], tcp_link=link_by_name[TCP_LINK],
                     hand_link=link_by_name[HAND_LINK], link_names=link_names)
    h.self_excluded = self_collision_exclusions(robot, client)
    for a, b in h.self_excluded:  # physics honours filters; getClosestPoints does not (verified), so the Verifier also skips them
        p.setCollisionFilterPair(robot, robot, a, b, 0, physicsClientId=client)
    h.bodies["table"] = table
    for obj in scene.objects:
        half = tuple(s / 2 for s in obj.size)
        if obj.shape == "cube":
            bid = _box(client, half, obj.pose.position, obj.pose.orientation, obj.mass_kg, COLORS[obj.color])
        else:
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=half[0], height=obj.size[2], physicsClientId=client)
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=half[0], length=obj.size[2],
                                      rgbaColor=COLORS[obj.color], physicsClientId=client)
            bid = p.createMultiBody(obj.mass_kg, col, vis, basePosition=obj.pose.position,
                                    baseOrientation=obj.pose.orientation, physicsClientId=client)
        h.bodies[obj.id] = bid
    for ob in scene.obstacles:
        h.bodies[ob.id] = _box(client, ob.half_extents, ob.pose.position, ob.pose.orientation, rgba=(0.4, 0.4, 0.45, 1))
    bn = scene.bin
    cx, cy, cz = bn.floor_center
    ix, iy = bn.inner_size_xy[0] / 2, bn.inner_size_xy[1] / 2
    t, hh = bn.wall_thickness, bn.wall_height / 2
    h.bodies["bin_floor"] = _box(client, (ix + t, iy + t, t / 2), (cx, cy, cz - t / 2), rgba=(0.3, 0.3, 0.8, 1))
    walls = [((t / 2, iy + t, hh), (cx + ix + t / 2, cy, cz + hh)), ((t / 2, iy + t, hh), (cx - ix - t / 2, cy, cz + hh)),
             ((ix, t / 2, hh), (cx, cy + iy + t / 2, cz + hh)), ((ix, t / 2, hh), (cx, cy - iy - t / 2, cz + hh))]
    for k, (half, pos) in enumerate(walls):
        h.bodies[f"bin_wall_{k}"] = _box(client, half, pos, rgba=(0.3, 0.3, 0.8, 1))
    h.names = {bid: name for name, bid in h.bodies.items()}
    h.names[robot] = "robot"
    set_arm(h, q)
    for j in h.finger_joints:
        p.resetJointState(robot, j, FINGER_OPEN_M, physicsClientId=client)
    return h


def set_arm(h: SceneHandles, q: np.ndarray) -> None:
    """Kinematic reset of the 7 arm joints (preview client only; the Executor never calls this mid-run).
    q: joint angles, rad, one per arm joint (length 7); raises ValueError otherwise."""
    if len(q) != len(h.arm_joints):
        raise ValueError(f"q must have {len(h.arm_joints)} joint values, got {len(q)}")
    for j, qi in zip(h.arm_joints, q):
        p.resetJointState(h.robot, j, float(qi), physicsClientId=h.client)
