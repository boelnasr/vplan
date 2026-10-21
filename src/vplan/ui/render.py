"""PyBullet images for the W3.5 UI: a scene overview, Verifier findings posed and highlighted, and the execution as an
animated GIF played back from the recorded 60 Hz streams (replay.replay: kinematic resets, no physics).

Every render opens its own DIRECT client and uses TinyRenderer (CPU, no GL context, no GPU), so the Verifier's preview
client and the physics client are never touched. Findings are re-posed from the plan and its rebuilt trajectory, as
evaluate.frames does for the shadow study: build_trajectory with the attempt's seed and verifier config reproduces the
verified trajectory (for an executed round the approval token proves it)."""
from __future__ import annotations

import io
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pybullet as p
from PIL import Image

from vplan.geometry import T_to_pose, pose_to_T
from vplan.records import CandidateTrajectory, ExecutionRecord, Finding, PickPlacePlan, Scene, Segment
from vplan.robot import HOME_Q
from vplan.sim.scene import SceneHandles, build_scene, set_arm
from vplan.ui.views import CHECKS, body_name, scene_names, stage_name

UI_SIZE_PX = 480  # sharp enough at 1080p; 512 px would add about 3 s to every execution render (384 px: 5.5 s)
FOV_DEG = 45.0
# (eye, target, up) in the world (= robot base) frame, m: the whole arm, the objects, obstacles and the bin, framed
# closely enough that the objects read at video size (tests keep every scene body in the picture).
OVERVIEW = ((1.5, -1.1, 1.1), (0.45, 0.05, 0.33), (0.0, 0.0, 1.0))
CLOSE_UP_OFFSET = (0.55, 0.1, 0.45)  # eye minus target for the close-up on highlighted bodies, m (from +x, above)
HIGHLIGHT_RGBA = (1.0, 0.0, 1.0, 1.0)  # magenta: no scene colour uses it (sim.scene.COLORS)
REPLAY_HZ = 60  # ExecutionRecord.q_replay rate (executor.REPLAY_EVERY at 240 Hz)
GIF_FPS = 10
HOLD_LAST_MS = 1500  # the final state stays on screen before the GIF loops
SEED_MODULUS = 2 ** 32  # run_task passes seed % 2**32 to the motion layer
BIN_BODIES = ("bin_floor", "bin_wall_0", "bin_wall_1", "bin_wall_2", "bin_wall_3")
_JOINT = re.compile(r"panda_joint(\d)")


@dataclass(frozen=True)
class Shot:
    """One rendered image and its caption; finding shots add a close-up on the highlighted bodies."""
    image: np.ndarray  # (H, W, 3) uint8 RGB
    caption: str
    close_up: np.ndarray | None = None


def _client() -> int:
    cid = p.connect(p.DIRECT)
    if cid < 0:
        raise RuntimeError("could not connect to a PyBullet DIRECT physics server")
    return cid


def render_rgb(h: SceneHandles, size: int = UI_SIZE_PX, camera=OVERVIEW) -> np.ndarray:
    eye, target, up = camera
    proj = p.computeProjectionMatrixFOV(FOV_DEG, 1.0, 0.02, 6.0, physicsClientId=h.client)
    view = p.computeViewMatrix(eye, target, up, physicsClientId=h.client)
    _, _, rgba, _, _ = p.getCameraImage(size, size, view, proj, renderer=p.ER_TINY_RENDERER, physicsClientId=h.client)
    return np.reshape(np.asarray(rgba, dtype=np.uint8), (size, size, 4))[:, :, :3].copy()


def scene_image(scene: Scene, size: int = UI_SIZE_PX) -> np.ndarray:
    """The scene at the start state (HOME_Q, fingers open)."""
    cid = _client()
    try:
        return render_rgb(build_scene(scene, cid, HOME_Q), size)
    finally:
        p.disconnect(cid)


def highlight(h: SceneHandles, names: Sequence[str]) -> list[str]:
    """Recolour the named bodies (scene ids, built-in names, or 'robot:<link>'); returns the names it found."""
    links = {v: k for k, v in h.link_names.items()}
    done = []
    for name in names:
        if name.startswith("robot:") and name[len("robot:"):] in links:
            p.changeVisualShape(h.robot, links[name[len("robot:"):]], rgbaColor=HIGHLIGHT_RGBA,
                                physicsClientId=h.client)
        elif name in h.bodies:
            p.changeVisualShape(h.bodies[name], -1, rgbaColor=HIGHLIGHT_RGBA, physicsClientId=h.client)
        else:
            continue
        done.append(name)
    return done


def _segment(traj: CandidateTrajectory, f: Finding) -> int | None:
    """The finding's segment: label and source field together identify it (transport segments share a label but not
    a source field); whole-plan checks name a plan field such as grasp.yaw_deg, so fall back to the first label. A
    collision after the last waypoint names transport_via (D74; place in earlier recordings): the leg into place."""
    field = "place" if (f.stage, f.plan_field) == ("transport", "transport_via") else f.plan_field
    exact = [i for i, s in enumerate(traj.segments) if s.label == f.stage and s.source_field == field]
    if exact:
        return exact[0]
    return next((i for i, s in enumerate(traj.segments) if s.label == f.stage), None)


def _tcp_T(h: SceneHandles) -> np.ndarray:
    ls = p.getLinkState(h.robot, h.tcp_link, computeForwardKinematics=True, physicsClientId=h.client)
    return pose_to_T(ls[4], ls[5])


def _pose_at(h: SceneHandles, plan: PickPlacePlan, scene: Scene, traj: CandidateTrajectory, seg: Segment,
             q: Sequence[float]) -> None:
    """Pose the arm at q; while the segment carries the object, it hangs from the closed fingers with its grasp-frame
    offset (as visual.gate poses its pre-release frame)."""
    obj = next((o for o in scene.objects if o.id == plan.target_id), None)
    if seg.carrying and obj is not None:
        set_arm(h, np.array(traj.segments[1].q[-1]))  # end of descend: the grasp pose
        attach = np.linalg.inv(_tcp_T(h)) @ pose_to_T(obj.pose.position, obj.pose.orientation)
        half_width = float(np.abs(attach[1, :3]) @ (np.asarray(obj.size) / 2))  # TCP y is the closing axis
        set_arm(h, np.array(q))
        pos, orn = T_to_pose(_tcp_T(h) @ attach)
        p.resetBasePositionAndOrientation(h.bodies[obj.id], pos, orn, physicsClientId=h.client)
        for j in h.finger_joints:
            p.resetJointState(h.robot, j, half_width, physicsClientId=h.client)
    else:
        set_arm(h, np.array(q))


def close_up(h: SceneHandles, names: Sequence[str]) -> tuple | None:
    """A camera aimed at the centre of the named bodies' bounding boxes, or None without any."""
    links = {v: k for k, v in h.link_names.items()}
    boxes = []
    for name in names:
        if name.startswith("robot:") and name[len("robot:"):] in links:
            boxes.append(p.getAABB(h.robot, links[name[len("robot:"):]], physicsClientId=h.client))
        elif name in h.bodies:
            boxes.append(p.getAABB(h.bodies[name], physicsClientId=h.client))
    if not boxes:
        return None
    target = np.mean([np.add(lo, hi) / 2 for lo, hi in boxes], axis=0)
    return tuple(target + CLOSE_UP_OFFSET), tuple(target), (0.0, 0.0, 1.0)


def offending(f: Finding, plan: PickPlacePlan) -> list[str]:
    """What to highlight for a finding: its bodies; for V-3 the link the named joint moves; for grasp and release
    findings the planned object (and the bin for release and an out-of-bin place)."""
    if f.bodies:
        return list(f.bodies)
    if f.check == "joint_limits" and (m := _JOINT.search(f.direction)):
        return [f"robot:panda_link{m.group(1)}"]
    if f.check == "grasp_preconditions":
        return [plan.target_id]
    if f.check == "release" or (f.check == "grounding" and f.plan_field == "place.position"):
        return [plan.target_id, *BIN_BODIES]
    return []


def finding_shot(plan: PickPlacePlan, scene: Scene, f: Finding, traj: CandidateTrajectory | None,
                 size: int = UI_SIZE_PX, names: dict[str, str] | None = None) -> Shot | None:
    """The scene posed where the finding was measured, offending bodies in magenta; None when there is nothing to
    pose (an unknown target, a Cosmos finding, or a stage the trajectory lacks). The caption names the check and the
    bodies in plain words (views.scene_names)."""
    names = scene_names(scene) if names is None else names
    if f.check == "visual" or (f.check == "grounding" and f.plan_field == "target_id"):
        return None
    cid = _client()
    try:
        h = build_scene(scene, cid, np.array(traj.q_start) if traj is not None else HOME_Q)
        where = "at the start"
        if f.stage != "plan":
            if traj is None:
                return None
            i = _segment(traj, f)
            if i is None:
                return None
            seg = traj.segments[i]
            k = f.sample_index if f.sample_index is not None and 0 <= f.sample_index < len(seg.q) else len(seg.q) - 1
            _pose_at(h, plan, scene, traj, seg, seg.q[k])
            where = f"during {stage_name(seg.label)} (sample {k})"
        lit = highlight(h, offending(f, plan))
        parts = list(dict.fromkeys(body_name(b, names) for b in lit))  # the bin's five bodies read as one or two
        title = next((t for _, c, t in CHECKS if c == f.check), f.check)
        # the parts first: the gallery shows one line per caption and cuts the rest. "↔" pairs the two bodies of a
        # collision; other checks only list the parts they concern
        join = " ↔ " if f.check == "collision" else ", "
        caption = f"{title}: " + (join.join(parts) if parts else "no single part to highlight") + f" {where}"
        camera = close_up(h, lit)
        return Shot(render_rgb(h, size), caption, None if camera is None else render_rgb(h, size, camera))
    finally:
        p.disconnect(cid)


def rebuild(plan: PickPlacePlan, scene: Scene, *, seed: int, verifier_cfg: dict) -> CandidateTrajectory:
    """The trajectory run_attempt built for this plan (same start, seed and verifier config)."""
    from vplan.motion.trajectory import build_trajectory
    return build_trajectory(plan, scene, HOME_Q, seed=seed % SEED_MODULUS, cfg=verifier_cfg)


def finding_shots(plan: PickPlacePlan, scene: Scene, findings: Sequence[Finding], *, seed: int, verifier_cfg: dict,
                  size: int = UI_SIZE_PX, label: str = "", enclosure: Sequence[str] = ()) -> list[Shot]:
    """One highlighted shot per renderable finding; a failure to render becomes a caption, never an exception.
    enclosure: the ids of a type-3 task's sealed box, named "enclosure" in captions."""
    findings = [f for f in findings if f.check != "visual"]
    if not findings:
        return []
    names = scene_names(scene, enclosure)
    try:
        traj = (rebuild(plan, scene, seed=seed, verifier_cfg=verifier_cfg)
                if any(f.stage != "plan" for f in findings) and plan.target_id in {o.id for o in scene.objects}
                else None)
        shots = [finding_shot(plan, scene, f, traj, size, names) for f in findings]
    except Exception as exc:  # display only: the recorded verdict stands without its picture
        return [Shot(np.zeros((size, size, 3), np.uint8), f"{label}picture failed: {type(exc).__name__}: {exc}"[:200])]
    return [Shot(s.image, label + s.caption, s.close_up) for s in shots if s is not None]


def execution_frames(record: ExecutionRecord, scene: Scene, *, fps: int = GIF_FPS, size: int = UI_SIZE_PX,
                     on_progress: Callable[[int, int], None] | None = None) -> list[np.ndarray]:
    """Every (60 / fps)-th recorded frame and the last one, rendered from the recorded joint and object streams.
    Raises ValueError when the record does not belong to the scene (replay._check)."""
    from vplan import replay

    every = max(1, round(REPLAY_HZ / fps))
    n = len(record.q_replay)
    frames: list[np.ndarray] = []

    def on_frame(i: int, h: SceneHandles) -> None:
        if i % every == 0 or i == n - 1:
            frames.append(render_rgb(h, size))
            if on_progress is not None:
                on_progress(i + 1, n)

    replay.replay(record, scene, on_frame=on_frame)
    return frames


def gif_bytes(frames: Sequence[np.ndarray], fps: int = GIF_FPS) -> bytes:
    if not frames:
        raise ValueError("no frames to encode")
    images = [Image.fromarray(f) for f in frames]
    durations = [round(1000 / fps)] * (len(images) - 1) + [HOLD_LAST_MS]
    buf = io.BytesIO()
    images[0].save(buf, format="GIF", save_all=True, append_images=images[1:], duration=durations, loop=0)
    return buf.getvalue()


def png_rgb(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(im.convert("RGB"))
