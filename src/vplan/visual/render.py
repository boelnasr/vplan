"""Key-frame renders for the Cosmos gate (spec §11): PyBullet TinyRenderer (CPU, deterministic, no GL context),
512x512 RGB. Cameras are fixed in the world (= robot base) frame and frame the table workspace and the bin."""
from __future__ import annotations

import io

import numpy as np
import pybullet as p
from PIL import Image

from vplan.sim.scene import SceneHandles

SIZE_PX = 512
FOV_DEG = 45.0
# name -> (eye, target, up), world frame, m. Objects sit at x 0.35-0.65, y -0.25..0.05; the bin at (0.45, 0.28).
CAMERAS: dict[str, tuple[tuple[float, float, float], ...]] = {
    "front_oblique": ((1.05, -0.45, 0.55), (0.5, 0.05, 0.04), (0.0, 0.0, 1.0)),
    # High and steep (~58 deg over the objects), not nadir: straight down sees only the hand over a top grasp.
    # From -y because top grasps close along world x, so the fingers would hide the object from +-x.
    "top": ((0.5, -0.7, 0.9), (0.5, 0.04, 0.0), (0.0, 0.0, 1.0)),
    "side": ((1.0, 0.65, 0.45), (0.5, 0.02, 0.04), (0.0, 0.0, 1.0)),
}


def render_views(h: SceneHandles, views: tuple[str, ...] = ("front_oblique", "top")) -> dict[str, np.ndarray]:
    """Current state of h's client, one (512, 512, 3) uint8 RGB image per view, in the given order."""
    proj = p.computeProjectionMatrixFOV(FOV_DEG, 1.0, 0.02, 5.0, physicsClientId=h.client)
    out = {}
    for name in views:
        eye, target, up = CAMERAS[name]
        view = p.computeViewMatrix(eye, target, up, physicsClientId=h.client)
        _, _, rgba, _, _ = p.getCameraImage(SIZE_PX, SIZE_PX, view, proj, renderer=p.ER_TINY_RENDERER,
                                            physicsClientId=h.client)
        out[name] = np.reshape(np.asarray(rgba, dtype=np.uint8), (SIZE_PX, SIZE_PX, 4))[:, :, :3].copy()
    return out


def to_png(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()
