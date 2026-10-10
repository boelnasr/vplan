"""Instruction templates and the referent resolver (spec §6). Planners never see the resolver's answer.

Templates: color ("the red cube"), shape ("the cylinder"), spatial ("the object nearest the bin", "the leftmost
object") and relational ("the object next to the green cube"). Left is world +y: the robot's left facing +x.
"""
from __future__ import annotations

import math
import re

from vplan.records import Scene

TEMPLATES = ("color", "shape", "nearest_bin", "leftmost", "next_to")
# Spatial and relational answers within this of the extreme value tie, so near-ties count as ambiguous.
TIE_M = 0.03
# "Next to" only names an object at most this far (center to center, xy) from the anchor.
NEXT_TO_MAX_M = 0.20

_SHAPES = "cube|cylinder"
_PHRASES = {
    "color": re.compile(rf"the (\w+) ({_SHAPES})"),
    "shape": re.compile(rf"the ({_SHAPES})"),
    "nearest_bin": re.compile(r"the object nearest the bin"),
    "leftmost": re.compile(r"the leftmost object"),
    "next_to": re.compile(rf"the object next to the (\w+) ({_SHAPES})"),
}


def instruction(template: str, *args: str) -> str:
    phrase = {"color": "the {} {}", "shape": "the {}", "nearest_bin": "the object nearest the bin",
              "leftmost": "the leftmost object", "next_to": "the object next to the {} {}"}[template]
    return f"Pick {phrase.format(*args)} and place it in the bin."


def _xy_dist(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _near(scores: dict[str, float]) -> list[str]:
    """All ids within TIE_M of the smallest score."""
    if not scores:
        return []
    best = min(scores.values())
    return [i for i, s in scores.items() if s <= best + TIE_M]


def resolve_referent(instruction_text: str, scene: Scene) -> list[str]:
    """Every object the instruction could mean: one id when unambiguous, none when absent, several when ambiguous.

    Raises ValueError for text outside the templates (that is a malformed task, not an absent referent)."""
    match = re.fullmatch(r"Pick (.+) and place it in the bin\.", instruction_text)
    phrase = match.group(1) if match else ""
    template, groups = next(((t, m.groups()) for t, rx in _PHRASES.items() if (m := rx.fullmatch(phrase))),
                            (None, ()))
    if template is None:
        raise ValueError(f"instruction does not follow a known template: {instruction_text!r}")
    objs = scene.objects
    if template == "color":
        return [o.id for o in objs if (o.color, o.shape) == groups]
    if template == "shape":
        return [o.id for o in objs if o.shape == groups[0]]
    if template == "nearest_bin":
        return _near({o.id: _xy_dist(o.pose.position, scene.bin.floor_center) for o in objs})
    if template == "leftmost":
        return _near({o.id: -o.pose.position[1] for o in objs})
    anchors = [o for o in objs if (o.color, o.shape) == groups]
    out: list[str] = []
    for anchor in anchors:  # several anchors make the phrase ambiguous: every anchor contributes its neighbours
        near = _near({o.id: _xy_dist(o.pose.position, anchor.pose.position) for o in objs if o is not anchor
                      and _xy_dist(o.pose.position, anchor.pose.position) <= NEXT_TO_MAX_M})
        out.extend(i for i in near if i not in out)
    return out
