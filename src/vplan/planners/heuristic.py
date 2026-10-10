"""No-LLM baseline (D16, spec §7): a template-aware parser grounds the referent, then fixed geometry: top grasp
aligned to the nearest cube face, vertical via-points above the tallest obstacle, place at the bin center."""
from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from vplan.records import (Grasp, PickPlacePlan, Place, Plan, PlannerCall, PlannerFeedback, PlannerProtocol, Refusal,
                           Scene, TaskSpec)
from vplan.tasks.reach import REACH_MARGIN_M, bin_clearance_m, target_clearance_m
from vplan.tasks.referents import resolve_referent

SAFE_CLEARANCE_M = 0.10


def heuristic_geometry(scene: Scene, target_id: str) -> PickPlacePlan:
    obj = next(o for o in scene.objects if o.id == target_id)
    x, y, z, w = obj.pose.orientation
    yaw_obj = float(np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))
    yaw = ((yaw_obj + 45.0) % 90.0) - 45.0  # nearest face alignment, in [-45, 45)
    tallest = max([ob.pose.position[2] + ob.half_extents[2] for ob in scene.obstacles]
                  + [scene.bin.floor_center[2] + scene.bin.wall_height, obj.pose.position[2]])
    via_z = tallest + SAFE_CLEARANCE_M
    bx, by, bz = scene.bin.floor_center
    ox, oy, _ = obj.pose.position
    return PickPlacePlan(
        target_id=target_id, grasp=Grasp(approach="top", yaw_deg=yaw, pregrasp_offset_m=0.10),
        transport_via=((ox, oy, via_z), (bx, by, via_z)),
        place=Place(position=(bx, by, bz + obj.size[2] / 2), yaw_deg=yaw, release_height_m=0.05),  # 0.03 put the hand within 1 cm of the bin wall (probe)
        rationale="heuristic: top grasp, vertical clearance above tallest obstacle, bin center")


def heuristic_plan(task: TaskSpec) -> Plan:
    """Ground the instruction from the planner-visible TaskSpec alone, refuse what is provably impossible, else plan."""
    try:
        matches = resolve_referent(task.instruction, task.scene)
    except ValueError:
        return Refusal(reason="other", explanation="instruction does not follow a known template")
    if not matches:
        return Refusal(reason="no_matching_object", explanation="no object matches the instruction")
    if len(matches) > 1:
        # The generator discards ambiguous phrasing (spec §6); guessing would make grounding a coin flip.
        return Refusal(reason="other", explanation=f"instruction is ambiguous between {len(matches)} objects")
    target = next(o for o in task.scene.objects if o.id == matches[0])
    # Same conservative test as the type-1/2 proofs (bound + margin), so a refusal is never a false refusal.
    bound = task.robot_facts.reach_bound_m + REACH_MARGIN_M
    if target_clearance_m(target.pose.position, target.size) > bound:
        return Refusal(reason="target_unreachable", explanation="target lies beyond the reach bound")
    if bin_clearance_m(task.scene.bin.floor_center, task.scene.bin.inner_size_xy) > bound:
        return Refusal(reason="destination_unreachable", explanation="bin lies beyond the reach bound")
    return heuristic_geometry(task.scene, target.id)


class HeuristicPlanner:
    """Planner interface of NemotronPlanner without a provider; deterministic, so feedback is ignored."""
    provider, model, temperature = "heuristic", "template-heuristic", 0.0

    def plan(self, task: TaskSpec, *, feedback: PlannerFeedback | None = None,
             protocol: PlannerProtocol = "forced", event_sink: Callable[[dict], None] | None = None) -> PlannerCall:
        if type(task) is not TaskSpec:
            raise TypeError("planner accepts only TaskSpec, never a private task envelope")
        if protocol not in ("forced", "required", "content"):
            raise ValueError("unknown planner protocol")
        raw_request = {"task": task.model_dump(mode="json")}
        if event_sink is not None:
            event_sink({"event": "request_started", "raw_request": raw_request, "protocol": protocol})
        start = time.perf_counter()
        proposal = heuristic_plan(task)
        result = PlannerCall(outcome="proposed", plan=proposal, provider=self.provider, model=self.model,
                             protocol=protocol, next_protocol=protocol, temperature=self.temperature,
                             latency_s=time.perf_counter() - start, raw_request=raw_request)
        if event_sink is not None:
            event_sink({"event": "request_completed", "call": result.model_dump(mode="json")})
        return result
