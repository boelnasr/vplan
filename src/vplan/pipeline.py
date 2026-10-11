"""One attempt without repair: plan -> motion -> verify -> (execute -> judge). W2.5 wraps it with repair."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from copy import deepcopy

import pybullet as p

from vplan.config import config_snapshot as read_config_snapshot
from vplan.execute.executor import approval_token, execute, init_physics
from vplan.judge.judge import judge
from vplan.motion.trajectory import build_trajectory
from vplan.records import (CandidateTrajectory, ExecutionRecord, Finding, JudgeResult, PickPlacePlan, Scene,
                           VisualGateReport)
from vplan.robot import HOME_Q
from vplan.verify import checks
from vplan.verify.verifier import verify


@dataclass
class AttemptResult:
    plan: PickPlacePlan
    findings: list[Finding] = field(default_factory=list)
    outcome: str = "blocked"  # "executed" | "blocked" | "visual_uncertain"; W2.5 adds the spec §10 outcomes
    record: ExecutionRecord | None = None
    judge: JudgeResult | None = None
    visual: VisualGateReport | None = None


def _connect() -> int:
    cid = p.connect(p.DIRECT)
    if cid < 0:
        raise RuntimeError("could not connect to a PyBullet DIRECT physics server")
    return cid


def run_attempt(plan: PickPlacePlan, scene: Scene, oracle_target: str, feasible: bool = True, verify_gate: bool = True,
                preview_client: int | None = None, physics_client: int | None = None, seed: int = 0, realtime: bool = False,
                config_snapshot: dict[str, dict] | None = None,
                visual_gate: Callable[[PickPlacePlan, CandidateTrajectory, Scene], VisualGateReport] | None = None,
                ) -> AttemptResult:
    """verify_gate=False is the evaluation-only, simulation-only ablation (config 1); the demo never uses it.
    realtime=True paces execution at wall-clock speed so a GUI physics_client can be watched.
    visual_gate is the demo-only live Cosmos gate (D47). It sees only plans the Verifier passed: fail blocks with
    its visual Finding, uncertain ends without executing, unavailable/skipped execute on the geometric checks."""
    cfgs = deepcopy(read_config_snapshot() if config_snapshot is None else config_snapshot)
    own: list[int] = []  # only clients this call opened, so a failed second connect cannot leak the first
    try:
        if preview_client is None:
            preview_client = _connect()
            own.append(preview_client)
        if physics_client is None:
            physics_client = _connect()
            own.append(physics_client)
        # an unknown target_id blocks in both modes before motion, which cannot be built without the object (M-11).
        # Other grounding findings (e.g. out-of-bin place) come from verify(), so unverified mode executes them (D29).
        findings = [f for f in checks.grounding(plan, scene, cfgs["verifier"]) if f.plan_field == "target_id"]
        if findings:
            return AttemptResult(plan=plan, findings=findings, outcome="blocked")
        traj = build_trajectory(plan, scene, HOME_Q, seed=seed, cfg=cfgs["verifier"])
        findings = verify(plan, traj, scene, preview_client, cfgs["verifier"]) if verify_gate else []
        if findings:
            return AttemptResult(plan=plan, findings=findings, outcome="blocked")
        visual = visual_gate(plan, traj, scene) if visual_gate is not None and verify_gate else None
        if visual is not None and visual.status == "fail":
            return AttemptResult(plan=plan, findings=list(visual.findings), outcome="blocked", visual=visual)
        if visual is not None and visual.status == "uncertain":
            return AttemptResult(plan=plan, outcome="visual_uncertain", visual=visual)
        h = init_physics(scene, physics_client, traj.q_start, cfgs["physics"])
        rec = execute(h, traj, scene, plan.target_id, cfgs, approval_token(traj, scene, plan.target_id, cfgs),
                      realtime=realtime)
        return AttemptResult(plan=plan, outcome="executed", record=rec, visual=visual,
                             judge=judge(rec, scene, oracle_target, "executed", feasible,
                                         planned_target=plan.target_id, cfg=cfgs["judge"]))
    finally:
        for c in own:
            p.disconnect(c)
