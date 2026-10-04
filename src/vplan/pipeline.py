"""One attempt without repair: plan -> motion -> verify -> (execute -> judge). W2.5 wraps it with repair."""
from __future__ import annotations

from dataclasses import dataclass, field

import pybullet as p

from vplan.config import config_snapshot
from vplan.execute.executor import approval_token, execute, init_physics
from vplan.judge.judge import judge
from vplan.motion.trajectory import build_trajectory
from vplan.records import ExecutionRecord, Finding, JudgeResult, PickPlacePlan, Scene
from vplan.robot import HOME_Q
from vplan.verify import checks
from vplan.verify.verifier import verify


@dataclass
class AttemptResult:
    plan: PickPlacePlan
    findings: list[Finding] = field(default_factory=list)
    outcome: str = "blocked"  # "executed" | "blocked"; W2.5 adds the spec §10 outcomes
    record: ExecutionRecord | None = None
    judge: JudgeResult | None = None


def _connect() -> int:
    cid = p.connect(p.DIRECT)
    if cid < 0:
        raise RuntimeError("could not connect to a PyBullet DIRECT physics server")
    return cid


def run_attempt(plan: PickPlacePlan, scene: Scene, oracle_target: str, feasible: bool = True, verify_gate: bool = True,
                preview_client: int | None = None, physics_client: int | None = None, seed: int = 0, realtime: bool = False) -> AttemptResult:
    """verify_gate=False is the evaluation-only, simulation-only ablation (config 1); the demo never uses it.
    realtime=True paces execution at wall-clock speed so a GUI physics_client can be watched."""
    cfgs = config_snapshot()  # one snapshot for verification, approval and execution (CR#4)
    own: list[int] = []  # only clients this call opened, so a failed second connect cannot leak the first
    try:
        if preview_client is None:
            preview_client = _connect()
            own.append(preview_client)
        if physics_client is None:
            physics_client = _connect()
            own.append(physics_client)
        if not verify_gate:
            # V-1 grounding still runs: a hallucinated target_id is an LLM error, not a simulation error (M-11)
            findings = checks.grounding(plan, scene, cfgs["verifier"])
            if findings:
                return AttemptResult(plan=plan, findings=findings, outcome="blocked")
        traj = build_trajectory(plan, scene, HOME_Q, seed=seed, cfg=cfgs["verifier"])
        findings = verify(plan, traj, scene, preview_client, cfgs["verifier"]) if verify_gate else []
        if findings:
            return AttemptResult(plan=plan, findings=findings, outcome="blocked")
        h = init_physics(scene, physics_client, traj.q_start, cfgs["physics"])
        rec = execute(h, traj, scene, plan.target_id, cfgs, approval_token(traj, scene, plan.target_id, cfgs),
                      realtime=realtime)
        return AttemptResult(plan=plan, outcome="executed", record=rec,
                             judge=judge(rec, scene, oracle_target, "executed", feasible,
                                         planned_target=plan.target_id, cfg=cfgs["judge"]))
    finally:
        for c in own:
            p.disconnect(c)
