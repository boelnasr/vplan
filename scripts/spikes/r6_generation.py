"""R6: does constructive generation yield enough feasible variety? 60-seed acceptance spike (plan W2.2).

Feasible seeds run the heuristic witness through run_attempt (verifier, physics, Judge); impossible seeds only need
their proof check. Prints acceptance per stratum (distractors x obstacles), per template and per impossibility type,
with rejection reasons, verifier finding checks and attach_rejected counts (D29 watch items).

    PYTHONPATH='src;.' python scripts/spikes/r6_generation.py [--seeds 60] [--workers 4] [--json runs/r6.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

from vplan.config import config_snapshot


def feasible_row(seed: int, cfgs: dict) -> dict:
    from vplan.tasks import GenerationRejected, draft_task, witness_rejection
    try:
        draft = draft_task(seed, "feasible")
    except GenerationRejected as exc:
        return {"seed": seed, "accepted": False, "reason": exc.reason}
    scene = draft.task.scene
    target = next(o for o in scene.objects if o.id == draft.target_id)
    row = {"seed": seed, "stratum": f"d{len(scene.objects) - 1}_o{len(scene.obstacles)}", "template": draft.template,
           "target_shape": target.shape}
    from vplan.pipeline import run_attempt
    from vplan.planners.heuristic import heuristic_geometry
    plan = heuristic_geometry(scene, draft.target_id)
    # run_attempt directly (as run_witness does) so the execution record's events stay available
    attempt = run_attempt(plan, scene, draft.target_id, seed=seed % 2**32, config_snapshot=cfgs)
    from vplan.records import WitnessEvaluation
    witness = WitnessEvaluation(plan=plan, outcome=attempt.outcome, findings=tuple(attempt.findings),
                                result=attempt.judge)
    reason = witness_rejection(witness, draft.target_id)
    row.update(accepted=reason is None, reason=reason,
               findings=[f"{f.check}:{f.stage}:{'/'.join(f.bodies)}" for f in attempt.findings],
               violations=[k for k, v in attempt.judge.predicates.items() if not v] if attempt.judge else [],
               attach_rejected=sum(e.kind == "attach_rejected" for e in attempt.record.events) if attempt.record else 0)
    return row


def impossible_row(seed: int, impossible_type: int) -> dict:
    from vplan.records import TaskOracle
    from vplan.tasks import GenerationRejected, check_impossibility, draft_task
    try:
        draft = draft_task(seed, "impossible", impossible_type)
    except GenerationRejected as exc:
        return {"seed": seed, "type": impossible_type, "accepted": False, "reason": exc.reason}
    oracle = TaskOracle(target_id=draft.target_id, feasibility="impossible", proof=draft.proof)
    ok = check_impossibility(draft.task, oracle)
    return {"seed": seed, "type": impossible_type, "accepted": ok, "reason": None if ok else "proof_failed",
            "template": draft.template}


def _rate(rows: list[dict]) -> str:
    n = len(rows)
    k = sum(r["accepted"] for r in rows)
    return f"{k}/{n} ({100 * k / n:.0f}%)" if n else "0/0"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=60)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--json", default=None, help="optional path for the per-seed rows")
    args = ap.parse_args()
    cfgs = config_snapshot()
    seeds = range(args.seeds)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        feasible = list(pool.map(feasible_row, seeds, [cfgs] * len(seeds)))
    impossible = [impossible_row(s, t) for t in (1, 2, 3, 4) for s in seeds]

    print(f"feasible: {_rate(feasible)} accepted")
    print("  rejection reasons:", dict(Counter(r["reason"] for r in feasible if not r["accepted"])))
    for key in ("stratum", "template", "target_shape"):
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in feasible:
            groups[r.get(key, "rejected_before_witness")].append(r)
        print(f"  by {key}:", {k: _rate(v) for k, v in sorted(groups.items())})
    for axis, idx in (("distractors", 1), ("obstacles", 4)):
        groups = defaultdict(list)
        for r in feasible:
            if "stratum" in r:
                groups[r["stratum"][idx]].append(r)
        print(f"  by {axis}:", {k: _rate(v) for k, v in sorted(groups.items())})
    print("  finding checks:", dict(Counter(f.split(":")[0] for r in feasible for f in r.get("findings", []))))
    print("  findings:", dict(Counter(f for r in feasible for f in r.get("findings", []))))
    print("  failed judge predicates:", dict(Counter(v for r in feasible for v in r.get("violations", []))))
    print("  attach_rejected events:", sum(r.get("attach_rejected", 0) for r in feasible))
    for t in (1, 2, 3, 4):
        rows = [r for r in impossible if r["type"] == t]
        print(f"impossible type {t}: {_rate(rows)} accepted;",
              "reasons:", dict(Counter(r["reason"] for r in rows if not r["accepted"])))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"feasible": feasible, "impossible": impossible}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
