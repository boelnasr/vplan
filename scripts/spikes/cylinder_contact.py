"""Reproduce D43's cylinder collision comparison without paid services.

Run from the repository root with PYTHONPATH=src;. (Windows) or src:. (POSIX):
    python scripts/spikes/cylinder_contact.py --variant builtin --cohort first
    python scripts/spikes/cylinder_contact.py --variant prism16 --cohort comparison

The first cohort has 21 cylinder drafts; seed 50 is Verifier-blocked, leaving
20 Judge-eligible runs. The comparison cohort has 45 cylinder drafts. These
seeds were used to choose 16 sides. The fresh cohort is seeds 144-263 and is
reserved for evaluating the already-selected 16-sided shape.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pybullet as p

import vplan.sim.scene as scene_module
from scripts.spikes.r6_generation import feasible_row
from vplan.config import config_snapshot
from vplan.tasks import GenerationRejected, draft_task


def cylinder_seeds(seeds: range) -> list[int]:
    """Select drafts whose target is a cylinder, before running any witness."""
    selected = []
    for seed in seeds:
        try:
            draft = draft_task(seed, "feasible")
        except GenerationRejected:
            continue
        target = next(o for o in draft.task.scene.objects if o.id == draft.target_id)
        if target.shape == "cylinder":
            selected.append(seed)
    return selected


def use_variant(variant: str) -> None:
    """Swap only cylinder collision construction; leave visuals and Judge unchanged."""
    if variant.startswith("prism"):
        scene_module.CYLINDER_SIDES = int(variant.removeprefix("prism"))
        return
    original = p.createCollisionShape

    def built_in(shape, *args, **kwargs):
        if shape == p.GEOM_MESH and "vertices" in kwargs:
            vertices = np.asarray(kwargs["vertices"])
            radius = (np.hypot(vertices[:, 0], vertices[:, 1]).max()
                      * math.cos(math.pi / scene_module.CYLINDER_SIDES) + scene_module.CONVEX_MARGIN_M)
            height = 2 * vertices[:, 2].max() + 2 * scene_module.CONVEX_MARGIN_M
            return original(p.GEOM_CYLINDER, radius=radius, height=height,
                            physicsClientId=kwargs.get("physicsClientId", 0))
        return original(shape, *args, **kwargs)

    p.createCollisionShape = built_in


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", choices=("builtin", "prism16", "prism24", "prism32"), required=True)
    parser.add_argument("--cohort", choices=("first", "comparison", "fresh"), required=True)
    parser.add_argument("--json", type=Path, help="optional per-seed output path")
    args = parser.parse_args()
    seed_range = {"first": range(60), "comparison": range(60, 144), "fresh": range(144, 264)}[args.cohort]
    seeds = cylinder_seeds(seed_range)
    use_variant(args.variant)
    cfg = config_snapshot()
    rows = [feasible_row(seed, cfg) for seed in seeds]
    eligible = [r for r in rows if r["reason"] != "witness_blocked"]
    passed = sum(r["accepted"] for r in eligible)
    print(f"{args.variant} {args.cohort}: {passed}/{len(eligible)} accepted; "
          f"{len(rows) - len(eligible)} Verifier-blocked")
    for row in rows:
        print(f"  seed {row['seed']}: {row['reason'] or 'accepted'} {row['violations']}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"variant": args.variant, "cohort": args.cohort, "rows": rows}, indent=2),
                             encoding="utf-8")


if __name__ == "__main__":
    main()
