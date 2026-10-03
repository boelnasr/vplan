"""Command-line entry points. Week 1: demo-once (the basic scene lives with the test fixtures until W2.2)."""
from __future__ import annotations

import argparse

from vplan.pipeline import run_attempt
from vplan.planners.heuristic import heuristic_geometry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vplan")
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo-once", help="run the heuristic plan on the basic scene")
    d.add_argument("--obstacle", action="store_true")
    args = parser.parse_args(argv)
    if args.cmd == "demo-once":
        from tests.conftest import make_basic_scene  # requires PYTHONPATH to include the repo root ('src;.')
        scene = make_basic_scene(with_obstacle=args.obstacle)
        res = run_attempt(heuristic_geometry(scene, "cube_red"), scene, "cube_red")
        print("outcome:", res.outcome)
        for f in res.findings:
            print("finding:", f.model_dump_json())
        if res.judge:
            print("judge:", res.judge.model_dump_json())
        return 0 if (res.judge and res.judge.success) else 1
    return 2
