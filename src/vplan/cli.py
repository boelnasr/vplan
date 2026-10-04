"""Command-line entry points. Week 1: demo-once (the basic scene lives with the test fixtures until W2.2)."""
from __future__ import annotations

import argparse

import pybullet as p

from vplan.pipeline import run_attempt
from vplan.planners.heuristic import heuristic_geometry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vplan")
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo-once", help="run the heuristic plan on the basic scene")
    d.add_argument("--obstacle", action="store_true")
    d.add_argument("--gui", action="store_true", help="watch execution in the PyBullet GUI at real-time speed")
    args = parser.parse_args(argv)
    if args.cmd == "demo-once":
        from tests.conftest import make_basic_scene  # requires PYTHONPATH to include the repo root ('src;.')
        scene = make_basic_scene(with_obstacle=args.obstacle)
        gui = None
        if args.gui:
            # only the execution client gets the GUI; the verifier's preview client stays DIRECT (one GUI per process)
            gui = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=gui)
            p.resetDebugVisualizerCamera(1.3, 45, -30, [0.45, 0.1, 0.1], physicsClientId=gui)
        try:
            res = run_attempt(heuristic_geometry(scene, "cube_red"), scene, "cube_red",
                              physics_client=gui, realtime=gui is not None)
            if gui is not None:
                input("done - press Enter to close the GUI ")
        finally:
            if gui is not None:
                p.disconnect(gui)
        print("outcome:", res.outcome)
        for f in res.findings:
            print("finding:", f.model_dump_json())
        if res.judge:
            print("judge:", res.judge.model_dump_json())
        return 0 if (res.judge and res.judge.success) else 1
    return 2
