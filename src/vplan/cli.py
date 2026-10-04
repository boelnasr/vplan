"""Command-line entry points: demo-once and the feedback pilot (prepare / run / report)."""
from __future__ import annotations

import argparse
import sys

import pybullet as p

from vplan.pipeline import run_attempt
from vplan.planners.heuristic import heuristic_geometry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vplan")
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo-once", help="run the heuristic plan on the basic scene")
    d.add_argument("--obstacle", action="store_true")
    d.add_argument("--gui", action="store_true", help="watch execution in the PyBullet GUI at real-time speed")
    pl = sub.add_parser("pilot", help="feasible-only development feedback pilot")
    psub = pl.add_subparsers(dest="pilot_cmd", required=True)
    pp = psub.add_parser("prepare", help="freeze a witness-validated cohort (no model calls)")
    pp.add_argument("--output", required=True)
    pp.add_argument("--n", type=int, default=20)
    pp.add_argument("--seed", type=int, default=0)
    pp.add_argument("--max-candidates", type=int, default=100)
    pr = psub.add_parser("run", help="run the paired repair/resampling experiment on a prepared cohort")
    pr.add_argument("--input", required=True)
    pr.add_argument("--env-file", required=True, help="KEY=VALUE file with LLM_BASE_URL, LLM_API_KEY, LLM_MODEL")
    pr.add_argument("--model", help="overrides LLM_MODEL")
    pr.add_argument("--temperature", type=float, default=0.7, help="used for every call in both arms")
    pr.add_argument("--limit", type=int, help="run only the first K tasks (smoke test)")
    pr.add_argument("--allow-source-drift", action="store_true", help="record source changes instead of refusing")
    pr.add_argument("--allow-paid", action="store_true", help="permit non-free or non-OpenRouter models")
    rp = psub.add_parser("report", help="rebuild summary.json and report.md for a run directory")
    rp.add_argument("--run", required=True)
    args = parser.parse_args(argv)
    if args.cmd == "pilot":
        from vplan.evaluate import pilot
        from vplan.tasks import CohortPreparationError
        try:
            for name in ("output", "input", "env_file", "run"):
                if getattr(args, name, None):
                    pilot.check_path_arg(getattr(args, name))
            if args.pilot_cmd == "prepare":
                pilot.prepare(args.output, n=args.n, seed=args.seed, max_candidates=args.max_candidates,
                              progress=lambda attempted, accepted, n: print(
                                  f"{accepted}/{n} accepted; {attempted} attempted", flush=True))
                print(f"prepared {args.output}")
            elif args.pilot_cmd == "run":
                run_dir = pilot.run_live(args.input, env_file=args.env_file, model=args.model,
                                         temperature=args.temperature, limit=args.limit,
                                         allow_source_drift=args.allow_source_drift, allow_paid=args.allow_paid)
                print(f"report: {run_dir}")
            else:
                summary = pilot.write_report(args.run)
                print("complete" if summary["complete"] else "INCOMPLETE" if not summary["finished"] else "limited")
                return 0 if summary["finished"] else 1
        except (pilot.PilotError, CohortPreparationError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        return 0
    if args.cmd == "demo-once":
        from vplan.tasks.scenes import make_basic_scene
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
