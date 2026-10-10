"""Command-line entry points: demo-once and the feedback pilot (prepare / run / report / show)."""
from __future__ import annotations

import argparse
import sys

import pybullet as p

from vplan.pipeline import run_attempt
from vplan.planners.heuristic import heuristic_geometry
from vplan.records import PickPlacePlan


def _open_gui() -> int:
    # only the execution client gets the GUI; the verifier's preview client stays DIRECT (one GUI per process)
    gui = p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=gui)
    p.resetDebugVisualizerCamera(1.3, 45, -30, [0.45, 0.1, 0.1], physicsClientId=gui)
    return gui


def _wait_and_close(gui: int) -> None:
    try:
        input("done - press Enter to close the GUI ")
    except EOFError:  # no interactive stdin (piped or backgrounded): nothing to wait for
        pass
    finally:
        if p.isConnected(gui):  # the user may already have closed the window
            p.disconnect(gui)


def _print_result(res) -> None:
    print("outcome:", res.outcome)
    for f in res.findings:
        print("finding:", f.model_dump_json())
    if res.judge:
        print("judge:", res.judge.model_dump_json())


def _replay_round(task, rnd) -> int:
    """Play back the stored ExecutionRecord of one round (W2.6): no physics, so it shows exactly what was judged."""
    from vplan import replay
    rec = rnd.evaluation.record if rnd.evaluation else None
    if rec is None:
        print("no execution record to replay (the round was not executed)")
        return 1
    print(f"replaying {len(rec.q_replay)} recorded frames")
    gui = _open_gui()
    try:
        replay.replay(rec, task.task.scene, gui=True, client=gui)
    finally:
        _wait_and_close(gui)
    if rnd.evaluation.judge:
        print("recorded judge:", rnd.evaluation.judge.model_dump_json())
    return 0


def _show_round(run: str, task_id: str, arm: str, round_index: int | None, *, unverified: bool,
                replay: bool = False) -> int:
    """Re-execute one recorded pilot plan in the GUI (re-simulation), or with replay=True play back its stored record."""
    from vplan.evaluate import pilot
    task, rnd = pilot.select_round(run, task_id, arm, round_index)
    print(f"{task_id} {arm} round {rnd.round_index}: recorded {rnd.evaluation.outcome if rnd.evaluation else rnd.call.outcome}")
    print("instruction:", task.task.instruction)
    if replay:
        return _replay_round(task, rnd)
    if not isinstance(rnd.call.plan, PickPlacePlan):
        print("nothing to execute:", rnd.call.plan.model_dump_json() if rnd.call.plan else rnd.call.outcome)
        return 1
    print("plan:", rnd.call.plan.model_dump_json())
    gui = _open_gui()
    try:
        # unverified executes even a blocked plan, to show the failure the verifier prevented
        res = run_attempt(rnd.call.plan, task.task.scene, task.oracle.target_id, verify_gate=not unverified,
                          physics_client=gui, realtime=True)
        if res.outcome == "blocked":
            print("blocked before motion (pass --unverified to watch it execute anyway)")
    finally:
        _wait_and_close(gui)
    _print_result(res)
    return 0 if (res.judge and res.judge.success) else 1


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
    pp.add_argument("--generator", choices=("pilot", "hard"), default="pilot",
                    help="pilot: the original clear/distractor/obstacle cohort; hard: dense W2.2 feasible tasks")
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
    sh = psub.add_parser("show", help="list a run's rounds, or re-execute one recorded plan in the PyBullet GUI")
    sh.add_argument("--run", required=True)
    sh.add_argument("--task", help="task id; omit to list every task's rounds")
    sh.add_argument("--arm", choices=("repair", "resample"), default="repair")
    sh.add_argument("--round", type=int, help="0 is the shared initial call; default: the arm's final round")
    sh.add_argument("--unverified", action="store_true", help="execute even if the verifier blocks the plan")
    sh.add_argument("--replay", action="store_true", help="play back the stored execution record instead of re-simulating")
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
                              generator=args.generator,
                              progress=lambda attempted, accepted, n: print(
                                  f"{accepted}/{n} accepted; {attempted} attempted", flush=True))
                print(f"prepared {args.output}")
            elif args.pilot_cmd == "run":
                run_dir = pilot.run_live(args.input, env_file=args.env_file, model=args.model,
                                         temperature=args.temperature, limit=args.limit,
                                         allow_source_drift=args.allow_source_drift, allow_paid=args.allow_paid)
                print(f"report: {run_dir}")
            elif args.pilot_cmd == "show":
                if args.task is None:
                    print("\n".join(pilot.list_rounds(args.run)))
                    return 0
                extra = {"replay": True} if args.replay else {}  # default call unchanged
                return _show_round(args.run, args.task, args.arm, args.round, unverified=args.unverified, **extra)
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
        gui = _open_gui() if args.gui else None
        try:
            res = run_attempt(heuristic_geometry(scene, "cube_red"), scene, "cube_red",
                              physics_client=gui, realtime=gui is not None)
        finally:
            if gui is not None:
                _wait_and_close(gui)
        _print_result(res)
        return 0 if (res.judge and res.judge.success) else 1
    return 2
