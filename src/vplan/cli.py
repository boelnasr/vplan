"""Command-line entry points: demo-once, the feedback pilot (prepare / run / report / show), the W3.1 evaluation
(eval prepare / eval run) and the spec §15 report (report build)."""
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


def _cosmos_gate(url: str | None, subject: str | None):
    """The demo-only live Cosmos gate (D47) bound to the instructed object, or None without --cosmos-url."""
    if url is None:
        return None
    from vplan.records import VisualGateReport
    from vplan.visual.gate import CosmosGate
    if subject is None:
        print("cosmos: skipped (the instruction names no single object)")
        return lambda plan, traj, scene: VisualGateReport(status="skipped", note="no single instructed object")
    gate = CosmosGate(url, on_check=lambda c: print(
        f"cosmos (learned check) {c.stage} [{', '.join(c.views)}]: {c.verdict.verdict} ({c.verdict.latency_s:.1f} s) - {c.verdict.reason}",
        flush=True))
    print(f"cosmos: learned visual check on the {subject} (not verification; D34 pilot balanced accuracy 0.70)")
    return lambda plan, traj, scene: gate(plan, traj, scene, subject=subject)


def _print_result(res) -> None:
    if res.visual is not None and res.visual.status == "unavailable":
        print(f"*** COSMOS UNAVAILABLE - executed on the geometric checks only ({res.visual.note}) ***")
    elif res.visual is not None:
        print("cosmos gate:", res.visual.status)
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
                replay: bool = False, cosmos_url: str | None = None) -> int:
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
    from vplan.visual.gate import subject
    if unverified and cosmos_url:
        print("cosmos: off - the visual gate runs only after the Verifier, and --unverified skips it")
    gate = None if unverified else _cosmos_gate(cosmos_url, subject(task.task))
    gui = _open_gui()
    try:
        # unverified executes even a blocked plan, to show the failure the verifier prevented
        res = run_attempt(rnd.call.plan, task.task.scene, task.oracle.target_id, verify_gate=not unverified,
                          physics_client=gui, realtime=True, visual_gate=gate)
        if res.outcome == "blocked" and res.visual is not None and res.visual.status == "fail":
            print("blocked by the Cosmos visual check (learned, not verification)")
        elif res.outcome == "blocked":
            print("blocked before motion (pass --unverified to watch it execute anyway)")
    finally:
        _wait_and_close(gui)
    _print_result(res)
    return 0 if (res.judge and res.judge.success) else 1


def _eval(args) -> int:
    """eval prepare / eval run. Exit codes: 0 done, 1 an incomplete run (resume it), 2 refused, 130 interrupted."""
    from vplan.evaluate import pilot, runner, seedsets
    try:
        for name in ("output", "seed_dir", "run_dir", "resume", "env_file"):
            if getattr(args, name, None):
                pilot.check_path_arg(getattr(args, name))
        if args.eval_cmd == "prepare":
            output = args.output or seedsets.default_seed_dir(args.seeds)
            manifest = seedsets.prepare_seed_set(
                output, args.seeds, feasible=args.feasible, impossible=args.impossible,
                max_candidates=args.max_candidates, allow_test_split=args.allow_test_split,
                progress=lambda attempted, accepted, n: print(f"{accepted}/{n} accepted; {attempted} attempted",
                                                              flush=True))
            print(f"prepared {output}: {manifest['n']} tasks, seed-set hash {manifest['seed_set_hash']}")
            return 0
        summary = runner.run_eval(
            args.config, args.seeds, seed_dir=args.seed_dir, run_dir=args.run_dir, resume=args.resume, n=args.n,
            env_file=args.env_file, model=args.model, temperature=args.temperature, allow_paid=args.allow_paid,
            allow_source_drift=args.allow_source_drift, timeout_s=args.timeout, max_rpm=args.max_rpm)
    except pilot.PilotError as exc:  # EvalError is a PilotError; messages never carry credentials
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted: the run directory printed above can be continued with --resume", file=sys.stderr)
        return 130
    counts = ", ".join(f"{k} {v}" for k, v in sorted(summary.outcomes.items()))
    if summary.status == "complete":
        print(f"complete: {summary.recorded}/{summary.planned} attempts ({counts}; {summary.successes} judged "
              f"successes) in {summary.run_dir}")
        return 0
    print(f"INCOMPLETE: {summary.recorded}/{summary.planned} attempts ({counts}); stopped: {summary.stop_reason}. "
          f"Continue with --resume {summary.run_dir}", file=sys.stderr)
    return 1


def _report(args) -> int:
    """report build. Exit codes, as for eval run: 0 built, 1 built from an incomplete run (titled INCOMPLETE; resume
    the run and rebuild), 2 refused."""
    from vplan.evaluate import collect, pilot
    from vplan.evaluate.metrics import MetricsError
    try:
        for value in [*args.run, args.output, args.seed_dir]:
            if value:
                pilot.check_path_arg(value)
        files, status = collect.build(args.run, args.output, seed_dir=args.seed_dir)
    except (pilot.PilotError, MetricsError) as exc:  # EvalError is a PilotError; messages never carry credentials
        print(f"error: {exc}", file=sys.stderr)
        return 2
    labels = " | ".join(status["title_prefixes"])  # ASCII: a Windows console without PYTHONUTF8 is cp1252
    print(f"report: {files.html} and {files.metrics}" + (f" ({labels})" if labels else ""))
    if status["incomplete"]:
        print("INCOMPLETE: a run recorded fewer attempts than it planned; continue it with eval run --resume RUN_DIR "
              "and build the report again", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vplan")
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo-once", help="run the heuristic plan on the basic scene")
    d.add_argument("--obstacle", action="store_true")
    d.add_argument("--gui", action="store_true", help="watch execution in the PyBullet GUI at real-time speed")
    d.add_argument("--cosmos-url", help="live Cosmos gate (demo only, D47), e.g. http://127.0.0.1:8000/v1")
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
    sh.add_argument("--cosmos-url", help="live Cosmos gate before re-execution (demo only, D47)")
    ev = sub.add_parser("eval", help="W3.1 evaluation: seed sets and supervised batch runs (spec §14)")
    esub = ev.add_subparsers(dest="eval_cmd", required=True)
    ep = esub.add_parser("prepare", help="generate and freeze a seed set (no model calls)")
    ep.add_argument("--seeds", required=True, choices=("dev", "test"))
    ep.add_argument("--output", help="default: runs/seeds/<split>")
    ep.add_argument("--feasible", type=int, help="override the D51 mix; pass with --impossible")
    ep.add_argument("--impossible", type=int, help="spread over types 1-4, lower types first")
    ep.add_argument("--max-candidates", type=int, help="candidate seeds to try in total (default 2n + 20)")
    ep.add_argument("--allow-test-split", action="store_true",
                    help="the held-out test split, generated only after the W3.3 freeze (spec §6)")
    er = esub.add_parser("run", help="run one spec §14 config over a prepared seed set")
    er.add_argument("--config", required=True, type=int, help="1, 3, 5, 6 or 7 (config 2 is round 1 of 3)")
    er.add_argument("--seeds", required=True, choices=("dev", "test"))
    er.add_argument("--n", type=int, help="run only the first N tasks (a stratified prefix)")
    er.add_argument("--seed-dir", help="the prepared seed set (default runs/seeds/<split>; on --resume, the run's)")
    er.add_argument("--run-dir", help="a new run directory (default runs/c<config>-<split>-<UTC time>)")
    er.add_argument("--resume", help="continue RUN_DIR: skip recorded attempts, refuse an incompatible run")
    er.add_argument("--env-file", help="KEY=VALUE file with LLM_BASE_URL, LLM_API_KEY, LLM_MODEL (LLM configs)")
    er.add_argument("--model", help="overrides LLM_MODEL; required for config 6")
    er.add_argument("--temperature", type=float, help="every planner call (default 0.7)")
    er.add_argument("--allow-paid", action="store_true", help="permit non-free or non-OpenRouter models")
    er.add_argument("--allow-source-drift", action="store_true", help="record source changes instead of refusing")
    er.add_argument("--timeout", type=float, default=600.0, help="per-attempt wall-clock limit, s")
    er.add_argument("--max-rpm", type=float, help="pace provider requests to at most this many per minute")
    rep = sub.add_parser("report", help="spec §15 report: metrics.json and one static HTML file")
    rsub = rep.add_subparsers(dest="report_cmd", required=True)
    rb = rsub.add_parser("build", help="build the report from W3.1 run directories over one seed set")
    rb.add_argument("--run", required=True, action="append",
                    help="a W3.1 run directory; repeat once per configuration (config 2 comes from config 3's run)")
    rb.add_argument("--output", required=True, help="the report directory: metrics.json and report.html")
    rb.add_argument("--seed-dir", help="the runs' seed set (default: the directory the first run recorded)")
    args = parser.parse_args(argv)
    if args.cmd == "eval":
        return _eval(args)
    if args.cmd == "report":
        return _report(args)
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
                if args.cosmos_url:
                    extra["cosmos_url"] = args.cosmos_url
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
        red = next(o for o in scene.objects if o.id == "cube_red")
        gate = _cosmos_gate(args.cosmos_url, f"{red.color} {red.shape}")
        gui = _open_gui() if args.gui else None
        try:
            res = run_attempt(heuristic_geometry(scene, "cube_red"), scene, "cube_red",
                              physics_client=gui, realtime=gui is not None, visual_gate=gate)
        finally:
            if gui is not None:
                _wait_and_close(gui)
        _print_result(res)
        return 0 if (res.judge and res.judge.success) else 1
    return 2
