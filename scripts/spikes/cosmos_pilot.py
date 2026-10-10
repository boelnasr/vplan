"""W2.1 Cosmos gate pilot (spec §11): 30 key frames, 10 per stage, balanced by the state predicates. A pilot, not
validation. Throwaway spike in two steps, because the local vLLM server is reachable only from WSL:

  frames (Windows Python, needs pybullet):
    PYTHONPATH='src;.' python scripts/spikes/cosmos_pilot.py frames --out runs/cosmos-pilot/frames
  ask (WSL venv, imports no pybullet):
    PYTHONPATH=src ~/venvs/cosmos/bin/python scripts/spikes/cosmos_pilot.py ask --frames runs/cosmos-pilot/frames \
        --base-url http://127.0.0.1:8000/v1 --thinking both

Labels are ALWAYS the computed predicate. Every predicate-false sample is an injected negative; there are no
natural negatives in this pilot.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

STAGES = ("pregrasp", "grasp", "prerelease")
KEY_POSE = {"pregrasp": "pregrasp", "grasp": "grasp", "prerelease": "release"}  # pre-release = TCP at release, closed
NEGATIVES = {"pregrasp": ("wrong_object",), "grasp": ("offset", "wrong_object"),
             "prerelease": ("outside_bin", "wrong_object")}
PER_CLASS = 5
MAX_DRAWS_PER_SAMPLE = 25
VIEWS = ("front_oblique", "top", "side")
ASK_VIEWS = (("front_oblique", "top"), ("front_oblique", "top", "side"))  # spec: uncertain -> add the third view
MODEL_REVISION = "344d602b128d1bbdacb43b08d0a3626f46343e29"
VLLM_VERSION = "0.31.0"
NOTE_NO_NATURAL = ("No natural negatives in this pilot: every predicate-false sample is an injected negative "
                   "(spec §11 reports natural and injected negatives separately).")


# ---- frames ------------------------------------------------------------------------------------
def _sample(seed: int, stage: str, kind: str | None, cid: int, vcfg: dict) -> tuple[dict | None, str]:
    """Pose the scene for one sample; returns (sample, "") or (None, discard reason). kind None = positive."""
    import numpy as np
    import pybullet as p

    from vplan.geometry import T_to_pose, pose_to_T, rot_angle
    from vplan.motion.poses import key_poses
    from vplan.planners.heuristic import heuristic_geometry
    from vplan.robot import HOME_Q, load_panda
    from vplan.sim.scene import build_scene
    from vplan.tasks import generate_candidate
    from vplan.visual.cosmos import stage_question
    from vplan.visual.predicates import grasp_truth, pregrasp_truth, prerelease_truth
    from vplan.visual.render import render_views

    stratum = ("distractor", "obstacle")[seed % 2]  # wrong-object negatives need distractors; clear scenes have none
    cand = generate_candidate(seed, stratum)
    scene, target = cand.task.scene, cand.target_id
    objs = {o.id: o for o in scene.objects}
    rng = np.random.default_rng(seed)
    posed = str(rng.choice(sorted(k for k in objs if k != target))) if kind == "wrong_object" else target
    m = load_panda()
    kps = {k.name: k.T for k in key_poses(heuristic_geometry(scene, posed), scene, m.fk(HOME_Q))}
    T = kps[KEY_POSE[stage]].copy()
    if kind == "offset":  # slid normal to the finger plane: the fingers straddle empty table beside the object
        T[:3, 3] += T[:3, 0] * rng.choice((-1.0, 1.0)) * rng.uniform(0.03, 0.04)
    elif kind == "outside_bin":  # carried past a bin wall along world x at the release height
        T[:3, 3] += np.array([rng.choice((-1.0, 1.0)) * rng.uniform(0.17, 0.20), 0.0, 0.0])
    np.random.seed(seed % 2**32)  # ManipulaPy IK perturbations use the global RNG
    q, _ = m.ik(T, HOME_Q, max_iterations=500)
    Tq = m.fk(q)
    if (not np.all(np.isfinite(q)) or np.linalg.norm(Tq[:3, 3] - T[:3, 3]) > vcfg["ik_pos_tol_m"]
            or np.degrees(rot_angle(Tq[:3, :3], T[:3, :3])) > vcfg["ik_rot_tol_deg"]):
        return None, "ik_failed"
    p.resetSimulation(physicsClientId=cid)
    h = build_scene(scene, cid, q)
    ls = p.getLinkState(h.robot, h.tcp_link, computeForwardKinematics=True, physicsClientId=cid)
    T_tcp = pose_to_T(ls[4], ls[5])  # simulator TCP, not the IK target: the label comes from what is rendered
    carried = None
    if stage == "prerelease":
        carried = posed
        o = objs[carried]
        attach = np.linalg.inv(kps["grasp"]) @ pose_to_T(o.pose.position, o.pose.orientation)
        pos, orn = T_to_pose(T_tcp @ attach)
        p.resetBasePositionAndOrientation(h.bodies[carried], pos, orn, physicsClientId=cid)
        for j in h.finger_joints:  # closed on the object
            p.resetJointState(h.robot, j, o.size[0] / 2, physicsClientId=cid)
    T_obj = {k: pose_to_T(*p.getBasePositionAndOrientation(h.bodies[k], physicsClientId=cid)) for k in objs}
    if stage == "pregrasp":
        truth = pregrasp_truth(T_tcp, {k: tuple(v[:3, 3]) for k, v in T_obj.items()}, target)
    elif stage == "grasp":
        truth = grasp_truth(T_tcp, T_obj[target], objs[target].size, vcfg)
    else:
        truth = prerelease_truth(carried, T_obj[carried], objs[carried].size, target, scene.bin)
    if truth != (kind is None):
        return None, "predicate_disagrees"
    t = objs[target]
    return {"seed": seed, "stratum": stratum, "stage": stage, "question": stage_question(stage, cand.task),
            "predicate": truth, "negative": None if kind is None else f"injected:{kind}",
            "task": {"instruction": cand.task.instruction, "color": t.color, "shape": t.shape},
            "posed_object": {"color": objs[posed].color, "shape": objs[posed].shape},
            "images": render_views(h, VIEWS)}, ""


def frames(out: Path, seed0: int) -> None:
    import pybullet as p

    from vplan.config import load_config
    from vplan.visual.render import CAMERAS, to_png

    out.mkdir(parents=True, exist_ok=True)
    vcfg = load_config("verifier")
    cid = p.connect(p.DIRECT)
    seed, discarded, rows = seed0, [], []
    try:
        for stage in STAGES:
            negs = NEGATIVES[stage]
            kinds = [None] * PER_CLASS + [negs[i % len(negs)] for i in range(PER_CLASS)]
            for k, kind in enumerate(kinds):
                for _ in range(MAX_DRAWS_PER_SAMPLE):
                    sample, why = _sample(seed, stage, kind, cid, vcfg)
                    seed += 1  # every draw consumes a seed, so all 30 scenes are distinct
                    if sample is not None:
                        break
                    discarded.append({"seed": seed - 1, "stage": stage, "intended": kind or "positive", "why": why})
                else:
                    raise RuntimeError(f"{stage} sample {k}: no valid draw in {MAX_DRAWS_PER_SAMPLE} seeds")
                sid = f"{stage}_{k:02d}"
                images = sample.pop("images")
                sample["views"] = {}
                for view, rgb in images.items():
                    (out / f"{sid}_{view}.png").write_bytes(to_png(rgb))
                    sample["views"][view] = f"{sid}_{view}.png"
                rows.append({"id": sid, **sample})
                print(f"{sid}: seed {sample['seed']} predicate={sample['predicate']} negative={sample['negative']}")
    finally:
        p.disconnect(cid)
    with open(out / "labels.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    per_stage = {s: dict(Counter(d["why"] for d in discarded if d["stage"] == s)) for s in STAGES}
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seed0": seed0,
                "seed_end_exclusive": seed, "n_samples": len(rows), "discarded_per_stage": per_stage,
                "discarded": discarded, "cameras": CAMERAS, "note": NOTE_NO_NATURAL}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} samples to {out}; discarded/resampled per stage: {per_stage}")
    print(NOTE_NO_NATURAL)


# ---- ask ---------------------------------------------------------------------------------------
def _report(mode: str, labels: list[dict], answers: list[dict], args: argparse.Namespace) -> str:
    from vplan.visual.cosmos import score

    final = {}
    for a in answers:  # the last record per sample is its final verdict
        final[a["id"]] = a["verdict"]["verdict"] if "verdict" in a else "unavailable"
    truths = [s["predicate"] for s in labels]
    verdicts = [final[s["id"]] for s in labels]
    sc = score(truths, verdicts)
    lat = [a["verdict"]["latency_s"] for a in answers if "verdict" in a]
    med = statistics.median(lat) if lat else float("inf")
    unavailable = sum(v == "unavailable" for v in verdicts)
    reasked = [a for a in answers if len(a["views"]) == 3]
    go = sc["balanced_accuracy"] >= 0.90 and med <= 5.0
    cols = ("pass", "fail", "uncertain", "unavailable")
    lines = [f"# {'DEGRADED: ' if unavailable else ''}Cosmos gate pilot, thinking {mode}", "",
             "A pilot, not validation (spec §11).", "",
             f"- Model: `{args.model}` at revision `{args.model_revision}`, served by vLLM {args.vllm_version} "
             f"at {args.base_url}",
             f"- Run: {datetime.now(timezone.utc).isoformat(timespec='seconds')}; frames: `{args.frames}` "
             f"({len(labels)} samples, {sum(truths)} predicate-true / {len(truths) - sum(truths)} predicate-false)",
             f"- {NOTE_NO_NATURAL}", "",
             "## Result", "",
             f"- Balanced accuracy (`pass` <=> predicate true; a final `uncertain` counts as wrong): "
             f"**{sc['balanced_accuracy']:.3f}** (TPR {sc['tpr']:.3f}, TNR {sc['tnr']:.3f})",
             f"- Abstention (final `uncertain`): {sc['abstention_rate']:.3f}",
             f"- Unavailable (transport/HTTP errors, counted wrong): {unavailable}",
             f"- Latency per request, over {len(lat)} requests including re-asks: median {med:.2f} s, "
             f"max {max(lat) if lat else float('nan'):.2f} s",
             f"- Re-asks with the side view after `uncertain`: {len(reasked)}, of which resolved to pass/fail: "
             f"{sum(final[a['id']] in ('pass', 'fail') for a in reasked)}",
             f"- **{'GO' if go else 'NO-GO'}**: go iff balanced accuracy >= 0.90 and median latency <= 5 s "
             f"(here {sc['balanced_accuracy']:.3f}, {med:.2f} s).",
             "- With 15/15 classes, one error moves balanced accuracy by 1/30 = 0.033.", "",
             "## Per-stage confusion (final verdict)", "",
             "| stage | predicate | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for stage in STAGES:
        for truth in (True, False):
            c = Counter(final[s["id"]] for s in labels if s["stage"] == stage and s["predicate"] == truth)
            lines.append(f"| {stage} | {truth} | " + " | ".join(str(c[k]) for k in cols) + " |")
    lines += ["", "## Injected negatives by type (correct = fail)", "",
              "| stage | type | n | " + " | ".join(cols) + " |", "|---|---|---|" + "---|" * len(cols)]
    for stage in STAGES:
        for kind in sorted({s["negative"] for s in labels if s["stage"] == stage and s["negative"]}):
            c = Counter(final[s["id"]] for s in labels if s["stage"] == stage and s["negative"] == kind)
            lines.append(f"| {stage} | {kind} | {sum(c.values())} | " + " | ".join(str(c[k]) for k in cols) + " |")
    return "\n".join(lines) + "\n"


def ask(args: argparse.Namespace) -> None:
    from vplan.visual.cosmos import CosmosUnavailable, ask_cosmos

    labels = [json.loads(line) for line in open(args.frames / "labels.jsonl", encoding="utf-8")]
    for mode in (("off", "on") if args.thinking == "both" else (args.thinking,)):
        answers = []
        with open(args.frames / f"answers_{mode}.jsonl", "w", encoding="utf-8") as f:
            for s in labels:
                for views in ASK_VIEWS:
                    rec = {"id": s["id"], "stage": s["stage"], "mode": mode, "views": list(views)}
                    try:
                        v = ask_cosmos({k: (args.frames / s["views"][k]).read_bytes() for k in views}, s["question"],
                                       args.base_url, model=args.model, enable_thinking=mode == "on",
                                       timeout=args.timeout)
                        rec["verdict"] = v.model_dump()
                    except CosmosUnavailable as e:
                        rec["error"] = str(e)
                    f.write(json.dumps(rec) + "\n")
                    answers.append(rec)
                    verdict = rec["verdict"]["verdict"] if "verdict" in rec else "unavailable"
                    print(f"[{mode}] {s['id']} {len(views)} views: {verdict} (predicate {s['predicate']})", flush=True)
                    if verdict != "uncertain":
                        break
        report = _report(mode, labels, answers, args)
        (args.frames / f"report_{mode}.md").write_text(report, encoding="utf-8")
        print(report)


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    fr = sub.add_parser("frames")
    fr.add_argument("--out", type=Path, required=True)
    fr.add_argument("--seed0", type=int, default=1000)  # the dev cohort uses seeds from 0; keep clear of them
    ak = sub.add_parser("ask")
    ak.add_argument("--frames", type=Path, required=True)
    ak.add_argument("--base-url", required=True)
    ak.add_argument("--thinking", choices=("off", "on", "both"), default="off")
    ak.add_argument("--model", default="nvidia/Cosmos3-Edge")
    ak.add_argument("--model-revision", default=MODEL_REVISION)
    ak.add_argument("--vllm-version", default=VLLM_VERSION)
    ak.add_argument("--timeout", type=float, default=30.0)  # s per request
    args = ap.parse_args(argv)
    if args.cmd == "frames":
        frames(args.out, args.seed0)
    else:
        ask(args)


if __name__ == "__main__":
    main(sys.argv[1:])
