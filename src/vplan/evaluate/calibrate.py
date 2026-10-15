"""W3.3 steps 1-3: sigma_thr calibrated without selection bias (spec §9 "sigma_thr" and V-4; spec §13 V3).

Sample: the dev seed set's feasible slots, filled exactly as `eval prepare --seeds dev` fills them (the D51 mix, the
slot order and the candidate seeds, through seedsets._candidate), but with sigma_min_threshold = 0 for the Verifier
and the Judge in the snapshot every witness is generated and accepted under. So no witness is filtered by the
threshold being calibrated. Impossible slots are filled too (their proofs ignore the thresholds), so candidate seeds
advance exactly as in prepare.

Per accepted witness:
- Verifier side: the minimum over every trajectory sample of verify.checks.sigma_min, the function V-4 uses. The
  trajectory is rebuilt as run_attempt built it and must reproduce the executed approval token (as frames.py does).
- Judge side: min(ExecutionRecord.sigma_min_steps), the values V3 compares.

Rule: sigma_thr = the 1st percentile (numpy.percentile, method="linear") of the per-witness Verifier minima, rounded
down to 0.001. The Judge threshold is min(sigma_thr, the same statistic of the Judge minima), so the Judge never fails
a witness the Verifier's threshold admits. `apply_thresholds` writes both into config/ and changes nothing else.
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import math
import os
import platform
from collections.abc import Callable, Sequence
from copy import deepcopy
from decimal import ROUND_FLOOR, Decimal
from pathlib import Path
from typing import NamedTuple

import numpy as np

from vplan.config import CONFIG_DIR, config_hash, config_snapshot
from vplan.evaluate.pilot import _git, _sha256, _utc_now, source_digest, source_hashes
from vplan.evaluate.seedsets import DEFAULT_MIX, SEED_BASE, EvalError, Slot, _candidate, mix_for, slot_order
from vplan.execute.executor import approval_token
from vplan.motion.trajectory import build_trajectory
from vplan.pipeline import AttemptResult
from vplan.records import SeedSetExclusion, SigmaWitness, TaskSpec, WitnessEvaluation
from vplan.robot import HOME_Q, load_panda
from vplan.tasks.generate import witness_attempt, witness_evaluation, witness_rejection
from vplan.verify.checks import sigma_min

SCHEMA_VERSION = 1
KIND = "sigma_calibration"
KEY = "sigma_min_threshold"
CONFIGS = ("verifier", "judge")
PERCENTILE = 1.0
PERCENTILE_METHOD = "linear"  # numpy's default (Hyndman and Fan type 7)
STEP = Decimal("0.001")
REPORTED_PERCENTILES = (1, 5, 50)
PACKAGES = ("numpy", "pybullet", "ManipulaPy", "pydantic")
SEED_MODULUS = 2 ** 32  # witness_attempt passes seed % 2**32 to the motion layer; the rebuild must use the same seed
RULE = {
    "sample": "every accepted witness of the dev seed set's feasible slots, filled as `vplan eval prepare --seeds dev` "
              "fills them (D51 mix, slot order and candidate seeds), with both singularity thresholds at 0",
    "calibration_overrides": {f"{name}.{KEY}": 0.0 for name in CONFIGS},
    "verifier_minimum": "min over every trajectory sample of sigma_min(J_s) (verify.checks.sigma_min, the V-4 "
                        "quantity), on the trajectory rebuilt from the plan and seed, which reproduces the executed "
                        "approval token",
    "judge_minimum": "min of ExecutionRecord.sigma_min_steps (the V3 quantity, every physics step)",
    "statistic": f"{PERCENTILE:g}st percentile of the per-witness minima, "
                 f"numpy.percentile(values, {PERCENTILE}, method='{PERCENTILE_METHOD}')",
    "rounding": f"down to {STEP} (decimal floor of the shortest float repr)",
    "verifier": f"sigma_thr = floor_{STEP}(P1(verifier minima))",
    "judge": f"min(sigma_thr, floor_{STEP}(P1(judge minima)))",
}


class CalibrationError(EvalError):
    """Measuring an accepted witness failed. Never turned into an exclusion: that would bias the sample."""


class Minima(NamedTuple):
    verifier: float
    verifier_segment: int
    verifier_stage: str
    verifier_sample: int
    judge: float
    judge_step: int


# (task, target_id, seed, config snapshot) -> the witness evaluation, and its minima when the witness is accepted
Measure = Callable[[TaskSpec, str, int, dict[str, dict]], tuple[WitnessEvaluation, Minima | None]]


# ---- the rule -------------------------------------------------------------------------------------------
def floor_to_step(value: float) -> float:
    """value rounded down to 0.001. A decimal floor of the shortest repr, so 0.035 stays 0.035, and the result never
    exceeds value (float rounding is monotone)."""
    x = float(value)
    if not math.isfinite(x) or x < 0:
        raise ValueError(f"a singular value is finite and non-negative, got {x!r}")
    return float(Decimal(repr(x)).quantize(STEP, rounding=ROUND_FLOOR))


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q, method=PERCENTILE_METHOD))


def calibrated_thresholds(verifier_minima: Sequence[float], judge_minima: Sequence[float]) -> dict[str, float]:
    if not verifier_minima or len(verifier_minima) != len(judge_minima):
        raise ValueError("need one Verifier and one Judge minimum per witness, and at least one witness")
    sigma_thr = floor_to_step(percentile(verifier_minima, PERCENTILE))
    return {"verifier": sigma_thr, "judge": min(sigma_thr, floor_to_step(percentile(judge_minima, PERCENTILE)))}


def zero_sigma_snapshot(cfgs: dict[str, dict]) -> dict[str, dict]:
    """A copy of cfgs with both singularity checks disabled (sigma_min >= 0 always holds)."""
    out = deepcopy(cfgs)
    for name in CONFIGS:
        out[name][KEY] = 0.0
    return out


def mix_is_default(split: str, feasible: int | None, impossible: int | None) -> bool:
    return (feasible, impossible) in ((None, None), DEFAULT_MIX[split])


# ---- measuring one witness ------------------------------------------------------------------------------
def witness_minima(task: TaskSpec, attempt: AttemptResult, *, seed: int, config_snapshot: dict[str, dict]) -> Minima:
    """Both minima of an executed witness attempt, run with seed and config_snapshot."""
    record = attempt.record
    if attempt.outcome != "executed" or record is None or not record.sigma_min_steps:
        raise CalibrationError(f"seed {seed}: only an executed witness with sigma evidence can be measured")
    plan, scene = attempt.plan, task.scene
    # As pipeline.run_attempt builds it: same plan, scene, start, seed and verifier config.
    traj = build_trajectory(plan, scene, HOME_Q, seed=seed % SEED_MODULUS, cfg=config_snapshot["verifier"])
    if approval_token(traj, scene, plan.target_id, config_snapshot) != record.approval:
        raise CalibrationError(f"seed {seed}: the rebuilt trajectory does not reproduce the executed approval token")
    panda, L = load_panda(), config_snapshot["verifier"]["jacobian_char_length_m"]
    best: tuple[float, int, str, int] | None = None
    for i, seg in enumerate(traj.segments):
        for k, q in enumerate(seg.q):
            value = sigma_min(q, panda, L)
            if best is None or value < best[0]:  # strict: the first occurrence, where V-4 would fire first
                best = (value, i, seg.label, k)
    assert best is not None  # a CandidateTrajectory has at least six segments of at least two samples
    steps = record.sigma_min_steps
    step = min(range(len(steps)), key=steps.__getitem__)
    return Minima(*best, judge=steps[step], judge_step=step)


def measure_witness(task: TaskSpec, target_id: str, seed: int,
                    cfgs: dict[str, dict]) -> tuple[WitnessEvaluation, Minima | None]:
    """The generator's witness under cfgs (generate.run_witness), plus its minima when it is accepted."""
    attempt = witness_attempt(task, target_id, seed=seed, config_snapshot=cfgs)
    evaluation = witness_evaluation(attempt)
    if witness_rejection(evaluation, target_id) is not None:
        return evaluation, None
    try:
        return evaluation, witness_minima(task, attempt, seed=seed, config_snapshot=cfgs)
    except CalibrationError:
        raise
    except Exception as exc:  # anything else from the rebuild is still a measurement failure, never an exclusion
        raise CalibrationError(f"seed {seed}: measuring the witness failed: {type(exc).__name__}: {exc}") from exc


# ---- the sample ---------------------------------------------------------------------------------------
def _exclusion(seed: int, split: str, index: int, slot: Slot, reason: str | None,
               witness: WitnessEvaluation | None) -> dict:
    row = SeedSetExclusion(seed=seed, split=split, slot=index, kind=slot.kind, distractors=slot.distractors,
                           obstacles=slot.obstacles, impossible_type=slot.impossible_type,
                           reason=reason or "unknown").model_dump(mode="json", exclude={"witness"})
    row["checks"] = [f.check for f in witness.findings] if witness is not None else []
    row["failed_predicates"] = (sorted(k for k, ok in witness.result.predicates.items() if not ok)
                                if witness is not None and witness.result is not None else [])
    return row


def _statistics(values: list[float]) -> dict:
    return {"n": len(values), "min": min(values),
            **{f"p{q}": percentile(values, q) for q in REPORTED_PERCENTILES}, "max": max(values)}


def _rejected(thresholds: dict[str, float], verifier: list[float], judge: list[float]) -> dict[str, int]:
    """Witnesses a threshold pair would reject: V-4 fires below the Verifier threshold, V3 fails below the Judge's."""
    return {"verifier": sum(v < thresholds["verifier"] for v in verifier),
            "judge": sum(j < thresholds["judge"] for j in judge),
            "either": sum(v < thresholds["verifier"] or j < thresholds["judge"] for v, j in zip(verifier, judge))}


def _manipulapy_source_digest() -> str | None:
    """A digest of the editable ManipulaPy sources (the Verifier's Jacobian). Its location is never recorded (D56)."""
    spec = importlib.util.find_spec("ManipulaPy")
    if spec is None or spec.origin is None:
        return None
    root = Path(spec.origin).parent
    return source_digest({p.relative_to(root).as_posix(): _sha256(p) for p in sorted(root.rglob("*.py"))})


def _provenance(hashes: dict[str, str], disk: dict[str, dict], used: dict[str, dict]) -> dict:
    packages: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"created_utc": _utc_now(), "git": _git(), "python": platform.python_version(),
            "os": platform.platform(), "packages": packages,
            "manipulapy_source_digest": _manipulapy_source_digest(),
            "config_hashes": {key: config_hash(value) for key, value in disk.items()},
            "calibration_config_hashes": {key: config_hash(value) for key, value in used.items()},
            "source_hashes": hashes, "source_digest": source_digest(hashes)}


def calibrate(split: str = "dev", *, feasible: int | None = None, impossible: int | None = None,
              max_candidates: int | None = None, measure: Measure | None = None,
              progress: Callable[[int, int, int], None] | None = None) -> dict:
    """Generate the unbiased sample and return the evidence (applied: None). No model calls, no files.

    feasible/impossible override the D51 mix (trial runs); measure replaces measure_witness in offline tests."""
    if split != "dev":
        raise EvalError("sigma_thr is calibrated on the dev seed set only (spec §9); test stays held out")
    mix = mix_for(split, feasible, impossible)
    n = mix.n
    max_candidates = 2 * n + 20 if max_candidates is None else max_candidates
    if not isinstance(max_candidates, int) or isinstance(max_candidates, bool) or max_candidates < n:
        raise EvalError(f"max candidates must be an integer >= the {n} slots")
    measure = measure_witness if measure is None else measure
    hashes, disk = source_hashes(), config_snapshot()
    cfgs = zero_sigma_snapshot(disk)
    base = SEED_BASE[split]
    sample: list[SigmaWitness] = []
    exclusions: list[dict] = []
    accepted = {"feasible": 0, "impossible": 0}
    attempted = 0
    for index, slot in enumerate(slot_order(mix)):
        while True:
            if attempted >= max_candidates:
                raise EvalError(f"filled {sum(accepted.values())}/{n} slots after {attempted} candidates "
                                f"(limit {max_candidates})")
            seed = base + attempted
            attempted += 1
            measured: list[Minima | None] = []
            failures: list[CalibrationError] = []

            def evaluate(task: TaskSpec, target_id: str, seed: int = seed) -> WitnessEvaluation:
                try:
                    evaluation, minima = measure(task, target_id, seed, cfgs)
                except CalibrationError as exc:  # _candidate would record it as an exclusion; re-raised below
                    failures.append(exc)
                    raise
                measured.append(minima)
                return evaluation

            pair, reason, witness = _candidate(seed, slot, cfgs, evaluate)
            if failures:
                raise failures[0]
            if pair is None:
                exclusions.append(_exclusion(seed, split, index, slot, reason, witness))
            else:
                accepted[slot.kind] += 1
                if slot.kind == "feasible":
                    minima = measured[-1] if measured else None
                    if minima is None:
                        raise CalibrationError(f"seed {seed}: an accepted witness came back without its minima")
                    spec = pair[0]
                    sample.append(SigmaWitness(
                        task_id=f"{split}_{seed:08d}", index=index, seed=seed,
                        distractors=len(spec.scene.objects) - 1, obstacles=len(spec.scene.obstacles),
                        verifier_sigma_min=minima.verifier, verifier_segment=minima.verifier_segment,
                        verifier_stage=minima.verifier_stage, verifier_sample=minima.verifier_sample,
                        judge_sigma_min=minima.judge, judge_step=minima.judge_step))
            if progress is not None:
                progress(attempted, sum(accepted.values()), n)
            if pair is not None:
                break
    if source_hashes() != hashes or config_snapshot() != disk:
        raise EvalError("source or config files changed during calibration; discard and calibrate again")
    if not sample:
        raise EvalError("the mix has no feasible slot, so there is no witness to calibrate on")
    verifier = [w.verifier_sigma_min for w in sample]
    judge = [w.judge_sigma_min for w in sample]
    previous = {name: disk[name][KEY] for name in CONFIGS}
    calibrated = calibrated_thresholds(verifier, judge)
    return {
        "schema_version": SCHEMA_VERSION, "kind": KIND, "split": split, "rule": deepcopy(RULE),
        "mix": {"feasible": sum(mix.feasible.values()), "impossible": sum(mix.impossible.values()),
                "d51_default": mix_is_default(split, feasible, impossible)},
        "seeds": {"first": base, "next": base + attempted, "attempted": attempted,
                  "accepted_feasible": accepted["feasible"], "accepted_impossible": accepted["impossible"],
                  "excluded": len(exclusions)},
        "statistics": {"verifier": _statistics(verifier), "judge": _statistics(judge)},
        "thresholds": {"previous": previous, "calibrated": calibrated,
                       "rejected": {"previous": _rejected(previous, verifier, judge),
                                    "calibrated": _rejected(calibrated, verifier, judge)}},
        "applied": None,
        "sample": [w.model_dump(mode="json") for w in sample],
        "exclusions": exclusions,
        "provenance": _provenance(hashes, disk, cfgs)}


# ---- --apply and the evidence file ----------------------------------------------------------------------
def config_bytes(cfg: dict) -> bytes:
    """A config file as committed: two-space indent, LF line endings, a trailing newline."""
    return (json.dumps(cfg, indent=2) + "\n").encode("utf-8")


def apply_thresholds(thresholds: dict[str, float], config_dir: Path | None = None) -> dict[str, str]:
    """Write sigma_min_threshold into config/verifier.json and config/judge.json, changing nothing else; returns the
    new config hashes. Both files are checked before either is written."""
    config_dir = CONFIG_DIR if config_dir is None else Path(config_dir)
    if set(thresholds) != set(CONFIGS):
        raise EvalError(f"need exactly the {' and '.join(CONFIGS)} thresholds")
    for name in CONFIGS:
        value = thresholds[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise EvalError(f"refusing {name} {KEY} = {value!r}: it must be a positive number (0 disables the check)")
    if thresholds["judge"] > thresholds["verifier"]:
        raise EvalError("the Judge threshold must not exceed sigma_thr, or the Judge fails witnesses the Verifier "
                        "admits")
    updated = {}
    for name in CONFIGS:
        cfg = json.loads((config_dir / f"{name}.json").read_text(encoding="utf-8"))
        if KEY not in cfg:
            raise EvalError(f"config/{name}.json has no {KEY}")
        cfg[KEY] = float(thresholds[name])
        updated[name] = cfg
    for name, cfg in updated.items():
        path = config_dir / f"{name}.json"
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(config_bytes(cfg))
        os.replace(tmp, path)
    return {name: config_hash(cfg) for name, cfg in updated.items()}


def run_calibration(output: Path, split: str = "dev", *, feasible: int | None = None, impossible: int | None = None,
                    max_candidates: int | None = None, apply: bool = False,
                    progress: Callable[[int, int, int], None] | None = None) -> dict:
    """`vplan eval calibrate`: calibrate, apply on request, then write the evidence JSON (LF, never overwritten)."""
    output = Path(output)
    if output.exists():
        raise EvalError(f"{output} exists; refusing to overwrite calibration evidence")
    if apply and (split not in DEFAULT_MIX or not mix_is_default(split, feasible, impossible)):
        raise EvalError("--apply needs the full D51 dev mix (75 feasible + 25 impossible): sigma_thr is defined over "
                        "the dev set's witnesses")
    evidence = calibrate(split, feasible=feasible, impossible=impossible, max_candidates=max_candidates,
                         progress=progress)
    if apply:
        evidence["applied"] = apply_thresholds(evidence["thresholds"]["calibrated"])
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    tmp.write_bytes((json.dumps(evidence, indent=2) + "\n").encode("utf-8"))
    os.replace(tmp, output)
    return evidence
