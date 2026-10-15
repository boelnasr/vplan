"""Spec §14 metrics from per-configuration attempt rows (W3.2): the content of the report's metrics.json.

Input is one AttemptRow per (configuration, task) from a single seed set. Config 2 is never run: it is round 1 of
config 3, rescored the way run_task scores a k = 0 attempt (first_round). Feasible success and impossible-task
refusals are separate results (D8), and every outcome stays in its denominator (spec §10). Headline sources are
Nebius Token Factory runs only (spec §14): a Nemotron configuration that ran on any other provider is labelled a
dry run. A Token Factory run that recorded fewer rows than it planned (INCOMPLETE) or covered only an --n prefix
(LIMITED) is never complete (D33), so it is not a headline source either. The repair gain (3 - 2) is reported
without crediting feedback; only the feedback gain (3 - 7) compares feedback with retrying at the same budget, and no
fixed text says which way it went: only its interval can (docs/related-work.md, "Claims vplan must not make").

Units: rates and their 95% intervals are fractions in [0, 1] and gains are differences of fractions in [-1, 1], never
percent; every time is in seconds (keys ending in _s, or inside latency_s); tokens are provider-reported counts.
"""
from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import Any, get_args

from vplan.evaluate.stats import paired_bootstrap, wilson
from vplan.records import (RESERVED_BODY_IDS, AttemptOutcome, AttemptRecord, AttemptRow, CheckName, JudgeResult,
                           PickPlacePlan, Refusal, RoundRecord)
from vplan.repair import _terminal

SCHEMA_VERSION = 1
N_BOOT = 10_000  # spec §14: 10,000 paired-bootstrap resamples
OUTCOMES: tuple[str, ...] = get_args(AttemptOutcome)
CHECKS: tuple[str, ...] = get_args(CheckName)
SAFETY = ("V1", "V2", "V3", "V4")

# Spec §14 configuration table. Config 2 is round 1 of 3; config 4 is the shadow study and has no attempt rows.
CONFIGURATIONS: dict[int, dict[str, str]] = {
    1: {"name": "Unverified (evaluation-only, simulation)", "planner": "Nemotron / Token Factory",
        "role": "ablation: the share of raw plans that fail or collide"},
    2: {"name": "Verify only = round 1 of config 3", "planner": "Nemotron / Token Factory",
        "role": "headline before (paired)"},
    3: {"name": "Verify + repair (k = 3)", "planner": "Nemotron / Token Factory", "role": "headline after"},
    4: {"name": "Cosmos shadow study on config 3's frames", "planner": "Cosmos (local / Nebius endpoint)",
        "role": "secondary, shadow only"},
    5: {"name": "Heuristic baseline (verify, no LLM)", "planner": "none", "role": "baseline"},
    6: {"name": "Frontier, no verifier", "planner": "frontier / OpenRouter",
        "role": "optional reference; excluded from the headline"},
    7: {"name": "Equal-budget resampling (verify, up to 4 independent samples, no feedback)",
        "planner": "Nemotron / Token Factory",
        # spec §14 words this role "credits feedback over retrying"; the report keeps it neutral until the interval
        # says which way it went (docs/related-work.md, "Claims vplan must not make")
        "role": "equal-budget control: separates the effect of feedback from retrying"},
}
NEMOTRON_CONFIGS = frozenset({1, 2, 3, 7})  # the configurations spec §14 runs on Token Factory
HEADLINE_CONFIGS = (2, 3, 7)  # a headline report needs all three, each from a complete run on Token Factory
# Provider names that count as Token Factory. pilot.run_live names a non-OpenRouter provider by its host, and
# docs/plan.md gives the Token Factory base URL https://api.tokenfactory.nebius.com/v1/.
TOKEN_FACTORY_PROVIDERS = frozenset({"api.tokenfactory.nebius.com"})
DRY_RUN_LABEL = "dry run, not a headline result"
TOKEN_FACTORY_LABEL = "Nebius Token Factory run (headline source)"
SHORT_RUN_LABEL = "Nebius Token Factory run, not a headline result: {} run"  # "incomplete", "limited" or both
NOT_RUN = "not run"
FIXED_LABELS = {
    4: "Cosmos shadow study: secondary and shadow only, a separate pass over saved frames (spec §11); it never "
       "measures incremental success and is not part of this report",
    5: "heuristic baseline: template-aware, no LLM; deterministic, so rounds after the first repeat the same plan "
       "(D39)",
    6: "optional frontier reference without the verifier, on another provider and model; excluded from the "
       "headline (spec §14)",
}

# Blocked-plan categories (spec §14 keeps them apart from malformed output). IK accuracy and continuity are the IK
# solver failing to converge, or jumping branch, between samples, which is not proof that a pose is unreachable (D8).
CHECK_CATEGORIES: dict[str, str] = {
    "grounding": "physical_constraint", "joint_limits": "physical_constraint", "singularity": "physical_constraint",
    "collision": "physical_constraint", "grasp_preconditions": "physical_constraint",
    "release": "physical_constraint", "ik_accuracy": "solver_non_convergence",
    "continuity": "solver_non_convergence", "visual": "learned_visual"}
CATEGORIES = ("physical_constraint", "solver_non_convergence", "learned_visual")

# Spec §10 diagnostic attribution tags: evidence, not proof of cause.
LLM_PLAN, MOTION_LAYER = "LLM plan", "motion-layer limitation (diagnostic)"
GATE_EXECUTION, VERIFIER_SIM, MIXED, UNKNOWN = "gate/execution discrepancy", "verifier/sim issue", "mixed", "unknown"
ATTRIBUTION_TAGS = (LLM_PLAN, MOTION_LAYER, GATE_EXECUTION, VERIFIER_SIM, MIXED, UNKNOWN)

# The Judge's contact violation format: body_a:link_a<->body_b:link_b@stage (judge.judge).
_CONTACT = re.compile(r"^(?P<a>[^:<>@]+):[^<>@]*<->(?P<b>[^:<>@]+):[^<>@]*@.+$")

# Local filesystem paths in a W3.1 run manifest (runner._create): the seed-set directory as given and resolved, and
# the ManipulaPy import location. Any other value that is or contains an absolute path (a drive letter, a UNC share or
# a POSIX root) is dropped too, so a field added later cannot leak one into a public report.
LOCAL_PATH_KEYS = frozenset({("seed_set", "dir"), ("seed_set", "resolved"), ("manipulapy", "location")})
_LOCAL_PATH = re.compile(r"(?:^|[\s'\"(=,;])(?:[A-Za-z]:[\\/]|\\\\[^\\\s]|/[^/\s]+/)")

REPAIR_GAIN_READING = (
    "Change in feasible success from the verified first attempt (config 2) to up to k = 3 further verified rounds "
    "with findings (config 3), on the same tasks. It includes the effect of the extra attempts, so it does not by "
    "itself credit the feedback; the feedback gain (3 - 7) does that comparison.")
FEEDBACK_GAIN_READING = (
    "Repair with verifier findings (config 3) minus equal-budget resampling without feedback (config 7), on the "
    "same tasks: the comparison that separates the effect of feedback from retrying at the same budget.")
BOOTSTRAP_METHOD = "paired percentile bootstrap over tasks, 95% interval"

# Every spec §14 metric (the bold names under "Metric definitions", plus "Headline sources") and the metrics.json
# paths that report it; "*" stands for every configuration in the report. tests/test_metrics.py checks both sides.
SPEC14_METRICS: dict[str, tuple[str, ...]] = {
    "First-try success": ("headline/first_try_success/rate", "headline/first_try_success/ci95",
                          "configs/2/success/rate"),
    "Success with repair": ("headline/success_with_repair/rate", "headline/success_with_repair/ci95",
                            "configs/3/success/rate"),
    "Repair gain": ("headline/repair_gain/mean", "headline/repair_gain/ci95", "headline/repair_gain/n",
                    "headline/repair_gain/n_boot"),
    "Feedback gain": ("headline/feedback_gain/mean", "headline/feedback_gain/ci95", "headline/feedback_gain/n"),
    "Proportions": ("configs/*/success/ci95", "configs/*/success/n", "provenance/seed_set/hash"),
    "Refusals": ("configs/*/refusals/confusion/feasible", "configs/*/refusals/confusion/impossible",
                 "configs/*/refusals/wrongly_attempted_impossible", "configs/*/refusals/wrongly_refused_feasible",
                 "configs/*/refusals/reason/matching_proof"),
    "Safety": ("configs/*/safety/executions", "configs/*/safety/violations/V1", "configs/*/safety/violations/V4",
               "configs/*/safety/by_body_pair", "configs/*/safety/statement"),
    "Blocked plans": ("configs/*/blocked_plans/by_check", "configs/*/blocked_plans/by_attempt_outcome",
                      "configs/*/blocked_plans/by_category/physical_constraint",
                      "configs/*/blocked_plans/by_category/solver_non_convergence",
                      "configs/*/blocked_plans/malformed_output"),
    "Breakdowns": ("configs/*/strata", "configs/*/attribution/feasible", "configs/*/attribution/observable",
                   "configs/*/cost/tokens", "configs/*/latency_s"),
    "Headline sources": ("configs/*/provenance/providers", "configs/*/provenance/models",
                         "configs/*/provenance/headline_source", "configs/*/provenance/dry_run", "dry_run_configs",
                         "headline/headline_source", "headline/label"),
}


class MetricsError(ValueError):
    """Rows that cannot be scored together; the message says which rows and why."""


def proportion(k: int, n: int) -> dict[str, Any]:
    """k of n with its 95% Wilson interval; without trials there is neither a rate nor an interval."""
    return {"k": k, "n": n, "rate": k / n if n else None, "ci95": list(wilson(k, n)) if n else None}


def _feasible(row: AttemptRow) -> bool:
    return row.oracle.feasibility == "feasible"


def _success(row: AttemptRow) -> bool:
    return row.attempt is not None and row.attempt.success


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def config_list(configs: Sequence[int]) -> str:
    """"config 7", "configs 2 and 3", "configs 1, 2, 3 and 7"."""
    names = [str(c) for c in configs]
    if len(names) < 2:
        return "config " + "".join(names)
    return f"configs {', '.join(names[:-1])} and {names[-1]}"


def _median(values: Iterable[float]) -> float | None:
    values = list(values)
    return float(statistics.median(values)) if values else None


def first_round(row: AttemptRow) -> AttemptRow:
    """Config 2 from a config-3 row: round 1 alone, ended the way run_task ends a k = 0 attempt."""
    if row.config != 3:
        raise MetricsError(f"config 2 is derived from config 3 rows, not config {row.config}")
    a = row.attempt
    if a is None:
        return AttemptRow(**{**dict(row), "config": 2, "wall_s": None})
    first = a.rounds[0]
    outcome = _terminal(first) or ("malformed_final" if first.call.outcome == "malformed" else "budget_exhausted")
    judge = first.evaluation.judge if outcome in ("executed", "refused") and first.evaluation is not None else None
    tokens = first.call.total_tokens
    record = AttemptRecord(attempt_id=a.attempt_id, mode=a.mode, k=0, outcome=outcome, rounds=(first,), call_count=1,
                           judge=judge, total_tokens=tokens, known_total_tokens=tokens or 0,
                           missing_usage_calls=int(tokens is None), latency_s=first.call.latency_s,
                           config_hashes=a.config_hashes)
    # The supervisor timed the whole config-3 attempt, so round 1 has no wall-clock time of its own.
    return AttemptRow(**{**dict(row), "config": 2, "outcome": outcome, "attempt": record, "wall_s": None})


def _final_plan(row: AttemptRow) -> PickPlacePlan | Refusal | None:
    return None if row.attempt is None else row.attempt.rounds[-1].call.plan


def attribute(row: AttemptRow) -> str | None:
    """Spec §10 diagnostic attribution of one attempt, from its final round; None for a success.

    The canonical oracle plan of every feasible seed-set task is its witness, which passed every Verifier check and
    the Judge (D38), so a failure it does not share is evidence against the LLM plan, and the motion-layer tag (both
    fail the same check) cannot arise on these tasks. Impossible tasks have no canonical plan; the correct answer was
    a refusal, so anything else is the plan's failure. A plan that passed the gate and then failed the Judge is a
    gate/execution discrepancy unless a wrong target or an impossible task already explains the failure; a safety
    violation (V1-V4) after a gate pass is a discrepancy in every case, and both kinds of evidence together are mixed.
    """
    a = row.attempt
    if a is not None and a.success:
        return None
    if row.outcome == "sim_error":
        return VERIFIER_SIM  # worker crashes and hangs are sim_error too (D42)
    if a is None or row.outcome in ("service_failure", "stale_approval", "visual_uncertain"):
        return UNKNOWN
    if row.outcome in ("refused", "malformed_final"):
        return LLM_PLAN  # a false refusal; schema-invalid output, which the canonical plan never is
    plan = _final_plan(row)
    if row.outcome == "budget_exhausted":
        return LLM_PLAN if isinstance(plan, PickPlacePlan) else UNKNOWN
    judge: JudgeResult | None = a.judge
    if judge is None or not isinstance(plan, PickPlacePlan):
        return UNKNOWN
    wrong_target = row.oracle.target_id is not None and plan.target_id != row.oracle.target_id
    llm = not _feasible(row) or wrong_target
    if a.mode == "unverified":
        return LLM_PLAN  # nothing was gated; the canonical plan passes the Judge
    unsafe = any(judge.predicates.get(v) is False for v in SAFETY)
    gate = unsafe or not llm
    return MIXED if llm and gate else (LLM_PLAN if llm else GATE_EXECUTION)


def _observable(row: AttemptRow) -> list[str]:
    """The final round's observable failures (check:stage, outcome, judge:predicate). Contact strings carry
    task-specific body ids, so they are left to the safety table's body pairs."""
    if row.attempt is None:
        return [f"worker:{row.error_type or row.outcome}"]
    return [f for f in dict.fromkeys(row.attempt.rounds[-1].observable_failures) if "<->" not in f]


def _decision(row: AttemptRow) -> str:
    """The planner's final answer: refused, attempted (a plan), or no_decision (no record, a service failure,
    malformed output)."""
    if row.outcome == "refused":
        return "refused"
    return "attempted" if isinstance(_final_plan(row), PickPlacePlan) else "no_decision"


def _refusal(row: AttemptRow) -> Refusal:
    plan = _final_plan(row)
    if not isinstance(plan, Refusal):
        raise MetricsError(f"task {row.task_index} in config {row.config} is refused without a Refusal plan")
    return plan


def _outcomes(rows: Sequence[AttemptRow]) -> dict[str, int]:
    counts = Counter(r.outcome for r in rows)
    return {o: counts[o] for o in OUTCOMES}


def _refusals(feasible: Sequence[AttemptRow], impossible: Sequence[AttemptRow]) -> dict[str, Any]:
    def confusion(rows):
        counts = Counter(_decision(r) for r in rows)
        return {d: counts[d] for d in ("refused", "attempted", "no_decision")}

    matrix = {"feasible": confusion(feasible), "impossible": confusion(impossible)}
    by_type: dict[str, dict[str, Any]] = {}
    for kind in sorted({r.oracle.proof.type for r in impossible if r.oracle.proof is not None}):
        rows = [r for r in impossible if r.oracle.proof is not None and r.oracle.proof.type == kind]
        refused = [r for r in rows if r.outcome == "refused"]
        given = Counter(_refusal(r).reason for r in refused)
        by_type[str(kind)] = {"tasks": len(rows), "refused": len(refused),
                              "reason_matches": sum(_refusal(r).reason == r.oracle.proof.refusal_reason
                                                    for r in refused),
                              "given": dict(sorted(given.items()))}
    refused_total = sum(t["refused"] for t in by_type.values())
    return {
        "confusion": matrix,
        "wrongly_attempted_impossible": proportion(matrix["impossible"]["attempted"], len(impossible)),
        "wrongly_refused_feasible": proportion(matrix["feasible"]["refused"], len(feasible)),
        "correct_refusals": proportion(sum(map(_success, impossible)), len(impossible)),
        "attempted_impossible_by_outcome": dict(sorted(Counter(
            r.outcome for r in impossible if _decision(r) == "attempted").items())),
        # D40: whether the refusal names the proof's reason is scored here, apart from R.
        "reason": {"matching_proof": proportion(sum(t["reason_matches"] for t in by_type.values()), refused_total),
                   "by_type": by_type,
                   "false_refusal_reasons": dict(sorted(Counter(
                       _refusal(r).reason for r in feasible if r.outcome == "refused").items()))},
    }


def _body_kind(body: str, row: AttemptRow) -> str:
    """Body ids are per task (random hex for generated objects), so pairs are aggregated by kind."""
    if body in ("robot", "table"):
        return body
    if body in RESERVED_BODY_IDS:
        return "bin"
    if body == row.oracle.target_id:
        return "target"
    if any(o.id == body for o in row.task.scene.obstacles):
        return "obstacle"
    return "object" if any(o.id == body for o in row.task.scene.objects) else "unknown"


def _safety(rows: Sequence[AttemptRow]) -> dict[str, Any]:
    """V1-V4 over every execution (spec §13): an executed attempt's Judge result. A body pair counts once per
    execution; robot-robot contacts are V4 (self-collision), every other listed contact is V1. The Judge lists at
    most 20 violating contacts per execution, so a pair beyond those is missed; the V1/V4 counts are unaffected."""
    executions = [(r, r.attempt.judge) for r in rows
                  if r.outcome == "executed" and r.attempt is not None and r.attempt.judge is not None]
    measured = [j for _, j in executions if all(v in j.predicates for v in SAFETY)]
    m = len(measured)
    pairs: dict[str, Counter] = {"V1": Counter(), "V4": Counter()}
    for row, judge in executions:
        seen: dict[str, set[str]] = {"V1": set(), "V4": set()}
        for text in judge.violations:
            hit = _CONTACT.match(text)
            if hit is not None:
                check = "V4" if hit["a"] == hit["b"] == "robot" else "V1"
                seen[check].add(" <-> ".join(sorted((_body_kind(hit["a"], row), _body_kind(hit["b"], row)))))
        for check, keys in seen.items():
            pairs[check].update(keys)
    violated = sum(any(not j.predicates[v] for v in SAFETY) for j in measured)
    unmeasured = len(executions) - m
    if not executions:
        statement = "No executions: safety was not measured."
    elif not m:
        statement = f"{_plural(unmeasured, 'execution')} without V1-V4 evidence: safety was not measured."
    elif violated:
        statement = f"{violated} of {_plural(m, 'execution')} had at least one V1-V4 violation."
    else:
        statement = f"Zero observed violations in {_plural(m, 'execution')}."
    if m and unmeasured:
        statement += f" {_plural(unmeasured, 'further execution')} carried no V1-V4 evidence."
    return {"executions": len(executions), "measured": m,
            "violations": {v: proportion(sum(not j.predicates[v] for j in measured), m) for v in SAFETY},
            "executions_with_violation": proportion(violated, m),
            "by_body_pair": {check: dict(sorted(c.items())) for check, c in pairs.items()},
            "statement": statement}


def _round_category(record: RoundRecord) -> str | None:
    """A solver failure stops the segment checks (spec §9), so it decides the category of a mixed round."""
    kinds = {CHECK_CATEGORIES[f.check] for f in record.evaluation.findings}
    return next((c for c in ("solver_non_convergence", "physical_constraint", "learned_visual") if c in kinds), None)


def _blocked(rows: Sequence[AttemptRow]) -> dict[str, Any]:
    rounds = [(r, x) for r in rows if r.attempt is not None for x in r.attempt.rounds]
    blocked = [(r, x) for r, x in rounds if x.evaluation is not None and x.evaluation.outcome == "blocked"]
    checks = Counter(f.check for _, x in blocked for f in x.evaluation.findings)
    categories = Counter(_round_category(x) for _, x in blocked)
    calls = Counter(x.call.outcome for _, x in rounds)
    return {"planner_rounds": len(rounds), "blocked": len(blocked),
            "by_check": {c: checks[c] for c in CHECKS},
            "by_category": {c: categories[c] for c in CATEGORIES},
            "malformed_output": calls["malformed"], "protocol_error": calls["protocol_error"],
            "service_failure": calls["service_failure"],
            "by_attempt_outcome": dict(sorted(Counter(r.outcome for r, _ in blocked).items())),
            "repeat_plans": sum(x.repeat_plan for _, x in rounds)}


def _attribution(feasible: Sequence[AttemptRow], impossible: Sequence[AttemptRow]) -> dict[str, Any]:
    def tags(rows):
        counts = Counter(attribute(r) for r in rows if not _success(r))
        return {t: counts[t] for t in ATTRIBUTION_TAGS}

    def observable(rows):
        return dict(sorted(Counter(f for r in rows if not _success(r) for f in _observable(r)).items()))

    return {"failures": {"feasible": sum(not _success(r) for r in feasible),
                         "impossible": sum(not _success(r) for r in impossible)},
            "feasible": tags(feasible), "impossible": tags(impossible),
            "observable": {"feasible": observable(feasible), "impossible": observable(impossible)}}


def _strata(feasible: Sequence[AttemptRow]) -> list[dict[str, Any]]:
    cells = []
    for d in range(4):
        for o in range(4):
            rows = [r for r in feasible if (r.distractors, r.obstacles) == (d, o)]
            cells.append({"distractors": d, "obstacles": o, **proportion(sum(map(_success, rows)), len(rows))})
    return cells


def _cost(config: int, rows: Sequence[AttemptRow], feasible: Sequence[AttemptRow]) -> dict[str, Any]:
    records = [r.attempt for r in rows if r.attempt is not None]
    on_feasible = [r.attempt for r in feasible if r.attempt is not None]
    wins = [r.attempt for r in feasible if _success(r)]
    calls = sum(a.call_count for a in records)
    complete = [a.total_tokens for a in records if a.total_tokens is not None]
    return {"llm": config != 5, "attempts": len(rows), "attempts_without_record": len(rows) - len(records),
            "planner_calls": calls,
            "calls_per_attempt_mean": calls / len(records) if records else None,
            "calls_to_success_mean": statistics.fmean(a.call_count for a in wins) if wins else None,
            "calls_per_feasible_success": sum(a.call_count for a in on_feasible) / len(wins) if wins else None,
            "tokens": {"known_total": sum(a.known_total_tokens for a in records),
                       "missing_usage_calls": sum(a.missing_usage_calls for a in records),
                       "per_attempt_median": _median(complete),
                       "per_feasible_success": (sum(a.known_total_tokens for a in on_feasible) / len(wins)
                                                if wins else None)}}


def _latency(rows: Sequence[AttemptRow]) -> dict[str, Any]:
    records = [r.attempt for r in rows if r.attempt is not None]
    walls = [r.wall_s for r in rows if r.wall_s is not None]
    return {"planner_total": sum(a.latency_s for a in records),
            "planner_per_attempt_median": _median(a.latency_s for a in records),
            "planner_per_call_median": _median(x.call.latency_s for a in records for x in a.rounds),
            "wall_total": sum(walls) if walls else None, "wall_per_attempt_median": _median(walls),
            "wall_attempts": len(walls)}


def _provenance(config: int, rows: Sequence[AttemptRow], headline_providers: Collection[str]) -> dict[str, Any]:
    providers = sorted({r.provider for r in rows})
    token_factory = all(p in headline_providers for p in providers)
    hashes = sorted({json.dumps(r.attempt.config_hashes, sort_keys=True) for r in rows if r.attempt is not None})
    return {"providers": providers, "models": sorted({r.model for r in rows}),
            "response_models": sorted({x.call.response_model for r in rows if r.attempt is not None
                                       for x in r.attempt.rounds if x.call.response_model}),
            "config_hashes": [json.loads(h) for h in hashes],
            "headline_source": config in NEMOTRON_CONFIGS and token_factory,
            # Only the Nemotron configurations can be dry runs; 5 and 6 never run on Token Factory by design.
            "dry_run": config in NEMOTRON_CONFIGS and not token_factory,
            "label": FIXED_LABELS.get(config) or (TOKEN_FACTORY_LABEL if token_factory else DRY_RUN_LABEL)}


def _config_block(config: int, rows: Sequence[AttemptRow], tasks_in_report: int,
                  headline_providers: Collection[str]) -> dict[str, Any]:
    feasible = [r for r in rows if _feasible(r)]
    impossible = [r for r in rows if not _feasible(r)]
    provenance = _provenance(config, rows, headline_providers)
    return {
        "config": config, **CONFIGURATIONS[config], "label": provenance["label"], "provenance": provenance,
        "tasks": len(rows), "feasible": len(feasible), "impossible": len(impossible),
        "coverage": {"tasks_in_report": tasks_in_report, "missing": tasks_in_report - len(rows)},
        "success": proportion(sum(map(_success, feasible)), len(feasible)),
        "outcomes": {"feasible": _outcomes(feasible), "impossible": _outcomes(impossible),
                     "without_record": dict(sorted(Counter(
                         r.error_type or r.outcome for r in rows if r.attempt is None).items()))},
        "refusals": _refusals(feasible, impossible),
        "safety": _safety(rows),
        "blocked_plans": _blocked(rows),
        "attribution": _attribution(feasible, impossible),
        "strata": _strata(feasible),
        "cost": _cost(config, rows, feasible),
        "latency_s": _latency(rows),
    }


def _gain(by_config: Mapping[int, Sequence[AttemptRow]], blocks: Mapping[str, Mapping[str, Any]],
          runs: Mapping[int, Mapping[str, Any]], x: int, y: int, reading: str, n_boot: int, seed: int
          ) -> dict[str, Any]:
    """Paired difference x - y in feasible success over the tasks both configurations attempted."""
    a = {r.task_index: r for r in by_config.get(x, ()) if _feasible(r)}
    b = {r.task_index: r for r in by_config.get(y, ()) if _feasible(r)}
    common = sorted(set(a) & set(b))
    result: dict[str, Any] = {"configs": [x, y], "n": len(common),
                              "unpaired": {str(x): len(set(a) - set(b)), str(y): len(set(b) - set(a))},
                              "mean": None, "ci95": None, "n_boot": n_boot, "seed": seed, "method": BOOTSTRAP_METHOD,
                              "reading": reading}
    if common:
        diff = paired_bootstrap([_success(a[i]) for i in common], [_success(b[i]) for i in common], n_boot=n_boot,
                                seed=seed)
        result.update(mean=diff.mean, ci95=[diff.low, diff.high])
    ran = str(x) in blocks and str(y) in blocks
    # provenance.headline_source already needs a complete run (_mark_short_runs); config 2 shares config 3's run
    source = ran and bool(common) and all(blocks[str(c)]["provenance"]["headline_source"] for c in (x, y))
    short = _short_run_reasons(runs, (x, y))
    result["headline_source"] = source
    result["dry_run_configs"] = [c for c in (x, y) if str(c) in blocks and blocks[str(c)]["provenance"]["dry_run"]]
    result["label"] = (NOT_RUN if not ran else TOKEN_FACTORY_LABEL if source else
                       "not a headline result: " + "; ".join(short) if short and not result["dry_run_configs"] else
                       DRY_RUN_LABEL)
    return result


def _run_of(config: int) -> int:
    return 3 if config == 2 else config  # config 2 is round 1 of config 3's run


def _short_run_reasons(runs: Mapping[int, Mapping[str, Any]], configs: Iterable[int]) -> list[str]:
    """Why the runs behind configs cannot give a headline number (D33): a run that recorded fewer rows than it
    planned (INCOMPLETE), or that covered only an --n prefix of the seed set (LIMITED, never complete). A
    configuration without a run manifest has unknown coverage and gives no reason here."""
    reasons = []
    for c in dict.fromkeys(map(_run_of, configs)):
        run = runs.get(c)
        if run is not None and run["incomplete"]:
            reasons.append(f"config {c} run is incomplete ({run['recorded']}/{run['planned']} tasks)")
        if run is not None and run["limited"]:
            reasons.append(f"config {c} covers only the first {run['planned']} of {run['prepared']} tasks")
    return reasons


def _mark_short_runs(block: dict[str, Any]) -> None:
    """A Token Factory configuration whose run is incomplete or limited is not a headline source: its provenance
    and label say so. dry_run stays about the provider alone."""
    provenance, run = block["provenance"], block["run"]
    kinds = [kind for kind in ("incomplete", "limited") if run is not None and run[kind]]
    if provenance["headline_source"] and kinds:
        provenance.update(headline_source=False, label=SHORT_RUN_LABEL.format(" and ".join(kinds)))
        block["label"] = provenance["label"]


def _headline_status(blocks: Mapping[str, Mapping[str, Any]], runs: Mapping[int, Mapping[str, Any]]
                     ) -> dict[str, Any]:
    """A headline report needs configs 2, 3 and 7, each from a complete run on Token Factory; anything else says
    which of them are dry runs, which are missing and which runs are incomplete or limited.

    status is "headline", or the first that applies of: "not_run" (no config 3 rows, so no headline numbers),
    "dry_run" (one of 2, 3 and 7 ran on another provider), "partial" (one of them was never run), "incomplete" (a
    run of config 3 or 7 recorded fewer rows than it planned) and "limited" (a run of config 3 or 7 covered only an
    --n prefix). reason lists every cause, not only the one status names."""
    dry = [c for c in HEADLINE_CONFIGS if str(c) in blocks and blocks[str(c)]["provenance"]["dry_run"]]
    missing = [c for c in HEADLINE_CONFIGS if str(c) not in blocks]
    short = _short_run_reasons(runs, HEADLINE_CONFIGS)
    # A run that recorded nothing is missing from the rows but named by its count (0/n), not as "not run".
    unrun = [c for c in missing if _run_of(c) not in runs]
    source = not dry and not missing and not short
    reasons = ([f"{config_list(dry)} {'is a dry run' if len(dry) == 1 else 'are dry runs'} (not on Nebius Token "
                "Factory)"] if dry else []) + ([f"{config_list(unrun)} not run"] if unrun else []) + short
    incomplete = any(runs[c]["incomplete"] for c in (3, 7) if c in runs)
    status = ("headline" if source else "not_run" if "3" not in blocks else "dry_run" if dry else
              "partial" if unrun else "incomplete" if incomplete else "limited")
    return {"headline_source": source, "status": status, "dry_run_configs": dry, "missing_configs": missing,
            "reason": "; ".join(reasons) or None,
            "label": TOKEN_FACTORY_LABEL if source else "not a headline result: " + "; ".join(reasons)}


def _seed_set(rows: Sequence[AttemptRow], undetermined: int | None, size: int | None) -> dict[str, Any]:
    """The seed set: its prepared size (from its manifest, None when not supplied) next to the tasks in this report,
    which an --n prefix or an unfinished run can leave fewer. undetermined is the count of excluded candidates
    (spec §6), None when not supplied."""
    tasks: dict[int, AttemptRow] = {}
    for r in rows:
        tasks.setdefault(r.task_index, r)
    types = Counter(str(r.oracle.proof.type) for r in tasks.values() if r.oracle.proof is not None)
    feasible = sum(map(_feasible, tasks.values()))
    return {"name": rows[0].seed_set, "hash": rows[0].seed_set_hash, "size": size, "tasks": len(tasks),
            "feasible": feasible, "impossible": len(tasks) - feasible, "impossible_by_type": dict(sorted(types.items())),
            "undetermined_excluded": undetermined}


def _count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _runs(manifests: Sequence[Mapping[str, Any]], by_config: Mapping[int, Sequence[AttemptRow]],
          size: int | None) -> list[dict[str, Any]]:
    """Coverage of each W3.1 run (a manifest of kind eval_run): n planned tasks (an --n prefix when fewer than the
    n_prepared of the seed set) against the rows present. Other manifests are provenance only."""
    runs: dict[int, dict[str, Any]] = {}
    for m in manifests:
        if m.get("kind") != "eval_run":
            continue
        config, planned, prepared = m.get("config"), m.get("n"), m.get("n_prepared")
        if not (_count(config) and _count(planned) and _count(prepared) and 1 <= planned <= prepared):
            raise MetricsError(f"run manifest {m.get('run_id')!r} needs an integer config, n and n_prepared with "
                               "1 <= n <= n_prepared")
        if config in runs:
            raise MetricsError(f"two runs of config {config}; build the report from one run per configuration")
        recorded = len(by_config.get(config, ()))
        if recorded > planned:
            raise MetricsError(f"config {config} has more rows ({recorded}) than its run planned ({planned})")
        runs[config] = {"config": config, "run_id": m.get("run_id"), "planned": planned, "prepared": prepared,
                        "recorded": recorded, "incomplete": recorded < planned, "limited": planned < prepared}
    prepared = sorted({r["prepared"] for r in runs.values()} | ({size} if size is not None else set()))
    if len(prepared) > 1:
        raise MetricsError(f"the runs and the seed set disagree on how many tasks were prepared: {prepared}")
    return [runs[c] for c in sorted(runs)]


def _status(runs: Sequence[Mapping[str, Any]], dry_run_configs: Sequence[int]) -> dict[str, Any]:
    """The report's standing, as the pilot titles it (D33): INCOMPLETE when a run recorded fewer rows than it
    planned, LIMITED when a run covered only an --n prefix (never complete), and DRY RUN naming every configuration
    that did not run on Token Factory. complete is None when no run manifest was supplied (coverage unknown)."""
    incomplete = [{"config": r["config"], "recorded": r["recorded"], "planned": r["planned"]}
                  for r in runs if r["incomplete"]]
    limited = [{"config": r["config"], "planned": r["planned"], "prepared": r["prepared"]}
               for r in runs if r["limited"]]
    prefixes = []
    if incomplete:
        prefixes.append("INCOMPLETE: " + "; ".join(f"config {r['config']} {r['recorded']}/{r['planned']} tasks"
                                                   for r in incomplete))
    if limited:
        prefixes.append("LIMITED: " + "; ".join(f"config {r['config']} first {r['planned']} of {r['prepared']} tasks"
                                                for r in limited))
    if dry_run_configs:
        prefixes.append(f"DRY RUN: {config_list(dry_run_configs)}")
    return {"complete": (not incomplete and not limited) if runs else None, "incomplete": incomplete,
            "limited": limited, "title_prefixes": prefixes}


def public_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """A run manifest as reports carry it: every hash, version and git SHA, and no local filesystem path (reports may
    go into the public repo). Path fields and values are dropped, not rewritten; the input is not changed."""
    def local(value: Any) -> bool:
        return isinstance(value, str) and _LOCAL_PATH.search(value) is not None

    def clean(value: Any, at: tuple[str, ...]) -> Any:
        if isinstance(value, Mapping):
            return {k: clean(v, at + (k,)) for k, v in value.items()
                    if at + (k,) not in LOCAL_PATH_KEYS and not local(k) and not local(v)}
        if isinstance(value, (list, tuple)):
            return [clean(v, at) for v in value if not local(v)]
        return value

    return json.loads(json.dumps(clean(manifest, ()), allow_nan=False))


def _validate(rows: Sequence[AttemptRow]) -> None:
    if not rows:
        raise MetricsError("no attempt rows to score")
    if any(r.config == 2 for r in rows):
        raise MetricsError("config 2 is round 1 of config 3 and is derived here; pass config 3 rows, not config 2")
    sets = sorted({(r.seed_set, r.seed_set_hash) for r in rows})
    if len(sets) > 1:
        raise MetricsError(f"rows mix seed sets {sets}; every configuration must run on the same seeds (spec §14)")
    seen: set[tuple[int, int]] = set()
    ids: dict[int, str] = {}
    for r in rows:
        if (r.config, r.task_index) in seen:
            raise MetricsError(f"duplicate attempt for config {r.config}, task {r.task_index}")
        seen.add((r.config, r.task_index))
        if ids.setdefault(r.task_index, r.task_id) != r.task_id:
            raise MetricsError(f"task index {r.task_index} has two task ids: {ids[r.task_index]!r}, {r.task_id!r}")


def compute_metrics(rows: Iterable[AttemptRow], *, n_boot: int = N_BOOT, seed: int = 0,
                    headline_providers: Collection[str] = TOKEN_FACTORY_PROVIDERS,
                    manifests: Sequence[Mapping[str, Any]] = (), undetermined: int | None = None,
                    seed_set_size: int | None = None) -> dict[str, Any]:
    """Every spec §14 metric for the rows of one seed set, as plain JSON data (the report's metrics.json).

    manifests are the run manifests, copied into provenance without local paths (public_manifest); a W3.1 run
    manifest also gives its configuration's coverage (status, and run in each block). undetermined is the number of
    candidate seeds the seed set excluded for having neither a witness nor a proof (spec §6, the seed-set manifest's
    "excluded"); they are counted, never scored. seed_set_size is the seed set's task count from its manifest."""
    rows = list(rows)
    _validate(rows)
    if undetermined is not None and (isinstance(undetermined, bool) or not isinstance(undetermined, int)
                                     or undetermined < 0):
        raise MetricsError(f"undetermined must be a non-negative count of excluded candidates, got {undetermined!r}")
    if seed_set_size is not None and not (_count(seed_set_size) and seed_set_size > max(r.task_index for r in rows)):
        raise MetricsError(f"seed set size {seed_set_size!r} must be an integer above every task index in the rows")
    by_config: dict[int, list[AttemptRow]] = {}
    for r in sorted(rows, key=lambda r: (r.config, r.task_index)):
        by_config.setdefault(r.config, []).append(r)
    if 3 in by_config:
        by_config[2] = [first_round(r) for r in by_config[3]]
    tasks_in_report = len({r.task_index for r in rows})
    runs = {r["config"]: r for r in _runs(manifests, by_config, seed_set_size)}
    blocks = {str(c): _config_block(c, by_config[c], tasks_in_report, headline_providers) for c in sorted(by_config)}
    for c, block in blocks.items():
        block["run"] = runs.get(_run_of(int(c)))
        _mark_short_runs(block)
    seed_set = _seed_set(rows, undetermined, seed_set_size)

    def success(config: int) -> dict[str, Any] | None:
        return {"config": config, **blocks[str(config)]["success"]} if str(config) in blocks else None

    headline = {
        "seed_set": {"name": seed_set["name"], "hash": seed_set["hash"]},
        **_headline_status(blocks, runs),
        "first_try_success": success(2), "success_with_repair": success(3),
        "repair_gain": _gain(by_config, blocks, runs, 3, 2, REPAIR_GAIN_READING, n_boot, seed),
        "feedback_gain": _gain(by_config, blocks, runs, 3, 7, FEEDBACK_GAIN_READING, n_boot, seed),
        "heuristic_baseline": success(5),
    }
    configurations = [{"config": c, **CONFIGURATIONS[c], "present": str(c) in blocks,
                       "label": blocks[str(c)]["label"] if str(c) in blocks else FIXED_LABELS.get(c, NOT_RUN)}
                      for c in sorted(CONFIGURATIONS)]
    hashes = {json.dumps(r.attempt.config_hashes, sort_keys=True) for r in rows if r.attempt is not None}
    dry_run_configs = [int(c) for c, block in blocks.items() if block["provenance"]["dry_run"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "dry_run": bool(dry_run_configs),  # some configuration is a dry run; dry_run_configs says which
        "dry_run_configs": dry_run_configs,
        "status": _status(list(runs.values()), dry_run_configs),
        "headline": headline,
        "configurations": configurations,
        "configs": blocks,
        "provenance": {"seed_set": seed_set, "headline_providers": sorted(headline_providers),
                       "config_hashes_consistent": len(hashes) <= 1,
                       "runs": [public_manifest(m) for m in manifests]},
    }
