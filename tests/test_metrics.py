"""W3.2 metrics: every spec §14 metric from per-config attempt rows, checked by hand on a seven-task scenario."""
import json
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from tests import eval_rows as ev
from vplan import records
from vplan.evaluate.metrics import (ATTRIBUTION_TAGS, CHECK_CATEGORIES, DRY_RUN_LABEL, SPEC14_METRICS,
                                    TOKEN_FACTORY_LABEL, MetricsError, attribute, compute_metrics, first_round)
from vplan.evaluate.stats import paired_bootstrap, wilson

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def rows():
    return ev.scenario_rows()


@pytest.fixture(scope="module")
def m(rows):
    return compute_metrics(rows)


def prop(k, n):
    return {"k": k, "n": n, "rate": k / n if n else None, "ci95": list(wilson(k, n)) if n else None}


def by_task(rows, config):
    return {r.task_index: r for r in rows if r.config == config}


# ---- spec §14 coverage ------------------------------------------------------------------------------

def _lookup(data, path):
    for key in path.split("/"):
        assert isinstance(data, dict) and key in data, f"missing {key!r} in {path}"
        data = data[key]
    return data


def test_metrics_json_keys_cover_every_spec_section_14_metric(m):
    spec = (ROOT / "docs" / "spec.md").read_text(encoding="utf-8")
    section = spec.split("## 14. Evaluation", 1)[1].split("\n## 15.", 1)[0]
    region = section.split("**Metric definitions", 1)[1].split("**Runner**", 1)[0]
    names = re.findall(r"^(?:- )?\*\*([^*]+?):\*\*", region, flags=re.M)
    assert len(names) == 10 and set(names) == set(SPEC14_METRICS)
    assert set(m["configs"]) == {"1", "2", "3", "5", "6", "7"}
    for paths in SPEC14_METRICS.values():
        assert paths
        for path in paths:
            for config in (m["configs"] if "/*/" in path else [None]):
                _lookup(m, path.replace("/*/", f"/{config}/"))


def test_metrics_are_plain_json_without_nan(m):
    assert json.loads(json.dumps(m, allow_nan=False)) == m


def test_every_verifier_check_has_one_blocked_plan_category():
    assert set(CHECK_CATEGORIES) == set(records.CheckName.__args__)
    assert set(CHECK_CATEGORIES.values()) == {"physical_constraint", "solver_non_convergence", "learned_visual"}
    assert {c for c, v in CHECK_CATEGORIES.items() if v == "solver_non_convergence"} == {"ik_accuracy", "continuity"}


# ---- headline: first try, repair, gains -------------------------------------------------------------

def test_first_try_success_is_round_one_of_config_3(m):
    assert m["headline"]["first_try_success"] == {"config": 2, **prop(1, 5)}
    assert m["configs"]["2"]["success"] == prop(1, 5)


def test_success_with_repair_keeps_every_outcome_in_the_denominator(m):
    # Tasks 0 and 1 succeed; the false refusal, the malformed attempt and the worker timeout stay in N.
    assert m["headline"]["success_with_repair"] == {"config": 3, **prop(2, 5)}
    assert m["configs"]["3"]["success"] == prop(2, 5)


def test_repair_and_feedback_gains_are_paired_bootstrap_differences(m):
    repair, feedback = m["headline"]["repair_gain"], m["headline"]["feedback_gain"]
    expected = paired_bootstrap([1, 1, 0, 0, 0], [1, 0, 0, 0, 0])  # feasible tasks 0, 1, 2, 3, 6
    assert repair["configs"] == [3, 2] and repair["n"] == 5 and repair["mean"] == pytest.approx(0.2)
    assert repair["ci95"] == [expected.low, expected.high]
    assert repair["n_boot"] == 10_000 and repair["seed"] == 0
    expected = paired_bootstrap([1, 1, 0, 0, 0], [1, 0, 1, 0, 0])
    assert feedback["configs"] == [3, 7] and feedback["n"] == 5 and feedback["mean"] == pytest.approx(0.0)
    assert feedback["ci95"] == [expected.low, expected.high]


# Fixed text must not claim what only a measured interval can show (docs/related-work.md, "Claims vplan must not
# make", 6 and 7): neither that feedback or repair beats resampling, nor that config 7 credits feedback.
CLAIMS = ("credits feedback", "credit feedback over", "feedback over retrying", "beats", "outperform", "better than",
          "superior to")


def strings(data):
    if isinstance(data, str):
        yield data
    elif isinstance(data, dict):
        for value in data.values():
            yield from strings(value)
    elif isinstance(data, list):
        for value in data:
            yield from strings(value)


def test_gain_wording_is_neutral_about_feedback_versus_resampling(m):
    repair, feedback = m["headline"]["repair_gain"]["reading"], m["headline"]["feedback_gain"]["reading"]
    assert "does not by itself credit the feedback" in repair
    assert repair.endswith("the feedback gain (3 - 7) addresses that question at equal budget.")
    assert feedback == ("Repair with verifier findings (config 3) minus equal-budget resampling without feedback "
                        "(config 7), on the same tasks: the comparison that separates the effect of feedback from "
                        "retrying at the same budget.")
    assert "separates the effect of feedback from retrying" in m["configs"]["7"]["role"]
    for text in strings(m):
        assert not any(claim in text.lower() for claim in CLAIMS), text


def test_gains_pair_only_tasks_present_in_both_configs(rows):
    kept = [r for r in rows if not (r.config == 7 and r.task_index in (0, 1))]
    feedback = compute_metrics(kept)["headline"]["feedback_gain"]
    assert feedback["n"] == 3 and feedback["unpaired"] == {"3": 2, "7": 0}
    assert feedback["mean"] == pytest.approx((0 - 1 + 0) / 3)  # tasks 2, 3, 6


def test_missing_configs_leave_the_headline_empty_but_well_formed(rows):
    only_heuristic = compute_metrics([r for r in rows if r.config == 5])
    headline = only_heuristic["headline"]
    assert headline["first_try_success"] is None and headline["success_with_repair"] is None
    assert headline["repair_gain"]["n"] == 0 and headline["repair_gain"]["mean"] is None
    assert headline["repair_gain"]["ci95"] is None
    assert headline["heuristic_baseline"] == {"config": 5, **prop(4, 5)}
    assert set(only_heuristic["configs"]) == {"5"}


def test_config_2_outcomes_replay_run_task_with_k_zero(rows):
    tasks = by_task(rows, 3)
    assert first_round(tasks[0]).outcome == "executed" and first_round(tasks[0]).attempt.success
    assert first_round(tasks[1]).outcome == "budget_exhausted"  # blocked first round
    assert first_round(tasks[2]).outcome == "refused"
    assert first_round(tasks[3]).outcome == "malformed_final"
    timed_out = first_round(tasks[6])
    assert timed_out.config == 2 and timed_out.attempt is None and timed_out.outcome == "sim_error"
    view = first_round(tasks[1])
    assert view.config == 2 and view.attempt.k == 0 and view.attempt.call_count == 1
    assert view.attempt.rounds == tasks[1].attempt.rounds[:1] and view.attempt.known_total_tokens == 100


# Config-3 attempts whose first round ends every way a round can; (oracle, scripted calls, scripted evaluations).
FIRST_ROUNDS = {
    "success": (ev.FEASIBLE, [ev.call(ev.GOOD)], [ev.judged()]),
    "judge failure": (ev.FEASIBLE, [ev.call(ev.GOOD)], [ev.judged("S1")]),
    "blocked, then repaired": (ev.FEASIBLE, [ev.call(ev.GOOD), ev.call(ev.FIXED)],
                               [ev.blocked(ev.COLLISION), ev.judged()]),
    "false refusal": (ev.FEASIBLE, [ev.call(ev.refusal("other"))], []),
    "malformed, then a plan": (ev.FEASIBLE, [ev.call(), ev.call(ev.GOOD)], [ev.judged()]),
    "protocol error, then a plan": (ev.FEASIBLE, [ev.call(outcome="protocol_error"), ev.call(ev.GOOD)],
                                    [ev.judged()]),
    "service failure": (ev.FEASIBLE, [ev.call(outcome="service_failure", tokens=None, status_code=503)], []),
    "stale approval": (ev.FEASIBLE, [ev.call(ev.GOOD)], [records.EvaluationResult(outcome="stale_approval")]),
    "simulation error": (ev.FEASIBLE, [ev.call(ev.GOOD)],
                         [records.EvaluationResult(outcome="sim_error", error_type="RuntimeError")]),
    "visual uncertain": (ev.FEASIBLE, [ev.call(ev.GOOD)], [records.EvaluationResult(outcome="visual_uncertain")]),
    "impossible, attempted": (ev.impossible(1), [ev.call(ev.GOOD)], [ev.judged()]),
    "impossible, refused": (ev.impossible(1), [ev.call(ev.refusal())], []),
}


@pytest.mark.parametrize("name", FIRST_ROUNDS)
def test_first_round_is_what_run_task_returns_with_k_zero_on_the_same_planner(name):
    # first_round reuses repair._terminal; this pins it to run_task itself so the two cannot drift apart silently.
    oracle, calls, evaluations = FIRST_ROUNDS[name]
    full = ev.scripted_row(3, 0, oracle, calls, evaluations)
    alone = ev.attempt(oracle, calls[:1], evaluations[:1], k=0, attempt_id=full.attempt.attempt_id)
    derived = first_round(full)
    assert derived.attempt == alone and derived.outcome == alone.outcome
    assert derived.attempt.success == alone.success


# ---- per-config tables ----------------------------------------------------------------------------

def test_success_per_config(m):
    expected = {"1": prop(1, 5), "2": prop(1, 5), "3": prop(2, 5), "5": prop(4, 5), "6": prop(1, 1), "7": prop(2, 5)}
    assert {c: block["success"] for c, block in m["configs"].items()} == expected


def test_outcomes_split_feasible_and_impossible(m):
    c3 = m["configs"]["3"]["outcomes"]
    assert set(c3["feasible"]) == set(records.AttemptOutcome.__args__)
    assert {k: v for k, v in c3["feasible"].items() if v} == {"executed": 2, "refused": 1, "malformed_final": 1,
                                                              "sim_error": 1}
    assert {k: v for k, v in c3["impossible"].items() if v} == {"refused": 1, "executed": 1}
    assert c3["without_record"] == {"worker_timeout": 1}


def test_refusal_confusion_matrix_and_wrong_decisions(m):
    r3 = m["configs"]["3"]["refusals"]
    assert r3["confusion"] == {"feasible": {"refused": 1, "attempted": 2, "no_decision": 2},
                               "impossible": {"refused": 1, "attempted": 1, "no_decision": 0}}
    assert r3["wrongly_refused_feasible"] == prop(1, 5)
    assert r3["wrongly_attempted_impossible"] == prop(1, 2)
    assert r3["correct_refusals"] == prop(1, 2)
    assert r3["attempted_impossible_by_outcome"] == {"executed": 1}
    r7 = m["configs"]["7"]["refusals"]
    assert r7["confusion"] == {"feasible": {"refused": 0, "attempted": 4, "no_decision": 1},
                               "impossible": {"refused": 1, "attempted": 1, "no_decision": 0}}
    assert r7["attempted_impossible_by_outcome"] == {"budget_exhausted": 1}


def test_refusal_reason_is_scored_against_the_proof(m):
    reason3, reason7 = m["configs"]["3"]["refusals"]["reason"], m["configs"]["7"]["refusals"]["reason"]
    assert reason3["matching_proof"] == prop(1, 1)
    assert reason3["by_type"] == {"1": {"tasks": 1, "refused": 1, "reason_matches": 1,
                                        "given": {"target_unreachable": 1}},
                                  "4": {"tasks": 1, "refused": 0, "reason_matches": 0, "given": {}}}
    assert reason3["false_refusal_reasons"] == {"other": 1}
    assert reason7["matching_proof"] == prop(0, 1)
    assert reason7["by_type"]["1"]["given"] == {"destination_unreachable": 1}
    assert m["configs"]["5"]["refusals"]["reason"]["matching_proof"] == prop(2, 2)


def test_safety_counts_v1_to_v4_per_execution_by_body_pair(m):
    s1, s3, s7 = (m["configs"][c]["safety"] for c in ("1", "3", "7"))
    assert s3["executions"] == 3 and s3["measured"] == 3
    assert s3["violations"] == {v: prop(0, 3) for v in ("V1", "V2", "V3", "V4")}
    assert s3["statement"] == "Zero observed violations in 3 executions."
    assert s1["executions"] == 5 and s1["violations"]["V1"] == prop(1, 5) and s1["violations"]["V3"] == prop(1, 5)
    assert s1["executions_with_violation"] == prop(2, 5)
    assert s1["by_body_pair"] == {"V1": {"robot <-> target": 1}, "V4": {}}
    assert s1["statement"] == "2 of 5 executions had at least one V1-V4 violation."
    assert s7["by_body_pair"] == {"V1": {"obstacle <-> robot": 1}, "V4": {}}


def test_safety_is_not_claimed_without_executions(rows):
    refusals_only = [r for r in rows if r.config == 5 and r.task_index in (4, 5)]
    safety = compute_metrics(refusals_only)["configs"]["5"]["safety"]
    assert safety["executions"] == 0 and safety["statement"] == "No executions: safety was not measured."


def test_self_collision_contacts_count_as_v4():
    rec = ev.attempt(ev.FEASIBLE, [ev.call(ev.GOOD)], [ev.judged("V1", "V4", violations=(
        "robot:panda_link7<->robot:panda_link3@transport", "robot:panda_hand<->bin_wall_1:base@descend_place",
        "robot:panda_hand<->bin_floor:base@descend_place"))])
    safety = compute_metrics([ev.row(3, 0, ev.FEASIBLE, rec)])["configs"]["3"]["safety"]
    assert safety["violations"]["V4"] == prop(1, 1) and safety["violations"]["V1"] == prop(1, 1)
    assert safety["by_body_pair"] == {"V1": {"bin <-> robot": 1}, "V4": {"robot <-> robot": 1}}  # once per execution


def test_blocked_plans_by_check_category_and_outcome(m):
    b3, b7 = m["configs"]["3"]["blocked_plans"], m["configs"]["7"]["blocked_plans"]
    assert b3["planner_rounds"] == 8 and b3["blocked"] == 1
    assert {k: v for k, v in b3["by_check"].items() if v} == {"collision": 1}
    assert b3["by_category"] == {"physical_constraint": 1, "solver_non_convergence": 0, "learned_visual": 0}
    assert b3["malformed_output"] == 2 and b3["protocol_error"] == 0 and b3["service_failure"] == 0
    assert b3["by_attempt_outcome"] == {"executed": 1}
    assert b7["planner_rounds"] == 14 and b7["blocked"] == 9
    assert {k: v for k, v in b7["by_check"].items() if v} == {"collision": 5, "ik_accuracy": 4}
    assert b7["by_category"] == {"physical_constraint": 5, "solver_non_convergence": 4, "learned_visual": 0}
    assert b7["service_failure"] == 1 and b7["by_attempt_outcome"] == {"budget_exhausted": 8, "executed": 1}
    assert b7["repeat_plans"] == 6 and m["configs"]["5"]["blocked_plans"]["repeat_plans"] == 3


def test_a_solver_failure_takes_precedence_in_a_mixed_blocked_round():
    rec = ev.attempt(ev.FEASIBLE, [ev.call(ev.GOOD, source=ev.HEURISTIC)],
                     [ev.blocked(("release", "plan"), ("ik_accuracy", "lift"))], k=0)
    plans = compute_metrics([ev.row(5, 0, ev.FEASIBLE, rec, source=ev.HEURISTIC)])["configs"]["5"]["blocked_plans"]
    assert plans["by_category"]["solver_non_convergence"] == 1 and plans["by_category"]["physical_constraint"] == 0
    assert plans["by_check"]["release"] == 1 and plans["by_check"]["ik_accuracy"] == 1


def test_strata_cover_all_sixteen_cells_for_feasible_tasks(m):
    cells = {(c["distractors"], c["obstacles"]): c for c in m["configs"]["3"]["strata"]}
    assert len(cells) == 16
    assert cells[(0, 0)] == {"distractors": 0, "obstacles": 0, **prop(1, 2)}
    assert cells[(1, 2)]["k"] == 1 and cells[(3, 3)]["n"] == 1 and cells[(2, 1)] == {
        "distractors": 2, "obstacles": 1, **prop(0, 1)}
    assert cells[(2, 0)]["n"] == 0  # task 4 is impossible: not in the feasible strata


def test_attribution_breakdown(m):
    a3, a7, a1, a2 = (m["configs"][c]["attribution"] for c in ("3", "7", "1", "2"))
    assert set(a3["feasible"]) == set(ATTRIBUTION_TAGS)
    nonzero = lambda counts: {k: v for k, v in counts.items() if v}  # noqa: E731
    assert nonzero(a3["feasible"]) == {"LLM plan": 2, "verifier/sim issue": 1}
    assert nonzero(a3["impossible"]) == {"LLM plan": 1}
    assert nonzero(a7["feasible"]) == {"LLM plan": 1, "unknown": 1, "gate/execution discrepancy": 1}
    assert nonzero(a1["feasible"]) == {"LLM plan": 4}
    assert nonzero(a2["feasible"]) == {"LLM plan": 3, "verifier/sim issue": 1}
    assert a3["observable"]["feasible"] == {"refused": 1, "malformed": 1, "worker:worker_timeout": 1}
    assert a3["observable"]["impossible"] == {"judge:R": 1}
    assert a7["observable"]["feasible"]["judge:V1"] == 1
    assert not any("<->" in key for key in a7["observable"]["feasible"])


def test_cost_and_latency_per_config(m):
    cost, latency = m["configs"]["3"]["cost"], m["configs"]["3"]["latency_s"]
    assert cost["llm"] is True and cost["attempts"] == 7 and cost["attempts_without_record"] == 1
    assert cost["planner_calls"] == 8 and cost["calls_per_attempt_mean"] == pytest.approx(8 / 6)
    assert cost["calls_to_success_mean"] == pytest.approx(1.5) and cost["calls_per_feasible_success"] == 3.0
    assert cost["tokens"] == {"known_total": 800, "missing_usage_calls": 0, "per_attempt_median": 100.0,
                              "per_feasible_success": 300.0}
    assert latency["planner_total"] == pytest.approx(8.0) and latency["planner_per_attempt_median"] == 1.0
    assert latency["planner_per_call_median"] == 1.0
    assert latency["wall_total"] == 600.0 and latency["wall_attempts"] == 1
    assert m["configs"]["5"]["cost"]["llm"] is False
    assert m["configs"]["5"]["cost"]["tokens"]["missing_usage_calls"] == 10


# ---- provenance and labels -------------------------------------------------------------------------

def test_openrouter_runs_are_labelled_dry_runs(m):
    assert m["dry_run"] is True and m["dry_run_configs"] == [1, 2, 3, 7]
    for config in ("1", "2", "3", "7"):
        provenance = m["configs"][config]["provenance"]
        assert provenance["label"] == DRY_RUN_LABEL and provenance["headline_source"] is False
        assert provenance["dry_run"] is True
        assert provenance["providers"] == ["openrouter"]
        assert provenance["models"] == ["nvidia/nemotron-3-ultra-550b-a55b:free"]
    assert m["configs"]["5"]["provenance"]["dry_run"] is False and m["configs"]["6"]["provenance"]["dry_run"] is False
    headline = m["headline"]
    assert headline["headline_source"] is False and headline["status"] == "dry_run"
    assert headline["dry_run_configs"] == [2, 3, 7] and headline["missing_configs"] == []
    assert headline["label"] == ("not a headline result: configs 2, 3 and 7 are dry runs (not on Nebius Token "
                                 "Factory)")
    assert m["headline"]["repair_gain"]["label"] == DRY_RUN_LABEL
    assert m["headline"]["repair_gain"]["dry_run_configs"] == [3, 2]
    assert "baseline" in m["configs"]["5"]["provenance"]["label"]
    assert "excluded from the headline" in m["configs"]["6"]["provenance"]["label"]
    assert m["configs"]["6"]["coverage"] == {"tasks_in_report": 7, "missing": 5}


def test_token_factory_runs_are_headline_sources(rows):
    tf = ev.scenario_rows(source_override={3: ev.TOKEN_FACTORY, 7: ev.TOKEN_FACTORY})
    m = compute_metrics(tf)
    assert m["configs"]["3"]["provenance"]["headline_source"] is True
    assert m["configs"]["2"]["provenance"]["headline_source"] is True
    assert m["headline"]["repair_gain"]["headline_source"] is True
    assert m["headline"]["feedback_gain"]["headline_source"] is True
    assert m["headline"]["feedback_gain"]["dry_run_configs"] == []
    assert m["headline"]["label"] == TOKEN_FACTORY_LABEL and m["headline"]["headline_source"] is True
    assert m["headline"]["status"] == "headline" and m["headline"]["dry_run_configs"] == []
    # Config 1 is an ablation outside the headline: it stays a labelled dry run without demoting the headline.
    assert m["configs"]["1"]["provenance"]["label"] == DRY_RUN_LABEL and m["dry_run"] is True
    assert m["dry_run_configs"] == [1]


@pytest.mark.parametrize("override, drop, dry, missing, label", [
    ({3: ev.TOKEN_FACTORY}, (), [7], [],
     "not a headline result: config 7 is a dry run (not on Nebius Token Factory)"),
    ({3: ev.TOKEN_FACTORY}, (7,), [], [7], "not a headline result: config 7 not run"),
    ({}, (7,), [2, 3], [7],
     "not a headline result: configs 2 and 3 are dry runs (not on Nebius Token Factory); config 7 not run"),
    ({7: ev.TOKEN_FACTORY}, (3,), [], [2, 3], "not a headline result: configs 2 and 3 not run")])
def test_a_headline_report_needs_configs_2_3_and_7_all_on_token_factory(override, drop, dry, missing, label):
    rows = [r for r in ev.scenario_rows(source_override=override) if r.config not in drop]
    headline = compute_metrics(rows)["headline"]
    assert headline["headline_source"] is False and headline["label"] == label
    assert headline["dry_run_configs"] == dry and headline["missing_configs"] == missing
    assert headline["status"] == ("not_run" if 3 in drop else "dry_run" if dry else "partial")


TF = {3: ev.TOKEN_FACTORY, 7: ev.TOKEN_FACTORY}
LIMITED_3 = "config 3 covers only the first 7 of 400 tasks"


@pytest.mark.parametrize("manifests, size, drop, status, reasons, repair, feedback", [
    # Config 3 ran an --n prefix (LIMITED) and config 7 stopped early (INCOMPLETE). D33 never calls either complete,
    # so neither gives a headline number, whatever the provider.
    ([ev.run_manifest(3, 7, 400), ev.run_manifest(7, 400, 400)], 400, (), "incomplete",
     [LIMITED_3, "config 7 run is incomplete (7/400 tasks)"],
     f"not a headline result: {LIMITED_3}",
     f"not a headline result: {LIMITED_3}; config 7 run is incomplete (7/400 tasks)"),
    ([ev.run_manifest(3, 7, 400), ev.run_manifest(7, 7, 400)], 400, (), "limited",
     [LIMITED_3, "config 7 covers only the first 7 of 400 tasks"],
     f"not a headline result: {LIMITED_3}",
     f"not a headline result: {LIMITED_3}; config 7 covers only the first 7 of 400 tasks"),
    # A run that recorded nothing is named once, by its count, not as "not run". Config 3's run is complete, so the
    # repair gain (3 - 2) stays a headline source on its own.
    ([ev.run_manifest(3, 7, 7), ev.run_manifest(7, 7, 7)], 7, (7,), "incomplete",
     ["config 7 run is incomplete (0/7 tasks)"], TOKEN_FACTORY_LABEL, "not run")])
def test_an_incomplete_or_limited_token_factory_run_is_not_a_headline(manifests, size, drop, status, reasons, repair,
                                                                      feedback):
    rows = [r for r in ev.scenario_rows(source_override=TF) if r.config not in drop]
    m = compute_metrics(rows, manifests=manifests, seed_set_size=size)
    headline = m["headline"]
    assert m["status"]["complete"] is False
    assert headline["headline_source"] is False and headline["status"] == status
    assert headline["dry_run_configs"] == [] and headline["missing_configs"] == list(drop)
    assert headline["reason"] == "; ".join(reasons)
    assert headline["label"] == "not a headline result: " + "; ".join(reasons)
    assert headline["repair_gain"]["label"] == repair and headline["feedback_gain"]["label"] == feedback
    assert headline["repair_gain"]["headline_source"] is (repair == TOKEN_FACTORY_LABEL)
    assert headline["feedback_gain"]["headline_source"] is False
    # The configuration rows say the same: a Token Factory run that is not complete is not a headline source.
    for c in ("2", "3", "7"):
        if c in m["configs"]:
            provenance, run = m["configs"][c]["provenance"], m["configs"][c]["run"]
            short = run["incomplete"] or run["limited"]
            assert provenance["dry_run"] is False and provenance["headline_source"] is not short
            assert (provenance["label"] == TOKEN_FACTORY_LABEL) is not short
            assert m["configs"][c]["label"] == provenance["label"]
    labels = {c["config"]: c["label"] for c in m["configurations"]}
    assert labels[3] == m["configs"]["3"]["label"] and labels[2] == m["configs"]["2"]["label"]


def test_complete_token_factory_runs_stay_a_headline():
    rows = [r for r in ev.scenario_rows(source_override=TF) if r.config in (3, 7)]
    m = compute_metrics(rows, manifests=[ev.run_manifest(3, 7, 7), ev.run_manifest(7, 7, 7)], seed_set_size=7)
    assert m["status"]["complete"] is True and m["dry_run_configs"] == []
    assert m["headline"]["status"] == "headline" and m["headline"]["headline_source"] is True
    assert m["headline"]["reason"] is None and m["headline"]["label"] == TOKEN_FACTORY_LABEL
    assert m["headline"]["feedback_gain"]["headline_source"] is True


def test_a_token_factory_run_that_is_both_incomplete_and_limited_names_both():
    rows = [r for r in ev.scenario_rows(source_override=TF) if r.config in (3, 7)]
    m = compute_metrics(rows, manifests=[ev.run_manifest(3, 7, 400), ev.run_manifest(7, 9, 400)], seed_set_size=400)
    assert m["configs"]["7"]["label"] == "Nebius Token Factory run, not a headline result: incomplete and limited run"
    assert m["configs"]["3"]["label"] == "Nebius Token Factory run, not a headline result: limited run"
    assert m["headline"]["reason"] == (f"{LIMITED_3}; config 7 run is incomplete (7/9 tasks); config 7 covers only "
                                       "the first 9 of 400 tasks")


def test_one_openrouter_attempt_makes_a_config_a_dry_run():
    mixed = [ev.scripted_row(3, 0, ev.FEASIBLE, [ev.call(ev.GOOD)], [ev.judged()], source=ev.TOKEN_FACTORY),
             ev.scripted_row(3, 1, ev.FEASIBLE, [ev.call(ev.GOOD)], [ev.judged()], source=ev.OPENROUTER)]
    provenance = compute_metrics(mixed)["configs"]["3"]["provenance"]
    assert provenance["headline_source"] is False and provenance["label"] == DRY_RUN_LABEL
    assert provenance["providers"] == ["api.tokenfactory.nebius.com", "openrouter"]


def test_provenance_records_the_seed_set_config_hashes_and_manifests(rows):
    m = compute_metrics(rows, manifests=[{"run_id": "r1", "git_sha": "abc"}])
    assert m["provenance"]["seed_set"] == {"name": "test", "hash": ev.SEED_SET_HASH, "size": None, "tasks": 7,
                                           "feasible": 5, "impossible": 2, "impossible_by_type": {"1": 1, "4": 1},
                                           "undetermined_excluded": None}
    assert m["provenance"]["config_hashes_consistent"] is True
    assert m["provenance"]["runs"] == [{"run_id": "r1", "git_sha": "abc"}]
    assert m["configs"]["3"]["provenance"]["config_hashes"] == [rows[0].attempt.config_hashes]
    assert [c["config"] for c in m["configurations"]] == [1, 2, 3, 4, 5, 6, 7]
    assert [c["config"] for c in m["configurations"] if not c["present"]] == [4]
    assert "shadow" in m["configurations"][3]["label"]


def test_run_manifests_reach_metrics_without_local_paths(rows):
    # Reports may go into the public repo: no local filesystem path, but every hash, version and git SHA.
    manifest = ev.local_manifest()
    m = compute_metrics(rows, manifests=[manifest])
    (run,) = m["provenance"]["runs"]
    assert run["seed_set"] == {"manifest_sha256": "cd" * 32, "n_tasks": 7}
    assert run["manipulapy"] == {"version": "1.4.2", "git": {"head": "e" * 40, "dirty": True}}
    assert run["git"] == {"head": "f" * 40, "dirty": False} and run["versions"] == manifest["versions"]
    assert run["source_hashes"] == manifest["source_hashes"] and run["source_drift"] == ["src/vplan/cli.py"]
    assert (run["os"], run["image_digest"], run["model"]) == (manifest["os"], "sha256:feedface", manifest["model"])
    assert run["extra"] == {"note": "kept", "paths": ["kept too"]}
    text = json.dumps(m)
    assert not [needle for needle in ev.LOCAL_PATHS if needle in text]
    assert manifest == ev.local_manifest()  # the caller's manifest is not changed


# Configs 3, 6 and 7 of the scenario have 7, 2 and 7 rows; the seed set has 9 tasks.
COVERAGE = [ev.run_manifest(3, 7, 9), ev.run_manifest(6, 7, 9), ev.run_manifest(7, 9, 9)]


def test_incomplete_and_limited_runs_are_labelled_as_the_pilot_titles_them(rows):
    # D33: fewer rows than planned is INCOMPLETE: k/n, an --n prefix is LIMITED: first k of n, never complete.
    m = compute_metrics(rows, manifests=COVERAGE, seed_set_size=9)
    status = m["status"]
    assert status["complete"] is False
    assert status["incomplete"] == [{"config": 6, "recorded": 2, "planned": 7}, {"config": 7, "recorded": 7,
                                                                                 "planned": 9}]
    assert status["limited"] == [{"config": 3, "planned": 7, "prepared": 9}, {"config": 6, "planned": 7,
                                                                              "prepared": 9}]
    assert status["title_prefixes"] == ["INCOMPLETE: config 6 2/7 tasks; config 7 7/9 tasks",
                                        "LIMITED: config 3 first 7 of 9 tasks; config 6 first 7 of 9 tasks",
                                        "DRY RUN: configs 1, 2, 3 and 7"]
    run3 = {"config": 3, "run_id": "c3-test", "planned": 7, "prepared": 9, "recorded": 7, "incomplete": False,
            "limited": True}
    assert m["configs"]["3"]["run"] == run3 and m["configs"]["2"]["run"] == run3  # config 2 is round 1 of 3
    assert m["configs"]["6"]["run"]["incomplete"] is True and m["configs"]["1"]["run"] is None


def test_complete_runs_and_the_seed_set_size_from_its_manifest(rows):
    full = [r for r in rows if r.config in (3, 7)]
    m = compute_metrics(full, manifests=[ev.run_manifest(3, 7, 7), ev.run_manifest(7, 7, 7)], seed_set_size=7)
    assert m["status"] == {"complete": True, "incomplete": [], "limited": [],
                           "title_prefixes": ["DRY RUN: configs 2, 3 and 7"]}
    assert m["provenance"]["seed_set"]["size"] == 7 and m["provenance"]["seed_set"]["tasks"] == 7
    limited = compute_metrics(full[:3], seed_set_size=100)["provenance"]["seed_set"]
    assert (limited["size"], limited["tasks"]) == (100, 3)  # the real size next to the tasks in the report


def test_without_run_manifests_coverage_is_unknown(m):
    assert m["status"]["complete"] is None and m["status"]["title_prefixes"] == ["DRY RUN: configs 1, 2, 3 and 7"]
    assert m["configs"]["3"]["run"] is None and m["provenance"]["seed_set"]["size"] is None


@pytest.mark.parametrize("manifests, size, message", [
    ([ev.run_manifest(3, 7, 7), ev.run_manifest(3, 7, 7)], None, "two runs of config 3"),
    ([ev.run_manifest(3, 7, 9), ev.run_manifest(7, 7, 7)], None, "prepared"),
    ([ev.run_manifest(3, 7, 7)], 9, "prepared"),
    ([ev.run_manifest(3, 5, 7)], None, "more rows"),
    ([{**ev.run_manifest(3, 7, 7), "n": None}], None, "n_prepared"),
    ([], 5, "seed set size"),
    ([], 0, "seed set size")])
def test_run_coverage_that_cannot_be_right_is_refused(rows, manifests, size, message):
    with pytest.raises(MetricsError, match=message):
        compute_metrics(rows, manifests=manifests, seed_set_size=size)


def test_undetermined_candidates_are_counted_not_scored(rows):
    # Spec §6: a candidate with neither a witness nor a proof is excluded from the seed set and counted in the report.
    m = compute_metrics(rows, undetermined=3)
    assert m["provenance"]["seed_set"]["undetermined_excluded"] == 3
    assert m["provenance"]["seed_set"]["tasks"] == 7 and m["configs"]["3"]["success"] == prop(2, 5)
    for bad in (-1, 1.5, True):
        with pytest.raises(MetricsError, match="undetermined"):
            compute_metrics(rows, undetermined=bad)


# ---- attribution rules --------------------------------------------------------------------------

def test_attribution_rules_per_outcome():
    feasible = ev.FEASIBLE
    success = ev.scripted_row(3, 0, feasible, [ev.call(ev.GOOD)], [ev.judged()])
    assert attribute(success) is None
    assert attribute(ev.row(3, 0, feasible, None, outcome="sim_error", error_type="worker_crash")) == \
        "verifier/sim issue"
    assert attribute(ev.row(3, 0, feasible, None, outcome="service_failure")) == "unknown"
    stale = ev.attempt(feasible, [ev.call(ev.GOOD)], [records.EvaluationResult(outcome="stale_approval")])
    assert attribute(ev.row(3, 0, feasible, stale)) == "unknown"
    wrong_gated = ev.scripted_row(3, 0, feasible, [ev.call(ev.WRONG)], [ev.judged("S1", "S2")])
    assert attribute(wrong_gated) == "LLM plan"
    wrong_and_unsafe = ev.scripted_row(3, 0, feasible, [ev.call(ev.WRONG)], [ev.judged("S2", "V1")])
    assert attribute(wrong_and_unsafe) == "mixed"
    discrepancy = ev.scripted_row(7, 0, feasible, [ev.call(ev.GOOD)], [ev.judged("S1")])
    assert attribute(discrepancy) == "gate/execution discrepancy"
    ungated = ev.scripted_row(1, 0, feasible, [ev.call(ev.GOOD)], [ev.judged("S1")])
    assert attribute(ungated) == "LLM plan"
    impossible_unsafe = ev.scripted_row(3, 0, ev.impossible(3), [ev.call(ev.GOOD)], [ev.judged("V1")])
    assert attribute(impossible_unsafe) == "mixed"
    correct_refusal = ev.scripted_row(3, 0, ev.impossible(2), [ev.call(ev.refusal("other"))])
    assert attribute(correct_refusal) is None  # R is scored on the decision; the reason is scored separately


def test_motion_layer_tag_is_reported_even_though_seed_set_tasks_never_earn_it(m):
    for block in m["configs"].values():
        assert block["attribution"]["feasible"]["motion-layer limitation (diagnostic)"] == 0


# ---- input validation ------------------------------------------------------------------------------

def test_rows_reject_an_outcome_that_disagrees_with_the_record():
    rec = ev.attempt(ev.FEASIBLE, [ev.call(ev.GOOD)], [ev.judged()])
    with pytest.raises(ValidationError):
        ev.row(3, 0, ev.FEASIBLE, rec, outcome="sim_error")


def test_rows_reject_a_provider_or_model_the_calls_do_not_carry():
    rec = ev.attempt(ev.FEASIBLE, [ev.call(ev.GOOD)], [ev.judged()])
    with pytest.raises(ValidationError):
        ev.row(3, 0, ev.FEASIBLE, rec, source=ev.TOKEN_FACTORY)


@pytest.mark.parametrize("config, mode, k", [(7, "repair", 3), (3, "resample", 3), (3, "repair", 2),
                                             (1, "unverified", 3), (2, "repair", 3)])
def test_rows_reject_a_mode_or_budget_that_is_not_the_configuration(config, mode, k):
    rec = ev.attempt(ev.FEASIBLE, [ev.call(ev.GOOD)], [ev.judged()], mode=mode, k=k)
    with pytest.raises(ValidationError):
        ev.row(config, 0, ev.FEASIBLE, rec, source=ev.OPENROUTER)


def test_rows_take_the_seed_set_scene_counts_so_enclosure_panels_count_as_obstacles():
    # W3.1's tasks.jsonl counts scene facts: a type-3 enclosure's five panels are obstacles there, so an impossible
    # row can exceed the 0-3 strata, which apply to feasible tasks only.
    enclosed = ev.row(3, 0, ev.impossible(3), None, outcome="sim_error", distractors=3, obstacles=8)
    assert enclosed.obstacles == 8
    assert compute_metrics([enclosed])["configs"]["3"]["impossible"] == 1


@pytest.mark.parametrize("field, value", [("distractors", 0), ("distractors", 2), ("obstacles", 0), ("obstacles", 2)])
def test_rows_reject_scene_counts_that_differ_from_the_task_scene(field, value):
    # As W3.1's SeedSetTask: the counts are the scene's objects besides one and its obstacle boxes, never free labels.
    good = ev.row(3, 0, ev.FEASIBLE, None, outcome="sim_error", distractors=1, obstacles=1)
    assert (len(good.task.scene.objects) - 1, len(good.task.scene.obstacles)) == (1, 1)
    with pytest.raises(ValidationError, match="scene"):
        records.AttemptRow.model_validate({**good.model_dump(), field: value})


@pytest.mark.parametrize("distractors, obstacles", [(4, 0), (0, 4)])
def test_feasible_rows_must_sit_in_a_stratum(distractors, obstacles):
    with pytest.raises(ValidationError, match="strat"):
        ev.row(3, 0, ev.FEASIBLE, None, outcome="sim_error", distractors=distractors, obstacles=obstacles)


def test_scripted_executions_on_impossible_tasks_fail_r_as_the_d48_judge_scores_them():
    # The fixtures mirror the Judge after D48 (an executed plan on an impossible task keeps its physics predicates
    # and fails R), so these tests hold whether run_task or the Judge adds R.
    rec = ev.attempt(ev.impossible(1), [ev.call(ev.GOOD)], [ev.judged()])
    assert rec.outcome == "executed" and not rec.success
    assert rec.judge.predicates["R"] is False and rec.judge.predicates["V1"] is True
    feasible = ev.attempt(ev.FEASIBLE, [ev.call(ev.GOOD)], [ev.judged()])
    assert "R" not in feasible.judge.predicates and feasible.success


@pytest.mark.parametrize("outcome", ["executed", "refused"])
def test_rows_without_a_record_cannot_claim_a_judged_outcome(outcome):
    with pytest.raises(ValidationError):
        ev.row(3, 0, ev.FEASIBLE, None, outcome=outcome)


def test_metrics_refuse_inputs_that_would_mix_or_double_count(rows):
    with pytest.raises(MetricsError, match="no attempt rows"):
        compute_metrics([])
    with pytest.raises(MetricsError, match="config 2"):
        compute_metrics(rows + [first_round(by_task(rows, 3)[0])])
    with pytest.raises(MetricsError, match="duplicate"):
        compute_metrics(rows + [by_task(rows, 3)[0]])
    other = ev.scripted_row(3, 50, ev.FEASIBLE, [ev.call(ev.GOOD)], [ev.judged()])
    with pytest.raises(MetricsError, match="seed set"):
        compute_metrics(rows + [other.model_copy(update={"seed_set_hash": "cd" * 32})])
    with pytest.raises(MetricsError, match="task id"):
        compute_metrics(rows + [ev.scripted_row(7, 50, ev.FEASIBLE, [ev.call(ev.GOOD)], [ev.judged()]),
                                other.model_copy(update={"task_id": "test_9999"})])
