"""W3.5 UI view models (spec §15): the check timeline, finding evidence, the Judge panel and the Cosmos banner are
plain data built from records, so they are tested here without Gradio or PyBullet."""
import pytest

from tests.test_repair import generated
from tests.test_run_task import call
from vplan import records
from vplan.ui import markup, views


def finding(check="collision", stage="transport", **kw):
    base = dict(stage=stage, check=check, plan_field="transport_via[0]", measured=0.004, threshold=0.01,
                direction="robot link panda_link5 too close to 'wall_1' (0.4 cm)", sample_index=12,
                bodies=("robot:panda_link5", "wall_1"))
    base.update(kw)
    return records.Finding(**base)


def blocked(*findings, visual=None):
    return records.EvaluationResult(outcome="blocked", findings=tuple(findings), visual=visual)


def judge(success=True, **predicates):
    preds = predicates or {"S1": success, "S2": True, "V1": True, "V2": True, "V3": True, "V4": True}
    return records.JudgeResult(success=success, predicates=preds, violations=() if success else ("S1",),
                               details={"S1": "inside the bin, at rest"})


def executed(success=True, visual=None):
    return records.EvaluationResult(outcome="executed", judge=judge(success), visual=visual)


def rnd(index, plan=None, evaluation=None, outcome=None, **kw):
    return records.RoundRecord(call_id=f"a:{index + 1}", round_index=index, call=call(plan, outcome),
                               evaluation=evaluation, **kw)


def attempt(*rounds, outcome="executed", mode="repair", k=3, judged=True):
    last = rounds[-1].evaluation
    return records.AttemptRecord(attempt_id="ui:basic", mode=mode, k=k, outcome=outcome, rounds=tuple(rounds),
                                 call_count=len(rounds), judge=last.judge if judged and last else None,
                                 total_tokens=10 * len(rounds), known_total_tokens=10 * len(rounds),
                                 missing_usage_calls=0, latency_s=0.0, config_hashes={})


PLAN = generated().oracle.reference_plan


def statuses(round_view):
    return {row.check: row.status for row in round_view.checks}


# ---- check timeline --------------------------------------------------------------------------------

def test_a_collision_round_fails_v6_with_the_offending_bodies_and_the_measured_clearance():
    view = views.round_view(rnd(0, PLAN, blocked(finding())), mode="repair")
    row = next(r for r in view.checks if r.check == "collision")
    assert (row.code, row.status) == ("V-6", "fail")
    assert "robot:panda_link5" in row.evidence and "wall_1" in row.evidence
    assert "0.40 cm" in row.evidence and "1.00 cm" in row.evidence  # measured clearance and the margin
    assert "transport" in row.evidence and "sample 12" in row.evidence
    assert row.finding is not None and row.finding.bodies == ("robot:panda_link5", "wall_1")
    assert [r.code for r in view.checks] == [f"V-{i}" for i in range(1, 9)]  # spec §9 order
    assert all(s == "pass" for c, s in statuses(view).items() if c != "collision")
    assert view.outcome == "blocked" and "Verifier" in view.summary


def test_penetration_is_reported_as_penetration_not_as_a_negative_clearance():
    view = views.round_view(rnd(0, PLAN, blocked(finding(measured=-0.003))), mode="repair")
    row = next(r for r in view.checks if r.check == "collision")
    assert "penetration 0.30 cm" in row.evidence and "-0.3" not in row.evidence


def test_segment_checks_after_an_ik_failure_are_only_partial():
    ik = finding("ik_accuracy", stage="lift", plan_field="grasp", measured=0.01, threshold=0.002, bodies=(),
                 direction="pose not reachable: residual 10.0 mm / 0.1 deg", sample_index=3)
    s = statuses(views.round_view(rnd(0, PLAN, blocked(ik)), mode="repair"))
    assert s["ik_accuracy"] == "fail"
    assert s["joint_limits"] == s["singularity"] == s["collision"] == "partial"
    assert s["grasp_preconditions"] == s["release"] == s["grounding"] == "pass"  # whole-plan checks still ran


def test_five_findings_leave_the_unreported_checks_unknown():
    five = [finding(c, plan_field="grasp", bodies=()) for c in
            ("grounding", "joint_limits", "singularity", "collision", "release")]
    s = statuses(views.round_view(rnd(0, PLAN, blocked(*five)), mode="repair"))
    assert [c for c, v in s.items() if v == "fail"] == ["grounding", "joint_limits", "singularity", "collision",
                                                        "release"]
    assert s["ik_accuracy"] == s["continuity"] == s["grasp_preconditions"] == "unknown"


def test_an_unknown_target_stops_everything_after_grounding():
    unknown = finding("grounding", stage="plan", plan_field="target_id", measured=None, threshold=None,
                      direction="target_id 'x' is not an object in the scene", sample_index=None, bodies=())
    s = statuses(views.round_view(rnd(0, PLAN, blocked(unknown)), mode="repair"))
    assert s["grounding"] == "fail" and {v for c, v in s.items() if c != "grounding"} == {"off"}


def test_unverified_rounds_show_the_verifier_off():
    view = views.round_view(rnd(0, PLAN, executed()), mode="unverified")
    s = statuses(view)
    assert s["grounding"] == "partial" and {v for c, v in s.items() if c != "grounding"} == {"off"}
    assert "unverified" in view.checks_note


def test_a_cosmos_rejection_means_every_geometric_check_passed():
    report = records.VisualGateReport(
        status="fail", findings=(records.Finding(stage="grasp", check="visual", plan_field="grasp", measured=None,
                                                 threshold=None, direction="learned visual check (Cosmos)"),),
        checks=(records.VisualCheck(stage="pregrasp", views=("front_oblique", "top"), verdict=records.VisualVerdict(
            verdict="pass", reason="ok", latency_s=1.0, raw="")),
                records.VisualCheck(stage="grasp", views=("front_oblique", "top"), verdict=records.VisualVerdict(
                    verdict="fail", reason="not held", latency_s=2.0, raw=""))))
    view = views.round_view(rnd(0, PLAN, blocked(*report.findings, visual=report)), mode="repair")
    assert set(statuses(view).values()) == {"pass"}
    assert view.visual_status == "fail" and [v.verdict for v in view.visual] == ["pass", "fail"]
    assert "Cosmos" in view.summary and "learned" in view.summary


def test_a_simulation_error_leaves_the_checks_unknown_and_names_the_error():
    error = records.EvaluationResult(outcome="sim_error", error_type="RuntimeError", error_message="NaN in solver")
    view = views.round_view(rnd(0, PLAN, error), mode="repair")
    assert set(statuses(view).values()) == {"unknown"}
    assert any("RuntimeError" in p for p in view.problems)


@pytest.mark.parametrize("outcome", ["malformed", "service_failure"])
def test_rounds_without_a_plan_have_no_timeline(outcome):
    view = views.round_view(rnd(0, None, None, outcome=outcome), mode="repair")
    assert view.checks == () and view.plan_json is None and view.outcome == outcome


def test_a_refusal_round_shows_the_reason_and_no_checks():
    refusal = records.Refusal(reason="target_enclosed", explanation="the cube sits in a closed box")
    j = records.JudgeResult(success=True, predicates={"R": True}, violations=(), details={})
    view = views.round_view(rnd(0, refusal, records.EvaluationResult(outcome="refused", judge=j)), mode="repair")
    assert view.plan_kind == "refusal" and view.checks == ()
    assert "target_enclosed" in view.summary and '"reason": "target_enclosed"' in view.plan_json


def test_plan_json_is_pretty_printed():
    view = views.round_view(rnd(0, PLAN, executed()), mode="repair")
    assert view.plan_json.startswith("{\n  ") and f'"target_id": "{PLAN.target_id}"' in view.plan_json


# ---- attempt, judge and banner -------------------------------------------------------------------------

def test_attempt_view_of_a_repaired_success():
    record = attempt(rnd(0, PLAN, blocked(finding())), rnd(1, PLAN, executed()))
    view = views.attempt_view(record)
    assert [r.outcome for r in view.rounds] == ["blocked", "executed"]
    assert view.success and view.judge.success and "success" in view.headline.lower()
    assert ("S1", True) in [(p.id, p.passed) for p in view.judge.predicates]


def test_a_correct_refusal_and_a_false_refusal_read_differently():
    refusal = records.Refusal(reason="target_unreachable", explanation="too far")
    ok = records.JudgeResult(success=True, predicates={"R": True}, violations=(), details={})
    bad = records.JudgeResult(success=False, predicates={"R": False}, violations=("R",), details={})
    good_view = views.attempt_view(attempt(rnd(0, refusal, records.EvaluationResult(outcome="refused", judge=ok)),
                                           outcome="refused"))
    bad_view = views.attempt_view(attempt(rnd(0, refusal, records.EvaluationResult(outcome="refused", judge=bad)),
                                          outcome="refused"))
    assert "correct refusal" in good_view.headline.lower()
    assert "false refusal" in bad_view.headline.lower()


def test_budget_exhausted_headline_counts_the_calls():
    record = attempt(*(rnd(i, PLAN, blocked(finding())) for i in range(4)), outcome="budget_exhausted", judged=False)
    view = views.attempt_view(record)
    assert not view.success and view.judge is None and "4 planner calls" in view.headline


def test_without_a_cosmos_url_the_banner_says_geometric_checks_only():
    banner = views.cosmos_banner(None, [])
    assert banner.level == "warning" and "geometric checks only" in banner.text


def test_an_unavailable_endpoint_gives_the_d58_banner_with_its_note():
    down = records.VisualGateReport(status="unavailable", note="APIConnectionError: offline")
    banner = views.cosmos_banner("http://127.0.0.1:8000/v1", [None, down])
    assert banner.level == "warning"
    assert banner.text.startswith("COSMOS UNAVAILABLE - proceeding on the geometric checks only")
    assert "offline" in banner.text


def test_a_working_gate_is_labelled_a_learned_check_not_verification():
    ok = records.VisualGateReport(status="pass")
    banner = views.cosmos_banner("http://127.0.0.1:8000/v1", [ok])
    assert banner.level == "info" and "not verification" in banner.text


def test_replay_banner_names_attempts_that_ran_without_the_gate():
    record = attempt(rnd(0, PLAN, executed()))
    assert "without the Cosmos gate" in views.replay_cosmos_banner(record).text
    down = records.VisualGateReport(status="unavailable", note="offline")
    assert views.replay_cosmos_banner(attempt(rnd(0, PLAN, executed(visual=down)))).text.startswith(
        "COSMOS UNAVAILABLE")


# ---- markup -----------------------------------------------------------------------------------------

def test_markup_escapes_planner_text():
    hostile = PLAN.model_copy(update={"rationale": "<script>alert(1)</script>"})
    html = markup.timeline_html(views.attempt_view(attempt(rnd(0, hostile, executed()))).rounds)
    assert "<script>" not in html and "&lt;script&gt;" in html


def test_markup_shows_status_badges_and_evidence():
    html = markup.timeline_html(views.attempt_view(attempt(rnd(0, PLAN, blocked(finding())),
                                                           rnd(1, PLAN, executed()))).rounds)
    assert html.count("Round ") == 2 and "V-6" in html and "robot:panda_link5" in html
    assert "fail" in html and "pass" in html


def test_judge_markup_lists_predicates_and_verdict():
    view = views.attempt_view(attempt(rnd(0, PLAN, executed(success=False))))
    html = markup.judge_html(view.judge)
    assert "FAILURE" in html and "S1" in html and "inside the bin" in html


def test_banner_markup_is_escaped_and_marked_by_level():
    html = markup.banner_html(views.Banner("warning", "COSMOS UNAVAILABLE - <offline>"))
    assert "&lt;offline&gt;" in html and "warning" in html
    assert markup.banner_html(None) == ""


def test_a_skipped_gate_reads_like_the_gate_being_off():
    skipped = records.VisualGateReport(status="skipped", note="no single instructed object")
    banner = views.cosmos_banner("http://127.0.0.1:8000/v1", [skipped])
    assert banner.level == "warning" and "geometric checks only" in banner.text and "skipped" in banner.text
    assert "learned visual check" not in banner.text
    record = attempt(rnd(0, PLAN, executed(visual=skipped)))
    assert "geometric checks only" in views.replay_cosmos_banner(record).text
