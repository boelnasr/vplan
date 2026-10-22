"""W3.5 UI view models (spec §15): the check timeline, finding evidence, the Judge panel and the Cosmos banner are
plain data built from records, so they are tested here without Gradio or PyBullet."""
import json

import pytest

from tests.test_repair import generated
from tests.test_run_task import call
from tests.ui_fixtures import JARGON, visible
from vplan import records
from vplan.tasks import make_basic_scene
from vplan.ui import markup, views

WALL_SCENE = views.scene_names(make_basic_scene(with_obstacle=True))


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
    view = views.round_view(rnd(0, PLAN, blocked(finding())), mode="repair", names=WALL_SCENE)
    row = next(r for r in view.checks if r.check == "collision")
    assert (row.code, row.title, row.status) == ("V-6", "Collision", "fail")
    assert "arm link 5 ↔ wall" in row.evidence and "robot:" not in row.evidence and "wall_1" not in row.evidence
    assert "0.40 cm" in row.evidence and "1.00 cm" in row.evidence  # measured clearance and the margin
    assert "transport" in row.evidence and "sample 12" in row.evidence
    assert row.finding is not None and row.finding.bodies == ("robot:panda_link5", "wall_1")  # the record is kept
    assert [r.code for r in view.checks] == [f"V-{i}" for i in range(1, 9)]  # spec §9 order
    assert [r.title for r in view.checks] == ["Target and drop-off", "Reach", "Joint limits", "Singularity margin",
                                              "Smooth joint motion", "Collision", "Grasp", "Release in bin"]
    assert all(s == "pass" for c, s in statuses(view).items() if c != "collision")
    assert view.outcome == "blocked" and view.summary == "rejected: 1 check failed"
    assert view.title == "Round 1: first plan"


def test_penetration_is_reported_as_penetration_not_as_a_negative_clearance():
    view = views.round_view(rnd(0, PLAN, blocked(finding(measured=-0.003))), mode="repair")
    row = next(r for r in view.checks if r.check == "collision")
    assert "penetration 0.30 cm" in row.evidence and "-0.3" not in row.evidence


def test_segment_checks_after_an_ik_failure_are_only_partial():
    ik = finding("ik_accuracy", stage="lift", plan_field="grasp", measured=0.01, threshold=0.002, bodies=(),
                 direction="pose not reachable: residual 10.0 mm / 0.1 deg", sample_index=3)
    s = statuses(views.round_view(rnd(0, PLAN, blocked(ik)), mode="repair"))
    assert s["ik_accuracy"] == "fail"
    assert s["continuity"] == "partial"
    assert s["joint_limits"] == s["singularity"] == s["collision"] == "partial"
    assert s["grasp_preconditions"] == s["release"] == s["grounding"] == "pass"  # whole-plan checks still ran


def test_ik_accuracy_after_a_continuity_failure_is_partial():
    continuity = finding("continuity", stage="lift", plan_field="grasp", measured=2.0, threshold=1.0, bodies=(),
                         direction="IK branch flip: large joint jump", sample_index=3)
    s = statuses(views.round_view(rnd(0, PLAN, blocked(continuity)), mode="repair"))
    assert s["continuity"] == "fail"
    assert s["ik_accuracy"] == "partial"
    assert s["joint_limits"] == s["singularity"] == s["collision"] == "partial"


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
    assert "without verification" in view.checks_note


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
    assert view.summary == "refused (object enclosed): the cube sits in a closed box"
    assert '"reason": "target_enclosed"' in view.plan_json  # the recorded JSON keeps its values


def test_plan_json_is_pretty_printed():
    view = views.round_view(rnd(0, PLAN, executed()), mode="repair")
    assert view.plan_json.startswith("{\n  ") and f'"target_id": "{PLAN.target_id}"' in view.plan_json



def test_plan_json_keeps_each_coordinate_list_on_one_line():
    view = views.round_view(rnd(0, PLAN, executed()), mode="repair")
    position = json.dumps(list(PLAN.place.position), separators=(", ", ": "))
    assert f'"position": {position}' in view.plan_json
    assert json.loads(view.plan_json) == PLAN.model_dump(mode="json")  # still the same, valid JSON

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
    assert banner.level == "warning" and "kinematic and collision checks only" in banner.text


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
    assert "without the Cosmos visual check" in views.replay_cosmos_banner(record).text
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
                                                           rnd(1, PLAN, executed())), names=WALL_SCENE).rounds)
    text = visible(html)
    assert text.count("Round ") == 2 and "Round 2: repair 1" in text
    assert "Collision" in text and "arm link 5 ↔ wall" in text
    assert "✕ Fail" in text and "■ Rejected" in text and "✓ Executed" in text
    # passed checks are compact chips; only the failing check gets a table row with its evidence
    assert "Passed ✓ Target and drop-off ✓ Reach" in text and text.count("✕ Fail") == 1
    assert 'title="Verifier check V-6' in html and 'title="Verifier check V-2' in html  # codes stay as tooltips
    assert not JARGON.search(text), JARGON.search(text)


def test_checks_that_did_not_run_are_grouped():
    unknown = finding("grounding", stage="plan", plan_field="target_id", measured=None, threshold=None,
                      direction="target_id 'x' is not an object in the scene", sample_index=None, bodies=())
    text = visible(markup.timeline_html([views.round_view(rnd(0, PLAN, blocked(unknown)), mode="repair")]))
    assert "Not run – Reach – Joint limits" in text and text.count("✕ Fail") == 1


def test_judge_markup_lists_predicates_and_verdict():
    view = views.attempt_view(attempt(rnd(0, PLAN, executed(success=False))))
    html = markup.judge_html(view.judge)
    text = visible(html)
    assert "✕ Failure" in text and "Object resting in the bin" in text and "Failed: Object resting in the bin" in text
    assert "Only the instructed object in the bin" in text  # S2: the target is in the bin and no other object is
    assert 'title="S1: the target is inside the bin, at rest"' in html
    assert not JARGON.search(text), JARGON.search(text)


def test_banner_markup_is_escaped_and_marked_by_level():
    html = markup.banner_html(views.Banner("warning", "COSMOS UNAVAILABLE - <offline>"))
    assert "&lt;offline&gt;" in html and "warning" in html
    assert markup.banner_html(None) == ""


def test_a_skipped_gate_reads_like_the_gate_being_off():
    skipped = records.VisualGateReport(status="skipped", note="no single instructed object")
    banner = views.cosmos_banner("http://127.0.0.1:8000/v1", [skipped])
    assert banner.level == "warning" and "kinematic and collision checks only" in banner.text and "skipped" in banner.text
    assert "learned visual check" not in banner.text
    record = attempt(rnd(0, PLAN, executed(visual=skipped)))
    assert "kinematic and collision checks only" in views.replay_cosmos_banner(record).text


# ---- plain names (W4.1) -------------------------------------------------------------------------------

def test_scene_names_turn_ids_into_what_a_viewer_sees():
    assert WALL_SCENE == {"cube_red": "red cube", "cube_green": "green cube", "wall_1": "wall"}
    assert views.body_name("robot:panda_link5", WALL_SCENE) == "arm link 5"
    assert views.body_name("robot:panda_hand", {}) == "gripper"
    assert views.body_name("robot:panda_leftfinger", {}) == "left finger"
    assert views.body_name("bin_wall_2", {}) == "bin wall" and views.body_name("bin_floor", {}) == "bin floor"
    assert views.body_name("table", {}) == "table"
    task = generated().task
    names = views.scene_names(task.scene)
    assert all(not v.startswith("o_") for v in names.values()) and set(names) >= {o.id for o in task.scene.objects}
    assert names[task.scene.objects[0].id] == f"{task.scene.objects[0].color} {task.scene.objects[0].shape}"


def test_obstacles_are_summarised_in_words():
    assert views.obstacle_summary(make_basic_scene(with_obstacle=True)) == "1 wall"
    assert views.obstacle_summary(make_basic_scene()) == "none"
    from vplan.ui import live
    enclosed = live.generated_task(1, "impossible 3: target enclosed")
    enclosure = {o.id for o in enclosed.oracle.proof.enclosure}
    names = views.scene_names(enclosed.task.scene, enclosure)
    assert {names[i] for i in enclosure} == {"enclosure"}
    assert views.obstacle_summary(enclosed.task.scene, enclosure).startswith("a closed box around the object")


def test_a_collision_direction_already_in_plain_words_is_left_as_written():
    """The Verifier now names the parts and bodies itself; the rewrite (kept for older recordings) must not garble it,
    apostrophes and all."""
    text = ("the carried object too close to an obstacle during transport (clearance 0.94 cm); its bottom is 0.050 m "
            "below the obstacle's top (top at z 0.150 m)")
    assert views.humanize(text, WALL_SCENE) == text


def test_verifier_text_is_rewritten_with_plain_names():
    text = views.humanize("robot link panda_link5 too close to 'wall_1' (0.4 cm)", WALL_SCENE)
    assert text == "arm link 5 too close to the wall (0.4 cm)"
    assert views.humanize("carried 'cube_red' too close to robot link panda_hand", WALL_SCENE) == \
        "carried red cube too close to the gripper"
    assert views.humanize("panda_joint4 within 2 deg of its limit", {}) == "joint 4 within 2 deg of its limit"
    # a model's explanation often repeats the id after the name
    names = {"o_400dc354a15f0aae": "yellow cylinder"}
    assert views.humanize("The yellow cylinder (o_400dc354a15f0aae) is beyond reach; o_400dc354a15f0aae is far.",
                          names) == "The yellow cylinder is beyond reach; yellow cylinder is far."
    ev = views.finding_evidence(finding("joint_limits", plan_field="transport_via[0]", measured=2.8, threshold=2.9,
                                        direction="panda_joint4 within 2 deg of its limit", bodies=()), WALL_SCENE)
    assert ev.startswith("joint 4 within 2 deg of its limit") and ev.endswith("in the plan: waypoint 1")


def test_round_titles_and_planner_names_read_plainly():
    assert views.round_title(0, "repair") == "Round 1: first plan"
    assert views.round_title(2, "repair") == "Round 3: repair 2"
    assert views.round_title(1, "resample") == "Round 2: new sample 1"
    assert views.planner_name("heuristic", "template-heuristic") == "Rule-based baseline"
    assert views.planner_name("api.tokenfactory.nebius.com", "nvidia/Nemotron-3_5-Lightning") == \
        "nvidia/Nemotron-3_5-Lightning via api.tokenfactory.nebius.com"


@pytest.mark.parametrize("outcome", ["executed", "refused", "budget_exhausted", "malformed_final", "service_failure",
                                     "stale_approval", "sim_error", "visual_uncertain"])
def test_headlines_and_banners_carry_no_internal_terms(outcome):
    refusal = records.Refusal(reason="other", explanation="x")
    j = records.JudgeResult(success=False, predicates={"R": False}, violations=("R",), details={})
    first = rnd(0, refusal, records.EvaluationResult(outcome="refused", judge=j)) if outcome == "refused" else \
        rnd(0, PLAN, executed(success=False))
    view = views.attempt_view(attempt(first, outcome=outcome, judged=outcome in ("executed", "refused")))
    assert not JARGON.search(view.headline), view.headline
    for text in (views.COSMOS_OFF, views.COSMOS_ON, views.COSMOS_SKIPPED, views.COSMOS_NONE):
        assert not JARGON.search(text), text


def test_the_cosmos_on_banner_cites_the_shadow_study_not_the_pilot():
    # The shadow study (D73) over the final run replaced the 30-frame pilot's 0.70 as the measured figure.
    assert "0.61" in views.COSMOS_ON and "0.52" in views.COSMOS_ON and "0.70" not in views.COSMOS_ON
    assert "not verification" in views.COSMOS_ON


def test_judge_contacts_and_failed_criteria_are_named():
    j = records.JudgeResult(success=False, predicates={"S1": True, "S2": True, "V1": False, "V2": True, "V3": True,
                                                      "V4": True},
                            violations=("robot:panda_link5<->wall_1:-1@transport", "cube_red:-1<->table:-1@lift"),
                            details={})
    view = views.judge_view(j, WALL_SCENE)
    assert view.violations == ("arm link 5 ↔ wall during transport", "red cube ↔ table during lift")
    text = visible(markup.judge_html(view))
    assert "Failed: No unsafe contact" in text and "Unsafe contacts: arm link 5 ↔ wall during transport" in text
    record = attempt(rnd(0, PLAN, records.EvaluationResult(outcome="executed", judge=j)))
    assert views.attempt_view(record).headline == ("Failure: the plan was executed, but the judge found a problem "
                                                   "(failed: No unsafe contact)")


def test_outcomes_pair_colour_with_an_icon_and_a_word():
    stopped = views.attempt_view(attempt(*(rnd(i, PLAN, blocked(finding())) for i in range(2)),
                                         outcome="budget_exhausted", judged=False))
    assert markup.attempt_status(stopped) == (stopped.headline, "warn", "■")
    html = markup.status_html([("Task", "x")], *markup.attempt_status(stopped)[:2], icon="■")
    assert visible(html).startswith(" ■ Blocked no plan passed the checks in 2 planner calls")
    success = views.attempt_view(attempt(rnd(0, PLAN, executed())))
    assert markup.attempt_status(success)[1:] == ("pass", "✓")


def test_the_css_wins_over_gradio_prose_rules():
    """gr.HTML content sits in .prose, whose two-class colour and table rules beat single-class selectors."""
    scoped = markup._scoped(markup._RULES)
    assert scoped in markup.CSS and scoped.count("{") == markup._RULES.count("{")
    assert all(sel.strip().startswith(markup.SCOPE.strip()) for line in scoped.splitlines()
               for sel in line.split("{")[0].split(","))


# ---- review round (W4.1) -------------------------------------------------------------------------------

def test_without_an_oracle_target_s1_and_s2_are_about_the_planned_object():
    """Type 4 (no matching object): the Judge scores S1/S2 on the object the planner picked (repair.run_task), so the
    panel must not claim a correct object that does not exist."""
    j = records.JudgeResult(success=False, predicates={"S1": True, "S2": True, "V1": True, "V2": True, "V3": True,
                                                      "V4": True, "R": False}, violations=("R",), details={})
    text = visible(markup.judge_html(views.judge_view(j, WALL_SCENE, no_target=True)))
    assert "Planned object resting in the bin" in text and "Only the planned object in the bin" in text
    assert "instructed" not in text and "Correct object" not in text
    record = attempt(rnd(0, PLAN, records.EvaluationResult(outcome="executed", judge=j)))
    view = views.attempt_view(record, WALL_SCENE, no_target=True)
    assert [p.name for p in view.judge.predicates][:2] == ["Planned object resting in the bin",
                                                         "Only the planned object in the bin"]


def test_an_executed_impossible_task_says_the_planner_should_have_refused():
    j = records.JudgeResult(success=False, predicates={"S1": True, "S2": True, "V1": True, "V2": True, "V3": True,
                                                      "V4": True, "R": False}, violations=("R",), details={})
    record = attempt(rnd(0, PLAN, records.EvaluationResult(outcome="executed", judge=j)))
    assert views.attempt_view(record).headline == "Failure: the task is impossible, so the planner should have refused"


def test_outcome_codes_and_protocol_names_never_reach_the_page():
    def no_plan(index, outcome, **update):
        return records.RoundRecord(call_id=f"a:{index + 1}", round_index=index,
                                   call=call(None, outcome).model_copy(update=update))

    malformed = no_plan(0, "malformed", errors=({"loc": ("plan", "grasp"), "msg": "Field required", "type": "missing"},))
    protocol = no_plan(1, "protocol_error", next_protocol="content")
    sim = rnd(2, PLAN, records.EvaluationResult(outcome="sim_error", error_type="RuntimeError",
                                                error_message="NaN in solver"))
    stale = rnd(3, PLAN, records.EvaluationResult(outcome="stale_approval", error_type="StaleApproval"))
    refusal = rnd(0, records.Refusal(reason="no_matching_object", explanation="there is no cube_green here"),
                  records.EvaluationResult(outcome="refused", judge=records.JudgeResult(
                      success=True, predicates={"R": True}, violations=(), details={"refusal": "correct"})))
    for r in (malformed, protocol, sim, stale, refusal):
        view = views.round_view(r, mode="repair", names=WALL_SCENE)
        text = visible(markup.round_html(view))
        assert not JARGON.search(text), (r.call.outcome, JARGON.search(text), text)
        assert "content" not in view.summary  # the protocol name
    assert views.round_view(malformed, mode="repair").problems == ("Invalid reply: plan.grasp: Field required",)
    assert views.round_view(sim, mode="repair").problems == ("Simulation error: RuntimeError: NaN in solver",)


def test_plan_fields_read_as_parts_of_the_plan():
    assert views.field_name("transport_via[1]") == "waypoint 2"
    assert views.field_name("place.position") == "the drop-off point"
    assert views.field_name("place.release_height_m") == "the release height"
    assert views.field_name("grasp.yaw_deg") == "the grasp angle"
    assert views.field_name("target_id") == "the target object"
    release = finding("release", stage="descend_place", plan_field="place.position", measured=0.01, threshold=0.005,
                      bodies=(), sample_index=None,
                      direction="place.position z must be the object's resting height 0.031 m (bin floor z + half the "
                                "object height); release_height_m sets the drop")
    ev = views.finding_evidence(release, WALL_SCENE)
    assert not JARGON.search(ev), ev
    assert ev.startswith("the drop-off height must be the object's resting height 0.031 m") and "the release height" in ev
    unknown = views.humanize("target_id 'cube_blue' is not an object in the scene; known ids: ['cube_green', "
                             "'cube_red']", WALL_SCENE)
    assert unknown == "the target name 'cube_blue' is not an object in the scene; objects here: green cube, red cube"


def test_judge_measurements_have_plain_names():
    j = records.JudgeResult(success=True, predicates={"S1": True}, violations=(), details={
        "in_bin": "{'cube_red': True, 'cube_green': False}", "at_rest": "True", "dwell_ok": "True",
        "dwell_containment": "True", "dwell_evidence_error": "", "sigma_evidence_steps": "2336"})
    text = visible(markup.judge_html(views.judge_view(j, WALL_SCENE)))
    assert not JARGON.search(text), text
    assert "Objects in the bin: red cube yes, green cube no" in text and "Settled at rest: yes" in text
    assert "Singularity samples recorded: 2336" in text and "problem" not in text  # an empty value is left out


def test_check_tooltips_explain_singularity_and_allowed_contacts():
    assert "locked" not in views.CHECK_HELP["singularity"]
    assert "loses the ability to move in some direction" in views.CHECK_HELP["singularity"]
    assert "fingers on the object" in views.CHECK_HELP["collision"] and "bin floor" in views.CHECK_HELP["collision"]


def test_the_baseline_shows_no_planner_latency():
    baseline = records.RoundRecord(call_id="a:1", round_index=0, evaluation=executed(), call=call(PLAN).model_copy(
        update={"provider": "heuristic", "model": "template-heuristic", "latency_s": 0.004}))
    html = markup.round_html(views.round_view(baseline, mode="repair"))
    assert "Rule-based baseline" in visible(html) and " s<" not in html and "0.0 s" not in visible(html)
