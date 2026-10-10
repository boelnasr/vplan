"""Regressions for the five approved review findings; no network calls."""
import json
import math

import pybullet as p
import pytest
from pydantic import ValidationError

from tests.conftest import make_basic_scene
from tests.test_execute import setup
from tests.test_judge import DT, DWELL, rec
from tests.test_planner import FakeClient, planner, response
from tests.test_repair import generated
from vplan.execute.executor import StaleApproval, _check_start_state, approval_token, execute, init_physics
from vplan.judge.judge import judge
from vplan.records import ExecutionRecord, Pose, Scene
from vplan.visual.cosmos import parse_verdict


@pytest.mark.parametrize("body", ["wall_1", "table", "bin_floor", "bin_wall_0", "bin_wall_1", "bin_wall_2", "bin_wall_3", "robot", "cube_red", "cube_green"])
@pytest.mark.parametrize("mutation", ["translate", "rotate", "remove"])
def test_live_scene_body_mutation_is_stale(client, body, mutation):
    scene, cfgs, traj = setup(make_basic_scene(with_obstacle=True))
    h = init_physics(scene, client, traj.q_start, cfgs["physics"])
    bid = h.robot if body == "robot" else h.bodies[body]
    pos, orn = p.getBasePositionAndOrientation(bid, physicsClientId=client)
    if mutation == "remove":
        p.removeBody(bid, physicsClientId=client)
    else:
        if mutation == "translate":
            pos = (pos[0] + .01, pos[1], pos[2])
        else:
            orn = p.getQuaternionFromEuler((0, 0, .1))
        p.resetBasePositionAndOrientation(bid, pos, orn, physicsClientId=client)
    with pytest.raises(StaleApproval, match=body):
        _check_start_state(h, traj, scene, cfgs)


def test_unchanged_scene_including_robot_com_offset_is_approved(client):
    scene, cfgs, traj = setup(make_basic_scene(with_obstacle=True))
    h = init_physics(scene, client, traj.q_start, cfgs["physics"])
    _check_start_state(h, traj, scene, cfgs)


def test_world_initialized_from_unrelated_scene_is_stale(client):
    scene, cfgs, traj = setup()
    other = Scene.model_validate({**scene.model_dump(), "table_top_z": .01})
    h = init_physics(other, client, traj.q_start, cfgs["physics"])
    with pytest.raises(StaleApproval, match="scene"):
        _check_start_state(h, traj, scene, cfgs)


def test_missing_body_handle_is_stale(client):
    scene, cfgs, traj = setup()
    h = init_physics(scene, client, traj.q_start, cfgs["physics"])
    h.bodies.pop("bin_floor")
    with pytest.raises(StaleApproval, match="bin_floor"):
        _check_start_state(h, traj, scene, cfgs)


def test_executor_records_all_dwell_poses_at_physics_frequency(client):
    scene, cfgs, traj = setup()
    h = init_physics(scene, client, traj.q_start, cfgs["physics"])
    record = execute(h, traj, scene, "cube_red", cfgs, approval_token(traj, scene, "cube_red", cfgs))
    for obj in scene.objects:
        assert len(record.rest_pose_hist[obj.id]) == len(record.rest_lin_hist[obj.id]) == DWELL
        assert len(record.rest_ang_hist[obj.id]) == DWELL
        assert record.rest_pose_hist[obj.id][-1] == record.final_poses[obj.id]


def stable_record():
    record = rec()
    data = record.model_dump()
    data["rest_pose_hist"] = {key: [pose.model_dump()] * DWELL for key, pose in record.final_poses.items()}
    return ExecutionRecord.model_validate(data)


def test_slow_boundary_crossing_ending_inside_fails_full_dwell_containment():
    # Bin right edge is .53; cube center must be <= .51. Move 2 mm over .5 s,
    # i.e. 4 mm/s, below the 10 mm/s rest limit, from just outside to just inside.
    positions = tuple(Pose(position=(.511 - .002 * k / (DWELL - 1), .28, .02)) for k in range(DWELL))
    record = rec(cube=positions[-1], lin=(.0041,) * DWELL)
    record = record.model_copy(update={"rest_pose_hist": {"cube_red": positions}})
    result = judge(record, make_basic_scene(), "cube_red")
    assert not result.predicates["S1"] and result.predicates["S2"]
    assert result.details["dwell_containment"] == "False"


@pytest.mark.parametrize("missing", ["poses", "short_poses", "short_linear", "short_angular", "final_mismatch"])
def test_missing_or_misaligned_dwell_evidence_fails_with_detail(missing):
    record = stable_record()
    if missing == "poses":
        record = record.model_copy(update={"rest_pose_hist": {}})
    elif missing == "short_poses":
        record = record.model_copy(update={"rest_pose_hist": {"cube_red": record.rest_pose_hist["cube_red"][:-1]}})
    elif missing == "final_mismatch":
        record = record.model_copy(update={"final_poses": {**record.final_poses, "cube_red": Pose(position=(.46, .28, .02))}})
    else:
        key = "rest_lin_hist" if missing == "short_linear" else "rest_ang_hist"
        record = record.model_copy(update={key: {"cube_red": (0.,) * (DWELL - 1)}})
    result = judge(record, make_basic_scene(), "cube_red")
    assert not result.predicates["S1"]
    assert result.details["dwell_evidence_error"]


def test_stable_full_dwell_passes_and_legacy_record_fails_safely():
    record = stable_record()
    assert judge(record, make_basic_scene(), "cube_red").success
    old = record.model_dump()
    del old["rest_pose_hist"]
    legacy = ExecutionRecord.model_validate(old)
    assert legacy.rest_pose_hist == {}
    assert not judge(legacy, make_basic_scene(), "cube_red").predicates["S1"]


DIMENSIONS = [("object", i) for i in range(3)] + [("obstacle", i) for i in range(3)] + [("inner", i) for i in range(2)] + [("height", None), ("thickness", None)]


@pytest.mark.parametrize("kind,index", DIMENSIONS)
@pytest.mark.parametrize("bad", [0., -.01, math.nan, math.inf, -math.inf])
def test_scene_rejects_nonpositive_and_nonfinite_dimensions(kind, index, bad):
    data = make_basic_scene(with_obstacle=True).model_dump(mode="json")
    if kind == "object":
        data["objects"][0]["size"][index] = bad
    elif kind == "obstacle":
        data["obstacles"][0]["half_extents"][index] = bad
    elif kind == "inner":
        data["bin"]["inner_size_xy"][index] = bad
    else:
        data["bin"]["wall_height" if kind == "height" else "wall_thickness"] = bad
    with pytest.raises(ValidationError):
        Scene.model_validate(data)


@pytest.mark.parametrize("mass", [-.01, math.nan, math.inf, -math.inf])
def test_scene_rejects_negative_or_nonfinite_mass(mass):
    data = make_basic_scene().model_dump()
    data["objects"][0]["mass_kg"] = mass
    with pytest.raises(ValidationError):
        Scene.model_validate(data)


def test_scene_valid_defaults_zero_mass_and_cylinder_consistency():
    scene = make_basic_scene(with_obstacle=True)
    assert Scene.model_validate_json(scene.model_dump_json()) == scene
    data = scene.model_dump()
    data["objects"][0].update(shape="cylinder", mass_kg=0)
    assert Scene.model_validate(data).objects[0].mass_kg == 0
    data["objects"][0]["size"] = (.04, .05, .04)
    with pytest.raises(ValidationError, match="diameter"):
        Scene.model_validate(data)


@pytest.mark.parametrize("raw", ["VERDICT: pass", "VERDICT: pass\nREASON:", "VERDICT: pass\nREASON:   ", "VERDICT: pass\nREASON: ok\nREASON: other", "VERDICT: pass\nVERDICT: pass\nREASON: ok", "VERDICT: pass\nREASON: ok\nextra", "extra\nVERDICT: pass\nREASON: ok", "REASON: ok\nVERDICT: pass", "<think>VERDICT: pass\nREASON: incomplete"])
def test_cosmos_requires_exactly_two_nonempty_answer_lines(raw):
    assert parse_verdict(raw) == ("uncertain", "unparseable reply")


def test_cosmos_accepts_reasoning_value_case_and_surrounding_whitespace():
    assert parse_verdict("<think>analysis</think>\n  VERDICT: PASS  \n REASON: visible \n") == ("pass", "visible")


@pytest.mark.parametrize("stale", [False, True])
def test_simulation_errors_preserve_sanitized_nested_trace_and_serialization(stale):
    from vplan.repair import evaluate_pair
    task, events = generated(), []
    client = FakeClient([response(task.oracle.reference_plan)])
    agent = planner(client)
    agent._secret = "known-credential-123"
    def inner_failure():
        private_local = "do-not-capture-local-values"
        error = StaleApproval if stale else RuntimeError
        raise error("solver failed; known-credential-123; Bearer abcXYZ; api_key=another-secret")
    def simulate(plan):
        inner_failure()
    result = evaluate_pair(task, agent, evaluator=simulate, event_sink=events.append)
    evaluation = result.initial.evaluation
    assert evaluation.outcome == ("stale_approval" if stale else "sim_error")
    assert "solver failed" in evaluation.error_message
    assert "inner_failure" in evaluation.error_traceback and "test_review_fixes.py" in evaluation.error_traceback
    assert "line " in evaluation.error_traceback
    serialized = result.model_dump_json() + json.dumps(events)
    for secret in ("known-credential-123", "abcXYZ", "another-secret", "do-not-capture-local-values"):
        assert secret not in serialized
    restored = type(result).model_validate_json(result.model_dump_json())
    assert restored.initial.evaluation == evaluation
    assert "error_traceback" not in json.dumps(client.requests)


def test_error_diagnostics_redact_basic_auth_and_unlabelled_provider_keys():
    from vplan.repair import evaluate_pair
    task = generated()
    def simulate(plan):
        raise RuntimeError("solver failed: Authorization: Basic YWxhZGRpbjpvcGVuc2VzYW1l; sk-or-v1-aBc123secretToken")
    result = evaluate_pair(task, planner(FakeClient([response(task.oracle.reference_plan)])), evaluator=simulate)
    serialized = result.model_dump_json()
    assert "YWxhZGRpbjpvcGVuc2VzYW1l" not in serialized
    assert "sk-or-v1-aBc123secretToken" not in serialized
    assert "solver failed" in result.initial.evaluation.error_message


def test_old_evaluation_deserializes_without_diagnostics():
    from vplan.records import EvaluationResult
    result = EvaluationResult.model_validate({"outcome": "sim_error", "error_type": "RuntimeError"})
    assert result.error_message is None and result.error_traceback is None


@pytest.mark.parametrize("assignment,fragments", [
    ('password="correct horse battery staple"', ("correct", "horse", "battery", "staple")),
    ("secret='two words; punctuation, stays: private'", ("two", "words", "punctuation", "private")),
    ('{"password": "json secret; horse, battery: staple"}', ("json secret", "horse", "battery", "staple")),
    (r'api_key="credentialhead \"credentialquote\" credentialsuffix; credentialtail"',
     ("credentialhead", "credentialquote", "credentialsuffix", "credentialtail")),
    (r"access_token='credentialhead \'credentialquote\' credentialsuffix; credentialtail'",
     ("credentialhead", "credentialquote", "credentialsuffix", "credentialtail")),
    (r'{"secret": "json \"quoted\" suffix; tail"}', ("json", "quoted", "suffix", "tail")),
])
def test_simulation_diagnostics_redact_complete_quoted_credentials(assignment, fragments):
    from vplan.repair import evaluate_pair
    task, events = generated(), []
    def inner_failure():
        raise RuntimeError("solver failed: " + assignment + "; mesh unavailable")
    def simulate(plan):
        inner_failure()
    result = evaluate_pair(task, planner(FakeClient([response(task.oracle.reference_plan)])),
                           evaluator=simulate, event_sink=events.append)
    diagnostic = result.initial.evaluation
    assert "solver failed" in diagnostic.error_message and "mesh unavailable" in diagnostic.error_message
    assert "inner_failure" in diagnostic.error_traceback and "test_review_fixes.py" in diagnostic.error_traceback
    assert "[REDACTED]" in diagnostic.error_message and "[REDACTED]" in diagnostic.error_traceback
    evidence = [diagnostic.error_message, diagnostic.error_traceback,
                result.model_dump_json(), json.dumps(events)]
    for serialized in evidence:
        for fragment in fragments:
            assert fragment not in serialized


@pytest.mark.parametrize("assignment", [
    'password="truncated-secret with private suffix',
    "secret='truncated-secret with private suffix",
    '{"password": "truncated-secret with private suffix',
    "{'secret': 'truncated-secret with private suffix",
    r'api_key="truncated-secret \"escaped quote\" private suffix',
    r"access_token='truncated-secret \'escaped quote\' private suffix",
    '{"secret": "truncated-secret with trailing escape\\',
])
@pytest.mark.parametrize("following_line", ["", "\nmesh unavailable"])
def test_simulation_diagnostics_redact_truncated_quoted_credentials(assignment, following_line):
    from vplan.repair import evaluate_pair
    task, events = generated(), []
    def inner_failure():
        raise RuntimeError("solver failed: " + assignment + following_line)
    def simulate(plan):
        inner_failure()
    result = evaluate_pair(task, planner(FakeClient([response(task.oracle.reference_plan)])),
                           evaluator=simulate, event_sink=events.append)
    diagnostic = result.initial.evaluation
    assert "solver failed" in diagnostic.error_message and "[REDACTED]" in diagnostic.error_message
    assert "inner_failure" in diagnostic.error_traceback and "test_review_fixes.py" in diagnostic.error_traceback
    if following_line:
        assert "mesh unavailable" in diagnostic.error_message and "mesh unavailable" in diagnostic.error_traceback
    evidence = [diagnostic.error_message, diagnostic.error_traceback,
                result.model_dump_json(), json.dumps(events)]
    for serialized in evidence:
        for fragment in ("truncated-secret", "private suffix", "escaped quote", "trailing escape"):
            assert fragment not in serialized
