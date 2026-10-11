"""Offline provider boundary tests: no credentials and no network."""
import json
from types import SimpleNamespace

import pytest

from vplan.planners.heuristic import heuristic_geometry
from vplan.records import Finding, SubmitPlanArgs
from vplan.tasks import generate_candidate


class FakeClient:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []
        self.options = []
        self.chat = SimpleNamespace(completions=self)

    def with_options(self, **kwargs):
        self.options.append(kwargs)
        return self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return reply


def response(plan=None, *, content=None, name="submit_plan", arguments=None):
    calls = [] if plan is None and arguments is None else [
        {"id": "call_1", "type": "function", "function": {"name": name,
         "arguments": arguments if arguments is not None else SubmitPlanArgs(plan=plan).model_dump_json()}}]
    return {"id": "reply_1", "model": "actual-provider-model", "choices": [{"index": 0,
            "finish_reason": "tool_calls" if calls else "stop", "message": {
                "role": "assistant", "content": content, "tool_calls": calls}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}


def planner(client):
    from vplan.planners.nemotron import NemotronPlanner
    return NemotronPlanner(model="free-model", provider="openrouter", client=client)


def fixture():
    candidate = generate_candidate(2, "obstacle")
    return candidate.task, heuristic_geometry(candidate.task.scene, candidate.target_id)


def test_forced_tool_schema_metadata_and_no_hidden_retries():
    task, plan = fixture()
    client = FakeClient([response(plan)])
    result = planner(client).plan(task)
    assert result.plan == plan and result.outcome == "proposed"
    assert len(client.requests) == 1 and client.options == [{"max_retries": 0, "timeout": 60.0}]
    request = client.requests[0]
    assert request["tool_choice"] == {"type": "function", "function": {"name": "submit_plan"}}
    assert request["tools"][0]["function"]["parameters"] == SubmitPlanArgs.model_json_schema()
    assert result.model == "free-model" and result.response_model == "actual-provider-model"
    assert result.provider == "openrouter" and result.total_tokens == 120 and result.latency_s >= 0
    payload = json.loads(request["messages"][1]["content"])
    assert payload == {"task": task.model_dump(mode="json")}
    assert "resting" in request["messages"][0]["content"]
    assert "counterclockwise" in request["messages"][0]["content"]
    assert "structured_outputs" not in json.dumps(request)


@pytest.mark.parametrize("bad", [
    {"choices": []}, response(content=""), response(arguments="not json"),
    response(arguments='{"plan":{"kind":"pick_place"}}'),
    response(arguments='{}', name="run_shell"),
])
def test_rejects_malformed_outputs_before_simulation(bad):
    task, _ = fixture()
    result = planner(FakeClient([bad])).plan(task, protocol="content")
    assert result.outcome == "malformed" and result.plan is None and result.errors


def test_rejects_multiple_tool_calls_even_when_content_is_valid():
    task, plan = fixture()
    reply = response(plan, content=SubmitPlanArgs(plan=plan).model_dump_json())
    reply["choices"][0]["message"]["tool_calls"] *= 2
    assert planner(FakeClient([reply])).plan(task).outcome == "malformed"


def test_required_and_content_fallback_each_make_one_request():
    task, plan = fixture()
    client = FakeClient([response(content=SubmitPlanArgs(plan=plan).model_dump_json()), response(content=plan.model_dump_json())])
    adapter = planner(client)
    assert adapter.plan(task, protocol="required").plan == plan
    assert adapter.plan(task, protocol="content").plan == plan
    assert client.requests[0]["tool_choice"] == "required"
    assert "tools" not in client.requests[1] and len(client.requests) == 2


def test_feedback_is_stateless_and_limited_without_private_data():
    from vplan.records import PlannerFeedback
    task, plan = fixture()
    finding = Finding(stage="transport", check="collision", plan_field="transport_via", measured=0,
                      threshold=.01, direction="below margin")
    client = FakeClient([response(plan), response(plan)])
    adapter = planner(client)
    adapter.plan(task, feedback=PlannerFeedback(previous_plan=plan, findings=(finding,), summaries=("round 0: blocked",)))
    adapter.plan(task)
    first = json.loads(client.requests[0]["messages"][1]["content"])
    second = json.loads(client.requests[1]["messages"][1]["content"])
    assert first["feedback"]["findings"][0]["check"] == "collision"
    assert second == {"task": task.model_dump(mode="json")}
    with pytest.raises(TypeError, match="TaskSpec"):
        adapter.plan(generate_candidate(2, "clear"))


def test_service_error_is_sanitized_and_no_retry():
    task, _ = fixture()
    client = FakeClient([RuntimeError("Authorization: Bearer secret-token")])
    result = planner(client).plan(task)
    assert result.outcome == "service_failure" and len(client.requests) == 1
    assert "secret-token" not in result.model_dump_json()
    assert result.total_tokens is None


@pytest.mark.parametrize("code, status", [(503, 503), (402, 402), ("overloaded", None)])
def test_http_200_error_body_is_a_service_failure_not_malformed(code, status):
    # Observed on OpenRouter: HTTP 200, choices null, error {code: 503, "provider_overloaded"} (D33 M9).
    task, _ = fixture()
    body = {"id": "gen-1", "choices": None, "model": None, "usage": None,
            "error": {"message": "Upstream error from Nvidia: Service temporarily overloaded", "code": code,
                      "metadata": {"error_type": "provider_overloaded"}}}
    result = planner(FakeClient([body])).plan(task)
    assert result.outcome == "service_failure" and result.status_code == status
    assert result.next_protocol == "forced" and result.raw_response["error"]["code"] == code


def test_unsupported_protocol_is_an_outcome_with_next_budgeted_format():
    task, _ = fixture()
    error = RuntimeError("tools unsupported")
    error.status_code = 400
    client = FakeClient([error])
    result = planner(client).plan(task)
    assert result.outcome == "protocol_error" and result.next_protocol == "required"
    assert len(client.requests) == 1


def test_empty_protocol_response_advances_fallback_but_bad_tool_arguments_do_not():
    task, _ = fixture()
    client = FakeClient([response(content="not usable JSON"), response(content=""), response(arguments="{}")])
    adapter = planner(client)
    forced = adapter.plan(task)
    required = adapter.plan(task, protocol="required")
    malformed = adapter.plan(task)
    assert forced.outcome == required.outcome == "protocol_error"
    assert forced.next_protocol == "required" and required.next_protocol == "content"
    assert malformed.outcome == "malformed" and malformed.next_protocol == "forced"


def test_auth_echoes_are_redacted_from_all_serializable_evidence():
    from vplan.planners.nemotron import NemotronPlanner
    task, plan = fixture()
    reply = response(plan.model_copy(update={"rationale": "key-is-test-secret Authorization: Bearer token-value"}))
    reply["authorization"] = "token-value"
    client = FakeClient([reply])
    events = []
    result = NemotronPlanner(model="free-model", api_key="key-is-test-secret", client=client).plan(task, event_sink=events.append)
    evidence = result.model_dump_json() + json.dumps(events)
    assert "key-is-test-secret" not in evidence and "token-value" not in evidence


def test_event_sink_failure_aborts_before_request_and_propagates_after_response():
    task, plan = fixture()
    client = FakeClient([response(plan)])
    def broken(event):
        raise OSError("evidence disk full")
    with pytest.raises(OSError, match="disk full"):
        planner(client).plan(task, event_sink=broken)
    assert not client.requests
    def after(event):
        if event["event"] == "request_completed":
            raise OSError("evidence disk full")
    with pytest.raises(OSError, match="disk full"):
        planner(client).plan(task, event_sink=after)
    assert len(client.requests) == 1


@pytest.mark.parametrize("patch", [
    {"choices": [None]}, {"choices": [{"message": None}]},
    {"choices": [{"message": {"tool_calls": [None]}}]},
    {"choices": [{"message": {"tool_calls": [{"type": "function", "function": None}]}}]},
    {"usage": "unusable"}, {"usage": [1]},
    {"usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": -1}},
    {"usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": "unknown"}},
    {"usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": True}},
    {"usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 12.5}},
    {"model": ["model"]}, {"model": 123},
])
def test_invalid_provider_envelope_or_metadata_is_logged_without_escaping(patch):
    task, plan = fixture()
    reply = {**response(plan), **patch, "authorization": "private-credential"}
    events = []
    result = planner(FakeClient([reply])).plan(task, event_sink=events.append)
    assert result.outcome == "malformed" and result.plan is None and result.errors
    assert [e["event"] for e in events] == ["request_started", "request_completed"]
    assert events[-1]["call"] == result.model_dump(mode="json")
    assert result.raw_response["authorization"] == "[REDACTED]"
    assert "private-credential" not in json.dumps(events)
    if "usage" in patch:
        assert result.total_tokens is None
        if isinstance(patch["usage"], dict):
            assert result.prompt_tokens == 100 and result.completion_tokens == 20
        else:
            assert result.prompt_tokens is None and result.completion_tokens is None
    else:
        assert result.total_tokens == 120
    if "model" in patch:
        assert result.response_model is None


@pytest.mark.parametrize("reply", [None, [], "bad envelope"])
def test_nonobject_provider_response_is_retained_as_logged_malformed(reply):
    task, _ = fixture()
    events = []
    result = planner(FakeClient([reply])).plan(task, event_sink=events.append)
    assert result.outcome == "malformed" and result.total_tokens is None
    assert result.raw_response == {"provider_response": reply}
    assert events[-1]["event"] == "request_completed"


def test_sdk_response_dump_error_still_completes_evidence_and_sink_failure_propagates():
    class BadDump:
        def model_dump(self, **kwargs):
            raise ValueError("Authorization: Bearer secret")
    task, _ = fixture()
    result = planner(FakeClient([BadDump()])).plan(task)
    assert result.outcome == "malformed" and "secret" not in result.model_dump_json()
    def broken(event):
        if event["event"] == "request_completed":
            raise OSError("evidence disk full")
    with pytest.raises(OSError, match="disk full"):
        planner(FakeClient([{"choices": [None]}])).plan(task, event_sink=broken)


def test_system_prompt_states_the_reach_center_matching_the_task_generator():
    """D48: reach_bound_m is measured from the shoulder (D38), so the planner must be told where that is."""
    from vplan.planners.nemotron import SYSTEM_PROMPT
    from vplan.tasks.reach import SHOULDER
    assert f"shoulder at ({SHOULDER[0]:g}, {SHOULDER[1]:g}, {SHOULDER[2]:g})" in SYSTEM_PROMPT
