"""Offline provider boundary tests: no credentials and no network."""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from vplan.planners.heuristic import heuristic_geometry
from vplan.records import Finding, Refusal, SubmitPlanArgs
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
    from vplan.planners.nemotron import plan_schema
    assert request["tools"][0]["function"]["parameters"] == plan_schema()
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


@pytest.mark.parametrize("protocol", ["forced", "content"])
@pytest.mark.parametrize("refusal", [False, True])
def test_a_plan_sent_as_a_json_string_is_decoded_once(protocol, refusal):
    # D64: Nemotron Super sends {"plan": "<the plan object as a JSON string>"} in every reply, also after the
    # schema retry. The evidence keeps the reply as sent.
    task, plan = fixture()
    if refusal:
        plan = Refusal(reason="target_unreachable", explanation="beyond the reach bound")
    wrapped = json.dumps({"plan": plan.model_dump_json()})
    reply = response(content=wrapped) if protocol == "content" else response(arguments=wrapped)
    result = planner(FakeClient([reply])).plan(task, protocol=protocol)
    assert result.outcome == "proposed" and result.plan == plan
    message = result.raw_response["choices"][0]["message"]
    sent = message["content"] if protocol == "content" else message["tool_calls"][0]["function"]["arguments"]
    assert sent == wrapped


def _schema_invalid(plan):
    return {**plan.model_dump(mode="json"), "grasp": "top"}


@pytest.mark.parametrize("protocol", ["forced", "content"])
@pytest.mark.parametrize("inner", [
    lambda plan: "not json", lambda plan: json.dumps("a string"), lambda plan: json.dumps([1, 2]),
    lambda plan: json.dumps(plan.model_dump_json()),  # encoded twice: decoding stops after one level
    lambda plan: json.dumps(_schema_invalid(plan)),  # decodes to an object that still fails the schema
])
def test_a_string_plan_that_is_not_one_valid_object_stays_malformed(protocol, inner):
    task, plan = fixture()
    wrapped = json.dumps({"plan": inner(plan)})
    reply = response(content=wrapped) if protocol == "content" else response(arguments=wrapped)
    result = planner(FakeClient([reply])).plan(task, protocol=protocol)
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


class StallingClient(FakeClient):
    """A provider that never finishes until released, like OpenRouter keeping a slow request alive (D63)."""
    def __init__(self, reply):
        super().__init__([reply])
        self.release = threading.Event()

    def create(self, **kwargs):
        self.requests.append(kwargs)
        self.release.wait(10)
        return next(self.replies)


def test_a_call_that_outlives_the_deadline_is_a_service_failure_and_a_late_reply_changes_nothing():
    # D63: the SDK timeout bounds each wait for data, so a 225 s call once passed a 60 s timeout and its attempt
    # died at the 600 s worker limit as a sim_error. The deadline bounds the whole call.
    from vplan.planners.nemotron import NemotronPlanner
    task, plan = fixture()
    client = StallingClient(response(plan))
    events = []
    try:
        result = NemotronPlanner(model="free-model", client=client, call_deadline_s=0.2).plan(task, event_sink=events.append)
    finally:
        client.release.set()
    assert result.outcome == "service_failure" and result.error_type == "CallDeadlineExceeded"
    assert result.status_code is None and result.next_protocol == "forced" and result.plan is None
    assert 0.2 <= result.latency_s < 5 and len(client.requests) == 1
    time.sleep(0.1)  # the abandoned request completes in the background; nothing reaches the evidence
    assert [e["event"] for e in events] == ["request_started", "request_completed"]
    assert events[1]["call"]["outcome"] == "service_failure"


def test_an_abandoned_call_does_not_hold_up_the_process_exit():
    # The eval worker is one subprocess per attempt; a stalled request must not keep it alive to the 600 s kill.
    code = ("import time; from vplan.planners.nemotron import _within_deadline, CallDeadlineExceeded\n"
            "try:\n    _within_deadline(lambda: time.sleep(60), 0.2)\nexcept CallDeadlineExceeded:\n    print('deadline')\n")
    start = time.perf_counter()
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30,
                          cwd=Path(__file__).resolve().parents[1],
                          env={**os.environ, "PYTHONPATH": os.pathsep.join(["src", "."])})
    assert done.returncode == 0 and done.stdout.strip() == "deadline", done.stderr
    assert time.perf_counter() - start < 20


def test_a_call_inside_the_deadline_is_unchanged():
    from vplan.planners.nemotron import NemotronPlanner
    task, plan = fixture()
    client = StallingClient(response(plan))
    client.release.set()
    result = NemotronPlanner(model="free-model", client=client, call_deadline_s=5).plan(task)
    assert result.outcome == "proposed" and result.plan == plan


def test_the_default_deadline_is_the_spec_per_call_timeout():
    from vplan.planners import nemotron
    assert nemotron.CALL_DEADLINE_S == 60.0
    assert nemotron.NemotronPlanner(model="m", client=FakeClient([])).call_deadline_s == 60.0
    assert "per-call timeout 60 s" in (Path(__file__).resolve().parents[1] / "docs" / "spec.md").read_text("utf-8")


def test_the_schema_the_model_sees_requires_each_plans_kind():
    # D67: pydantic leaves the defaulted `kind` out of `required`, so the schema told the model it could omit the tag
    # the parser needs; Nemotron Super did, and its replies ended malformed. Only the required lists change.
    from vplan.planners.nemotron import plan_schema
    schema, original = plan_schema(), SubmitPlanArgs.model_json_schema()
    targets = schema["properties"]["plan"]["discriminator"]["mapping"].values()
    names = sorted(ref.rsplit("/", 1)[-1] for ref in targets)
    assert names == ["PickPlacePlan", "Refusal"]
    for name in names:
        assert "kind" in schema["$defs"][name]["required"]
        assert "kind" not in original["$defs"][name].get("required", [])
        original["$defs"][name]["required"] = ["kind", *original["$defs"][name].get("required", [])]
    assert schema == original
    assert plan_schema() is not plan_schema()  # a fresh copy per request: nothing can mutate a shared schema


@pytest.mark.parametrize("protocol", ["forced", "required", "content"])
def test_every_protocol_sends_the_kind_required_schema(protocol):
    from vplan.planners.nemotron import plan_schema
    task, plan = fixture()
    reply = response(content=SubmitPlanArgs(plan=plan).model_dump_json()) if protocol == "content" else response(plan)
    client = FakeClient([reply])
    planner(client).plan(task, protocol=protocol)
    request = client.requests[0]
    if protocol == "content":
        assert json.dumps(plan_schema()) in request["messages"][0]["content"] and "tools" not in request
    else:
        assert request["tools"][0]["function"]["parameters"] == plan_schema()


def test_a_longer_deadline_also_lengthens_the_sdk_read_timeout():
    # D65: a server silent until the reply is done (GMI, AIHubMix) would otherwise fail at the SDK's 60 s first.
    from vplan.planners.nemotron import NemotronPlanner
    for deadline, read_timeout in ((180, 180.0), (30, 60.0)):
        client = FakeClient([])
        NemotronPlanner(model="m", client=client, call_deadline_s=deadline)
        assert client.options == [{"max_retries": 0, "timeout": read_timeout}]


@pytest.mark.parametrize("bad", [0, -1.0, float("nan"), float("inf"), True, threading.TIMEOUT_MAX * 2])
def test_an_unusable_deadline_is_refused(bad):
    from vplan.planners.nemotron import NemotronPlanner
    with pytest.raises(ValueError, match="deadline"):
        NemotronPlanner(model="m", client=FakeClient([]), call_deadline_s=bad)


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
