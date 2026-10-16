"""Stateless OpenAI-compatible planner. One invocation is exactly one provider request."""
from __future__ import annotations

import json
import math
import re
import threading
import time
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from vplan.records import (PLAN_ADAPTER, PlannerCall, PlannerFeedback, PlannerProtocol,
                           SubmitPlanArgs, TaskSpec)

SYSTEM_PROMPT = """Plan a single robot pick-and-place task using only the supplied task and feedback.
Submit exactly one submit_plan with {"plan": <pick_place or refusal>}. Do not call any other tool.
World frame equals robot base frame, z up, positions in meters, yaw in degrees.
For top grasps yaw is the gripper closing-axis heading counterclockwise from world +x about +z.
For side grasps yaw rotates the closing axis about the approach axis, measured from world +z.
place.position is the object's resting center: z = bin floor top + half the object height.
release_height_m is the additional drop height; prefer 0.02–0.10 m because zero is legal but brittle.
Between grasp and release the gripper turns
about world +z by place.yaw_deg minus grasp.yaw_deg. transport_via holds up to three world points.
robot_facts.reach_bound_m is the maximum gripper-center distance from the shoulder at (0, 0, 0.333), not from the base.
Use only object IDs in the scene. Feedback reports violated constraints, not suggested fixes.
If no valid plan can be found, return the refusal schema. Rationale is for logging only."""


def _sanitize(value: Any, secret: str | None = None) -> Any:
    """Keep provider payloads, but never persist auth fields or known credential strings."""
    if isinstance(value, dict):
        return {str(k): "[REDACTED]" if str(k).lower().replace("-", "_") in {
            "authorization", "proxy_authorization", "api_key", "x_api_key", "access_token"}
            else _sanitize(v, secret) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v, secret) for v in value]
    if isinstance(value, str):
        if secret:
            value = value.replace(secret, "[REDACTED]")
        return re.sub(r"(?i)\bbearer\s+[^\s\"'<>]+", "Bearer [REDACTED]", value)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return {"unserializable_type": type(value).__name__}


def _message(raw: dict) -> dict:
    choices = raw.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("expected exactly one response choice object")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("expected response message object")
    return message


def _no_tools(raw: dict) -> bool:
    try:
        calls = _message(raw).get("tool_calls")
    except ValueError:
        return False
    return calls is None or calls == []


def _decode_string_plan(payload: Any) -> Any:
    """D64: Nemotron Super sends {"plan": "<the plan object as a JSON string>"}. Decode that string once when it
    holds a JSON object; anything else is returned unchanged for validation to reject."""
    if isinstance(payload, dict) and isinstance(payload.get("plan"), str):
        try:
            plan = json.loads(payload["plan"])
        except (ValueError, RecursionError):
            return payload
        if isinstance(plan, dict):
            return {**payload, "plan": plan}
    return payload


def _parse(raw: dict):
    message = _message(raw)
    calls = message.get("tool_calls")
    if calls is None:
        calls = []
    if not isinstance(calls, list):
        raise ValueError("expected tool call array")
    if calls:
        if len(calls) != 1 or not isinstance(calls[0], dict) or calls[0].get("type") != "function":
            raise ValueError("expected exactly one function call")
        function = calls[0].get("function", {})
        if not isinstance(function, dict) or function.get("name") != "submit_plan":
            raise ValueError("unknown tool name")
        arguments = function.get("arguments", "")
        if isinstance(arguments, str):
            try:
                loaded = json.loads(arguments)
            except (ValueError, RecursionError):
                loaded = None  # left for model_validate_json to report, as before
            decoded = _decode_string_plan(loaded)
            if decoded is not loaded:  # only the string-plan case is re-encoded; every other reply validates as sent
                arguments = json.dumps(decoded)
        return SubmitPlanArgs.model_validate_json(arguments).plan
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("empty response content")
    payload = _decode_string_plan(json.loads(content))
    return SubmitPlanArgs.model_validate(payload).plan if isinstance(payload, dict) and "plan" in payload else PLAN_ADAPTER.validate_python(payload)


def _response_call(response, common: dict, secret: str | None) -> PlannerCall:
    """Guard the complete received-response boundary, excluding persistence callbacks."""
    raw = None
    metadata = dict(response_model=None, prompt_tokens=None, completion_tokens=None, total_tokens=None)
    try:
        if hasattr(response, "model_dump"):
            response = response.model_dump(mode="json")
        sanitized = _sanitize(response, secret)
        raw = sanitized if isinstance(sanitized, dict) else {"provider_response": sanitized}
        if not isinstance(sanitized, dict):
            raise ValueError("response must be an object")
        provider_error = raw.get("error")
        if provider_error and not raw.get("choices"):
            # OpenRouter can return HTTP 200 with an error body (e.g. upstream 503); that is the provider failing, not the model.
            code = provider_error.get("code") if isinstance(provider_error, dict) else None
            return PlannerCall(**common, **metadata, raw_response=raw, outcome="service_failure",
                               error_type="provider_error_body", status_code=code if type(code) is int else None)
        errors = []
        model = raw.get("model")
        if model is None or isinstance(model, str):
            metadata["response_model"] = model
        else:
            errors.append({"type": "invalid_response_metadata", "field": "model"})
        usage = raw.get("usage")
        if usage is None:
            usage = {}
        if not isinstance(usage, dict):
            errors.append({"type": "invalid_response_metadata", "field": "usage"})
            usage = {}
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(field)
            if value is None or type(value) is int and value >= 0:
                metadata[field] = value
            else:
                errors.append({"type": "invalid_response_metadata", "field": f"usage.{field}"})
        if errors:
            return PlannerCall(**common, **metadata, raw_response=raw, outcome="malformed", errors=tuple(errors))
        try:
            proposal = _parse(raw)
        except ValidationError as exc:
            return PlannerCall(**common, **metadata, raw_response=raw, outcome="malformed",
                errors=tuple(exc.errors(include_url=False, include_context=False, include_input=False)))
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            fallback = common["protocol"] != "content" and _no_tools(raw)
            if fallback:
                common = {**common, "next_protocol": "required" if common["protocol"] == "forced" else "content"}
            return PlannerCall(**common, **metadata, raw_response=raw,
                outcome="protocol_error" if fallback else "malformed", errors=({"type": type(exc).__name__,
                "message": "invalid submit_plan response structure or JSON"},))
        return PlannerCall(**common, **metadata, raw_response=raw, outcome="proposed", plan=proposal)
    except Exception as exc:
        # Neither provider payload shape nor SDK serialization may erase a spent request.
        return PlannerCall(**common, **metadata, raw_response=raw, outcome="malformed",
            errors=({"type": type(exc).__name__, "message": "invalid provider response envelope or metadata"},))


# Spec §10 per-call timeout, as wall-clock time for the whole call (D63). The SDK timeout below only bounds each wait
# for data, and OpenRouter keeps a slow request alive, so calls of 178-225 s passed it.
CALL_DEADLINE_S = 60.0


class CallDeadlineExceeded(TimeoutError):
    """No complete response within the call deadline; the request is abandoned, never retried."""


def _within_deadline(call: Callable[[], Any], deadline_s: float) -> Any:
    """Run call on a daemon thread and give up after deadline_s. The abandoned request finishes or times out on its
    own and its result is dropped; a daemon thread never holds up the worker's exit."""
    outcome: dict[str, Any] = {}
    finished = threading.Event()

    def run() -> None:
        try:
            outcome["response"] = call()
        except BaseException as exc:  # relayed to the caller, which classifies it as before
            outcome["error"] = exc
        finally:
            finished.set()

    threading.Thread(target=run, name="planner-call", daemon=True).start()
    if not finished.wait(deadline_s):
        raise CallDeadlineExceeded(f"no complete provider response within {deadline_s:g} s")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["response"]


class NemotronPlanner:
    def __init__(self, *, model: str, provider: str = "openrouter", base_url: str | None = None,
                 api_key: str | None = None, client=None, temperature: float = .2,
                 call_deadline_s: float = CALL_DEADLINE_S):
        if type(call_deadline_s) not in (int, float) or not 0 < call_deadline_s <= threading.TIMEOUT_MAX:
            raise ValueError("call deadline must be a positive, finite number of seconds")
        if client is None:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0, timeout=60.0)
        # Enforce this even on an injected real SDK client configured with its default retries.
        self._client = client.with_options(max_retries=0, timeout=60.0)
        self._secret = api_key
        self.model, self.provider, self.temperature = model, provider, temperature
        self.call_deadline_s = float(call_deadline_s)

    def plan(self, task: TaskSpec, *, feedback: PlannerFeedback | None = None,
             protocol: PlannerProtocol = "forced", event_sink: Callable[[dict], None] | None = None) -> PlannerCall:
        if type(task) is not TaskSpec:
            raise TypeError("planner accepts only TaskSpec, never a private task envelope")
        if protocol not in ("forced", "required", "content"):
            raise ValueError("unknown planner protocol")
        payload = {"task": task.model_dump(mode="json")}
        if feedback is not None:
            payload["feedback"] = feedback.model_dump(mode="json")
        system = SYSTEM_PROMPT
        if protocol == "content":
            system = system.replace('Submit exactly one submit_plan with {"plan": <pick_place or refusal>}. Do not call any other tool.',
                                    'Return one JSON object {"plan": <pick_place or refusal>} in message content. Do not call tools.')
            system += "\nReturn only a JSON object matching this schema: " + json.dumps(SubmitPlanArgs.model_json_schema())
        request = {"model": self.model, "temperature": self.temperature, "messages": [
            {"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload)}]}
        if protocol != "content":
            request["tools"] = [{"type": "function", "function": {
                "name": "submit_plan", "description": "Submit one manipulation plan or refusal.",
                "parameters": SubmitPlanArgs.model_json_schema()}}]
            request["tool_choice"] = "required" if protocol == "required" else {
                "type": "function", "function": {"name": "submit_plan"}}
        raw_request = _sanitize(request, self._secret)
        if event_sink is not None:
            event_sink({"event": "request_started", "raw_request": raw_request, "protocol": protocol})
        start = time.perf_counter()
        common = dict(provider=self.provider, model=self.model, protocol=protocol,
                      next_protocol=protocol, temperature=self.temperature, raw_request=raw_request)
        try:
            response = _within_deadline(lambda: self._client.chat.completions.create(**request), self.call_deadline_s)
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            status = status if isinstance(status, int) else None
            # Unsupported tool choice is a logged request, not an uncounted internal retry.
            description = str(exc).lower()
            unsupported = status in (400, 422) and any(s in description for s in ("tool", "function"))
            fallback = unsupported and protocol != "content"
            if fallback:
                common["next_protocol"] = "required" if protocol == "forced" else "content"
            result = PlannerCall(**common, outcome="protocol_error" if fallback else "service_failure",
                                 latency_s=time.perf_counter() - start, error_type=type(exc).__name__, status_code=status)
        else:
            common["latency_s"] = time.perf_counter() - start
            result = _response_call(response, common, self._secret)
        if event_sink is not None:
            event_sink({"event": "request_completed", "call": result.model_dump(mode="json")})
        return result
