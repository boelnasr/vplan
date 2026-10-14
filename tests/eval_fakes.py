"""Test-only worker module for the W3.1 runner, loaded with --import: no request ever leaves the machine.

openai.OpenAI becomes a fake client whose every reply is a refusal; the reply id echoes the API key it was built with,
so a test can prove the key is redacted before anything reaches the run directory.
"""
from __future__ import annotations

import openai

from tests.test_planner import FakeClient, response
from vplan.records import Refusal


def _fake_openai(*, api_key=None, **kwargs):
    reply = {**response(Refusal(reason="other", explanation="fake provider")), "id": f"reply {api_key}"}
    return FakeClient(iter(lambda: reply, None))


openai.OpenAI = _fake_openai
