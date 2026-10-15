"""Offline CLI outage reporting: the user sees the fallback before motion can start."""
import io

import pytest

from tests.test_repair import generated
from tests.test_visual_gate import FakeAsk, fake_render
from vplan import cli, pipeline, records
from vplan.visual import gate as visual_gate
from vplan.visual.cosmos import CosmosUnavailable


def test_cosmos_outage_warning_is_flushed_before_execution(monkeypatch):
    class Output(io.StringIO):
        flushed = ""

        def flush(self):
            self.flushed = self.getvalue()

    class StopBeforeMotion(Exception):
        pass

    output = Output()
    monkeypatch.setattr("sys.stdout", output)
    real_gate = visual_gate.CosmosGate
    monkeypatch.setattr(visual_gate, "CosmosGate", lambda url, **kw: real_gate(
        url, ask=FakeAsk([CosmosUnavailable("offline")]), render=fake_render, **kw))

    def before_motion(*args, **kwargs):
        assert "COSMOS UNAVAILABLE" in output.flushed
        assert "geometric checks only" in output.flushed
        assert "offline" in output.flushed
        assert "executed" not in output.flushed
        raise StopBeforeMotion

    monkeypatch.setattr(pipeline, "execute", before_motion)
    task = generated()
    gate = cli._cosmos_gate("http://cosmos.invalid/v1", "red cube")
    with pytest.raises(StopBeforeMotion):
        pipeline.run_attempt(task.oracle.reference_plan, task.task.scene, task.oracle.target_id, visual_gate=gate)


@pytest.mark.parametrize("outcome", ["stale_approval", "sim_error"])
def test_outage_result_does_not_claim_execution_when_execution_failed(capsys, outcome):
    result = pipeline.AttemptResult(plan=generated().oracle.reference_plan, outcome=outcome,
        visual=records.VisualGateReport(status="unavailable", note="offline"))
    cli._print_result(result)
    text = capsys.readouterr().out
    assert outcome in text
    assert "unavailable" in text.lower()
    assert "executed" not in text
