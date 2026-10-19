"""Config-4 Cosmos shadow study (spec §11 "Shadow study", §14): `vplan eval shadow` over a W3.1 run's saved key frames.

One real executed attempt (the heuristic plan for cube_red on the basic scene with its obstacle; cube_green is the only
other object) is simulated once per module and its key frames exported as `eval run` exports them. Cosmos is faked
(a scripted ask_cosmos); the seed set is built in memory, so only the shadow study's own loading is exercised.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from tests import eval_rows as ev
from vplan import records
from vplan.config import config_snapshot
from vplan.evaluate.seedsets import EvalError, SeedSet
from vplan.visual.cosmos import CosmosUnavailable

SEED, INDEX = 7, 4
ATTEMPT = f"3:dev:{INDEX}"
SPEC = ev.SPEC.model_copy(update={"instruction": "Pick the red cube and place it in the bin."})  # a template phrase
HASH = "cd" * 32
STAGES = ("pregrasp", "grasp", "prerelease")
GATE_VIEWS, RETRY_VIEWS = ("front_oblique", "top"), ("front_oblique", "top", "side")
# (stage, source) of every question about the executed round, in the order they are asked
ORDER = [("pregrasp", "natural"), ("grasp", "natural"), ("prerelease", "natural"),
         ("pregrasp", "wrong_object"), ("grasp", "wrong_object"), ("prerelease", "wrong_object"),
         ("grasp", "misaligned_grasp")]


def verdict(v: str, latency_s: float = 0.1) -> records.VisualVerdict:
    return records.VisualVerdict(verdict=v, reason="r", latency_s=latency_s, raw=f"VERDICT: {v}\nREASON: r")


class FakeAsk:
    """Scripted ask_cosmos: one reply (a verdict name or an exception) per request, in order; records each request."""

    def __init__(self, replies):
        self.replies, self.sent = list(replies), []

    def __call__(self, images, question, base_url, **kwargs):
        self.sent.append({"views": tuple(images), "images": dict(images), "question": question,
                          "base_url": base_url, **kwargs})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return verdict(reply)


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    """A W3.1-style config 3 run directory: task 0 refused (no frames), task 4 executed with its 9 key frames."""
    from vplan.evaluate.frames import export_frames
    from vplan.planners.heuristic import HeuristicPlanner
    from vplan.repair import run_task
    executed = run_task(SPEC, ev.FEASIBLE, HeuristicPlanner(), k=3, mode="repair", attempt_id=ATTEMPT, seed=SEED,
                        config_snapshot=config_snapshot())
    assert executed.outcome == "executed" and executed.success
    run = tmp_path_factory.mktemp("shadow") / "c3-dev"
    frames = export_frames(executed, SPEC, seed=SEED, config_snapshot=config_snapshot(),
                           out_dir=run / "frames" / f"{INDEX:04d}", rel_dir=f"frames/{INDEX:04d}")
    assert frames.status == "written"
    refused = ev.attempt(ev.impossible(1), [ev.call(ev.refusal())], attempt_id="3:dev:0")
    rows = [records.EvalAttempt(attempt_id="3:dev:0", config=3, split="dev", task_index=0, task_id="dev_0000",
                                session=0, outcome="refused", tries=1, wall_s=1.0, record=refused,
                                frames=records.FrameExport(status="none")),
            records.EvalAttempt(attempt_id=ATTEMPT, config=3, split="dev", task_index=INDEX, task_id=f"dev_{INDEX:04d}",
                                session=0, outcome="executed", tries=1, wall_s=5.0, record=executed, frames=frames)]
    (run / "attempts.jsonl").write_text("".join(r.model_dump_json() + "\n" for r in rows), encoding="utf-8")
    manifest = {"schema_version": 1, "kind": "eval_run", "run_id": "c3-dev", "config": 3, "split": "dev",
                "seed_set_hash": HASH, "provider": "openrouter", "model": "m", "n": 5, "n_prepared": 5,
                "frames": True, "seed_set": {"dir": "seeds-not-used"}}
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run


def seed_set(seed_set_hash: str = HASH) -> SeedSet:
    tasks = tuple(records.SeedSetTask(task_id=f"dev_{i:04d}", index=i, seed=SEED if i == INDEX else i, split="dev",
                                      distractors=1, obstacles=1, task=SPEC) for i in range(5))
    return SeedSet(path=Path("seeds"), split="dev", tasks=tasks, oracles=(ev.impossible(1),) + (ev.FEASIBLE,) * 4,
                   manifest={}, manifest_sha256="", seed_set_hash=seed_set_hash, config=config_snapshot(),
                   source_hashes={}, drift=[])


@pytest.fixture(autouse=True)
def seeds(monkeypatch):
    from vplan.evaluate import shadow
    loaded = []
    monkeypatch.setattr(shadow, "load_seed_set", lambda path, **kw: loaded.append((path, kw)) or seed_set())
    return loaded


def shadow_run(run_dir, out, replies, **kw):
    from vplan.evaluate import shadow
    ask = FakeAsk(replies)
    result = shadow.run_shadow(run_dir, out, base_url="http://cosmos.invalid/v1", ask=ask, echo=lambda *a: None, **kw)
    return result, ask


def lines(out) -> list[records.ShadowRecord]:
    return [records.ShadowRecord.model_validate_json(line)
            for line in (out / "shadow.jsonl").read_text(encoding="utf-8").splitlines()]


REPLIES = ["pass",                    # natural pre-grasp
           "uncertain", "fail",       # natural grasp: asked again with the side view, then a false rejection
           "pass",                    # natural pre-release
           "fail",                    # wrong-object pre-grasp: caught
           "pass",                    # wrong-object grasp: missed
           "uncertain", "uncertain",  # wrong-object pre-release: still uncertain with the side view
           "fail"]                    # misaligned grasp: caught


@pytest.fixture(scope="module")
def studied(run_dir, tmp_path_factory):
    from vplan.evaluate import shadow
    out = tmp_path_factory.mktemp("out") / "shadow"
    loaded = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(shadow, "load_seed_set", lambda path, **kw: loaded.append((path, kw)) or seed_set())
        result, ask = shadow_run(run_dir, out, REPLIES)
    return result, ask, out, loaded


# ---- one complete study ---------------------------------------------------------------------------------------
def test_every_executed_round_gets_natural_and_injected_questions_in_stage_order(studied):
    result, ask, out, _ = studied
    got = lines(out)
    assert result.status == "complete" and not ask.replies
    assert [(r.stage, r.source) for r in got] == ORDER
    assert all((r.attempt_id, r.task_index, r.round_index) == (ATTEMPT, INDEX, 0) for r in got)
    assert len({r.question_id for r in got}) == len(ORDER)
    assert [r.posed for r in got] == ["cube_red"] * 3 + ["cube_green"] * 3 + ["cube_red"]


def test_truth_is_the_state_predicate_for_the_oracle_target(studied):
    # The canonical plan's own frames satisfy every predicate; every injected frame violates its stage's predicate.
    assert [r.truth for r in lines(studied[2])] == [True] * 3 + [False] * 4


def test_the_gate_protocol_is_replayed_per_question(studied):
    _, ask, out, _ = studied
    got = lines(out)
    assert [r.verdict for r in got] == ["pass", "fail", "pass", "fail", "pass", "uncertain", "fail"]
    assert [[c.views for c in r.checks] for r in got] == [
        [GATE_VIEWS], [GATE_VIEWS, RETRY_VIEWS], [GATE_VIEWS], [GATE_VIEWS], [GATE_VIEWS], [GATE_VIEWS, RETRY_VIEWS],
        [GATE_VIEWS]]
    assert all(r.error is None and r.skipped is None for r in got)
    # The live gate's settings (D47): thinking on and its 90 s timeout; the object named by color and shape only.
    assert all(s["enable_thinking"] is True and s["timeout"] == 90.0 and s["model"] == "nvidia/Cosmos3-Edge"
               for s in ask.sent)
    assert all("red cube" in s["question"] and "cube_red" not in s["question"] for s in ask.sent)
    assert all(s["base_url"] == "http://cosmos.invalid/v1" for s in ask.sent)


def test_natural_questions_send_the_runs_own_frames_and_injected_ones_their_renders(studied, run_dir):
    _, ask, out, _ = studied
    got = lines(out)
    index = records.FrameIndex.model_validate_json((run_dir / "frames" / "0004" / "index.json").read_bytes())
    for r in got[:3]:
        assert r.frames == tuple(f"frames/0004/r0-{r.stage}-{v}.png" for v in RETRY_VIEWS)
    for r in got[3:]:
        assert r.frames == tuple(f"injected/0004/r0-{r.stage}-{r.source}-{v}.png" for v in RETRY_VIEWS)
        assert all((out / f).is_file() for f in r.frames)
    natural = {(f.stage, f.view): (run_dir / "frames" / "0004" / f.file).read_bytes() for f in index.frames}
    first = ask.sent[0]  # natural pre-grasp, gate views
    assert first["images"] == {v: natural[("pregrasp", v)] for v in GATE_VIEWS}
    wrong = ask.sent[4]  # wrong-object pre-grasp (the natural grasp took two requests)
    assert wrong["images"] == {v: (out / f"injected/0004/r0-pregrasp-wrong_object-{v}.png").read_bytes()
                               for v in GATE_VIEWS}
    assert wrong["images"]["top"] != natural[("pregrasp", "top")]


def test_the_seed_set_defaults_to_the_one_the_run_recorded(studied):
    assert [(str(path), kw) for path, kw in studied[3]] == [("seeds-not-used", {"allow_source_drift": True})]


def test_the_summary_reports_natural_and_injected_negatives_separately(studied):
    s = json.loads((studied[2] / "summary.json").read_text(encoding="utf-8"))
    assert s == studied[0].summary
    c = s["classes"]
    assert {k: c["predicate_true"]["all"][k] for k in ("n", "pass", "fail", "uncertain", "errors")} == {
        "n": 3, "pass": 2, "fail": 1, "uncertain": 0, "errors": 0}
    assert c["predicate_true"]["grasp"]["fail_rate"] == 1.0
    assert c["natural_negative"]["all"]["n"] == 0
    injected = c["injected_negative"]
    assert {k: injected["all"][k] for k in ("n", "pass", "fail", "uncertain")} == {
        "n": 4, "pass": 1, "fail": 2, "uncertain": 1}
    assert injected["all"]["uncertain_rate"] == 0.25 and injected["grasp"]["n"] == 2
    assert c["wrong_object"]["all"]["n"] == 3 and c["misaligned_grasp"]["all"]["fail"] == 1
    assert s["balanced_accuracy"]["natural"]["all"] is None  # no natural negatives here
    assert s["balanced_accuracy"]["injected"]["all"] == pytest.approx((2 / 3 + 2 / 4) / 2)
    # Replaying the live gate (stop at the first stage that does not pass) on the natural frames: the correct plan
    # would have been stopped at the grasp.
    assert s["gate"]["predicate_true"] == {"n": 1, "pass": 0, "fail": 1, "uncertain": 0,
                                           "stopped_at": {"pregrasp": 0, "grasp": 1, "prerelease": 0}}
    assert s["latency_s"]["request"] == {"n": 9, "median": 0.1, "p90": 0.1, "max": 0.1}
    md = (studied[2] / "summary.md").read_text(encoding="utf-8")
    assert "not verification" in md and "incremental" in md


def test_an_injected_pose_that_misses_its_ik_target_is_skipped_not_asked(run_dir, tmp_path, monkeypatch):
    from vplan.evaluate import shadow
    real = shadow._goal

    def raised(traj, stage):  # 5 cm above what the trajectory was solved for: the posed TCP cannot be there
        goal = real(traj, stage).copy()
        goal[2, 3] += 0.05
        return goal
    monkeypatch.setattr(shadow, "_goal", raised)
    result, ask = shadow_run(run_dir, tmp_path / "shadow", ["pass"] * 4)
    got = {r.source: r for r in lines(tmp_path / "shadow")}
    assert result.status == "complete" and len(ask.sent) == 4  # 3 natural, and the misaligned grasp solved anew
    wrong = [r for r in lines(tmp_path / "shadow") if r.source == "wrong_object"]
    assert len(wrong) == 3 and all(r.skipped == "the pose missed its IK target" and not r.checks for r in wrong)
    assert got["misaligned_grasp"].verdict == "pass"
    assert result.summary["questions"]["skipped"] == {"the pose missed its IK target": 3}
    assert result.summary["classes"]["wrong_object"]["all"]["n"] == 0


def test_an_injected_frame_whose_predicate_holds_is_skipped_not_asked(run_dir, tmp_path, monkeypatch):
    from vplan.evaluate import shadow
    monkeypatch.setattr(shadow, "OFFSET_M", (0.0, 0.0))  # no offset: the grasp is the executed one, in the band
    result, ask = shadow_run(run_dir, tmp_path / "shadow", ["pass"] * 6)
    (misaligned,) = [r for r in lines(tmp_path / "shadow") if r.source == "misaligned_grasp"]
    assert misaligned.truth is True and misaligned.skipped == "the state predicate holds" and len(ask.sent) == 6
    assert result.summary["classes"]["misaligned_grasp"]["all"]["n"] == 0


# ---- outages and reruns ---------------------------------------------------------------------------------------
def test_an_outage_stops_after_three_errors_in_a_row_and_a_rerun_asks_only_what_is_missing(run_dir, tmp_path):
    out = tmp_path / "shadow"
    down = CosmosUnavailable("APIConnectionError: Connection error.")
    first, ask = shadow_run(run_dir, out, ["pass", down, down, down, "pass"])
    assert first.status == "incomplete" and "3 endpoint errors in a row" in first.stop_reason
    assert len(ask.sent) == 4 and ask.replies == ["pass"]  # the fourth question was never asked
    got = lines(out)
    assert [r.verdict for r in got] == ["pass", None, None, None]
    assert [r.error for r in got[1:]] == ["APIConnectionError: Connection error."] * 3
    assert first.summary["questions"]["errors"] == 3
    with open(out / "shadow.jsonl", "a", encoding="utf-8") as f:
        f.write('{"question_id": "3:dev:4/r0/gr')  # a session killed mid-write leaves a partial line
    second, ask = shadow_run(run_dir, out, ["pass"] * 6)
    assert second.status == "complete" and len(ask.sent) == 6 and not ask.replies
    again = lines(out)
    assert [(r.stage, r.source) for r in again] == ORDER and again[0] == got[0]
    assert all(r.verdict is not None and r.error is None for r in again)
    third, ask = shadow_run(run_dir, out, [])
    assert third.status == "complete" and ask.sent == [] and lines(out) == again


def test_limit_caps_the_questions_asked_in_one_session(run_dir, tmp_path):
    result, ask = shadow_run(run_dir, tmp_path / "shadow", ["pass"] * 7, limit=2)
    assert result.status == "incomplete" and "limit" in result.stop_reason
    assert len(ask.sent) == 2 and len(lines(tmp_path / "shadow")) == 2


# ---- refusals -------------------------------------------------------------------------------------------------
def test_a_rerun_with_other_settings_is_refused(studied, run_dir):
    with pytest.raises(EvalError, match="model"):
        shadow_run(run_dir, studied[2], [], model="nvidia/Cosmos3-Nano")


def test_a_frame_that_differs_from_its_index_is_refused(run_dir, tmp_path):
    copy = tmp_path / "c3-dev"
    shutil.copytree(run_dir, copy)
    png = copy / "frames" / "0004" / "r0-grasp-top.png"
    png.write_bytes(png.read_bytes() + b"x")
    with pytest.raises(EvalError, match="r0-grasp-top.png"):
        shadow_run(copy, tmp_path / "shadow", [])


def test_the_output_never_goes_inside_the_run_directory(run_dir):
    with pytest.raises(EvalError, match="inside the run directory"):
        shadow_run(run_dir, run_dir / "shadow", [])
    assert not (run_dir / "shadow").exists()


def test_a_run_over_another_seed_set_is_refused(run_dir, tmp_path, monkeypatch):
    from vplan.evaluate import shadow
    monkeypatch.setattr(shadow, "load_seed_set", lambda path, **kw: seed_set("ef" * 32))
    with pytest.raises(EvalError, match="seed-set hash"):
        shadow_run(run_dir, tmp_path / "shadow", [])


# ---- records and the summary --------------------------------------------------------------------------------
def record(stage, source, truth, verdicts=(), *, error=None, skipped=None, latency_s=1.0, attempt="3:test:1"):
    checks = tuple(records.VisualCheck(stage=stage, views=GATE_VIEWS if i == 0 else RETRY_VIEWS,
                                       verdict=verdict(v, latency_s)) for i, v in enumerate(verdicts))
    return records.ShadowRecord(question_id=f"{attempt}/r0/{stage}/{source}", attempt_id=attempt, task_index=1,
                                round_index=0, stage=stage, source=source, subject="red cube", posed="cube_red",
                                frames=("a.png",), truth=truth, checks=checks,
                                verdict=verdicts[-1] if verdicts and error is None else None, error=error,
                                skipped=skipped)


def test_a_shadow_record_is_answered_failed_or_skipped_exactly_once():
    record("grasp", "natural", True, ["pass"])
    record("grasp", "natural", True, error="down")
    record("grasp", "wrong_object", True, skipped="predicate holds")
    for bad in ({}, {"error": "down", "skipped": "x"}):
        with pytest.raises(ValidationError, match="exactly one"):
            records.ShadowRecord(**{**record("grasp", "natural", True, ["pass"]).model_dump(), "verdict": None, **bad})
    with pytest.raises(ValidationError, match="last check"):
        records.ShadowRecord(**{**record("grasp", "natural", True, ["pass"]).model_dump(), "verdict": "fail"})


def test_the_summary_counts_rejections_misses_uncertain_and_latency_by_class_and_stage():
    from vplan.evaluate.shadow import summarize
    rs = [record("pregrasp", "natural", True, ["pass"], latency_s=1.0, attempt="3:test:1"),
          record("grasp", "natural", True, ["pass"], latency_s=2.0, attempt="3:test:1"),
          record("prerelease", "natural", True, ["pass"], latency_s=3.0, attempt="3:test:1"),
          record("pregrasp", "natural", False, ["fail"], latency_s=4.0, attempt="3:test:2"),
          record("grasp", "natural", False, ["uncertain", "pass"], latency_s=5.0, attempt="3:test:2"),
          record("prerelease", "natural", False, ["pass"], latency_s=6.0, attempt="3:test:2"),
          record("grasp", "misaligned_grasp", False, ["pass"], latency_s=7.0, attempt="3:test:2"),
          record("grasp", "wrong_object", False, error="down", attempt="3:test:2"),
          record("prerelease", "wrong_object", True, skipped="predicate holds", attempt="3:test:2")]
    s = summarize(rs)
    assert s["questions"] == {"recorded": 9, "answered": 7, "errors": 1, "skipped": {"predicate holds": 1}}
    neg = s["classes"]["natural_negative"]
    assert {k: neg["all"][k] for k in ("n", "pass", "fail", "uncertain", "errors")} == {
        "n": 3, "pass": 2, "fail": 1, "uncertain": 0, "errors": 0}
    assert neg["all"]["pass_rate"] == pytest.approx(2 / 3) and neg["pregrasp"]["fail_rate"] == 1.0
    assert s["classes"]["injected_negative"]["grasp"] == {
        "n": 1, "pass": 1, "fail": 0, "uncertain": 0, "errors": 1, "pass_rate": 1.0, "fail_rate": 0.0,
        "uncertain_rate": 0.0, "latency_s": {"n": 1, "median": 7.0, "p90": 7.0, "max": 7.0}}
    assert s["classes"]["predicate_true"]["prerelease"]["n"] == 1  # the skipped injected frame is not a question
    assert s["classes"]["wrong_object"]["all"]["n"] == 0
    assert s["balanced_accuracy"]["natural"]["all"] == pytest.approx((1.0 + 1 / 3) / 2)
    assert s["balanced_accuracy"]["injected"]["grasp"] == pytest.approx((1.0 + 0.0) / 2)
    assert s["balanced_accuracy"]["injected"]["pregrasp"] is None
    assert s["gate"]["predicate_false"] == {"n": 1, "pass": 0, "fail": 1, "uncertain": 0,
                                            "stopped_at": {"pregrasp": 1, "grasp": 0, "prerelease": 0}}
    assert s["gate"]["predicate_true"]["pass"] == 1
    # Per request: 1, 2, 3, 4, 5, 5, 6, 7 s (the re-asked question counts twice); per question: 1, 2, 3, 4, 10, 6, 7 s.
    assert s["latency_s"]["request"] == {"n": 8, "median": 4.5, "p90": pytest.approx(6.3), "max": 7.0}
    assert s["latency_s"]["question"]["median"] == 4.0 and s["latency_s"]["question"]["max"] == 10.0


# ---- the command line -----------------------------------------------------------------------------------------
def test_the_cli_survives_an_unreachable_endpoint_and_says_how_to_continue(run_dir, tmp_path, capsys):
    from vplan import cli
    out = tmp_path / "shadow"
    code = cli.main(["eval", "shadow", "--run", str(run_dir), "--output", str(out),
                     "--cosmos-url", "http://127.0.0.1:9/v1", "--timeout", "5"])
    err = capsys.readouterr().err
    assert code == 1 and "127.0.0.1:9" in err and "rerun" in err.lower()
    got = lines(out)
    assert len(got) == 3 and all(r.error and r.verdict is None for r in got)
    assert (out / "summary.md").is_file()


def test_the_cli_refuses_a_run_without_frames(run_dir, tmp_path, capsys):
    from vplan import cli
    copy = tmp_path / "c5-dev"
    shutil.copytree(run_dir, copy, ignore=shutil.ignore_patterns("frames"))
    manifest = json.loads((copy / "manifest.json").read_text(encoding="utf-8"))
    (copy / "manifest.json").write_text(json.dumps({**manifest, "frames": False}), encoding="utf-8")
    code = cli.main(["eval", "shadow", "--run", str(copy), "--output", str(tmp_path / "shadow"),
                     "--cosmos-url", "http://127.0.0.1:9/v1"])
    assert code == 2 and "frames" in capsys.readouterr().err
