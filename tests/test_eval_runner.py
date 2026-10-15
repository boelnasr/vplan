"""W3.1 evaluation runner: configs, provider guard, manifest, resume, stop rules and the run_task worker job.

Most tests replace the supervised worker with an instant fake (no subprocess, no simulation); the two at the end run
real worker subprocesses, one with real simulation and one with a fake provider that echoes the API key.
"""
import json
import os

import pytest

from tests.test_generate import _fake_witness
from tests.test_pilot import SECRET, env_file
from vplan import records
from vplan.config import config_hash, config_snapshot

FREE = "nvidia/nemotron-3-ultra-550b-a55b:free"


@pytest.fixture(scope="module")
def seed_dir(tmp_path_factory):
    """Five tasks in slot order: feasible, feasible, impossible type 1, feasible, impossible type 2."""
    from vplan.evaluate import seedsets
    out = tmp_path_factory.mktemp("eval") / "seeds"
    seedsets.prepare_seed_set(out, "dev", feasible=3, impossible=2, witness_evaluator=_fake_witness)
    return out


# ---- an instant stand-in for supervisor.supervise -------------------------------------------------------
def _call(outcome="proposed", plan=None, status_code=None):
    return records.PlannerCall(outcome=outcome, plan=plan, provider="fake", model="fake", protocol="forced",
                               next_protocol="forced", temperature=0.0, latency_s=0.0, raw_request={},
                               status_code=status_code)


PLAN = records.PickPlacePlan(target_id="o_x", grasp=records.Grasp(approach="top", yaw_deg=0, pregrasp_offset_m=.1),
                             place=records.Place(position=(.45, .3, .03), yaw_deg=0, release_height_m=.05))
BLOCKED = records.EvaluationResult(outcome="blocked", findings=(records.Finding(
    stage="transport", check="collision", plan_field="transport_via", measured=0., threshold=.01,
    direction="below margin"),))


def record_for(job, script):
    """The AttemptRecord run_task would return for a scripted list of rounds: 'refused', 'blocked' or 'sfNNN'."""
    p = job.params
    rounds, outcome, judge = [], None, None
    for i, step in enumerate(script):
        if step == "refused":
            judge = records.JudgeResult(success=True, predicates={"R": True}, violations=(), details={})
            rnd = dict(call=_call(plan=records.Refusal(reason="other", explanation="x")),
                       evaluation=records.EvaluationResult(outcome="refused", judge=judge))
            outcome = "refused"
        elif step == "blocked":
            rnd = dict(call=_call(plan=PLAN), evaluation=BLOCKED)
            outcome = "budget_exhausted"
        else:
            rnd = dict(call=_call("service_failure", status_code=int(step[2:])))
            outcome = "service_failure"
        rounds.append(records.RoundRecord(call_id=f"{job.attempt_id}:{i + 1}", round_index=i, **rnd))
    return records.AttemptRecord(
        attempt_id=job.attempt_id, mode=p["mode"], k=p["k"], outcome=outcome, rounds=tuple(rounds),
        call_count=len(rounds), judge=judge, total_tokens=None, known_total_tokens=0,
        missing_usage_calls=len(rounds), latency_s=0.0,
        config_hashes={k: config_hash(v) for k, v in p["config_snapshot"].items()})


class FakeSupervise:
    """script[i] is the round list for the i-th attempt this fake runs (default: one refusal), or 'crash'."""

    def __init__(self, script=(), interrupt_at=None):
        self.script, self.interrupt_at, self.jobs, self.stems = list(script), interrupt_at, [], []

    def __call__(self, job, stem, **kwargs):
        if self.interrupt_at is not None and len(self.jobs) == self.interrupt_at:
            raise KeyboardInterrupt
        self.jobs.append(job)
        self.stems.append(stem)
        steps = self.script[len(self.jobs) - 1] if len(self.jobs) <= len(self.script) else ["refused"]
        if steps == "crash":
            return records.SupervisedAttempt(attempt_id=job.attempt_id, job=job.job, outcome="sim_error", tries=1,
                                             wall_s=.01, exit_code=70, error_type="worker_crash")
        record = record_for(job, steps)
        failed = [r.call.status_code for r in record.rounds if r.call.outcome == "service_failure"]
        return records.SupervisedAttempt(attempt_id=job.attempt_id, job=job.job, outcome=record.outcome, tries=1,
                                         wall_s=.01, exit_code=0, status_code=failed[0] if failed else None,
                                         payload={"record": record.model_dump(mode="json")})


def run(seed_dir, tmp_path, config=5, fake=None, **kw):
    from vplan.evaluate import runner
    kw.setdefault("run_dir", None if "resume" in kw else tmp_path / "run")
    kw.setdefault("echo", lambda *a: None)
    return runner.run_eval(config, "dev", seed_dir=seed_dir, supervise_attempt=fake or FakeSupervise(), **kw)


def rows(run_dir):
    return [records.EvalAttempt.model_validate_json(line)
            for line in (run_dir / "attempts.jsonl").read_text(encoding="utf-8").splitlines()]


def sessions(run_dir):
    return [json.loads(line) for line in (run_dir / "sessions.jsonl").read_text(encoding="utf-8").splitlines()]


def manifest(run_dir):
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


# ---- configs and planner settings ------------------------------------------------------------------------
def test_config_table_maps_each_config_to_planner_mode_and_k():
    from vplan.evaluate import runner
    assert {n: (c.planner, c.mode, c.k) for n, c in runner.CONFIGS.items()} == {
        1: ("nemotron", "unverified", 0), 3: ("nemotron", "repair", 3), 5: ("heuristic", "repair", 3),
        6: ("frontier", "unverified", 0), 7: ("nemotron", "resample", 3)}
    assert runner.eval_config(3) is runner.CONFIGS[3]


@pytest.mark.parametrize("number, message", [(2, "round 1 of config 3"), (4, "shadow study"), (8, "unknown")])
def test_configs_that_are_not_runs_are_refused_with_a_reason(number, message):
    from vplan.evaluate import runner
    with pytest.raises(runner.EvalError, match=message):
        runner.eval_config(number)


def test_heuristic_config_uses_no_llm_and_refuses_llm_options(tmp_path):
    from vplan.evaluate import runner
    s = runner.planner_settings(runner.CONFIGS[5])
    assert (s.kind, s.provider, s.model, s.temperature, s.env_file) == ("heuristic", "heuristic",
                                                                        "template-heuristic", 0.0, None)
    for kw in ({"env_file": env_file(tmp_path)}, {"model": FREE}, {"temperature": 0.0}, {"allow_paid": True}):
        with pytest.raises(runner.EvalError, match="without an LLM"):
            runner.planner_settings(runner.CONFIGS[5], **kw)


@pytest.mark.parametrize("config, base, model, flags, message", [
    (3, "https://openrouter.ai/api/v1", "nvidia/nemotron-3-ultra-550b-a55b", {}, "--allow-paid"),
    (1, "https://api.tokenfactory.nebius.com/v1/", FREE, {}, "--allow-paid"),
    (7, "https://openrouter.ai/api/v1", "meta-llama/llama-3.3-70b-instruct:free", {}, "--allow-paid"),
    (6, "https://openrouter.ai/api/v1", FREE, {}, "--model"),
    (6, "https://openrouter.ai/api/v1", FREE, {"model": "anthropic/claude-sonnet-5.5"}, "--allow-paid"),
    (3, "https://openrouter.ai/api/v1", FREE, {"temperature": -0.1}, "temperature"),
    (3, "https://openrouter.ai/api/v1", FREE, {"temperature": float("nan")}, "temperature")])
def test_provider_guard_refuses_paid_or_unnamed_models(tmp_path, config, base, model, flags, message):
    from vplan.evaluate import runner
    with pytest.raises(runner.EvalError, match=message):
        runner.planner_settings(runner.CONFIGS[config], env_file=env_file(tmp_path, base, model), **flags)


def test_llm_configs_need_an_env_file():
    from vplan.evaluate import runner
    with pytest.raises(runner.EvalError, match="--env-file"):
        runner.planner_settings(runner.CONFIGS[3])


def test_llm_settings_never_hold_the_key(tmp_path):
    from vplan.evaluate import runner
    path = env_file(tmp_path)
    s = runner.planner_settings(runner.CONFIGS[3], env_file=path)
    assert (s.kind, s.provider, s.model, s.temperature, s.base_url_host) == ("llm", "openrouter", FREE, .7,
                                                                             "openrouter.ai")
    assert SECRET not in json.dumps(s.params()) + json.dumps(s.describe()) + repr(s)
    assert s.params()["env_file"] == str(path.resolve()) and "env_file" not in s.describe()
    paid = runner.planner_settings(runner.CONFIGS[6], env_file=path, model="anthropic/claude-sonnet-5.5",
                                   allow_paid=True)
    assert paid.model == "anthropic/claude-sonnet-5.5"  # config 6 never falls back to LLM_MODEL


# ---- new runs ---------------------------------------------------------------------------------------------
def test_new_run_records_every_attempt_once_with_a_spec_manifest(seed_dir, tmp_path, monkeypatch):
    from vplan.evaluate import seedsets
    monkeypatch.setenv("VPLAN_IMAGE_DIGEST", "sha256:feedface")
    fake = FakeSupervise()
    summary = run(seed_dir, tmp_path, fake=fake)
    run_dir = tmp_path / "run"
    assert summary.status == "complete" and summary.stop_reason is None and summary.run_dir == run_dir
    assert (summary.planned, summary.recorded, summary.outcomes) == (5, 5, {"refused": 5})
    got = rows(run_dir)
    assert [r.attempt_id for r in got] == [f"5:dev:{i}" for i in range(5)]
    loaded = seedsets.load_seed_set(seed_dir)
    assert [(r.task_index, r.task_id) for r in got] == [(t.index, t.task_id) for t in loaded.tasks]
    assert all(r.config == 5 and r.split == "dev" and r.session == 0 and r.record.mode == "repair" for r in got)
    assert [s.name for s in fake.stems] == [f"{i:04d}" for i in range(5)]
    assert all(s.parent == run_dir / "workers" for s in fake.stems)
    m = manifest(run_dir)
    assert (m["kind"], m["config"], m["mode"], m["k"], m["split"]) == ("eval_run", 5, "repair", 3, "dev")
    assert (m["provider"], m["model"], m["temperature"]) == ("heuristic", "template-heuristic", 0.0)
    assert m["seed_set_hash"] == loaded.seed_set_hash and m["seed_set"]["n_tasks"] == 5
    assert (m["n"], m["n_prepared"], m["limited"]) == (5, 5, False)
    assert m["config_hashes"] == {k: config_hash(v) for k, v in config_snapshot().items()}
    assert m["image_digest"] == "sha256:feedface" and m["timeout_s"] == 600.0
    assert {"head", "dirty", "source"} >= set(m["git"] or {}) and {"version", "location", "git"} <= set(m["manipulapy"])
    assert {"openai", "pybullet", "ManipulaPy", "numpy", "pydantic"} <= set(m["versions"])
    assert {"python", "os", "source_digest", "source_hashes", "created_utc"} <= set(m)
    start, end = sessions(run_dir)
    assert (start["event"], start["session"], start["pending"]) == ("start", 0, 5)
    assert (end["event"], end["status"], end["recorded"]) == ("end", "complete", 5)


def test_image_digest_is_null_when_unset(seed_dir, tmp_path, monkeypatch):
    monkeypatch.delenv("VPLAN_IMAGE_DIGEST", raising=False)
    run(seed_dir, tmp_path)
    assert manifest(tmp_path / "run")["image_digest"] is None


def test_n_runs_a_prefix_and_marks_the_run_limited(seed_dir, tmp_path):
    summary = run(seed_dir, tmp_path, n=2)
    assert summary.status == "complete" and [r.task_index for r in rows(tmp_path / "run")] == [0, 1]
    m = manifest(tmp_path / "run")
    assert (m["n"], m["n_prepared"], m["limited"]) == (2, 5, True)


@pytest.mark.parametrize("n", [0, 6])
def test_n_outside_the_seed_set_is_refused(seed_dir, tmp_path, n):
    from vplan.evaluate import runner
    with pytest.raises(runner.EvalError, match="--n"):
        run(seed_dir, tmp_path, n=n)
    assert not (tmp_path / "run").exists()


def test_new_run_refuses_a_used_directory_and_a_wrong_split(seed_dir, tmp_path):
    from vplan.evaluate import runner
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "keep.txt").write_text("evidence", encoding="utf-8")
    with pytest.raises(runner.EvalError, match="not empty"):
        run(seed_dir, tmp_path)
    with pytest.raises(runner.EvalError, match="dev split"):
        runner.run_eval(5, "test", seed_dir=seed_dir, run_dir=tmp_path / "other", echo=lambda *a: None,
                        supervise_attempt=FakeSupervise())


def test_job_params_carry_task_snapshot_and_planner_but_never_the_key(seed_dir, tmp_path):
    fake = FakeSupervise()
    run(seed_dir, tmp_path, config=1, fake=fake, env_file=env_file(tmp_path), max_rpm=30)
    job = fake.jobs[2]
    p = job.params
    assert job.job == "run_task" and job.attempt_id == p["attempt_id"] == "1:dev:2"
    assert (p["mode"], p["k"]) == ("unverified", 0) and p["config_snapshot"] == config_snapshot()
    task = records.TaskSpec.model_validate(p["task"])
    oracle = records.TaskOracle.model_validate(p["oracle"])
    assert oracle.feasibility == "impossible" and oracle.proof.type == 1 and task.instruction
    assert p["planner"]["model"] == FREE and p["planner"]["env_file"] == str((tmp_path / "test.env").resolve())
    assert p["pacing"]["min_interval_s"] == 2.0 and p["pacing"]["path"].endswith("pacing.json")
    for j in fake.jobs:
        assert SECRET not in json.dumps(j.params)
    assert all(SECRET.encode() not in f.read_bytes() for f in (tmp_path / "run").rglob("*") if f.is_file())


# ---- resume -----------------------------------------------------------------------------------------------
def test_resume_discards_a_partial_final_line_and_completes_without_duplicates(seed_dir, tmp_path):
    run(seed_dir, tmp_path)
    run_dir = tmp_path / "run"
    path = run_dir / "attempts.jsonl"
    full = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text("".join(full[:2]) + full[2][:40], encoding="utf-8", newline="")  # killed mid-write
    fake = FakeSupervise()
    summary = run(seed_dir, tmp_path, fake=fake, resume=run_dir)
    assert summary.status == "complete" and (summary.planned, summary.recorded) == (5, 5)
    assert [j.attempt_id for j in fake.jobs] == ["5:dev:2", "5:dev:3", "5:dev:4"]
    assert [s.name for s in fake.stems] == ["0002.s1", "0003.s1", "0004.s1"]  # never the killed session's files
    got = rows(run_dir)
    assert sorted(r.attempt_id for r in got) == [f"5:dev:{i}" for i in range(5)]
    assert [r.session for r in got] == [0, 0, 1, 1, 1]
    start = [s for s in sessions(run_dir) if s["event"] == "start"]
    assert [s["session"] for s in start] == [0, 1] and start[1]["discarded_partial_bytes"] == 40
    assert start[1]["pending"] == 3 and start[1]["resumed"] is True


def test_interrupted_run_is_marked_and_resumes(seed_dir, tmp_path):
    with pytest.raises(KeyboardInterrupt):
        run(seed_dir, tmp_path, fake=FakeSupervise(interrupt_at=2))
    run_dir = tmp_path / "run"
    assert len(rows(run_dir)) == 2 and sessions(run_dir)[-1]["status"] == "interrupted"
    summary = run(seed_dir, tmp_path, resume=run_dir)
    assert summary.status == "complete" and summary.session == 1
    assert sorted(r.task_index for r in rows(run_dir)) == [0, 1, 2, 3, 4]


def test_resume_of_a_complete_run_runs_nothing(seed_dir, tmp_path):
    run(seed_dir, tmp_path)
    fake = FakeSupervise()
    summary = run(seed_dir, tmp_path, fake=fake, resume=tmp_path / "run")
    assert fake.jobs == [] and summary.status == "complete" and summary.outcomes == {"refused": 5}


def test_resume_uses_the_runs_own_seed_set_by_default(seed_dir, tmp_path):
    from vplan.evaluate import runner
    with pytest.raises(KeyboardInterrupt):
        run(seed_dir, tmp_path, fake=FakeSupervise(interrupt_at=1))
    summary = runner.run_eval(5, "dev", resume=tmp_path / "run", supervise_attempt=FakeSupervise(),
                              echo=lambda *a: None)
    assert summary.status == "complete" and len(rows(tmp_path / "run")) == 5


@pytest.mark.parametrize("damage, message", [("duplicate", "duplicate"), ("foreign", "not part of this run"),
                                             ("corrupt", "line 2")])
def test_resume_refuses_damaged_attempt_rows(seed_dir, tmp_path, damage, message):
    from vplan.evaluate import runner
    run(seed_dir, tmp_path, n=3)
    path = tmp_path / "run" / "attempts.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    if damage == "duplicate":
        lines.append(lines[0])
    elif damage == "foreign":
        row = json.loads(lines[0])
        lines.append(json.dumps({**row, "attempt_id": "5:dev:4", "task_index": 4,
                                 "record": {**row["record"], "attempt_id": "5:dev:4"}}) + "\n")
    else:
        lines[1] = '{"attempt_id": "5:dev:1"}\n'
    path.write_text("".join(lines), encoding="utf-8")
    with pytest.raises(runner.EvalError, match=message):
        run(seed_dir, tmp_path, resume=tmp_path / "run")


@pytest.mark.parametrize("change, message", [
    ("config", "config"), ("model", "model"), ("temperature", "temperature"), ("n", "--n"),
    ("seed_set", "seed-set hash"), ("configs", "configuration"), ("source", "source changed"),
    ("not_a_run", "not an evaluation run")])
def test_resume_refuses_an_incompatible_run(seed_dir, tmp_path, monkeypatch, change, message):
    from vplan.evaluate import runner, seedsets
    path = env_file(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        run(seed_dir, tmp_path, config=1, env_file=path, fake=FakeSupervise(interrupt_at=1))
    kw = dict(config=1, env_file=path, resume=tmp_path / "run")
    if change == "config":
        kw["config"] = 7
    elif change == "model":
        kw["model"] = "nvidia/nemotron-3.5-lightning:free"
    elif change == "temperature":
        kw["temperature"] = 0.2
    elif change == "n":
        kw["n"] = 2
    elif change == "seed_set":
        other = tmp_path / "other-seeds"
        seedsets.prepare_seed_set(other, "dev", feasible=3, impossible=1, witness_evaluator=_fake_witness)
        kw["seed_dir"] = other
    elif change == "configs":
        snapshot = config_snapshot()
        snapshot["judge"] = {**snapshot["judge"], "tampered": 1}
        monkeypatch.setattr(seedsets, "config_snapshot", lambda: snapshot)
        monkeypatch.setattr(runner, "config_snapshot", lambda: snapshot)
    elif change == "source":
        current = runner.source_hashes()
        drifted = {**current, "src/vplan/repair.py": "0" * 64}
        monkeypatch.setattr(seedsets, "source_hashes", lambda: drifted)
        monkeypatch.setattr(runner, "source_hashes", lambda: drifted)
    else:
        (tmp_path / "run" / "manifest.json").unlink()
    config, seeds = kw.pop("config"), kw.pop("seed_dir", seed_dir)
    fake = FakeSupervise()
    with pytest.raises(runner.EvalError, match=message):
        run(seeds, tmp_path, config=config, fake=fake, **kw)
    assert fake.jobs == [] and len(rows(tmp_path / "run")) == 1


def test_resume_with_allowed_source_drift_records_it(seed_dir, tmp_path, monkeypatch):
    from vplan.evaluate import runner, seedsets
    with pytest.raises(KeyboardInterrupt):
        run(seed_dir, tmp_path, fake=FakeSupervise(interrupt_at=1))
    drifted = {**runner.source_hashes(), "src/vplan/repair.py": "0" * 64}
    monkeypatch.setattr(seedsets, "source_hashes", lambda: drifted)
    monkeypatch.setattr(runner, "source_hashes", lambda: drifted)
    summary = run(seed_dir, tmp_path, resume=tmp_path / "run", allow_source_drift=True)
    assert summary.status == "complete"
    assert sessions(tmp_path / "run")[-2]["source_drift"] == ["src/vplan/repair.py"]


# ---- stop rules (D33, D42) ----------------------------------------------------------------------------------
def test_fatal_provider_status_stops_at_once(seed_dir, tmp_path):
    fake = FakeSupervise([["sf401"]])
    summary = run(seed_dir, tmp_path, fake=fake)
    assert summary.status == "incomplete" and "401" in summary.stop_reason and len(fake.jobs) == 1
    assert len(rows(tmp_path / "run")) == 1 and sessions(tmp_path / "run")[-1]["status"] == "incomplete"


def test_three_consecutive_failed_requests_stop_the_run(seed_dir, tmp_path):
    summary = run(seed_dir, tmp_path, fake=FakeSupervise([["sf429"], ["sf503"], ["sf429"]]))
    assert summary.status == "incomplete" and "consecutive" in summary.stop_reason and summary.recorded == 3


def test_requests_that_reached_the_provider_reset_the_count(seed_dir, tmp_path):
    # Per request, as D33 counts: the third attempt's first request succeeded, so the count restarts there.
    script = [["sf429"], ["sf429"], ["blocked", "sf429"], ["sf429"], ["refused"]]
    summary = run(seed_dir, tmp_path, fake=FakeSupervise(script))
    assert summary.status == "complete" and summary.recorded == 5


def test_a_worker_crash_resets_the_count_as_in_d42(seed_dir, tmp_path):
    summary = run(seed_dir, tmp_path, fake=FakeSupervise([["sf429"], ["sf429"], "crash", ["sf429"], ["sf429"]]))
    assert summary.status == "complete" and summary.outcomes == {"service_failure": 4, "sim_error": 1}
    crash = rows(tmp_path / "run")[2]
    assert crash.record is None and crash.error_type == "worker_crash" and crash.exit_code == 70


def test_a_source_change_during_the_run_stops_it(seed_dir, tmp_path, monkeypatch):
    from vplan.evaluate import runner
    real, calls = runner.source_hashes(), []

    def hashes():
        calls.append(1)
        return real if len(calls) <= 3 else {**real, "src/vplan/judge/judge.py": "0" * 64}
    monkeypatch.setattr(runner, "source_hashes", hashes)
    summary = run(seed_dir, tmp_path)
    assert summary.status == "incomplete" and "src/vplan/judge/judge.py" in summary.stop_reason
    assert 1 <= summary.recorded < 5


@pytest.mark.parametrize("tamper", ["attempt_id", "mode", "config_hashes"])
def test_an_inconsistent_record_becomes_sim_error(seed_dir, tmp_path, tamper):
    class Tampered(FakeSupervise):
        def __call__(self, job, stem, **kwargs):
            out = super().__call__(job, stem, **kwargs)
            record = dict(out.payload["record"])
            record.update({"attempt_id": "5:dev:99"} if tamper == "attempt_id" else
                          {"mode": "resample"} if tamper == "mode" else {"config_hashes": {"judge": "0"}})
            return records.SupervisedAttempt.model_validate({**out.model_dump(), "payload": {"record": record}})
    run(seed_dir, tmp_path, fake=Tampered(), n=1)
    (row,) = rows(tmp_path / "run")
    assert row.outcome == "sim_error" and row.error_type == "invalid_attempt_record" and row.record is None


# ---- pacing -------------------------------------------------------------------------------------------------
def test_request_pacer_spaces_request_starts_across_worker_processes(tmp_path):
    from vplan.evaluate import runner
    now, slept = [100.0], []

    def sleep(s):
        slept.append(s)
        now[0] += s
    path = tmp_path / "pacing.json"
    first = runner.RequestPacer(path, 3.0, clock=lambda: now[0], sleep=sleep)
    assert first.wait() == 0.0
    now[0] = 101.0
    second = runner.RequestPacer(path, 3.0, clock=lambda: now[0], sleep=sleep)  # the next worker process
    assert second.wait() == pytest.approx(2.0) and slept == [pytest.approx(2.0)]
    now[0] = 200.0
    assert second.wait() == 0.0
    path.write_text(json.dumps({"last_request_start_unix_s": 1e12}), encoding="utf-8")  # a clock jump
    assert second.wait() == 3.0
    path.write_text("not json", encoding="utf-8")
    assert second.wait() == 0.0


def test_paced_planner_waits_before_each_request_and_passes_attributes_through():
    from vplan.evaluate import runner
    from vplan.planners.heuristic import HeuristicPlanner
    events = []

    class Pacer:
        def wait(self):
            events.append("wait")
            return 0.0

    class Inner(HeuristicPlanner):
        _secret = "s3cret"

        def plan(self, task, **kw):
            events.append("plan")
            return "called"
    paced = runner.PacedPlanner(Inner(), Pacer())
    assert paced.plan("task", protocol="forced") == "called" and events == ["wait", "plan"]
    assert (paced.model, paced.provider, paced._secret) == ("template-heuristic", "heuristic", "s3cret")


# ---- the run_task worker job ----------------------------------------------------------------------------------
def job_params(seed_dir, index, config=5, planner=None):
    from vplan.evaluate import runner, seedsets
    loaded = seedsets.load_seed_set(seed_dir)
    spec = runner.CONFIGS[config]
    return {"attempt_id": f"{config}:dev:{index}", "task": loaded.tasks[index].task.model_dump(mode="json"),
            "oracle": loaded.oracles[index].model_dump(mode="json"), "seed": loaded.tasks[index].seed,
            "config_snapshot": config_snapshot(), "mode": spec.mode, "k": spec.k,
            "planner": planner or runner.planner_settings(spec).params(), "pacing": None}


def test_run_task_job_returns_the_attempt_record(seed_dir):
    from vplan.evaluate import worker
    out = worker.JOBS["run_task"](job_params(seed_dir, 2))  # impossible type 1: the heuristic refuses, no physics
    record = records.AttemptRecord.model_validate(out["payload"]["record"])
    assert out["outcome"] == record.outcome == "refused" and record.success and out["status_code"] is None
    assert (record.attempt_id, record.mode, record.k) == ("5:dev:2", "repair", 3)


def test_run_task_job_sanitizes_an_escaping_exception(seed_dir, tmp_path, monkeypatch):
    from vplan import repair
    from vplan.evaluate import runner
    settings = runner.planner_settings(runner.CONFIGS[1], env_file=env_file(tmp_path))

    def boom(*a, **kw):
        raise RuntimeError(f"exploded near {SECRET}")
    monkeypatch.setattr(repair, "run_task", boom)
    out = runner.run_task_job(job_params(seed_dir, 0, config=1, planner=settings.params()))
    assert out["outcome"] == "sim_error" and out["error_type"] == "RuntimeError"
    assert SECRET not in json.dumps(out) and "[REDACTED]" in out["error_message"]


# ---- real worker subprocesses ---------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def two_task_seeds(tmp_path_factory):
    """Index 0 feasible (cell 0,0), index 1 impossible type 1."""
    from vplan.evaluate import seedsets
    out = tmp_path_factory.mktemp("eval2") / "seeds"
    seedsets.prepare_seed_set(out, "dev", feasible=1, impossible=1, witness_evaluator=_fake_witness)
    return out


def test_config_5_runs_real_workers_with_simulation(two_task_seeds, tmp_path):
    from vplan.evaluate import runner
    summary = runner.run_eval(5, "dev", seed_dir=two_task_seeds, run_dir=tmp_path / "run", echo=lambda *a: None)
    assert summary.status == "complete" and summary.recorded == 2
    feasible, impossible = rows(tmp_path / "run")
    assert feasible.record is not None, feasible.error_traceback
    assert feasible.record.config_hashes == manifest(tmp_path / "run")["config_hashes"]
    if feasible.outcome == "executed":  # D41: the record keeps what replay needs
        assert feasible.record.rounds[-1].evaluation.record.q_replay
    assert impossible.outcome == "refused" and impossible.record.success
    names = {p.name for p in (tmp_path / "run" / "workers").iterdir()}
    assert {"0000.params.json", "0000.try1.result.json", "0001.try1.stdout.log"} <= names


def test_the_api_key_never_reaches_the_run_directory_or_the_console(two_task_seeds, tmp_path, capsys):
    from vplan.evaluate import runner
    path = env_file(tmp_path)
    summary = runner.run_eval(1, "dev", seed_dir=two_task_seeds, run_dir=tmp_path / "run", env_file=path,
                              import_modules=("tests.eval_fakes",))
    assert summary.status == "complete", rows(tmp_path / "run")[0].error_traceback
    got = rows(tmp_path / "run")
    assert [r.outcome for r in got] == ["refused", "refused"]
    raw = got[0].record.rounds[0].call.raw_response
    assert raw["id"] == "reply [REDACTED]"  # the fake provider echoed the key; the planner redacted it
    files = [f for f in (tmp_path / "run").rglob("*") if f.is_file()]
    assert len(files) > 6 and all(SECRET.encode() not in f.read_bytes() for f in files)
    out = capsys.readouterr()
    assert SECRET not in out.out + out.err
    assert os.environ.get("LLM_API_KEY") != SECRET


# ---- CLI ------------------------------------------------------------------------------------------------------
def test_cli_eval_prepare_wiring(monkeypatch, capsys):
    from vplan.cli import main
    from vplan.evaluate import seedsets
    calls = {}

    def fake(output, split, **kw):
        calls.update(output=str(output), split=split, **kw)
        return {"n": 3, "seed_set_hash": "ab" * 32}
    monkeypatch.setattr(seedsets, "prepare_seed_set", fake)
    assert main(["eval", "prepare", "--seeds", "dev", "--feasible", "2", "--impossible", "1",
                 "--max-candidates", "9"]) == 0
    assert calls["output"] == os.path.join("runs", "seeds", "dev") and calls["split"] == "dev"
    assert (calls["feasible"], calls["impossible"], calls["max_candidates"], calls["allow_test_split"]) == (
        2, 1, 9, False)
    assert "ab" * 32 in capsys.readouterr().out


def test_cli_eval_run_wiring_and_exit_codes(monkeypatch, tmp_path, capsys):
    from vplan.cli import main
    from vplan.evaluate import runner
    seen, status = [], ["complete"]

    def fake(config, split, **kw):
        seen.append((config, split, kw))
        done = status[0] == "complete"
        return runner.RunSummary(run_dir=tmp_path, status=status[0], stop_reason=None if done else "3 consecutive",
                                 planned=5, recorded=5 if done else 3, session=0, outcomes={"refused": 5},
                                 successes=5)
    monkeypatch.setattr(runner, "run_eval", fake)
    assert main(["eval", "run", "--config", "5", "--seeds", "dev", "--n", "3", "--seed-dir", "s",
                 "--run-dir", "r"]) == 0
    assert seen[0] == (5, "dev", {"seed_dir": "s", "run_dir": "r", "resume": None, "n": 3, "env_file": None,
                                  "model": None, "temperature": None, "allow_paid": False,
                                  "allow_source_drift": False, "timeout_s": 600.0, "max_rpm": None})
    status[0] = "incomplete"
    assert main(["eval", "run", "--config", "3", "--seeds", "dev", "--env-file", "e.env", "--resume", "r",
                 "--max-rpm", "20", "--timeout", "300", "--temperature", "0.5", "--allow-paid"]) == 1
    kw = seen[1][2]
    assert (kw["resume"], kw["env_file"], kw["max_rpm"], kw["timeout_s"], kw["temperature"], kw["allow_paid"]) == (
        "r", "e.env", 20.0, 300.0, 0.5, True)
    assert "--resume" in capsys.readouterr().err


def test_cli_eval_refusals_exit_2_with_a_clean_message(capsys):
    from vplan.cli import main
    assert main(["eval", "run", "--config", "2", "--seeds", "dev"]) == 2
    err = capsys.readouterr().err
    assert "round 1 of config 3" in err and "Traceback" not in err


def test_cli_eval_refuses_wsl_paths(monkeypatch, capsys):
    from vplan.cli import main
    from vplan.evaluate import pilot, seedsets
    monkeypatch.setattr(pilot, "_WINDOWS", True)
    monkeypatch.setattr(seedsets, "prepare_seed_set", lambda *a, **kw: pytest.fail("must refuse first"))
    assert main(["eval", "prepare", "--seeds", "dev", "--output", "/mnt/c/x"]) == 2
    assert "Windows" in capsys.readouterr().err
