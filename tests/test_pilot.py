"""Pilot artifacts, validation and reporting, fully offline (fake cohort, fake provider, fake evaluator)."""
import hashlib
import json

import pytest

from tests.test_planner import FakeClient, response
from tests.test_repair import blocked, executed
from vplan import records
from vplan.config import config_hash, config_snapshot
from vplan.planners.heuristic import heuristic_geometry
from vplan.tasks import generate_candidate

SECRET = "sk-or-v1-FAKE-test-key-0123456789"
CURRENT = {"plan": None}  # the fake provider answers with the current task's reference plan


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generated(seed, stratum):
    candidate = generate_candidate(seed, stratum)
    plan = heuristic_geometry(candidate.task.scene, candidate.target_id)
    return records.GeneratedTask(metadata=candidate.metadata, task=candidate.task,
        oracle=records.TaskOracle(target_id=candidate.target_id, reference_plan=plan,
            reference_result=records.JudgeResult(success=True, predicates={"S1": True}, violations=(), details={})))


def fake_cohort(n=3, seed=0, max_candidates=10):
    tasks = tuple(generated(seed + i, ("clear", "distractor", "obstacle")[i % 3]) for i in range(n))
    exclusion = records.CandidateExclusion(metadata=records.TaskMetadata(task_id="dev_x", seed=99, stratum="clear"),
                                           reason="judge_failure")
    return records.PreparedCohort(seed=seed, requested=n, max_candidates=max_candidates, attempted=n + 1,
                                  tasks=tasks, exclusions=(exclusion,),
                                  config_hashes={k: config_hash(v) for k, v in config_snapshot().items()})


def prepare_fake(tmp_path, monkeypatch, n):
    from vplan.evaluate import pilot
    monkeypatch.setattr(pilot, "prepare_cohort", lambda **kw: fake_cohort(n=n))
    out = tmp_path / "pilot"
    pilot.prepare(out, n=n, seed=0, max_candidates=10)
    return out


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    return prepare_fake(tmp_path, monkeypatch, 3)


def status_error(code):
    error = RuntimeError(f"provider returned {code}")
    error.status_code = code
    return error


def scripted_provider(monkeypatch, replies):
    import openai
    client = FakeClient(replies)
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: client)
    return client


@pytest.fixture
def fake_provider(monkeypatch):
    """Real NemotronPlanner over a fake SDK client; the reply echoes the key to prove redaction."""
    import openai
    client = FakeClient(iter(lambda: {**response(CURRENT["plan"]), "id": f"reply {SECRET}"}, None))
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: client)
    return client


@pytest.fixture
def fake_evaluation(monkeypatch):
    """Real evaluate_pair, fake one-plan evaluator: no simulation. Each plan is answered from the oracle."""
    from vplan.evaluate import pilot
    from vplan.repair import evaluate_pair
    seen = []

    def wrapped(task, planner, **kw):
        assert set(kw) == {"config_snapshot", "event_sink", "arm_order"}
        seen.append(task.metadata.task_id)
        CURRENT["plan"] = task.oracle.reference_plan
        return evaluate_pair(task, planner, evaluator=lambda plan: executed(), **kw)
    monkeypatch.setattr(pilot, "evaluate_pair", wrapped)
    return seen


def env_file(tmp_path, base="https://openrouter.ai/api/v1", model="nvidia/nemotron-3-ultra-550b-a55b:free"):
    path = tmp_path / "test.env"
    path.write_text(f'# comment\n\nLLM_BASE_URL={base}\nLLM_API_KEY="{SECRET}"\nLLM_MODEL=\'{model}\'\n',
                    encoding="utf-8")
    return path


def run_dirs(out):
    return sorted((out / "runs").iterdir()) if (out / "runs").exists() else []


# ---- prepare ------------------------------------------------------------------------------------
def test_prepare_writes_separate_public_and_private_files_with_hashes(prepared):
    from vplan.evaluate import pilot
    public = [json.loads(line) for line in (prepared / "tasks.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(public) == 3 and all(set(row) == {"task_id", "seed", "stratum", "task"} for row in public)
    text = (prepared / "tasks.jsonl").read_text(encoding="utf-8")
    assert "target_id" not in text and "oracle" not in text and "reference" not in text
    private = [json.loads(line) for line in (prepared / "oracle.jsonl").read_text(encoding="utf-8").splitlines()]
    cohort = fake_cohort()
    assert [r["task_id"] for r in private] == [r["task_id"] for r in public]
    assert [r["oracle"]["target_id"] for r in private] == [t.oracle.target_id for t in cohort.tasks]
    assert len((prepared / "exclusions.jsonl").read_text(encoding="utf-8").splitlines()) == 1

    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"] == {name: sha(prepared / name)
                                 for name in ("tasks.jsonl", "oracle.jsonl", "exclusions.jsonl")}
    assert manifest["config_hashes"] == cohort.config_hashes
    assert (manifest["n"], manifest["seed"], manifest["max_candidates"], manifest["attempted"]) == (3, 0, 10, 4)
    assert manifest["strata"] == {"clear": 1, "distractor": 1, "obstacle": 1}
    hashes = manifest["source_hashes"]
    assert hashes["src/vplan/evaluate/pilot.py"] == sha(pilot.ROOT / "src/vplan/evaluate/pilot.py")
    assert hashes["config/judge.json"] == sha(pilot.ROOT / "config/judge.json")
    assert hashes == pilot.source_hashes() and manifest["source_digest"] == pilot.source_digest(hashes)
    assert {"schema_version", "created_utc", "python", "manipulapy", "git"} <= set(manifest)


def test_prepare_refuses_non_empty_output_without_generating(tmp_path, monkeypatch):
    from vplan.evaluate import pilot
    calls = []
    monkeypatch.setattr(pilot, "prepare_cohort", lambda **kw: calls.append(kw))
    (tmp_path / "keep.txt").write_text("evidence", encoding="utf-8")
    with pytest.raises(pilot.PilotError, match="not empty"):
        pilot.prepare(tmp_path, n=3)
    assert calls == [] and (tmp_path / "keep.txt").read_text(encoding="utf-8") == "evidence"


def test_failed_preparation_leaves_no_manifest(tmp_path, monkeypatch):
    from vplan.evaluate import pilot
    from vplan.tasks import CohortPreparationError
    def fail(**kw):
        raise CohortPreparationError(fake_cohort().model_copy(update={"requested": 5}))
    monkeypatch.setattr(pilot, "prepare_cohort", fail)
    with pytest.raises(CohortPreparationError):
        pilot.prepare(tmp_path / "pilot", n=5)
    assert not (tmp_path / "pilot" / "manifest.json").exists()
    # Rejections are still evidence when the cohort cannot be filled.
    assert len((tmp_path / "pilot" / "exclusions.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_source_change_during_preparation_leaves_no_manifest(tmp_path, monkeypatch):
    from vplan.evaluate import pilot
    monkeypatch.setattr(pilot, "prepare_cohort", lambda **kw: fake_cohort())
    snapshots = iter([{"src/vplan/a.py": "1"}, {"src/vplan/a.py": "2"}])
    monkeypatch.setattr(pilot, "source_hashes", lambda: next(snapshots))
    with pytest.raises(pilot.PilotError, match="during preparation"):
        pilot.prepare(tmp_path / "pilot", n=3)
    assert not (tmp_path / "pilot" / "manifest.json").exists()


# ---- run validation -----------------------------------------------------------------------------
@pytest.mark.parametrize("tamper, message", [
    ("tasks", "tasks.jsonl"), ("oracle", "oracle.jsonl"), ("manifest", "manifest.json"),
    ("config", "configuration"), ("source", "src/vplan/repair.py")])
def test_run_refuses_mismatched_preparation_before_any_request(prepared, monkeypatch, fake_provider, tamper, message):
    from vplan.evaluate import pilot
    if tamper in ("tasks", "oracle"):
        with open(prepared / f"{tamper}.jsonl", "a", encoding="utf-8") as f:
            f.write(" ")
    elif tamper == "manifest":
        (prepared / "manifest.json").unlink()
    elif tamper == "config":
        snapshot = config_snapshot()
        snapshot["judge"] = {**snapshot["judge"], "tampered": 1}
        monkeypatch.setattr(pilot, "config_snapshot", lambda: snapshot)
    else:
        current = pilot.source_hashes()
        monkeypatch.setattr(pilot, "source_hashes", lambda: {**current, "src/vplan/repair.py": "0" * 64})
    with pytest.raises(pilot.PilotError, match=message):
        pilot.run_live(prepared, env_file=env_file(prepared.parent))
    assert fake_provider.requests == [] and run_dirs(prepared) == []


def test_allowed_source_drift_is_recorded(prepared, monkeypatch, fake_provider, fake_evaluation):
    from vplan.evaluate import pilot
    current = pilot.source_hashes()
    monkeypatch.setattr(pilot, "source_hashes", lambda: {**current, "src/vplan/repair.py": "0" * 64})
    run = pilot.run_live(prepared, env_file=env_file(prepared.parent), allow_source_drift=True)
    meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
    assert meta["source_drift"] == ["src/vplan/repair.py"] and meta["status"] == "complete"
    assert "source drift" in (run / "report.md").read_text(encoding="utf-8").lower()


@pytest.mark.parametrize("base, model", [
    ("https://openrouter.ai/api/v1", "nvidia/nemotron-3-ultra-550b-a55b"),
    ("https://api.tokenfactory.nebius.com/v1/", "nvidia/nemotron-3-ultra-550b-a55b:free"),
    ("http://openrouter.ai.evil.example/v1", "x:free"),
    ("https://openrouter.ai/api/v1", "meta-llama/llama-3.3-70b-instruct:free")])
def test_run_refuses_paid_or_non_openrouter_without_opt_in(prepared, fake_provider, base, model):
    from vplan.evaluate import pilot
    with pytest.raises(pilot.PilotError, match="--allow-paid"):
        pilot.run_live(prepared, env_file=env_file(prepared.parent, base, model))
    assert fake_provider.requests == [] and run_dirs(prepared) == []


def test_allow_paid_and_model_override(prepared, fake_provider, fake_evaluation):
    from vplan.evaluate import pilot
    run = pilot.run_live(prepared, env_file=env_file(prepared.parent, model="paid-model"),
                         model="other/paid", allow_paid=True, limit=1)
    assert json.loads((run / "run.json").read_text(encoding="utf-8"))["model"] == "other/paid"
    assert fake_provider.requests[0]["model"] == "other/paid"


def test_env_file_parser(tmp_path):
    from vplan.evaluate import pilot
    assert pilot.read_env_file(env_file(tmp_path)) == {
        "LLM_BASE_URL": "https://openrouter.ai/api/v1", "LLM_API_KEY": SECRET,
        "LLM_MODEL": "nvidia/nemotron-3-ultra-550b-a55b:free"}
    exported = tmp_path / "exported.env"
    exported.write_text('\ufeffexport LLM_MODEL=nvidia/x:free # dev model\nLLM_API_KEY="a #b" # quoted\n',
                        encoding="utf-8")
    assert pilot.read_env_file(exported) == {"LLM_MODEL": "nvidia/x:free", "LLM_API_KEY": "a #b"}
    bad = tmp_path / "bad.env"
    bad.write_text(f"{SECRET}\n", encoding="utf-8")
    with pytest.raises(pilot.PilotError) as info:
        pilot.read_env_file(bad)
    assert SECRET not in str(info.value)


# ---- run evidence -------------------------------------------------------------------------------
def test_cli_run_writes_complete_evidence_and_never_persists_key(prepared, fake_provider, fake_evaluation, capsys):
    from vplan.cli import main
    code = main(["pilot", "run", "--input", str(prepared), "--env-file", str(env_file(prepared.parent))])
    assert code == 0
    [run] = run_dirs(prepared)
    meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
    assert meta["status"] == "complete" and meta["temperature"] == .7 and meta["base_url_host"] == "openrouter.ai"
    assert meta["manifest_sha256"] == sha(prepared / "manifest.json")
    assert all(r["temperature"] == .7 for r in fake_provider.requests) and len(fake_provider.requests) == 3
    results = (run / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert [records.PairedResult.model_validate_json(r).task_id for r in results] == fake_evaluation
    events = [json.loads(e) for e in (run / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [e["event"] for e in events].count("pair_completed") == 3
    report = (run / "report.md").read_text(encoding="utf-8")
    assert report.startswith("# Development pilot (feasible tasks only)") and "INCOMPLETE" not in report
    assert json.loads((run / "summary.json").read_text(encoding="utf-8"))["complete"] is True
    written = [p for p in prepared.parent.rglob("*") if p.is_file()]
    assert all(SECRET.encode() not in p.read_bytes() for p in written if p.suffix != ".env")
    output = capsys.readouterr()
    assert SECRET not in output.out + output.err


def test_crash_after_two_tasks_keeps_partial_evidence_marked_incomplete(prepared, fake_provider, monkeypatch):
    from vplan.evaluate import pilot
    from vplan.repair import evaluate_pair
    def crashing(task, planner, **kw):
        if crashing.calls == 2:
            raise RuntimeError("simulated crash")
        crashing.calls += 1
        CURRENT["plan"] = task.oracle.reference_plan
        return evaluate_pair(task, planner, evaluator=lambda plan: executed(), **kw)
    crashing.calls = 0
    monkeypatch.setattr(pilot, "evaluate_pair", crashing)
    with pytest.raises(RuntimeError, match="simulated crash"):
        pilot.run_live(prepared, env_file=env_file(prepared.parent))
    [run] = run_dirs(prepared)
    assert len((run / "results.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    assert json.loads((run / "run.json").read_text(encoding="utf-8"))["status"] != "complete"
    assert (run / "report.md").read_text(encoding="utf-8").startswith("# INCOMPLETE: 2/3 tasks")
    assert json.loads((run / "summary.json").read_text(encoding="utf-8"))["complete"] is False


def test_hard_crash_report_regenerates_as_incomplete(prepared, fake_provider, fake_evaluation):
    from vplan.evaluate import pilot
    run = pilot.run_live(prepared, env_file=env_file(prepared.parent))
    meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
    (run / "run.json").write_text(json.dumps({**meta, "status": "running"}), encoding="utf-8")
    pilot.write_report(run)
    assert (run / "report.md").read_text(encoding="utf-8").startswith("# INCOMPLETE: 3/3 tasks")


def test_existing_run_dir_is_never_overwritten(prepared, fake_provider, fake_evaluation):
    from vplan.evaluate import pilot
    (prepared / "runs" / "fixed").mkdir(parents=True)
    (prepared / "runs" / "fixed" / "results.jsonl").write_text("old\n", encoding="utf-8")
    with pytest.raises(pilot.PilotError, match="exists"):
        pilot.run_live(prepared, env_file=env_file(prepared.parent), run_id="fixed")
    assert (prepared / "runs" / "fixed" / "results.jsonl").read_text(encoding="utf-8") == "old\n"
    assert fake_provider.requests == []


def test_limit_runs_first_tasks_and_report_says_so(prepared, fake_provider, fake_evaluation):
    from vplan.evaluate import pilot
    run = pilot.run_live(prepared, env_file=env_file(prepared.parent), limit=1)
    assert len(fake_evaluation) == 1
    report = (run / "report.md").read_text(encoding="utf-8")
    assert "INCOMPLETE" not in report and report.startswith("# LIMITED: first 1 of 3 tasks")
    assert json.loads((run / "summary.json").read_text(encoding="utf-8"))["complete"] is False


# ---- summary ------------------------------------------------------------------------------------
def paired_results():
    """A: initial success. B: blocked, repair fixes it, resampling exhausts. C: service failure."""
    from vplan.planners.nemotron import NemotronPlanner
    from vplan.repair import evaluate_pair
    a, b, c = (generated(i, "obstacle") for i in (1, 2, 3))
    plan_b = b.oracle.reference_plan
    fixed = plan_b.model_copy(update={"transport_via": ((.5, .1, .4),)})
    def run(task, replies, evaluator):
        return evaluate_pair(task, NemotronPlanner(model="m:free", client=FakeClient(replies)), evaluator=evaluator)
    return [run(a, [response(a.oracle.reference_plan)], lambda p: executed()),
            run(b, [response(plan_b), response(fixed)] + [response(plan_b)] * 3,
                lambda p: executed() if p == fixed else blocked()),
            run(c, [RuntimeError("transport")], lambda p: executed())]


def test_summarize_keeps_failures_in_denominator_and_counts_pairs():
    from vplan.evaluate.pilot import summarize
    s = summarize(paired_results(), n_expected=3)
    assert s["complete"] is True and s["tasks_completed"] == 3
    assert s["success"]["first_attempt"] == {"count": 1, "denominator": 3, "rate": 1 / 3}
    assert s["success"]["repair"]["count"] == 2 and s["success"]["resample"]["count"] == 1
    assert s["success"]["resample"]["denominator"] == 3
    # Only B's arms continued past the shared initial call; C's service failure ended both arms.
    assert s["paired"] == {"repair_only": 1, "resample_only": 0, "both": 1, "neither": 1,
                           "repair_opportunities": 1}
    assert s["initial_terminal_failures"] == {"service_failure": 1}
    assert s["service_failures"] == {"total": 1, "initial": {"transport": 1}, "repair": {"transport": 1},
                                     "resample": {"transport": 1}}
    assert s["outcomes"]["resample"] == {"executed": 1, "budget_exhausted": 1, "service_failure": 1}
    assert s["finding_checks"]["first_round"] == {"collision": 1}
    assert s["finding_checks"]["resample"] == {"collision": 4}
    assert s["repeat_plans"]["resample"] == 3


def test_summarize_counts_shared_initial_call_once():
    from vplan.evaluate.pilot import summarize
    cost = summarize(paired_results(), n_expected=3)["cost"]
    assert cost["logical_calls"] == {"repair": 4, "resample": 6}
    assert cost["unique_requests"] == 7  # 4 + 6 minus the three shared initial requests
    assert cost["tokens"]["unique"] == {"known_total": 6 * 120, "missing_usage_calls": 1}
    assert cost["latency_s"]["unique"]["total"] >= 0 and "median_per_request" in cost["latency_s"]["unique"]


def test_summarize_marks_partial_results_incomplete():
    from vplan.evaluate.pilot import summarize
    s = summarize(paired_results()[:2], n_expected=20)
    assert s["complete"] is False and s["tasks_completed"] == 2 and s["success"]["repair"]["denominator"] == 2


# ---- CLI wiring ---------------------------------------------------------------------------------
def test_cli_pilot_argument_wiring(monkeypatch, tmp_path, capsys):
    from vplan.cli import main
    from vplan.evaluate import pilot
    calls = {}
    monkeypatch.setattr(pilot, "prepare", lambda out, **kw: calls.setdefault("prepare", (str(out), kw)))
    monkeypatch.setattr(pilot, "run_live", lambda inp, **kw: calls.setdefault("run", (str(inp), kw)) and tmp_path)
    assert main(["pilot", "prepare", "--output", "d", "--n", "5", "--seed", "3", "--max-candidates", "7"]) == 0
    assert calls["prepare"][0] == "d"
    assert {k: calls["prepare"][1][k] for k in ("n", "seed", "max_candidates")} == {"n": 5, "seed": 3, "max_candidates": 7}
    assert main(["pilot", "run", "--input", "d", "--env-file", "e.env", "--limit", "2", "--allow-source-drift"]) == 0
    assert calls["run"] == ("d", {"env_file": "e.env", "model": None, "temperature": .7, "limit": 2,
                                  "allow_source_drift": True, "allow_paid": False})


def test_cli_pilot_defaults_and_clear_error(monkeypatch, tmp_path, capsys):
    from vplan.cli import main
    from vplan.evaluate import pilot
    seen = {}
    def refuse(out, **kw):
        seen.update(kw)
        raise pilot.PilotError("output directory is not empty")
    monkeypatch.setattr(pilot, "prepare", refuse)
    assert main(["pilot", "prepare", "--output", str(tmp_path)]) == 2
    assert (seen["n"], seen["seed"], seen["max_candidates"]) == (20, 0, 100)
    assert "not empty" in capsys.readouterr().err


def test_report_skips_a_truncated_final_result_line_and_flags_it(prepared, fake_provider, fake_evaluation):
    from vplan.evaluate import pilot
    run = pilot.run_live(prepared, env_file=env_file(prepared.parent))
    path = run / "results.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + '{"task_id": "cut-off', encoding="utf-8")  # killed mid-write
    summary = pilot.write_report(run)
    assert summary["truncated_result_lines"] == 1 and summary["complete"] is False
    assert (run / "report.md").read_text(encoding="utf-8").startswith("# INCOMPLETE")


# ---- review fixes: degraded runs, arm alternation, guards ------------------------------------------
def test_auth_failure_stops_the_run_incomplete_and_degraded(prepared, monkeypatch):
    from vplan.evaluate import pilot
    client = scripted_provider(monkeypatch, [status_error(401)] * 5)
    with pytest.raises(pilot.PilotError, match="401"):
        pilot.run_live(prepared, env_file=env_file(prepared.parent))
    [run] = run_dirs(prepared)
    assert len(client.requests) == 1
    assert len((run / "results.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
    assert meta["status"] == "incomplete" and "401" in meta["stop_reason"]
    report = (run / "report.md").read_text(encoding="utf-8")
    assert report.startswith("# INCOMPLETE: 1/3 tasks") and "DEGRADED: 1 service failures" in report.splitlines()[0]


def test_three_consecutive_rate_limits_across_tasks_stop_the_run(tmp_path, monkeypatch):
    from vplan.evaluate import pilot
    prepared = prepare_fake(tmp_path, monkeypatch, 5)
    client = scripted_provider(monkeypatch, [status_error(429)] * 10)
    with pytest.raises(pilot.PilotError, match="consecutive"):
        pilot.run_live(prepared, env_file=env_file(prepared.parent))
    [run] = run_dirs(prepared)
    assert len(client.requests) == 3
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["complete"] is False and summary["service_failures"]["initial"] == {"429": 3}
    assert (run / "report.md").read_text(encoding="utf-8").startswith("# INCOMPLETE: 3/5 tasks")


def test_end_to_end_multi_round_and_service_failure_with_alternating_arms(prepared, monkeypatch):
    """Task 0 succeeds at once; task 1 (resample first) is repaired but not resampled; task 2 hits a 500."""
    from vplan.evaluate import pilot
    from vplan.repair import evaluate_pair
    tasks = fake_cohort().tasks
    bad = tasks[1].oracle.reference_plan.model_copy(update={"rationale": "blocked-me"})
    client = scripted_provider(monkeypatch, [response(tasks[0].oracle.reference_plan)]
                               + [response(bad)] * 4 + [response(tasks[1].oracle.reference_plan)]
                               + [status_error(500)])
    monkeypatch.setattr(pilot, "evaluate_pair", lambda task, planner, **kw: evaluate_pair(
        task, planner, evaluator=lambda p: blocked() if p.rationale == "blocked-me" else executed(), **kw))
    run = pilot.run_live(prepared, env_file=env_file(prepared.parent))
    assert len(client.requests) == 7
    results = [records.PairedResult.model_validate_json(line)
               for line in (run / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r.arm_order for r in results] == [("repair", "resample"), ("resample", "repair"), ("repair", "resample")]
    events = [json.loads(e) for e in (run / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [e["arm"] for e in events if e["event"] == "request_started" and e["task_id"] == results[1].task_id] \
        == ["initial", "resample", "resample", "resample", "repair"]
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert [summary["success"][k]["count"] for k in ("first_attempt", "repair", "resample")] == [1, 2, 1]
    assert summary["success"]["repair"]["denominator"] == 3
    assert summary["paired"]["repair_only"] == 1 and summary["paired"]["repair_opportunities"] == 1
    assert summary["initial_terminal_failures"] == {"service_failure": 1}
    assert summary["service_failures"]["initial"] == {"500": 1} and summary["complete"] is True
    assert summary["cost"]["logical_calls"] == {"repair": 4, "resample": 6} and summary["cost"]["unique_requests"] == 7
    report = (run / "report.md").read_text(encoding="utf-8")
    assert report.startswith("# DEGRADED: 1 service failures") and "| repair | 2 | 3 |" in report


def test_run_json_records_provenance_and_report_warns_on_manipulapy_change(prepared, monkeypatch,
                                                                            fake_provider, fake_evaluation):
    from vplan.evaluate import pilot
    versions = pilot._versions()
    monkeypatch.setattr(pilot, "_versions", lambda: {**versions, "ManipulaPy": "0.0.0"})
    run = pilot.run_live(prepared, env_file=env_file(prepared.parent))
    meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
    assert meta["source_digest"] == pilot.source_digest(pilot.source_hashes())
    assert {"git", "python"} <= set(meta) and set(meta["versions"]) == {"openai", "pybullet", "ManipulaPy"}
    assert "ManipulaPy version differs" in (run / "report.md").read_text(encoding="utf-8")


def test_cli_report_exit_code_is_nonzero_only_for_incomplete_runs(prepared, fake_provider, fake_evaluation):
    from vplan.cli import main
    from vplan.evaluate import pilot
    limited = pilot.run_live(prepared, env_file=env_file(prepared.parent), limit=1, run_id="limited")
    assert main(["pilot", "report", "--run", str(limited)]) == 0
    meta = json.loads((limited / "run.json").read_text(encoding="utf-8"))
    (limited / "run.json").write_text(json.dumps({**meta, "status": "running"}), encoding="utf-8")
    assert main(["pilot", "report", "--run", str(limited)]) == 1


def test_missing_inputs_give_clean_errors(prepared, tmp_path, capsys):
    from vplan.cli import main
    assert main(["pilot", "run", "--input", str(prepared), "--env-file", str(tmp_path / "absent.env")]) == 2
    assert main(["pilot", "report", "--run", str(tmp_path / "no-run")]) == 2
    (prepared / "manifest.json").write_text('{"schema_version": 1}', encoding="utf-8")
    assert main(["pilot", "run", "--input", str(prepared), "--env-file", str(env_file(tmp_path))]) == 2
    err = capsys.readouterr().err
    assert err.count("error: ") == 3 and "Traceback" not in err and "manifest.json" in err


def test_wsl_paths_are_rejected_for_the_windows_interpreter(monkeypatch, capsys):
    from vplan.cli import main
    from vplan.evaluate import pilot
    monkeypatch.setattr(pilot, "_WINDOWS", True)
    monkeypatch.setattr(pilot, "prepare", lambda out, **kw: pytest.fail("must refuse before preparing"))
    assert main(["pilot", "prepare", "--output", "/mnt/c/Users/x/pilot"]) == 2
    assert "Windows" in capsys.readouterr().err


def multi_round_run(prepared, monkeypatch):
    """Task 1 is blocked once in the repair arm, then repaired (same script as the end-to-end test)."""
    from vplan.evaluate import pilot
    from vplan.repair import evaluate_pair
    tasks = fake_cohort().tasks
    bad = tasks[1].oracle.reference_plan.model_copy(update={"rationale": "blocked-me"})
    scripted_provider(monkeypatch, [response(tasks[0].oracle.reference_plan)] + [response(bad)] * 4
                      + [response(tasks[1].oracle.reference_plan)] + [status_error(500)])
    monkeypatch.setattr(pilot, "evaluate_pair", lambda task, planner, **kw: evaluate_pair(
        task, planner, evaluator=lambda p: blocked() if p.rationale == "blocked-me" else executed(), **kw))
    return pilot.run_live(prepared, env_file=env_file(prepared.parent)), tasks


def test_select_round_returns_the_recorded_plan_and_task(prepared, monkeypatch):
    from vplan.evaluate import pilot
    run, tasks = multi_round_run(prepared, monkeypatch)
    task_id = tasks[1].metadata.task_id
    task, rnd = pilot.select_round(run, task_id, "repair", 0)
    assert task.metadata.task_id == task_id and rnd.round_index == 0
    assert rnd.call.plan.rationale == "blocked-me" and rnd.evaluation.outcome == "blocked"
    task, last = pilot.select_round(run, task_id, "repair")  # default: the arm's final round
    assert last.round_index == 1 and last.evaluation.outcome == "executed"
    assert last.call.plan == tasks[1].oracle.reference_plan


def test_select_round_errors_are_clean(prepared, monkeypatch):
    from vplan.evaluate import pilot
    run, tasks = multi_round_run(prepared, monkeypatch)
    with pytest.raises(pilot.PilotError, match="no task"):
        pilot.select_round(run, "dev_missing", "repair")
    with pytest.raises(pilot.PilotError, match="round 3"):
        pilot.select_round(run, tasks[1].metadata.task_id, "repair", 3)


def test_list_rounds_summarizes_each_task_and_arm(prepared, monkeypatch):
    from vplan.evaluate import pilot
    run, tasks = multi_round_run(prepared, monkeypatch)
    lines = pilot.list_rounds(run)
    assert len(lines) == 3
    assert lines[1].startswith(tasks[1].metadata.task_id) and "repair: blocked -> executed" in lines[1]
    assert "service_failure" in lines[2]


def test_cli_pilot_show_wiring(monkeypatch, tmp_path, capsys):
    from vplan import cli
    from vplan.evaluate import pilot
    calls = []
    monkeypatch.setattr(pilot, "list_rounds", lambda run: ["dev_a  repair: executed"])
    monkeypatch.setattr(cli, "_show_round", lambda *a, **kw: calls.append((a, kw)) or 0)
    assert cli.main(["pilot", "show", "--run", str(tmp_path)]) == 0
    assert "dev_a" in capsys.readouterr().out and calls == []
    assert cli.main(["pilot", "show", "--run", str(tmp_path), "--task", "dev_a", "--arm", "resample",
                     "--round", "1", "--unverified"]) == 0
    assert calls == [((str(tmp_path), "dev_a", "resample", 1), {"unverified": True})]


def test_show_round_survives_closed_window_and_closed_stdin(monkeypatch, capsys):
    """Closing the GUI window or running without a terminal must still print the outcome."""
    import builtins
    import pybullet as p
    from vplan import cli
    from vplan.evaluate import pilot
    task = fake_cohort(n=1).tasks[0]
    rnd = records.RoundRecord(call_id="c0", round_index=0, call=records.PlannerCall.model_validate_json(
        json.dumps({"outcome": "proposed", "plan": task.oracle.reference_plan.model_dump(mode="json"),
                    "provider": "x", "model": "m", "protocol": "forced", "next_protocol": "forced",
                    "temperature": 0.7, "latency_s": 0.0, "raw_request": {}})))
    monkeypatch.setattr(pilot, "select_round", lambda *a: (task, rnd))
    client = p.connect(p.DIRECT)
    monkeypatch.setattr(cli, "_open_gui", lambda: client)

    def fake_attempt(*a, **kw):
        p.disconnect(client)  # the user closed the window
        return type("R", (), {"outcome": "executed", "findings": [], "judge": None, "visual": None})()
    monkeypatch.setattr(cli, "run_attempt", fake_attempt)
    monkeypatch.setattr(builtins, "input", lambda *a: (_ for _ in ()).throw(EOFError()))
    assert cli._show_round("run", task.metadata.task_id, "repair", 0, unverified=False) == 1
    assert "outcome: executed" in capsys.readouterr().out
