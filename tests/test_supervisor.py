"""R9 supervisor fault injection: crash, kill, hang, fake 429 and garbage stdout, with real worker subprocesses."""
import json
import os

import pytest

from vplan import records
from vplan.evaluate import supervisor
from vplan.evaluate.supervisor import AttemptJob, SupervisorError, run_batch

FAULTS = ("tests.supervisor_jobs",)


def batch(tmp_path, jobs, **kw):
    kw.setdefault("timeout_s", 60.0)
    kw.setdefault("import_modules", FAULTS)
    kw.setdefault("sleep", lambda s: None)
    return run_batch(jobs, tmp_path / "run", **kw)


def lines(tmp_path):
    text = (tmp_path / "run" / "results.jsonl").read_text(encoding="utf-8")
    return [records.SupervisedAttempt.model_validate_json(line) for line in text.splitlines()]


def test_batch_survives_crash_kill_hang_and_error(tmp_path):
    jobs = [AttemptJob("a:0", "fault_crash"), AttemptJob("a:1", "fault_kill"), AttemptJob("a:2", "fault_hang"),
            AttemptJob("a:3", "fault_raise"), AttemptJob("a:4", "fault_ok", {"echo": "still alive"})]
    out = batch(tmp_path, jobs, timeout_s=8.0)
    assert out.status == "complete" and out.stop_reason is None
    got = {a.attempt_id: a for a in out.attempts}
    assert [a.attempt_id for a in lines(tmp_path)] == ["a:0", "a:1", "a:2", "a:3", "a:4"]
    assert got["a:0"].outcome == "sim_error" and got["a:0"].error_type == "worker_crash"
    assert got["a:0"].exit_code == 70
    assert got["a:1"].outcome == "sim_error" and got["a:1"].error_type == "worker_crash"
    assert got["a:2"].outcome == "sim_error" and got["a:2"].error_type == "worker_timeout"
    assert 8.0 <= got["a:2"].wall_s < 40.0
    assert got["a:3"].outcome == "sim_error" and got["a:3"].error_type == "RuntimeError"
    assert "motion layer exploded" in got["a:3"].error_traceback
    assert got["a:4"].outcome == "executed" and got["a:4"].payload == {"echo": "still alive"}


def test_an_after_step_runs_once_the_result_is_written_and_can_never_lose_it(tmp_path):
    # The result file is written before the after step (a key-frame export), so a crash or a hang there keeps the
    # attempt's result; only the after step's own payload is missing.
    jobs = [AttemptJob("f:0", "fault_after_ok"), AttemptJob("f:1", "fault_after_raise"),
            AttemptJob("f:2", "fault_after_crash"), AttemptJob("f:3", "fault_after_hang")]
    got = {a.attempt_id: a for a in batch(tmp_path, jobs, timeout_s=8.0).attempts}
    assert all(a.outcome == "executed" and a.error_type is None for a in got.values())
    assert got["f:0"].payload == {"echo": "base", "extra": True} and got["f:0"].exit_code == 0
    assert got["f:1"].payload == {"echo": "base", "after_error": {"error_type": "RuntimeError",
                                                                 "error_message": "renderer exploded"}}
    assert got["f:2"].payload == {"echo": "base"} and got["f:2"].exit_code == 70
    assert got["f:3"].payload == {"echo": "base"} and got["f:3"].exit_code is None and got["f:3"].wall_s >= 8.0


def test_in_process_run_includes_the_after_step():
    import tests.supervisor_jobs  # noqa: F401 (registers the fault jobs)
    from vplan.evaluate import worker
    assert worker.run("fault_after_ok", "f:0", {}).payload == {"echo": "base", "extra": True}
    assert worker.run("fault_after_raise", "f:1", {}).payload["after_error"]["error_type"] == "RuntimeError"


def test_garbage_stdout_never_reaches_results(tmp_path):
    out = batch(tmp_path, [AttemptJob("g", "fault_garbage_stdout")])
    (a,) = out.attempts
    assert a.outcome == "executed" and a.payload == {"clean": True}
    logs = list((tmp_path / "run" / "workers").glob("*.stdout.log"))
    assert logs and b"banner" in logs[0].read_bytes()  # kept for diagnosis, never parsed


def test_invalid_job_return_is_sim_error(tmp_path):
    (a,) = batch(tmp_path, [AttemptJob("b", "fault_bad_return")]).attempts
    assert a.outcome == "sim_error" and a.error_type == "ValidationError"


def test_unknown_job_is_sim_error(tmp_path):
    (a,) = batch(tmp_path, [AttemptJob("u", "no_such_job")]).attempts
    assert a.outcome == "sim_error" and a.error_type == "unknown_job"


def test_429_without_retries_is_service_failure(tmp_path):
    sleeps = []
    out = batch(tmp_path, [AttemptJob("r", "fault_429"), AttemptJob("ok", "fault_ok")], sleep=sleeps.append)
    r, ok = out.attempts
    assert r.outcome == "service_failure" and r.status_code == 429 and r.tries == 1
    assert ok.outcome == "executed" and out.status == "complete"


def test_429_retry_backs_off_and_recovers(tmp_path):
    sleeps = []
    job = AttemptJob("f", "fault_flaky_429", {"marker": str(tmp_path / "marker")})
    (a,) = batch(tmp_path, [job], service_retries=2, backoff_s=2.0, sleep=sleeps.append).attempts
    assert a.outcome == "executed" and a.tries == 2 and a.payload == {"recovered": True}
    assert sleeps == [2.0]  # max(backoff, Retry-After 0.5)


def test_429_retries_exhausted_with_exponential_backoff(tmp_path):
    sleeps = []
    job = AttemptJob("r", "fault_429", {"retry_after_s": 3.0})
    (a,) = batch(tmp_path, [job], service_retries=2, backoff_s=1.0, max_backoff_s=10.0, sleep=sleeps.append).attempts
    assert a.outcome == "service_failure" and a.tries == 3
    assert sleeps == [3.0, 3.0]  # Retry-After dominates 1 s, 2 s


def test_consecutive_service_failures_stop_the_batch(tmp_path):
    jobs = [AttemptJob(f"r{i}", "fault_429") for i in range(3)] + [AttemptJob("never", "fault_ok")]
    out = batch(tmp_path, jobs)
    assert out.status == "incomplete" and "consecutive" in out.stop_reason
    assert [a.attempt_id for a in lines(tmp_path)] == ["r0", "r1", "r2"]


def test_success_resets_the_consecutive_count(tmp_path):
    jobs = [AttemptJob("r0", "fault_429"), AttemptJob("r1", "fault_429"), AttemptJob("ok", "fault_ok"),
            AttemptJob("r2", "fault_429")]
    assert batch(tmp_path, jobs).status == "complete"


def test_fatal_status_stops_immediately_without_retry(tmp_path):
    jobs = [AttemptJob("auth", "fault_429", {"status": 401}), AttemptJob("never", "fault_ok")]
    out = batch(tmp_path, jobs, service_retries=3)
    assert out.status == "incomplete" and "401" in out.stop_reason
    assert [(a.attempt_id, a.tries) for a in out.attempts] == [("auth", 1)]


def test_fatal_status_mirrors_pilot_stop_rule():
    from vplan.evaluate import pilot
    assert supervisor.FATAL_STATUS == pilot.FATAL_STATUS
    assert supervisor.MAX_CONSECUTIVE_SERVICE_FAILURES == pilot.MAX_CONSECUTIVE_SERVICE_FAILURES


def test_parent_is_the_only_results_writer_and_refuses_reuse(tmp_path):
    batch(tmp_path, [AttemptJob("x", "fault_ok")])
    run = tmp_path / "run"
    worker_files = {p.name for p in (run / "workers").iterdir()}
    assert "results.jsonl" not in worker_files and any(n.endswith(".result.json") for n in worker_files)
    with pytest.raises(SupervisorError, match="already"):
        batch(tmp_path, [AttemptJob("y", "fault_ok")])


def test_duplicate_attempt_ids_refused(tmp_path):
    with pytest.raises(SupervisorError, match="duplicate"):
        batch(tmp_path, [AttemptJob("d", "fault_ok"), AttemptJob("d", "fault_ok")])
    assert not (tmp_path / "run").exists()


def test_attempt_ids_with_colons_are_safe_file_names(tmp_path):
    # W3.1 IDs look like "3:dev:7"; ':' is not a valid Windows file-name character.
    (a,) = batch(tmp_path, [AttemptJob("3:dev:7", "fault_ok")]).attempts
    assert a.attempt_id == "3:dev:7" and a.outcome == "executed"


def test_child_env_forces_utf8_and_finds_the_package():
    env = supervisor.child_env({"PYTHONPATH": "elsewhere", "PYTHONUTF8": "0"})
    assert env["PYTHONUTF8"] == "1"
    parts = env["PYTHONPATH"].split(os.pathsep)
    assert parts[-1] == "elsewhere" and os.path.isdir(os.path.join(parts[0], "vplan"))


def test_worker_command_is_an_argument_list():
    cmd = supervisor.worker_command("basic_attempt", "a:0", "params.json", "out.json", ("tests.supervisor_jobs",))
    assert cmd[1:3] == ["-m", "vplan.evaluate.worker"] and all(isinstance(c, str) for c in cmd)


def test_read_result_rejects_garbage_and_mismatched_ids(tmp_path):
    path = tmp_path / "r.json"
    path.write_bytes(b"\xff not json")
    assert supervisor.read_result(path, "a") is None
    good = records.WorkerResult(attempt_id="b", job="fault_ok", outcome="executed")
    path.write_text(good.model_dump_json(), encoding="utf-8")
    assert supervisor.read_result(path, "a") is None
    assert supervisor.read_result(path, "b") == good
    assert supervisor.read_result(tmp_path / "missing.json", "b") is None


def test_supervise_runs_one_attempt_at_any_stem_without_a_results_file(tmp_path):
    # W3.1 reuses the per-attempt loop with its own sink (attempts.jsonl) and resume-safe file stems.
    (tmp_path / "w").mkdir()
    sleeps = []
    job = AttemptJob("5:dev:12", "fault_flaky_429", {"marker": str(tmp_path / "marker")})
    a = supervisor.supervise(job, tmp_path / "w" / "0012.s1", timeout_s=60.0, service_retries=1, backoff_s=1.0,
                             import_modules=FAULTS, sleep=sleeps.append)
    assert isinstance(a, records.SupervisedAttempt) and a.attempt_id == "5:dev:12"
    assert a.outcome == "executed" and a.tries == 2 and sleeps == [1.0]
    names = {p.name for p in (tmp_path / "w").iterdir()}
    assert {"0012.s1.params.json", "0012.s1.try1.result.json", "0012.s1.try2.result.json"} <= names
    assert json.loads((tmp_path / "w" / "0012.s1.params.json").read_text(encoding="utf-8")) == job.params
    assert not any(n.endswith(".jsonl") for n in names)


def test_basic_attempt_job_runs_heuristic_plan_in_a_worker(tmp_path):
    (a,) = batch(tmp_path, [AttemptJob("basic:0", "basic_attempt", {"target_id": "cube_red"})],
                 timeout_s=300.0, import_modules=()).attempts
    assert a.outcome == "executed", a.error_traceback
    assert a.payload["judge_success"] is True and a.payload["findings"] == []
    json.dumps(a.payload)
