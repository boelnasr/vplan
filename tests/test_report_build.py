"""vplan report build: W3.1 run directories joined with their seed set into report rows, then metrics and HTML.

The runs are real W3.1 run directories written by runner.run_eval; only the supervised worker is replaced by an instant
stand-in that runs the real run_task with a scripted planner answering as the job's planner. No subprocess, simulation,
network or model call.
"""
import json
import os

import pytest

from tests import eval_rows as ev
from tests.test_eval_runner import FakeSupervise, known_git  # noqa: F401 (module-scoped autouse: a pinned git SHA)
from tests.test_generate import _fake_witness
from tests.test_pilot import env_file
from vplan import records
from vplan.planners.heuristic import heuristic_geometry
from vplan.repair import run_task


WALL_S, RENDER_S = 0.25, 0.125  # a scripted attempt's supervisor time, and its key-frame export's share of it


class ScriptedSupervise:
    """Stand-in for supervisor.supervise: feasible tasks get the canonical plan and a passing Judge, impossible tasks
    the proof's refusal; every call carries the job's provider and model, as the real planner's would. With render_s,
    a job that asks for frames also gets the worker's export (written for an execution, none for a refusal), which
    took render_s of the attempt's WALL_S; without it, the worker reports no export, as after a crash while
    rendering."""

    def __init__(self, interrupt_at=None, execution=None, render_s=None):
        self.interrupt_at, self.jobs, self.execution, self.render_s = interrupt_at, [], execution, render_s

    def __call__(self, job, stem, **kwargs):
        if self.interrupt_at is not None and len(self.jobs) == self.interrupt_at:
            raise KeyboardInterrupt
        self.jobs.append(job)
        p = job.params
        task, oracle = records.TaskSpec.model_validate(p["task"]), records.TaskOracle.model_validate(p["oracle"])
        source = (p["planner"]["provider"], p["planner"]["model"])
        if oracle.feasibility == "feasible":
            calls, evaluations = [ev.call(heuristic_geometry(task.scene, oracle.target_id), source=source)], [
                ev.judged().model_copy(update={"record": self.execution})]
        else:
            calls, evaluations = [ev.call(ev.refusal(oracle.proof.refusal_reason), source=source)], []
        record = run_task(task, oracle, ev.Scripted(calls), k=p["k"], mode=p["mode"],
                          evaluator=lambda plan: evaluations.pop(0), attempt_id=job.attempt_id,
                          config_snapshot=p["config_snapshot"])
        payload = {"record": record.model_dump(mode="json")}
        if p.get("frames") is not None and self.render_s is not None:
            export = (records.FrameExport(status="written", index=f"{p['frames']['rel']}/index.json",
                                          index_sha256="ab" * 32, rounds=(0,), files=9, bytes=1000,
                                          render_s=self.render_s)
                      if record.outcome == "executed" else records.FrameExport(status="none", render_s=self.render_s))
            payload["frames"] = export.model_dump(mode="json")
        return records.SupervisedAttempt(attempt_id=job.attempt_id, job=job.job, outcome=record.outcome, tries=1,
                                         wall_s=WALL_S, exit_code=0, payload=payload)


def run(config, seed_dir, run_dir, *, fake=None, **kw):
    from vplan.evaluate import runner
    if config != 5:
        kw.setdefault("env_file", env_file(run_dir.parent))
    return runner.run_eval(config, "dev", seed_dir=seed_dir, run_dir=run_dir, echo=lambda *a: None,
                           supervise_attempt=fake or ScriptedSupervise(), **kw)


@pytest.fixture(scope="module")
def seed_dir(tmp_path_factory):
    """Five tasks in slot order: feasible, feasible, impossible type 1, feasible, impossible type 2."""
    from vplan.evaluate import seedsets
    out = tmp_path_factory.mktemp("build") / "seeds"
    seedsets.prepare_seed_set(out, "dev", feasible=3, impossible=2, witness_evaluator=_fake_witness)
    return out


@pytest.fixture(scope="module")
def other_seed_dir(tmp_path_factory):
    from vplan.evaluate import seedsets
    out = tmp_path_factory.mktemp("other") / "seeds"
    seedsets.prepare_seed_set(out, "dev", feasible=3, impossible=1, witness_evaluator=_fake_witness)
    return out


@pytest.fixture(scope="module")
def runs(seed_dir, tmp_path_factory):
    """Complete runs of configs 5, 3 and 7 over the five tasks; config 3 renders key frames, as by default."""
    root = tmp_path_factory.mktemp("runs")
    out = {}
    for config in (5, 3, 7):
        out[config] = root / f"c{config}"
        assert run(config, seed_dir, out[config], fake=ScriptedSupervise(render_s=RENDER_S)).status == "complete"
    return out


def attempts(run_dir):
    return [records.EvalAttempt.model_validate_json(line)
            for line in (run_dir / "attempts.jsonl").read_text(encoding="utf-8").splitlines()]


# ---- the loader --------------------------------------------------------------------------------------------
def test_every_attempt_becomes_a_report_row_joined_with_its_seed_set_task(runs, seed_dir):
    from vplan.evaluate import collect, seedsets
    got = collect.collect_runs([runs[5], runs[3], runs[7]])
    seeds = seedsets.load_seed_set(seed_dir)
    assert got.seed_set_size == 5 and got.undetermined == seeds.manifest["excluded"]
    assert [m["config"] for m in got.manifests] == [5, 3, 7]
    assert len(got.rows) == 15
    for config, run_dir in runs.items():
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        rows = [r for r in got.rows if r.config == config]
        for row, attempt in zip(rows, attempts(run_dir), strict=True):
            task = seeds.tasks[attempt.task_index]
            assert (row.config, row.task_index, row.task_id, row.outcome, row.error_type) == (
                attempt.config, attempt.task_index, attempt.task_id, attempt.outcome, attempt.error_type)
            # The row keeps the supervisor's time; the report row leaves config 3's key-frame export out of it.
            assert attempt.wall_s == WALL_S and row.wall_s == (WALL_S - RENDER_S if config == 3 else WALL_S)
            assert row.seed_set == attempt.split == "dev" and row.attempt == attempt.record
            assert (row.seed_set_hash, row.provider, row.model) == (manifest["seed_set_hash"], manifest["provider"],
                                                                    manifest["model"])
            assert (row.task, row.distractors, row.obstacles) == (task.task, task.distractors, task.obstacles)
            assert row.oracle == seeds.oracles[attempt.task_index]
    assert {(r.provider, r.model) for r in got.rows if r.config == 5} == {("heuristic", "template-heuristic")}


STREAMS = ("q_replay", "finger_replay", "object_replay", "sigma_min_steps", "rest_lin_hist", "rest_ang_hist",
           "rest_pose_hist")


def test_report_rows_drop_the_per_step_streams_and_keep_every_metric(seed_dir, tmp_path):
    # D56: each row kept about 1.2 MB of streams that no metric reads, so a 400-task report needed 1.5-2 GB of RAM.
    from vplan.evaluate import collect
    from vplan.evaluate.metrics import compute_metrics
    run_dir = tmp_path / "c3"
    assert run(3, seed_dir, run_dir, fake=ScriptedSupervise(execution=ev.execution_record())).status == "complete"
    got = collect.collect_runs([run_dir])
    executed = [r.attempt.rounds[-1].evaluation.record for r in got.rows if r.outcome == "executed"]
    assert len(executed) == 3 and all(rec.approval == "ab" * 32 and rec.events for rec in executed)
    assert all(not getattr(rec, name) for rec in executed for name in STREAMS)
    full = [r.model_copy(update={"attempt": a.record}) for r, a in zip(got.rows, attempts(run_dir), strict=True)]
    assert all(getattr(r.attempt.rounds[-1].evaluation.record, name) for r in full if r.outcome == "executed"
               for name in STREAMS)
    assert [collect.without_streams(r.attempt) for r in full] == [r.attempt for r in got.rows]
    kw = dict(manifests=got.manifests, undetermined=got.undetermined, seed_set_size=got.seed_set_size)
    assert json.dumps(compute_metrics(got.rows, **kw)) == json.dumps(compute_metrics(full, **kw))


def test_without_streams_keeps_records_without_an_execution_unchanged():
    from vplan.evaluate import collect
    assert collect.without_streams(None) is None
    record = ev.attempt(ev.FEASIBLE, [ev.call(ev.GOOD)], [ev.blocked(ev.COLLISION)], k=0)
    assert collect.without_streams(record) is record


def test_report_wall_time_leaves_out_the_key_frame_export(runs):
    # The worker renders frames inside the supervised try, after the result is final: about 1.3 s per executed
    # attempt on the acceptance runs, and only config 3 renders by default. Raw wall times would bias the spec §14
    # latency-per-configuration comparison, including the config 3 vs 7 equal-budget pair.
    from vplan.evaluate import collect
    from vplan.evaluate.metrics import compute_metrics
    assert all(a.frames.render_s == RENDER_S and a.wall_s == WALL_S for a in attempts(runs[3]))
    assert all(a.frames is None for c in (5, 7) for a in attempts(runs[c]))
    got = collect.collect_runs([runs[5], runs[3], runs[7]])
    m = compute_metrics(got.rows, manifests=got.manifests, undetermined=got.undetermined,
                        seed_set_size=got.seed_set_size)
    latency = {c: m["configs"][str(c)]["latency_s"] for c in (3, 5, 7)}
    assert (latency[3]["wall_per_attempt_median"], latency[3]["wall_total"], latency[3]["wall_attempts"]) == (
        WALL_S - RENDER_S, 5 * (WALL_S - RENDER_S), 5)
    assert latency[5]["wall_per_attempt_median"] == latency[7]["wall_per_attempt_median"] == WALL_S


def test_an_attempt_whose_export_did_not_report_its_time_has_no_report_wall_time(seed_dir, tmp_path):
    # A crash or hang while rendering keeps the attempt (the result was written first), but its wall time then holds
    # an unknown render share, up to the attempt timeout; the report counts it among attempts without wall time.
    from vplan.evaluate import collect
    from vplan.evaluate.metrics import compute_metrics
    run_dir = tmp_path / "c3"
    assert run(3, seed_dir, run_dir).status == "complete"  # the scripted worker reports no export
    assert all(a.frames.status == "failed" and a.frames.render_s is None for a in attempts(run_dir))
    got = collect.collect_runs([run_dir])
    assert [r.wall_s for r in got.rows] == [None] * 5
    latency = compute_metrics(got.rows)["configs"]["3"]["latency_s"]
    assert (latency["wall_attempts"], latency["wall_per_attempt_median"]) == (0, None)


def test_the_seed_set_defaults_to_the_one_the_runs_recorded(runs, seed_dir):
    from vplan.evaluate import collect
    assert collect.collect_runs([runs[5]], seed_dir=seed_dir).rows == collect.collect_runs([runs[5]]).rows


@pytest.mark.parametrize("damage", ["absent", "omitted"])
def test_attempts_without_matching_session_provenance_are_marked_unknown(seed_dir, tmp_path, damage):
    from vplan.evaluate import collect
    run_dir = tmp_path / "c5"
    assert run(5, seed_dir, run_dir).status == "complete"
    sessions = run_dir / "sessions.jsonl"
    if damage == "absent":
        sessions.unlink()
    else:
        sessions.write_text("", encoding="utf-8")
    got = collect.collect_runs([run_dir])
    assert got.manifests[0]["sessions"] == []
    assert got.manifests[0]["session_provenance_complete"] is False


@pytest.mark.parametrize("damage", ["bool_session", "float_session", "string_session", "missing_source_digest"])
def test_malformed_session_provenance_is_rejected(seed_dir, tmp_path, damage):
    from vplan.evaluate import collect
    run_dir = tmp_path / "c5"
    assert run(5, seed_dir, run_dir).status == "complete"
    path = run_dir / "sessions.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    event = json.loads(lines[0])
    if damage == "bool_session":
        event["session"] = False
    elif damage == "float_session":
        event["session"] = 0.0
    elif damage == "string_session":
        event["session"] = "0"
    else:
        del event["source_digest"]
    lines[0] = json.dumps(event)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(collect.EvalError, match="sessions.jsonl line 1"):
        collect.collect_runs([run_dir])


def test_a_report_built_from_runs_is_labelled_and_carries_no_local_path(runs, seed_dir, tmp_path):
    from vplan.evaluate import collect
    files, status = collect.build([runs[5], runs[3], runs[7]], tmp_path / "report")
    m = json.loads(files.metrics.read_text(encoding="utf-8"))
    assert status == m["status"] and status["complete"] is True
    assert set(m["configs"]) == {"2", "3", "5", "7"} and m["dry_run_configs"] == [2, 3, 7]
    assert m["headline"]["headline_source"] is False and m["headline"]["status"] == "dry_run"
    assert m["provenance"]["seed_set"]["size"] == 5 and m["provenance"]["seed_set"]["tasks"] == 5
    assert m["configs"]["3"]["success"]["k"] == 3 and m["configs"]["5"]["refusals"]["correct_refusals"]["k"] == 2
    assert [r["config"] for r in m["provenance"]["runs"]] == [5, 3, 7]
    assert m["provenance"]["runs"][0]["sessions"] == [{
        "session": 0, "git": json.loads((runs[5] / "sessions.jsonl").read_text(encoding="utf-8").splitlines()[0])["git"],
        "source_digest": json.loads((runs[5] / "sessions.jsonl").read_text(encoding="utf-8").splitlines()[0])[
            "source_digest"], "source_drift": [], "contributed": True}]
    assert m["provenance"]["runs"][0]["prepared_source_digest"] == json.loads(
        (seed_dir / "manifest.json").read_text(encoding="utf-8"))["source_digest"]
    page = files.html.read_text(encoding="utf-8")
    assert "DRY RUN: configs 2, 3 and 7 — vplan evaluation report</title>" in page
    for path in (files.metrics, files.html):
        text = path.read_text(encoding="utf-8")
        for local in (str(runs[5].parent), str(tmp_path), os.path.expanduser("~")):
            assert local not in text and json.dumps(local)[1:-1] not in text, (path.name, local)


def test_complete_token_factory_runs_on_dev_seeds_are_not_a_headline_report(seed_dir, tmp_path):
    from vplan.evaluate import collect
    (tmp_path / "tf").mkdir()  # run() writes its default env file into the runs' parent directory
    tf = env_file(tmp_path / "tf", "https://api.tokenfactory.nebius.com/v1/", "nvidia/Nemotron-3-Ultra-550b-a55b")
    for config in (3, 7):
        assert run(config, seed_dir, tmp_path / f"c{config}", env_file=tf, allow_paid=True).status == "complete"
    files, status = collect.build([tmp_path / "c3", tmp_path / "c7"], tmp_path / "report")
    m = json.loads(files.metrics.read_text(encoding="utf-8"))
    assert status["complete"] is True and m["dry_run_configs"] == []
    assert m["configs"]["3"]["provenance"]["providers"] == ["api.tokenfactory.nebius.com"]
    assert m["headline"]["status"] == "development" and m["headline"]["headline_source"] is False
    assert m["headline"]["label"] == "not a headline result: development seeds, not the held-out test set"


def test_incomplete_and_limited_runs_are_labelled(seed_dir, runs, tmp_path):
    from vplan.evaluate import collect
    with pytest.raises(KeyboardInterrupt):
        run(7, seed_dir, tmp_path / "c7", fake=ScriptedSupervise(interrupt_at=2))
    assert run(5, seed_dir, tmp_path / "c5", n=3).status == "complete"
    files, status = collect.build([runs[3], tmp_path / "c7", tmp_path / "c5"], tmp_path / "report")
    assert status["complete"] is False
    assert status["incomplete"] == [{"config": 7, "recorded": 2, "planned": 5}]
    assert status["limited"] == [{"config": 5, "planned": 3, "prepared": 5}]
    page = files.html.read_text(encoding="utf-8")
    assert "<h1>INCOMPLETE: config 7 2/5 tasks — LIMITED: config 5 first 3 of 5 tasks — DRY RUN" in page


# ---- refusals ----------------------------------------------------------------------------------------------
def test_runs_over_different_seed_sets_are_refused(runs, other_seed_dir, tmp_path):
    from vplan.evaluate import collect
    run(7, other_seed_dir, tmp_path / "c7")
    with pytest.raises(collect.EvalError, match="different seed sets"):
        collect.collect_runs([runs[3], tmp_path / "c7"])


def test_a_seed_set_that_is_not_the_runs_is_refused(runs, other_seed_dir):
    from vplan.evaluate import collect
    with pytest.raises(collect.EvalError, match="seed-set hash"):
        collect.collect_runs([runs[3]], seed_dir=other_seed_dir)


def test_two_runs_of_the_same_config_are_refused(runs, seed_dir, tmp_path):
    from vplan.evaluate import collect
    run(5, seed_dir, tmp_path / "again")
    with pytest.raises(collect.EvalError, match="two runs of config 5"):
        collect.collect_runs([runs[5], tmp_path / "again"])


@pytest.mark.parametrize("damage, message", [("missing", "no manifest.json"), ("not_json", "not valid JSON"),
                                             ("not_a_run", "not an evaluation run")])
def test_a_run_directory_without_a_valid_manifest_is_refused(runs, tmp_path, damage, message):
    from vplan.evaluate import collect
    bad = tmp_path / "bad"
    bad.mkdir()
    if damage == "not_json":
        (bad / "manifest.json").write_text("{", encoding="utf-8")
    elif damage == "not_a_run":
        (bad / "manifest.json").write_text(json.dumps({"kind": "seed_set"}), encoding="utf-8")
    with pytest.raises(collect.EvalError, match=message):
        collect.collect_runs([runs[5], bad])


@pytest.mark.parametrize("key, value", [("n", "5"), ("n", True), ("n_prepared", None), ("config", 5.0),
                                        ("seed_set_hash", ["ab"]), ("split", 1), ("provider", None),
                                        ("model", {"name": "x"})])
def test_a_manifest_field_of_the_wrong_type_is_refused_before_it_is_used(runs, tmp_path, key, value):
    # A hand-edited manifest must give the exit-2 message, not a TypeError traceback from the loader.
    from vplan.evaluate import collect
    run_dir = _copy_run(runs[5], tmp_path / "c5")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    (run_dir / "manifest.json").write_text(json.dumps({**manifest, key: value}), encoding="utf-8")
    with pytest.raises(collect.EvalError, match=f"{key} must be"):
        collect.collect_runs([run_dir])


def test_a_seed_set_without_an_excluded_count_is_refused(runs, seed_dir, tmp_path, monkeypatch):
    from vplan.evaluate import collect, seedsets
    real = seedsets.load_seed_set

    def without_excluded(path, **kw):
        seeds = real(path, **kw)
        return seeds._replace(manifest={k: v for k, v in seeds.manifest.items() if k != "excluded"})
    monkeypatch.setattr(collect, "load_seed_set", without_excluded)
    with pytest.raises(collect.EvalError, match="excluded"):
        collect.collect_runs([runs[5]], seed_dir=seed_dir)


def _copy_run(run_dir, dest):
    dest.mkdir()
    for name in ("manifest.json", "attempts.jsonl"):
        (dest / name).write_bytes((run_dir / name).read_bytes())
    return dest


@pytest.mark.parametrize("damage, message", [
    ("corrupt", "line 2 does not validate"), ("partial", "line 5 does not validate"),
    ("duplicate", "twice"), ("task_id", "task id"), ("config", "config 3"), ("beyond", "not part of")])
def test_rows_that_fail_validation_are_refused(runs, tmp_path, damage, message):
    from vplan.evaluate import collect
    run_dir = _copy_run(runs[5], tmp_path / "c5")
    path = run_dir / "attempts.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    row = json.loads(lines[1])
    if damage == "corrupt":
        lines[1] = '{"attempt_id": "5:dev:1"}\n'
    elif damage == "partial":
        lines[4] = lines[4][:40]
    elif damage == "duplicate":
        lines.append(lines[1])
    elif damage == "task_id":
        lines[1] = json.dumps({**row, "task_id": "dev_99999999"}) + "\n"
    elif damage == "config":
        record = {**row["record"], "attempt_id": "3:dev:1"}
        lines[1] = json.dumps({**row, "config": 3, "attempt_id": "3:dev:1", "record": record}) + "\n"
    else:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        (run_dir / "manifest.json").write_text(json.dumps({**manifest, "n": 1, "limited": True}), encoding="utf-8")
    path.write_text("".join(lines), encoding="utf-8")
    with pytest.raises(collect.EvalError, match=message):
        collect.collect_runs([run_dir])


@pytest.mark.parametrize("n, n_prepared, extra_row", [(6, 6, True), (6, 6, False), (5, 6, False), (3, 4, False)])
def test_a_manifest_that_does_not_fit_the_seed_set_is_refused_before_rows_are_read(runs, tmp_path, n, n_prepared,
                                                                                   extra_row):
    # A hand-edited n beyond the seed set used to reach seeds.tasks[index] and end in an IndexError traceback.
    from vplan.evaluate import collect
    run_dir = _copy_run(runs[5], tmp_path / "c5")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    (run_dir / "manifest.json").write_text(json.dumps({**manifest, "n": n, "n_prepared": n_prepared}),
                                           encoding="utf-8")
    if extra_row:
        row = json.loads((run_dir / "attempts.jsonl").read_text(encoding="utf-8").splitlines()[0])
        moved = {**row, "attempt_id": "5:dev:5", "task_index": 5, "record": {**row["record"], "attempt_id": "5:dev:5"}}
        with open(run_dir / "attempts.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(moved) + "\n")
    with pytest.raises(collect.EvalError, match="seed set's 5 tasks"):
        collect.collect_runs([run_dir])


def test_a_run_labelled_with_another_split_than_its_seed_set_is_refused(runs, tmp_path):
    # Headline status needs the test split (spec §14). The rows' split was only checked against the run manifest, so
    # a run over dev seeds relabelled "test" throughout would have passed; the seed set's own split decides.
    from vplan.evaluate import collect
    run_dir = _copy_run(runs[5], tmp_path / "c5")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    (run_dir / "manifest.json").write_text(json.dumps({**manifest, "split": "test"}), encoding="utf-8")
    lines = []
    for line in (run_dir / "attempts.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        attempt_id = row["attempt_id"].replace(":dev:", ":test:")
        lines.append(json.dumps({**row, "split": "test", "attempt_id": attempt_id,
                                 "record": {**row["record"], "attempt_id": attempt_id}}))
    (run_dir / "attempts.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(collect.EvalError, match="test split, but its seed set holds the dev split"):
        collect.collect_runs([run_dir])


def test_a_record_that_does_not_carry_the_runs_planner_is_refused(seed_dir, tmp_path):
    from vplan.evaluate import collect
    run(5, seed_dir, tmp_path / "c5", fake=FakeSupervise())  # its records name provider "fake", not "heuristic"
    with pytest.raises(collect.EvalError, match="line 1 does not join its seed-set task"):
        collect.collect_runs([tmp_path / "c5"])


# ---- CLI ---------------------------------------------------------------------------------------------------
def test_cli_report_build_writes_the_report_and_exits_0(runs, tmp_path, capsys):
    from vplan.cli import main
    out = tmp_path / "report"
    assert main(["report", "build", "--run", str(runs[5]), "--run", str(runs[3]), "--run", str(runs[7]),
                 "--output", str(out)]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["metrics.json", "report.html"]
    printed = capsys.readouterr().out
    assert "report.html" in printed and "DRY RUN: configs 2, 3 and 7" in printed


def test_cli_report_build_passes_every_run_and_the_seed_dir(monkeypatch, tmp_path):
    from vplan.cli import main
    from vplan.evaluate import collect
    from vplan.evaluate.report import ReportFiles
    seen = []

    def fake(run_dirs, output, seed_dir=None):
        seen.append((run_dirs, output, seed_dir))
        return ReportFiles(tmp_path / "metrics.json", tmp_path / "report.html"), {
            "complete": False, "incomplete": [], "limited": [{"config": 5, "planned": 3, "prepared": 5}],
            "title_prefixes": ["LIMITED: config 5 first 3 of 5 tasks"]}
    monkeypatch.setattr(collect, "build", fake)
    assert main(["report", "build", "--run", "a", "--run", "b", "--output", "out", "--seed-dir", "s"]) == 0
    assert seen == [(["a", "b"], "out", "s")]


def test_cli_report_build_exits_1_for_an_incomplete_run_and_still_writes_it(seed_dir, runs, tmp_path, capsys):
    from vplan.cli import main
    with pytest.raises(KeyboardInterrupt):
        run(7, seed_dir, tmp_path / "c7", fake=ScriptedSupervise(interrupt_at=2))
    out = tmp_path / "report"
    assert main(["report", "build", "--run", str(runs[3]), "--run", str(tmp_path / "c7"), "--output", str(out)]) == 1
    assert (out / "report.html").is_file()
    captured = capsys.readouterr()
    assert "INCOMPLETE: config 7 2/5 tasks" in captured.out and "--resume" in captured.err


@pytest.mark.parametrize("case, message", [("same_config", "two runs of config 5"),
                                           ("no_manifest", "no manifest.json"), ("no_rows", "no attempt rows")])
def test_cli_report_build_refusals_exit_2_with_a_clean_message(seed_dir, runs, tmp_path, capsys, case, message):
    from vplan.cli import main
    if case == "same_config":
        dirs = [runs[5], runs[5]]
    elif case == "no_manifest":
        (tmp_path / "empty").mkdir()
        dirs = [runs[5], tmp_path / "empty"]
    else:
        with pytest.raises(KeyboardInterrupt):
            run(5, seed_dir, tmp_path / "c5", fake=ScriptedSupervise(interrupt_at=0))
        dirs = [tmp_path / "c5"]
    argv = ["report", "build", "--output", str(tmp_path / "report")]
    for d in dirs:
        argv += ["--run", str(d)]
    assert main(argv) == 2
    err = capsys.readouterr().err
    assert message in err and err.startswith("error: ") and "Traceback" not in err
    assert not (tmp_path / "report").exists()


@pytest.mark.parametrize("where", ["file", "under_a_file"])
def test_cli_report_build_refuses_an_output_that_is_a_file(runs, tmp_path, capsys, where):
    from vplan.cli import main
    taken = tmp_path / "taken"
    taken.write_text("not a directory", encoding="utf-8")
    out = taken if where == "file" else taken / "report"
    assert main(["report", "build", "--run", str(runs[5]), "--output", str(out)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "--output" in err and "Traceback" not in err
    assert taken.read_text(encoding="utf-8") == "not a directory"


def test_cli_report_build_refuses_wsl_paths(monkeypatch, capsys):
    from vplan.cli import main
    from vplan.evaluate import collect, pilot
    monkeypatch.setattr(pilot, "_WINDOWS", True)
    monkeypatch.setattr(collect, "build", lambda *a, **kw: pytest.fail("must refuse first"))
    assert main(["report", "build", "--run", "/mnt/c/run", "--output", "out"]) == 2
    assert "Windows" in capsys.readouterr().err
