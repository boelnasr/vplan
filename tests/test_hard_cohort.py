"""Hard feasible pilot cohort from the W2.2 generator (dense distractors/obstacles), offline unless noted."""
import json

import pytest

from tests.test_pilot import env_file, fake_cohort, run_dirs, sha
from tests.test_tasks import _witness
from vplan import records

HARD = ("dense_distractors", "dense_obstacles", "dense_both")
COUNTS = {"dense_distractors": (3, 1), "dense_obstacles": (1, 3), "dense_both": (3, 3)}


def test_hard_cohort_rotates_dense_strata_by_accepted_index_and_records_rejections():
    from vplan.tasks import prepare_hard_cohort

    def evaluate(candidate):
        if candidate.metadata.seed == 4:
            raise RuntimeError("physics unavailable")
        if candidate.metadata.seed == 5:
            return _witness(candidate, outcome="blocked")
        return _witness(candidate, success=candidate.metadata.seed != 3)

    cohort = prepare_hard_cohort(n=4, seed=3, max_candidates=10, witness_evaluator=evaluate)
    assert [t.metadata.seed for t in cohort.tasks] == [6, 7, 8, 9]
    assert [t.metadata.stratum for t in cohort.tasks] == list(HARD) + ["dense_distractors"]
    assert [(e.metadata.seed, e.reason.split(":")[0]) for e in cohort.exclusions] == [
        (3, "judge_failure"), (4, "simulation_error"), (5, "witness_blocked")]
    assert cohort.attempted == 7
    for t in cohort.tasks:
        n_d, n_o = COUNTS[t.metadata.stratum]
        assert (len(t.task.scene.objects), len(t.task.scene.obstacles)) == (1 + n_d, n_o)
        assert t.oracle.feasibility == "feasible" and t.oracle.reference_result.success
        assert t.oracle.reference_plan.target_id == t.oracle.target_id
        assert set(t.task.model_dump()) == {"instruction", "scene", "robot_facts"}


def test_hard_cohort_is_deterministic():
    from vplan.tasks import prepare_hard_cohort
    assert (prepare_hard_cohort(n=3, seed=7, witness_evaluator=_witness)
            == prepare_hard_cohort(n=3, seed=7, witness_evaluator=_witness))


def test_draft_rejection_reason_is_recorded_without_evaluation(monkeypatch):
    from vplan.tasks import GenerationRejected, generator, prepare_hard_cohort
    real = generator.draft_task

    def draft(seed, kind, **kw):
        assert kind == "feasible"
        if seed == 0:
            raise GenerationRejected("no_room_for_obstacle")
        return real(seed, kind, **kw)
    monkeypatch.setattr(generator, "draft_task", draft)
    evaluated = []
    cohort = prepare_hard_cohort(n=1, seed=0, witness_evaluator=lambda c: evaluated.append(c) or _witness(c))
    assert [(e.metadata.seed, e.metadata.stratum, e.reason) for e in cohort.exclusions] == [
        (0, "dense_distractors", "no_room_for_obstacle")]
    assert [c.metadata.seed for c in evaluated] == [1]


def test_hard_cohort_is_bounded_and_keeps_exclusions_on_failure():
    from vplan.tasks import CohortPreparationError, prepare_hard_cohort
    with pytest.raises(CohortPreparationError, match="0/2") as caught:
        prepare_hard_cohort(n=2, seed=0, max_candidates=3, witness_evaluator=lambda c: _witness(c, success=False))
    assert caught.value.cohort.attempted == 3 and len(caught.value.cohort.exclusions) == 3


def test_hard_task_ids_are_distinct_from_pilot_ids():
    from vplan.tasks import STRATA, prepare_cohort, prepare_hard_cohort
    hard = {t.metadata.task_id for t in prepare_hard_cohort(n=6, witness_evaluator=_witness).tasks}
    pilot = {t.metadata.task_id for t in prepare_cohort(n=6, witness_evaluator=_witness).tasks}
    assert len(hard) == 6 and not hard & pilot
    from vplan.tasks.generator import _metadata
    assert not hard & {_metadata(s, st).task_id for s in range(6) for st in STRATA}


def test_hard_default_evaluator_receives_one_captured_snapshot(monkeypatch):
    from vplan.config import config_snapshot
    from vplan.tasks import generator
    captured = config_snapshot()
    seen = []
    monkeypatch.setattr(generator, "evaluate_witness",
                        lambda candidate, *, config_snapshot=None: seen.append(config_snapshot) or _witness(candidate))
    monkeypatch.setattr(generator, "config_snapshot", lambda: captured)
    generator.prepare_hard_cohort(n=2)
    assert seen == [captured, captured]


# ---- pilot artifacts ----------------------------------------------------------------------------
def fake_hard_cohort(n=3):
    from vplan.tasks import prepare_hard_cohort
    return prepare_hard_cohort(n=n, witness_evaluator=_witness)


@pytest.fixture
def hard_prepared(tmp_path, monkeypatch):
    from vplan.evaluate import pilot
    monkeypatch.setattr(pilot, "prepare_cohort", lambda **kw: pytest.fail("pilot generator must not run"))
    monkeypatch.setattr(pilot, "prepare_hard_cohort", lambda **kw: fake_hard_cohort())
    out = tmp_path / "hard"
    manifest = pilot.prepare(out, n=3, generator="hard")
    assert manifest["generator"] == "hard"
    return out


def test_prepare_hard_records_generator_and_strata(hard_prepared):
    from vplan.evaluate import pilot
    manifest = json.loads((hard_prepared / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["generator"] == "hard"
    assert manifest["strata"] == {s: 1 for s in HARD}
    prepared = pilot.load_prepared(hard_prepared)
    assert prepared.generator == "hard"
    assert [t.metadata.stratum for t in prepared.tasks] == list(HARD)
    assert "reference" not in (hard_prepared / "tasks.jsonl").read_text(encoding="utf-8")


def test_prepare_defaults_to_the_pilot_generator(tmp_path, monkeypatch):
    from vplan.evaluate import pilot
    monkeypatch.setattr(pilot, "prepare_hard_cohort", lambda **kw: pytest.fail("hard generator must not run"))
    monkeypatch.setattr(pilot, "prepare_cohort", lambda **kw: fake_cohort())
    manifest = pilot.prepare(tmp_path / "p", n=3)
    assert manifest["generator"] == "pilot" and pilot.load_prepared(tmp_path / "p").generator == "pilot"
    with pytest.raises(pilot.PilotError, match="generator"):
        pilot.prepare(tmp_path / "q", n=3, generator="bogus")


def _rewrite_manifest(directory, **changes):
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(changes)
    for key in [k for k, v in changes.items() if v is None]:
        del manifest[key]
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_load_prepared_treats_a_missing_generator_as_pilot(tmp_path, monkeypatch):
    from vplan.evaluate import pilot
    monkeypatch.setattr(pilot, "prepare_cohort", lambda **kw: fake_cohort())
    pilot.prepare(tmp_path / "old", n=3)
    _rewrite_manifest(tmp_path / "old", generator=None)  # cohorts prepared before the field existed
    assert pilot.load_prepared(tmp_path / "old").generator == "pilot"


@pytest.mark.parametrize("generator, message", [("bogus", "unknown generator"), ("pilot", "stratum")])
def test_load_prepared_refuses_unknown_generator_or_foreign_strata(hard_prepared, generator, message):
    from vplan.evaluate import pilot
    _rewrite_manifest(hard_prepared, generator=generator)
    with pytest.raises(pilot.PilotError, match=message):
        pilot.load_prepared(hard_prepared)


def test_hard_cohort_runs_lists_and_selects_rounds(hard_prepared, monkeypatch):
    from tests.test_pilot import scripted_provider, response
    from tests.test_repair import executed
    from vplan.evaluate import pilot
    from vplan.repair import evaluate_pair
    tasks = pilot.load_prepared(hard_prepared).tasks
    seen = []
    scripted_provider(monkeypatch, [response(t.oracle.reference_plan) for t in tasks])

    def evaluate(task, planner, **kw):
        seen.append(task)
        return evaluate_pair(task, planner, evaluator=lambda plan: executed(), **kw)
    monkeypatch.setattr(pilot, "evaluate_pair", evaluate)
    run = pilot.run_live(hard_prepared, env_file=env_file(hard_prepared.parent))
    meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
    assert meta["status"] == "complete" and meta["generator"] == "hard"
    assert meta["manifest_sha256"] == sha(hard_prepared / "manifest.json")
    assert [r.split()[0] for r in pilot.list_rounds(run)] == [t.metadata.task_id for t in tasks]
    task, rnd = pilot.select_round(run, tasks[2].metadata.task_id, "repair")
    assert task.metadata.stratum == "dense_both" and rnd.call.plan == tasks[2].oracle.reference_plan
    assert run_dirs(hard_prepared) == [run]


def test_cli_generator_flag_reaches_prepare(monkeypatch):
    from vplan.cli import main
    from vplan.evaluate import pilot
    calls = []
    monkeypatch.setattr(pilot, "prepare", lambda out, **kw: calls.append(kw))
    assert main(["pilot", "prepare", "--output", "d"]) == 0
    assert main(["pilot", "prepare", "--output", "d", "--generator", "hard"]) == 0
    assert [kw["generator"] for kw in calls] == ["pilot", "hard"]
    with pytest.raises(SystemExit):
        main(["pilot", "prepare", "--output", "d", "--generator", "bogus"])


def test_task_metadata_accepts_the_hard_strata():
    for stratum in HARD + ("clear", "distractor", "obstacle"):
        assert records.TaskMetadata(task_id="dev_x", seed=0, stratum=stratum).stratum == stratum


def test_real_hard_cohort_smoke():
    """Real simulation: three dense tasks whose stored witness succeeded under the captured snapshot."""
    from vplan.tasks import prepare_hard_cohort
    from vplan.tasks.referents import resolve_referent
    cohort = prepare_hard_cohort(n=3, seed=0, max_candidates=15)
    assert [t.metadata.stratum for t in cohort.tasks] == list(HARD)
    for t in cohort.tasks:
        assert resolve_referent(t.task.instruction, t.task.scene) == [t.oracle.target_id]
        assert t.oracle.reference_result.success
