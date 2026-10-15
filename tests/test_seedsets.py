"""W3.1 evaluation seed sets: the D51 mix, deterministic preparation and the validating loader (no simulation)."""
import hashlib
import json
from collections import Counter

import pytest

from tests.test_generate import _fake_witness
from vplan import records
from vplan.config import config_snapshot

CELLS_4X4 = {(d, o) for d in range(4) for o in range(4)}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lines(path):
    return path.read_text(encoding="utf-8").splitlines()


def prepare(tmp_path, name="seeds", split="dev", feasible=6, impossible=4, **kw):
    from vplan.evaluate import seedsets
    kw.setdefault("witness_evaluator", _fake_witness)
    out = tmp_path / name
    seedsets.prepare_seed_set(out, split, feasible=feasible, impossible=impossible, **kw)
    return out


def manifest(out):
    return json.loads((out / "manifest.json").read_text(encoding="utf-8"))


def failed_witness(task, target_id):
    ok = _fake_witness(task, target_id)
    return records.WitnessEvaluation(plan=ok.plan, outcome="executed", result=records.JudgeResult(
        success=False, predicates={"S1": False}, violations=("S1",), details={}))


def margins(feasible):
    rows = [sum(feasible[(d, o)] for o in range(4)) for d in range(4)]
    cols = [sum(feasible[(d, o)] for d in range(4)) for o in range(4)]
    return rows, cols


# ---- mix arithmetic (D51) -------------------------------------------------------------------------
def test_default_dev_mix_is_the_d51_mix():
    from vplan.evaluate import seedsets
    mix = seedsets.mix_for("dev")
    assert set(mix.feasible) == CELLS_4X4 and sum(mix.feasible.values()) == 75
    assert Counter(mix.feasible.values()) == {5: 11, 4: 5}
    assert mix.impossible == {1: 7, 2: 6, 3: 6, 4: 6} and mix.n == 100
    rows, cols = margins(mix.feasible)
    assert max(rows) - min(rows) <= 1 and max(cols) - min(cols) <= 1


def test_default_test_mix_is_stratified_with_balanced_margins():
    from vplan.evaluate import seedsets
    mix = seedsets.mix_for("test")
    assert Counter(mix.feasible.values()) == {19: 12, 18: 4} and sum(mix.feasible.values()) == 300
    assert mix.impossible == {1: 25, 2: 25, 3: 25, 4: 25}
    # The 12 larger cells are three whole diagonals, so every distractor and every obstacle count gets 75 tasks.
    assert margins(mix.feasible) == ([75] * 4, [75] * 4)


@pytest.mark.parametrize("feasible, impossible", [(38, 12), (1, 0), (0, 3), (17, 5), (64, 4), (5, 1)])
def test_custom_mix_is_as_even_as_possible(feasible, impossible):
    from vplan.evaluate import seedsets
    mix = seedsets.mix_for("dev", feasible=feasible, impossible=impossible)
    cells = list(mix.feasible.values())
    assert sum(cells) == feasible and max(cells) - min(cells) <= 1
    rows, cols = margins(mix.feasible)
    assert max(rows) - min(rows) <= 1 and max(cols) - min(cols) <= 1
    types = [mix.impossible[t] for t in (1, 2, 3, 4)]
    assert sum(types) == impossible and max(types) - min(types) <= 1 and types == sorted(types, reverse=True)


@pytest.mark.parametrize("split, feasible, impossible, message", [
    ("val", None, None, "split"), ("dev", -1, 3, "non-negative"), ("dev", 0, 0, "at least one"),
    ("dev", 5, None, "both"), ("dev", True, 2, "non-negative")])
def test_mix_refuses_bad_requests(split, feasible, impossible, message):
    from vplan.evaluate import seedsets
    with pytest.raises(seedsets.EvalError, match=message):
        seedsets.mix_for(split, feasible=feasible, impossible=impossible)


def test_slot_order_interleaves_evenly_and_cycles_cells_and_types():
    from vplan.evaluate import seedsets
    mix = seedsets.mix_for("dev")
    slots = seedsets.slot_order(mix)
    kinds = [s.kind for s in slots]
    assert len(slots) == 100 and kinds[:8] == ["feasible"] * 3 + ["impossible"] + ["feasible"] * 3 + ["impossible"]
    # Every prefix keeps the 3:1 ratio to within one task, so `eval run --n` samples both kinds.
    assert all(abs(kinds[:i].count("impossible") - i / 4) < 1 for i in range(1, 101))
    cells = [(s.distractors, s.obstacles) for s in slots if s.kind == "feasible"]
    assert set(cells[:16]) == CELLS_4X4 and Counter(cells) == Counter(mix.feasible)
    assert all(s.impossible_type is None for s in slots if s.kind == "feasible")
    types = [s.impossible_type for s in slots if s.kind == "impossible"]
    assert types[:8] == [1, 2, 3, 4, 1, 2, 3, 4] and Counter(types) == Counter(mix.impossible)
    assert all(s.distractors is None and s.obstacles is None for s in slots if s.kind == "impossible")


# ---- preparation ----------------------------------------------------------------------------------
def test_prepare_is_byte_identical_for_the_same_command(tmp_path):
    a, b = prepare(tmp_path, "a"), prepare(tmp_path, "b")
    for name in ("tasks.jsonl", "oracle.jsonl", "exclusions.jsonl"):
        assert (a / name).read_bytes() == (b / name).read_bytes()
    assert manifest(a)["seed_set_hash"] == manifest(b)["seed_set_hash"]
    assert b"\r\n" not in (a / "tasks.jsonl").read_bytes()  # LF on every OS, so the hash is OS-independent


def test_prepared_tasks_follow_the_slot_order_and_account_for_every_seed(tmp_path):
    from vplan.evaluate import seedsets
    from vplan.tasks import check_impossibility
    out = prepare(tmp_path, feasible=6, impossible=4)
    tasks = [records.SeedSetTask.model_validate_json(x) for x in lines(out / "tasks.jsonl")]
    oracles = [records.SeedSetOracle.model_validate_json(x) for x in lines(out / "oracle.jsonl")]
    excluded = [records.SeedSetExclusion.model_validate_json(x) for x in lines(out / "exclusions.jsonl")]
    m = manifest(out)
    assert [t.index for t in tasks] == list(range(10)) and [o.index for o in oracles] == list(range(10))
    assert [o.task_id for o in oracles] == [t.task_id for t in tasks] and len(set(t.task_id for t in tasks)) == 10
    # Candidate seeds start at 0 for dev and are consumed one by one: each was accepted or excluded, in order.
    assert [t.seed for t in tasks] == sorted(t.seed for t in tasks)
    assert sorted([t.seed for t in tasks] + [e.seed for e in excluded]) == list(range(m["seeds"]["first"],
                                                                                         m["seeds"]["next"]))
    assert m["seeds"]["first"] == 0 and m["seeds"]["attempted"] == 10 + len(excluded)
    for task, row, slot in zip(tasks, oracles, seedsets.slot_order(seedsets.mix_for("dev", 6, 4))):
        assert task.split == "dev"
        assert task.distractors == len(task.task.scene.objects) - 1
        assert task.obstacles == len(task.task.scene.obstacles)
        if slot.kind == "feasible":
            assert row.oracle.feasibility == "feasible" and row.oracle.reference_result.success
            assert (task.distractors, task.obstacles) == (slot.distractors, slot.obstacles)
        else:
            assert row.oracle.feasibility == "impossible" and row.oracle.proof.type == slot.impossible_type
            assert check_impossibility(task.task, row.oracle)


def test_private_fields_live_only_in_oracle_jsonl(tmp_path):
    out = prepare(tmp_path, feasible=4, impossible=4)
    public = (out / "tasks.jsonl").read_text(encoding="utf-8")
    for word in ("feasib", "impossib", "proof", "oracle", "reference", "target_id", "refusal", "witness"):
        assert word not in public, word
    assert all(set(json.loads(x)) == {"task_id", "index", "seed", "split", "distractors", "obstacles", "task"}
               for x in lines(out / "tasks.jsonl"))
    private = (out / "oracle.jsonl").read_text(encoding="utf-8")
    assert '"feasibility":"impossible"' in private and '"proof"' in private and '"reference_plan"' in private


def test_manifest_records_mix_seeds_hashes_and_provenance(tmp_path):
    from vplan.evaluate import pilot, seedsets
    out = prepare(tmp_path, feasible=5, impossible=2)
    m = manifest(out)
    assert m["kind"] == "seed_set" and m["schema_version"] == 1 and m["split"] == "dev" and m["n"] == 7
    assert m["requested"]["feasible"] == 5 and m["requested"]["impossible"] == 2
    assert m["requested"]["impossible_types"] == {"1": 1, "2": 1, "3": 0, "4": 0}
    assert sum(map(sum, m["requested"]["feasible_cells"])) == 5 and len(m["requested"]["feasible_cells"]) == 4
    assert m["files"] == {name: sha(out / name) for name in ("tasks.jsonl", "oracle.jsonl", "exclusions.jsonl")}
    assert m["seed_set_hash"] == seedsets.seed_set_hash(sha(out / "tasks.jsonl"), sha(out / "oracle.jsonl"))
    assert m["source_hashes"] == pilot.source_hashes() and m["source_digest"] == pilot.source_digest(m["source_hashes"])
    assert set(m["config_hashes"]) == {"verifier", "judge", "physics"}
    assert m["seeds"]["max_candidates"] == 2 * 7 + 20
    assert {"created_utc", "python", "manipulapy", "git"} <= set(m)


def test_rejected_candidate_is_excluded_and_the_next_seed_fills_the_same_slot(tmp_path):
    calls = []

    def first_fails(task, target_id):
        calls.append(target_id)
        return failed_witness(task, target_id) if len(calls) == 1 else _fake_witness(task, target_id)
    out = prepare(tmp_path, feasible=2, impossible=1, witness_evaluator=first_fails)
    excluded = [records.SeedSetExclusion.model_validate_json(x) for x in lines(out / "exclusions.jsonl")]
    judged = [e for e in excluded if e.reason == "judge_failure"]
    assert len(judged) == 1 and judged[0].slot == 0 and judged[0].kind == "feasible"
    assert judged[0].witness is not None and judged[0].witness.result.success is False
    first = records.SeedSetTask.model_validate_json(lines(out / "tasks.jsonl")[0])
    assert first.seed > judged[0].seed and (first.distractors, first.obstacles) == (judged[0].distractors,
                                                                                    judged[0].obstacles)


def test_draft_rejections_are_excluded_with_their_reason(tmp_path, monkeypatch):
    from vplan.evaluate import seedsets
    from vplan.tasks.generate import GenerationRejected, draft_task
    real = seedsets.generate_task

    def no_room_for_seed_0(seed, kind, impossible_type=None, **kw):
        return None if seed == 0 else real(seed, kind, impossible_type, **kw)

    def draft(seed, kind, impossible_type=None, **kw):
        if seed == 0:
            raise GenerationRejected("no_room_for_obstacle")
        return draft_task(seed, kind, impossible_type, **kw)
    monkeypatch.setattr(seedsets, "generate_task", no_room_for_seed_0)
    monkeypatch.setattr(seedsets, "draft_task", draft)
    out = prepare(tmp_path, feasible=1, impossible=0)
    (excluded,) = [records.SeedSetExclusion.model_validate_json(x) for x in lines(out / "exclusions.jsonl")]
    assert excluded.seed == 0 and excluded.reason == "no_room_for_obstacle" and excluded.witness is None


def test_candidate_limit_keeps_exclusions_and_writes_no_manifest(tmp_path):
    from vplan.evaluate import seedsets
    with pytest.raises(seedsets.EvalError, match="0/2 tasks after 3 candidates"):
        prepare(tmp_path, feasible=2, impossible=0, max_candidates=3, witness_evaluator=failed_witness)
    assert not (tmp_path / "seeds" / "manifest.json").exists()
    assert len(lines(tmp_path / "seeds" / "exclusions.jsonl")) == 3


def test_prepare_refuses_a_non_empty_output_without_generating(tmp_path):
    from vplan.evaluate import seedsets
    (tmp_path / "seeds").mkdir()
    (tmp_path / "seeds" / "keep.txt").write_text("evidence", encoding="utf-8")
    with pytest.raises(seedsets.EvalError, match="not empty"):
        prepare(tmp_path, witness_evaluator=lambda *a: pytest.fail("must refuse before generating"))
    assert (tmp_path / "seeds" / "keep.txt").read_text(encoding="utf-8") == "evidence"


def test_test_split_needs_an_explicit_opt_in_and_uses_its_own_seed_range(tmp_path):
    from vplan.evaluate import seedsets
    with pytest.raises(seedsets.EvalError, match="W3.3"):
        prepare(tmp_path, "t0", split="test", feasible=1, impossible=1)
    assert not (tmp_path / "t0").exists()
    out = prepare(tmp_path, "t1", split="test", feasible=1, impossible=1, allow_test_split=True)
    tasks = [records.SeedSetTask.model_validate_json(x) for x in lines(out / "tasks.jsonl")]
    assert all(t.split == "test" and t.seed >= 10_000_000 for t in tasks)
    assert manifest(out)["seeds"]["first"] == 10_000_000


def test_source_change_during_preparation_leaves_no_manifest(tmp_path, monkeypatch):
    from vplan.evaluate import seedsets
    snapshots = iter([{"src/vplan/a.py": "1"}, {"src/vplan/a.py": "2"}])
    monkeypatch.setattr(seedsets, "source_hashes", lambda: next(snapshots))
    with pytest.raises(seedsets.EvalError, match="during preparation"):
        prepare(tmp_path)
    assert not (tmp_path / "seeds" / "manifest.json").exists()


def test_config_change_during_preparation_leaves_no_manifest(tmp_path, monkeypatch):
    from vplan.evaluate import seedsets
    first, changed = config_snapshot(), config_snapshot()
    changed["judge"] = {**changed["judge"], "tampered": 1}
    snapshots = iter([first, changed])
    monkeypatch.setattr(seedsets, "config_snapshot", lambda: next(snapshots))
    with pytest.raises(seedsets.EvalError, match="during preparation"):
        prepare(tmp_path)
    assert not (tmp_path / "seeds" / "manifest.json").exists()


# ---- loader ---------------------------------------------------------------------------------------
def test_loader_returns_aligned_public_and_private_tasks(tmp_path):
    from vplan.evaluate import seedsets
    out = prepare(tmp_path)
    loaded = seedsets.load_seed_set(out)
    assert loaded.split == "dev" and len(loaded.tasks) == len(loaded.oracles) == 10 and loaded.drift == []
    assert loaded.seed_set_hash == manifest(out)["seed_set_hash"]
    assert loaded.manifest_sha256 == sha(out / "manifest.json")
    assert all(type(t) is records.SeedSetTask and type(o) is records.TaskOracle
               for t, o in zip(loaded.tasks, loaded.oracles))


@pytest.mark.parametrize("tamper, message", [
    ("tasks", "tasks.jsonl"), ("oracle", "oracle.jsonl"), ("exclusions", "exclusions.jsonl"),
    ("missing", "manifest.json is missing"), ("hash", "seed-set hash"), ("config", "configuration"),
    ("source", "src/vplan/repair.py")])
def test_loader_refuses_a_changed_seed_set_or_checkout(tmp_path, monkeypatch, tamper, message):
    from vplan.evaluate import seedsets
    out = prepare(tmp_path)
    if tamper in ("tasks", "oracle", "exclusions"):
        with open(out / f"{tamper}.jsonl", "a", encoding="utf-8") as f:
            f.write(" ")
    elif tamper == "missing":
        (out / "manifest.json").unlink()
    elif tamper == "hash":
        m = manifest(out)
        (out / "manifest.json").write_text(json.dumps({**m, "seed_set_hash": "0" * 64}), encoding="utf-8")
    elif tamper == "config":
        snapshot = config_snapshot()
        snapshot["judge"] = {**snapshot["judge"], "tampered": 1}
        monkeypatch.setattr(seedsets, "config_snapshot", lambda: snapshot)
    else:
        current = seedsets.source_hashes()
        monkeypatch.setattr(seedsets, "source_hashes", lambda: {**current, "src/vplan/repair.py": "0" * 64})
    with pytest.raises(seedsets.EvalError, match=message):
        seedsets.load_seed_set(out)


def test_loader_records_allowed_source_drift(tmp_path, monkeypatch):
    from vplan.evaluate import seedsets
    out = prepare(tmp_path)
    current = seedsets.source_hashes()
    monkeypatch.setattr(seedsets, "source_hashes", lambda: {**current, "src/vplan/repair.py": "0" * 64})
    assert seedsets.load_seed_set(out, allow_source_drift=True).drift == ["src/vplan/repair.py"]


def test_loader_refuses_a_pilot_cohort_and_the_pilot_refuses_a_seed_set(tmp_path, monkeypatch):
    from tests.test_pilot import prepare_fake
    from vplan.evaluate import pilot, seedsets
    cohort = prepare_fake(tmp_path, monkeypatch, 3)
    with pytest.raises(seedsets.EvalError, match="not an evaluation seed set"):
        seedsets.load_seed_set(cohort)
    with pytest.raises(pilot.PilotError, match="generator"):
        pilot.load_prepared(prepare(tmp_path))
