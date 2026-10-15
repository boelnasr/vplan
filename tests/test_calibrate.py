"""W3.3 sigma_thr calibration (spec §9 sigma_thr, V-4; spec §13 V3): the rule, the unbiased sample and --apply."""
import json
import math
import shutil
from copy import deepcopy

import numpy as np
import pytest

from tests.test_generate import _fake_witness
from vplan import records
from vplan.config import CONFIG_DIR, config_snapshot


def fake_rejects(task, target_id) -> bool:
    """A deterministic, content-based witness failure, so prepare_seed_set and calibrate see the same verdicts."""
    target = next(o for o in task.scene.objects if o.id == target_id)
    return int(round(target.pose.position[0] * 1000)) % 3 == 0


def fake_evaluation(task, target_id):
    ok = _fake_witness(task, target_id)
    if not fake_rejects(task, target_id):
        return ok
    return records.WitnessEvaluation(plan=ok.plan, outcome="blocked", findings=(records.Finding(
        stage="lift", check="collision", plan_field="grasp", measured=0.0, threshold=0.01, direction="fake"),))


def fake_measure(seen=None, values=None):
    """measure(task, target_id, seed, cfgs) without simulation; minima come from values(seed) when accepted."""
    from vplan.evaluate import calibrate

    def measure(task, target_id, seed, cfgs):
        if seen is not None:
            seen.append((seed, deepcopy(cfgs)))
        evaluation = fake_evaluation(task, target_id)
        if evaluation.findings:
            return evaluation, None
        v, j = values(seed) if values else (0.05 + seed / 1000, 0.04 + seed / 1000)
        return evaluation, calibrate.Minima(verifier=v, verifier_segment=2, verifier_stage="lift",
                                            verifier_sample=7, judge=j, judge_step=123)
    return measure


# ---- the rule ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("value, expected", [
    (0.0349999, 0.034), (0.035, 0.035), (0.0350001, 0.035), (0.057, 0.057), (0.029, 0.029), (0.0009, 0.0),
    (0.1, 0.1), (np.float64(0.0219), 0.021), (1e-5, 0.0), (0.0, 0.0)])
def test_floor_rounds_down_to_a_thousandth_and_never_up(value, expected):
    from vplan.evaluate import calibrate
    out = calibrate.floor_to_step(value)
    assert out == expected and type(out) is float
    assert out <= float(value)  # the floored threshold never rejects the value it came from


@pytest.mark.parametrize("value", [-0.001, float("nan"), float("inf")])
def test_floor_refuses_values_that_are_not_a_singular_value(value):
    from vplan.evaluate import calibrate
    with pytest.raises(ValueError):
        calibrate.floor_to_step(value)


def test_percentile_is_numpy_linear_interpolation():
    from vplan.evaluate import calibrate
    assert calibrate.PERCENTILE == 1.0 and calibrate.PERCENTILE_METHOD == "linear"
    assert calibrate.percentile(list(range(101)), 1.0) == pytest.approx(1.0)
    # 75 witnesses (the dev mix): the 1st percentile sits 0.74 of the way from the smallest to the second smallest.
    values = [0.03, 0.02] + [0.1] * 73
    assert calibrate.percentile(values, 1.0) == pytest.approx(0.02 + 0.74 * 0.01)
    assert calibrate.percentile(values, 1.0) == pytest.approx(np.percentile(values, 1.0, method="linear"))


def test_thresholds_take_the_floored_1st_percentile_and_the_judge_never_exceeds_the_verifier():
    from vplan.evaluate import calibrate
    verifier = [0.0457] + [0.2] * 74  # P1 = 0.0457 + 0.74 * (0.2 - 0.0457) = 0.159882 -> 0.159
    assert calibrate.calibrated_thresholds(verifier, [0.3] * 75) == {"verifier": 0.159, "judge": 0.159}
    assert calibrate.calibrated_thresholds(verifier, [0.0123] * 75) == {"verifier": 0.159, "judge": 0.012}
    with pytest.raises(ValueError):
        calibrate.calibrated_thresholds([], [])
    with pytest.raises(ValueError):
        calibrate.calibrated_thresholds([0.1, 0.2], [0.1])  # one pair of minima per witness


def test_zero_override_disables_only_the_two_singularity_thresholds():
    from vplan.evaluate import calibrate
    cfgs = config_snapshot()
    before = deepcopy(cfgs)
    zero = calibrate.zero_sigma_snapshot(cfgs)
    assert cfgs == before  # the caller's snapshot is not mutated
    assert zero["verifier"]["sigma_min_threshold"] == 0.0 and zero["judge"]["sigma_min_threshold"] == 0.0
    for name in cfgs:
        assert {k: v for k, v in zero[name].items() if k != "sigma_min_threshold"} == \
               {k: v for k, v in cfgs[name].items() if k != "sigma_min_threshold"}
    assert "sigma_min_threshold" not in zero["physics"]


# ---- the sample ---------------------------------------------------------------------------------------
def test_every_witness_is_generated_and_accepted_with_both_thresholds_at_zero():
    from vplan.evaluate import calibrate
    seen = []
    evidence = calibrate.calibrate("dev", feasible=5, impossible=2, measure=fake_measure(seen))
    assert seen, "feasible slots must reach the witness"
    disk = config_snapshot()
    for _, cfgs in seen:
        assert cfgs == calibrate.zero_sigma_snapshot(disk)
    assert evidence["rule"]["calibration_overrides"] == {"verifier.sigma_min_threshold": 0.0,
                                                         "judge.sigma_min_threshold": 0.0}


def test_the_sample_fills_the_same_slots_with_the_same_seeds_as_eval_prepare(tmp_path):
    """The calibration sample is the dev seed set as prepare would build it with both thresholds at zero."""
    from vplan.evaluate import calibrate, seedsets
    seedsets.prepare_seed_set(tmp_path / "seeds", "dev", feasible=9, impossible=3,
                              witness_evaluator=fake_evaluation)
    tasks = [records.SeedSetTask.model_validate_json(x)
             for x in (tmp_path / "seeds" / "tasks.jsonl").read_text(encoding="utf-8").splitlines()]
    excluded = [records.SeedSetExclusion.model_validate_json(x)
                for x in (tmp_path / "seeds" / "exclusions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert excluded, "the fake must reject some witnesses for this test to mean anything"
    evidence = calibrate.calibrate("dev", feasible=9, impossible=3, measure=fake_measure())
    sample = [records.SigmaWitness.model_validate(row) for row in evidence["sample"]]
    feasible = [t for t in tasks if seedsets.slot_order(seedsets.mix_for("dev", 9, 3))[t.index].kind == "feasible"]
    assert [(w.task_id, w.index, w.seed, w.distractors, w.obstacles) for w in sample] == \
           [(t.task_id, t.index, t.seed, t.distractors, t.obstacles) for t in feasible]
    assert [(e["seed"], e["slot"], e["reason"]) for e in evidence["exclusions"]] == \
           [(e.seed, e.slot, e.reason) for e in excluded]
    assert evidence["seeds"] == {"first": 0, "next": tasks[-1].seed + 1, "attempted": len(tasks) + len(excluded),
                                 "accepted_feasible": 9, "accepted_impossible": 3, "excluded": len(excluded)}
    blocked = [e for e in evidence["exclusions"] if e["reason"] == "witness_blocked"]
    assert blocked and all(e["kind"] == "feasible" and e["checks"] == ["collision"] for e in blocked)


def test_evidence_reports_the_rule_the_statistics_and_both_thresholds():
    from vplan.evaluate import calibrate
    values = {}

    def minima(seed):
        values[seed] = (0.01 + 0.001 * len(values), 0.012 + 0.001 * len(values))
        return values[seed]
    evidence = calibrate.calibrate("dev", feasible=12, impossible=0, measure=fake_measure(values=minima))
    v = [w["verifier_sigma_min"] for w in evidence["sample"]]
    j = [w["judge_sigma_min"] for w in evidence["sample"]]
    assert len(v) == 12
    assert evidence["thresholds"]["calibrated"] == calibrate.calibrated_thresholds(v, j)
    disk = config_snapshot()
    assert evidence["thresholds"]["previous"] == {"verifier": disk["verifier"]["sigma_min_threshold"],
                                                  "judge": disk["judge"]["sigma_min_threshold"]}
    stats = evidence["statistics"]["verifier"]
    assert stats["n"] == 12 and stats["min"] == min(v) and stats["max"] == max(v)
    for q in (1, 5, 50):
        assert stats[f"p{q}"] == pytest.approx(np.percentile(v, q, method="linear"))
    assert evidence["statistics"]["judge"]["p1"] == pytest.approx(np.percentile(j, 1, method="linear"))
    new, old = evidence["thresholds"]["calibrated"], evidence["thresholds"]["previous"]
    assert evidence["thresholds"]["rejected"]["calibrated"] == {
        "verifier": sum(x < new["verifier"] for x in v), "judge": sum(x < new["judge"] for x in j),
        "either": sum(a < new["verifier"] or b < new["judge"] for a, b in zip(v, j))}
    assert evidence["thresholds"]["rejected"]["previous"]["either"] == sum(
        a < old["verifier"] or b < old["judge"] for a, b in zip(v, j))
    rule = evidence["rule"]
    assert "linear" in rule["statistic"] and "0.001" in rule["rounding"] and "min(" in rule["judge"]
    assert evidence["kind"] == "sigma_calibration" and evidence["split"] == "dev"
    assert evidence["mix"] == {"feasible": 12, "impossible": 0, "d51_default": False}


def test_evidence_has_provenance_but_no_local_paths():
    from vplan.evaluate import calibrate
    from vplan.evaluate.pilot import ROOT
    evidence = calibrate.calibrate("dev", feasible=3, impossible=1, measure=fake_measure())
    prov = evidence["provenance"]
    assert {"created_utc", "git", "python", "packages", "config_hashes", "calibration_config_hashes",
            "source_hashes", "source_digest", "manipulapy_source_digest"} <= set(prov)
    assert set(prov["packages"]) >= {"numpy", "pybullet", "ManipulaPy"}
    assert prov["config_hashes"] != prov["calibration_config_hashes"]
    text = json.dumps(evidence)
    for local in (str(ROOT), ROOT.as_posix(), "/mnt/", ":\\\\", "C:/", "/home/"):
        assert local not in text


def test_calibration_refuses_the_test_split_and_a_candidate_limit():
    from vplan.evaluate import calibrate, seedsets
    with pytest.raises(seedsets.EvalError, match="dev"):
        calibrate.calibrate("test", feasible=3, impossible=1, measure=fake_measure())

    def always_rejects(task, target_id, seed, cfgs):
        evaluation = fake_evaluation(task, target_id)
        return records.WitnessEvaluation(plan=evaluation.plan, outcome="blocked", findings=(records.Finding(
            stage="lift", check="collision", plan_field="grasp", measured=0.0, threshold=0.01, direction="x"),)), None
    with pytest.raises(seedsets.EvalError, match="candidates"):
        calibrate.calibrate("dev", feasible=2, impossible=0, max_candidates=5, measure=always_rejects)


# ---- --apply ----------------------------------------------------------------------------------------------
def config_copy(tmp_path):
    out = tmp_path / "config"
    shutil.copytree(CONFIG_DIR, out)
    return out


def test_config_files_round_trip_so_apply_changes_one_line_each(tmp_path):
    from vplan.evaluate import calibrate
    for name in ("verifier", "judge"):
        raw = (CONFIG_DIR / f"{name}.json").read_bytes()
        assert calibrate.config_bytes(json.loads(raw)) == raw  # no formatting churn in the committed diff
    cfg_dir = config_copy(tmp_path)
    calibrate.apply_thresholds({"verifier": 0.013, "judge": 0.011}, config_dir=cfg_dir)
    for name, value in (("verifier", 0.013), ("judge", 0.011)):
        old = (CONFIG_DIR / f"{name}.json").read_bytes().split(b"\n")
        new = (cfg_dir / f"{name}.json").read_bytes().split(b"\n")
        changed = [(a, b) for a, b in zip(old, new) if a != b]
        assert len(old) == len(new) and len(changed) == 1
        assert changed[0][1] == f'  "sigma_min_threshold": {value},'.encode()
        assert b"\r" not in (cfg_dir / f"{name}.json").read_bytes()


def test_apply_changes_only_sigma_min_threshold(tmp_path):
    from vplan.evaluate import calibrate
    cfg_dir = config_copy(tmp_path)
    before = {n: json.loads((cfg_dir / f"{n}.json").read_text(encoding="utf-8")) for n in ("verifier", "judge")}
    physics = (cfg_dir / "physics.json").read_bytes()
    hashes = calibrate.apply_thresholds({"verifier": 0.017, "judge": 0.015}, config_dir=cfg_dir)
    after = {n: json.loads((cfg_dir / f"{n}.json").read_text(encoding="utf-8")) for n in ("verifier", "judge")}
    assert after["verifier"] == dict(before["verifier"], sigma_min_threshold=0.017)
    assert after["judge"] == dict(before["judge"], sigma_min_threshold=0.015)
    assert list(after["verifier"]) == list(before["verifier"])  # key order kept
    assert (cfg_dir / "physics.json").read_bytes() == physics
    from vplan.config import config_hash
    assert hashes == {"verifier": config_hash(after["verifier"]), "judge": config_hash(after["judge"])}


@pytest.mark.parametrize("thresholds", [{"verifier": 0.0, "judge": 0.0}, {"verifier": 0.02, "judge": 0.03},
                                        {"verifier": float("nan"), "judge": 0.01}, {"verifier": 0.02}])
def test_apply_refuses_thresholds_that_disable_the_check_or_break_the_judge_rule(tmp_path, thresholds):
    from vplan.evaluate import calibrate, seedsets
    cfg_dir = config_copy(tmp_path)
    before = {p.name: p.read_bytes() for p in cfg_dir.iterdir()}
    with pytest.raises(seedsets.EvalError):
        calibrate.apply_thresholds(thresholds, config_dir=cfg_dir)
    assert {p.name: p.read_bytes() for p in cfg_dir.iterdir()} == before


def test_run_calibration_writes_the_evidence_and_applies_only_on_request(tmp_path, monkeypatch):
    from vplan.evaluate import calibrate
    applied = []
    monkeypatch.setattr(calibrate, "measure_witness", fake_measure())
    monkeypatch.setattr(calibrate, "apply_thresholds",
                        lambda thresholds, config_dir=CONFIG_DIR: applied.append(thresholds) or {"verifier": "v",
                                                                                                 "judge": "j"})
    out = tmp_path / "evidence.json"
    evidence = calibrate.run_calibration(out, "dev", feasible=3, impossible=1)
    raw = out.read_bytes()
    assert json.loads(raw) == evidence and raw.endswith(b"}\n") and b"\r\n" not in raw
    assert evidence["applied"] is None and not applied
    with pytest.raises(calibrate.EvalError, match="exists"):
        calibrate.run_calibration(out, "dev", feasible=3, impossible=1)


def test_run_calibration_applies_only_the_full_dev_mix(tmp_path, monkeypatch):
    from vplan.evaluate import calibrate
    monkeypatch.setattr(calibrate, "measure_witness", lambda *a: pytest.fail("must refuse before any witness"))
    with pytest.raises(calibrate.EvalError, match="D51"):
        calibrate.run_calibration(tmp_path / "e.json", "dev", feasible=3, impossible=1, apply=True)
    assert not (tmp_path / "e.json").exists()


def test_run_calibration_apply_records_the_new_config_hashes(tmp_path, monkeypatch):
    from vplan.evaluate import calibrate
    cfg_dir = config_copy(tmp_path)
    monkeypatch.setattr(calibrate, "measure_witness", fake_measure())
    monkeypatch.setattr(calibrate, "CONFIG_DIR", cfg_dir)
    monkeypatch.setattr(calibrate, "mix_is_default", lambda split, feasible, impossible: True)
    evidence = calibrate.run_calibration(tmp_path / "e.json", "dev", feasible=4, impossible=1, apply=True)
    new = evidence["thresholds"]["calibrated"]
    assert json.loads((cfg_dir / "verifier.json").read_text(encoding="utf-8"))["sigma_min_threshold"] == new["verifier"]
    assert json.loads((cfg_dir / "judge.json").read_text(encoding="utf-8"))["sigma_min_threshold"] == new["judge"]
    from vplan.config import config_hash
    assert evidence["applied"] == {
        "verifier": config_hash(json.loads((cfg_dir / "verifier.json").read_text(encoding="utf-8"))),
        "judge": config_hash(json.loads((cfg_dir / "judge.json").read_text(encoding="utf-8")))}
    assert json.loads((tmp_path / "e.json").read_text(encoding="utf-8"))["applied"] == evidence["applied"]
    # the repository's own configs are untouched
    assert config_snapshot()["verifier"]["sigma_min_threshold"] == \
        json.loads((CONFIG_DIR / "verifier.json").read_text(encoding="utf-8"))["sigma_min_threshold"]


# ---- CLI --------------------------------------------------------------------------------------------------
def test_cli_eval_calibrate_wiring(monkeypatch, capsys):
    from vplan.cli import main
    from vplan.evaluate import calibrate
    calls = {}

    def fake(output, split, **kw):
        calls.update(output=str(output), split=split, **kw)
        return {"sample": [{}] * 75, "thresholds": {"calibrated": {"verifier": 0.012, "judge": 0.011}},
                "applied": {"verifier": "a", "judge": "b"} if kw["apply"] else None}
    monkeypatch.setattr(calibrate, "run_calibration", fake)
    assert main(["eval", "calibrate", "--seeds", "dev", "--output", "runs/sigma.json", "--apply"]) == 0
    assert calls["output"] == "runs/sigma.json" and calls["split"] == "dev" and calls["apply"] is True
    assert (calls["feasible"], calls["impossible"], calls["max_candidates"]) == (None, None, None)
    out = capsys.readouterr().out
    assert "0.012" in out and "0.011" in out and "75" in out and "applied" in out


def test_cli_eval_calibrate_refusal_exits_2(tmp_path, capsys, monkeypatch):
    from vplan.cli import main
    from vplan.evaluate import calibrate
    monkeypatch.setattr(calibrate, "measure_witness", lambda *a: pytest.fail("must refuse before any witness"))
    assert main(["eval", "calibrate", "--seeds", "dev", "--output", str(tmp_path / "e.json"), "--feasible", "3",
                 "--impossible", "1", "--apply"]) == 2
    err = capsys.readouterr().err
    assert "D51" in err and "Traceback" not in err
    with pytest.raises(SystemExit):
        main(["eval", "calibrate", "--seeds", "test", "--output", str(tmp_path / "e.json")])


# ---- the measured quantities (real simulation) ------------------------------------------------------------
def test_witness_minima_are_the_quantities_v4_and_v3_compare():
    """One real witness at zero thresholds: V-4 passes at exactly the Verifier minimum and fires a hair above it, at
    the recorded sample; V3 likewise at the Judge minimum."""
    from vplan.evaluate import calibrate, seedsets
    from vplan.judge.judge import judge
    from vplan.motion.trajectory import build_trajectory
    from vplan.robot import HOME_Q, load_panda
    from vplan.tasks import draft_task
    from vplan.tasks.generate import witness_attempt
    from vplan.verify import checks

    seed = 1  # accepted in the R6 spike (test_generate)
    draft = draft_task(seed, "feasible")
    task, target = draft.task, draft.target_id
    cfgs = calibrate.zero_sigma_snapshot(config_snapshot())
    attempt = witness_attempt(task, target, seed=seed, config_snapshot=cfgs)
    assert attempt.outcome == "executed" and attempt.judge is not None and attempt.judge.success
    m = calibrate.witness_minima(task, attempt, seed=seed, config_snapshot=cfgs)

    traj = build_trajectory(attempt.plan, task.scene, HOME_Q, seed=seed, cfg=cfgs["verifier"])
    panda = load_panda()
    at = dict(cfgs["verifier"], sigma_min_threshold=m.verifier)
    assert all(checks.singularity(seg, panda, at) is None for seg in traj.segments)
    above = dict(cfgs["verifier"], sigma_min_threshold=math.nextafter(m.verifier, 1.0))
    first = next((i, f) for i, f in enumerate(checks.singularity(seg, panda, above) for seg in traj.segments) if f)
    assert first[0] == m.verifier_segment and traj.segments[first[0]].label == m.verifier_stage
    assert first[1].sample_index == m.verifier_sample and first[1].measured == m.verifier

    rec = attempt.record
    assert m.judge == min(rec.sigma_min_steps) and rec.sigma_min_steps[m.judge_step] == m.judge
    for thr, ok in ((m.judge, True), (math.nextafter(m.judge, 1.0), False)):
        result = judge(rec, task.scene, target, "executed", True, planned_target=target,
                       cfg=dict(cfgs["judge"], sigma_min_threshold=thr))
        assert result.predicates["V3"] is ok

    # Another snapshot does not reproduce the executed trajectory's approval token, so its minima are refused.
    other = deepcopy(cfgs)
    other["verifier"]["collision_margin_m"] += 0.001
    with pytest.raises(seedsets.EvalError, match="approval"):
        calibrate.witness_minima(task, attempt, seed=seed, config_snapshot=other)

    evaluation, minima = calibrate.measure_witness(task, target, seed, cfgs)
    assert minima == m and evaluation.result == attempt.judge and evaluation.plan == attempt.plan
