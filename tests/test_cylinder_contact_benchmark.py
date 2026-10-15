import json
import re
import sys

from scripts.spikes.cylinder_contact import cylinder_seeds


def test_cylinder_benchmark_selects_the_documented_cohorts() -> None:
    first = cylinder_seeds(range(60))
    comparison = cylinder_seeds(range(60, 144))
    assert len(first) == 21  # seed 50 is subsequently blocked by the Verifier
    assert (first[0], first[-1]) == (5, 58)
    assert len(comparison) == 45
    assert (comparison[0], comparison[-1]) == (60, 143)


def _strings(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, str):
        yield value


def test_benchmark_provenance_binds_inputs_source_and_runtime(monkeypatch) -> None:
    import importlib.util
    from pathlib import Path

    import scripts.spikes.cylinder_contact as benchmark
    from vplan.config import config_hash, config_snapshot
    from vplan.evaluate.pilot import ROOT, source_digest

    # Windows git cannot follow a WSL worktree's gitdir (D53), so the revision comes from a stub here.
    monkeypatch.setattr(benchmark, "_git", lambda *a, **k: {"head": "a" * 40, "dirty": False})
    cfg = config_snapshot()
    result = benchmark.benchmark_provenance(cfg, [5, 8])
    assert result["git"] == {"head": "a" * 40, "dirty": False}
    assert result["seeds"] == [5, 8]
    assert result["configs"] == cfg and result["configs"] is not cfg
    assert result["config_hashes"] == {key: config_hash(value) for key, value in cfg.items()}
    assert result["python"] and result["os"] and result["captured_utc"]
    hashes = result["source_hashes"]
    for path in ("scripts/spikes/cylinder_contact.py", "scripts/spikes/r6_generation.py", "pyproject.toml",
                 "src/vplan/sim/scene.py", "config/verifier.json"):
        assert re.fullmatch(r"[0-9a-f]{64}", hashes[path]), path
    assert result["source_digest"] == source_digest(hashes)
    for name in ("numpy", "pybullet", "ManipulaPy"):
        assert result["packages"][name]
    # The editable upstream checkout can change without changing its advertised version.
    assert re.fullmatch(r"[0-9a-f]{64}", result["manipulapy"]["source_digest"])
    assert re.fullmatch(r"[0-9a-f]{64}", result["pybullet_binary_sha256"])
    assert result["seed_set_sha256"] != benchmark.benchmark_provenance(cfg, [8, 5])["seed_set_sha256"]
    # Benchmark output is committed under docs/evidence: no local paths (as D56 strips them from reports). Decoded
    # strings are searched, since JSON text escapes the backslashes of a Windows path.
    manipulapy = Path(importlib.util.find_spec("ManipulaPy").origin).parent
    found = [s for s in _strings(result) for local in (ROOT, manipulapy) if str(local) in s or local.as_posix() in s]
    assert not found


def test_benchmark_provenance_records_a_missing_revision_as_null(monkeypatch) -> None:
    import scripts.spikes.cylinder_contact as benchmark
    from vplan.config import config_snapshot

    monkeypatch.setattr(benchmark, "_git", lambda *a, **k: None)
    assert benchmark.benchmark_provenance(config_snapshot(), [5])["git"] is None


def test_benchmark_json_carries_provenance_captured_before_any_run(monkeypatch, tmp_path) -> None:
    import scripts.spikes.cylinder_contact as benchmark

    order = []
    monkeypatch.setattr(benchmark, "cylinder_seeds", lambda seeds: [5])
    monkeypatch.setattr(benchmark, "use_variant", lambda variant: None)  # keep the collision shapes untouched
    monkeypatch.setattr(benchmark, "benchmark_provenance",
                        lambda cfg, seeds: order.append("provenance") or {"seeds": list(seeds)})
    monkeypatch.setattr(benchmark, "feasible_row", lambda seed, cfg: order.append(seed) or {
        "seed": seed, "accepted": True, "reason": None, "violations": []})
    out = tmp_path / "bench.json"
    monkeypatch.setattr(sys, "argv", ["cylinder_contact.py", "--variant", "prism16", "--cohort", "first",
                                      "--json", str(out)])
    benchmark.main()
    assert order == ["provenance", 5]
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["provenance"] == {"seeds": [5]} and [r["seed"] for r in data["rows"]] == [5]
    assert (data["variant"], data["cohort"]) == ("prism16", "first")
