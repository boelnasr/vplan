"""The Hugging Face Space folder (scripts/space): curated replay data, the pins, the README front matter and the entry
point. Built from a tiny fixture run, so nothing here needs the real runs or the network (app.py itself runs on the
Space only; here it is compiled and read, never executed)."""
from __future__ import annotations

import ast
import hashlib
import json
import py_compile
import re
import shutil
from pathlib import Path

import pytest

from scripts.space import build_space
from tests.ui_fixtures import basic_result, basic_task, write_eval_run
from vplan.evaluate.seedsets import seed_set_hash
from vplan.ui import sources

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # Python 3.10: tomli is a locked pytest dependency there
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
SPACE = ROOT / "scripts" / "space"
# The first blockquote of the repo README: the Space states the headline in the same words (D70)
HEADLINE = ("Bounded repair with verifier feedback raised Nemotron 3.5 Lightning's task success from 24.7% (verified "
            "first attempts) to 65.0% across 300 held-out, reference-solvable generated tasks. At equal budget, "
            "feedback beat resampling by 29.3 points (95% CI 23–36).")

# Seed-set rows 2 and up: (feasible, impossible type, obstacles). Rows 0 and 1 are the real task (feasible, one
# obstacle); the attempt "3:test:1" replays it, as an attempt id must end in its task index.
TASKS = [(True, None, 0), (True, None, 2), (True, None, 3), (False, 1, 0), (False, 2, 0), (False, 3, 5),
         (False, 4, 0)]
# (attempt id, task index, outcome, rounds, Judge success); "3:test:1" is the real, replayable attempt
ATTEMPTS = [
    ("3:test:1", 1, "executed", 1, True),           # always in; also the first-round success
    ("3:test:2", 2, "executed", 2, True),           # a 2-round repair without obstacles ...
    ("3:test:3", 5, "refused", 1, True),            # always in; also the type 1 refusal
    ("3:test:4", 3, "executed", 2, True),           # ... loses to this one, which has obstacles
    ("3:test:5", 2, "executed", 3, True),           # a 3-round repair without obstacles ...
    ("3:test:6", 4, "executed", 3, True),           # ... loses to this one, which has obstacles
    ("3:test:7", 0, "executed", 1, True),           # a first-round success that would win on rank: one is already in
    ("3:test:8", 5, "refused", 1, True),            # a second type 1 refusal: the first is already in
    ("3:test:9", 6, "refused", 3, True),            # type 2 refusal
    ("3:test:10", 2, "budget_exhausted", 4, None),  # a feasible task nobody solved
    ("3:test:11", 7, "budget_exhausted", 4, None),  # an impossible task: not the budget example
    ("3:test:12", 8, "refused", 1, True),           # type 4 refusal
    ("3:test:13", 4, "executed", 1, False),         # a feasible task the Judge failed
    ("3:test:14", 7, "executed", 1, False),         # an impossible one: not the Judge-failed example
    ("3:test:15", 3, "service_failure", 1, None),   # no record to replay
]
CHOSEN = ["3:test:1", "3:test:3", "3:test:4", "3:test:6", "3:test:9", "3:test:10", "3:test:12", "3:test:13"]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(rows) -> bytes:
    return b"".join(json.dumps(row, separators=(",", ":")).encode() + b"\n" for row in rows)


def _attempt_lines(template: dict) -> list[bytes]:
    """One line per ATTEMPTS entry, made from the real attempt so that "3:test:1" replays."""
    lines = []
    for attempt_id, index, outcome, rounds, success in ATTEMPTS:
        row = json.loads(json.dumps(template))
        row.update(attempt_id=attempt_id, task_index=index, task_id=f"test_{index:08d}", outcome=outcome)
        if attempt_id == "3:test:15":
            row["record"] = None
        elif attempt_id != "3:test:1":
            record = row["record"]
            record.update(attempt_id=attempt_id, outcome=outcome, rounds=[record["rounds"][0]] * rounds,
                          judge=None if success is None else {**record["judge"], "success": success})
        else:
            row["record"]["attempt_id"] = attempt_id
        lines.append(_jsonl([row]))
    return lines


@pytest.fixture
def runs(tmp_path):
    """A runs folder shaped like the real one: two runs of the same attempts, and the seed set they were made on."""
    result, _ = basic_result()
    work = tmp_path / "work"
    run, seeds = write_eval_run(work, basic_task(), result.record, config=3, split="test")
    template = json.loads((run / "attempts.jsonl").read_text(encoding="utf-8"))
    for name in ("tasks.jsonl", "oracle.jsonl"):  # row 1: the real task again, under its own index
        row = json.loads((seeds / name).read_text(encoding="utf-8"))
        row.update(index=1, task_id="test_00000001")
        with open(seeds / name, "ab") as f:
            f.write(_jsonl([row]))
    with open(seeds / "tasks.jsonl", "ab") as tasks, open(seeds / "oracle.jsonl", "ab") as oracles:
        for index, (feasible, kind, obstacles) in enumerate(TASKS, 2):
            task_id = f"test_{index:08d}"
            tasks.write(_jsonl([{"task_id": task_id, "index": index, "obstacles": obstacles}]))
            oracle = ({"feasibility": "feasible"} if feasible
                      else {"feasibility": "impossible", "proof": {"type": kind}})
            oracles.write(_jsonl([{"task_id": task_id, "index": index, "oracle": oracle}]))
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    manifest["seed_set_hash"] = seed_set_hash(_sha(seeds / "tasks.jsonl"), _sha(seeds / "oracle.jsonl"))
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (seeds / "manifest.json").write_text(json.dumps({"kind": "seed_set"}), encoding="utf-8")
    (seeds / "exclusions.jsonl").write_text("", encoding="utf-8")
    (run / "attempts.jsonl").write_bytes(b"".join(_attempt_lines(template)))
    (run / "sessions.jsonl").write_text('{"event": "start"}\n', encoding="utf-8")
    for name in ("frames", "workers"):  # large and not needed to replay
        (run / name).mkdir()
        (run / name / "x.bin").write_bytes(b"x")
    root = tmp_path / "runs"
    shutil.copytree(seeds, root / "seeds" / "test")
    for _, source in build_space.RUNS:
        shutil.copytree(run, root / source)
    return root


@pytest.fixture
def built(runs, tmp_path):
    out = tmp_path / "space"
    assert build_space.main(["--out", str(out), "--runs", str(runs)]) == 0
    return out


# ---- curation -------------------------------------------------------------------------------------------------------

def test_the_curation_picks_by_rule_in_file_order(runs, capsys, built):
    for dest, _ in build_space.RUNS:
        lines = (built / "data" / "runs" / dest / "attempts.jsonl").read_bytes().splitlines()
        assert [json.loads(line)["attempt_id"] for line in lines] == CHOSEN
    out = capsys.readouterr().out
    assert all(attempt_id in out for attempt_id in CHOSEN) and "MB" in out
    assert "3:test:5" not in out  # a candidate that lost to one with obstacles is not reported as chosen


def test_the_curated_lines_are_the_originals_byte_for_byte(runs, built):
    source = {json.loads(line)["attempt_id"]: line for line in
              (runs / build_space.RUNS[0][1] / "attempts.jsonl").read_bytes().splitlines(keepends=True)}
    kept = (built / "data" / "runs" / build_space.RUNS[0][0] / "attempts.jsonl").read_bytes()
    assert kept == b"".join(source[attempt_id] for attempt_id in CHOSEN)


def test_a_run_without_an_always_included_attempt_is_an_error(runs, tmp_path):
    path = runs / build_space.RUNS[0][1] / "attempts.jsonl"
    path.write_bytes(b"".join(line for line in path.read_bytes().splitlines(keepends=True)
                              if b'"3:test:3"' not in line))
    with pytest.raises(SystemExit, match="3:test:3"):
        build_space.main(["--out", str(tmp_path / "out"), "--runs", str(runs)])


def test_the_build_refuses_a_non_empty_output_folder(runs, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("mine", encoding="utf-8")
    with pytest.raises(SystemExit, match="not empty"):
        build_space.main(["--out", str(out), "--runs", str(runs)])
    assert (out / "keep.txt").read_text(encoding="utf-8") == "mine"


# ---- the folder -----------------------------------------------------------------------------------------------------

def test_the_built_folder_holds_the_app_the_code_the_configs_and_the_replay_data(runs, built):
    files = {p.relative_to(built).as_posix() for p in built.rglob("*") if p.is_file()}
    package = {p.relative_to(ROOT).as_posix() for p in (ROOT / "src" / "vplan").rglob("*")
               if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    configs = {p.relative_to(ROOT).as_posix() for p in (ROOT / "config").glob("*.json")}
    data = {f"data/runs/{dest}/{name}" for dest, _ in build_space.RUNS
            for name in ("manifest.json", "sessions.jsonl", "attempts.jsonl")}
    data |= {f"data/seeds/test/{name}" for name in ("tasks.jsonl", "oracle.jsonl", "manifest.json", "exclusions.jsonl")}
    assert files == {"app.py", "requirements.txt", "README.md", "LICENSE"} | package | configs | data
    assert (built / "LICENSE").read_bytes() == (ROOT / "LICENSE").read_bytes()
    for dest, source in build_space.RUNS:  # copied as they are
        for name in ("manifest.json", "sessions.jsonl"):
            assert (built / "data" / "runs" / dest / name).read_bytes() == (runs / source / name).read_bytes()
    for name in ("tasks.jsonl", "oracle.jsonl", "manifest.json", "exclusions.jsonl"):
        assert (built / "data" / "seeds" / "test" / name).read_bytes() == (runs / "seeds" / "test" / name).read_bytes()


def test_a_built_run_replays_with_its_bundled_seed_set(built):
    seeds = built / "data" / "seeds" / "test"
    for dest, _ in build_space.RUNS:
        run = built / "data" / "runs" / dest
        roots = (str(run), str(seeds))
        assert [c.case_id for c in sources.list_cases(run, roots=roots)] == CHOSEN
        case = sources.load_case(run, "3:test:1", seed_dir=seeds, roots=roots)
        assert case.record.outcome == "executed" and case.record.success and case.task == basic_task().task


# ---- the files that run on the Space --------------------------------------------------------------------------------

def test_app_py_compiles_and_reads_as_the_space_entry_point(tmp_path):
    py_compile.compile(str(SPACE / "app.py"), cfile=str(tmp_path / "app.pyc"), doraise=True)
    tree = ast.parse((SPACE / "app.py").read_text(encoding="utf-8"))
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert isinstance(imports[0], ast.Import) and imports[0].names[0].name == "spaces"  # first, and not in a try
    gpu = [n for n in tree.body if isinstance(n, ast.FunctionDef) and any(
        ast.unparse(d) == "spaces.GPU(duration=1)" for d in n.decorator_list)]
    assert len(gpu) == 1  # ZeroGPU will not start without one decorated function
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) == "UISettings"]
    assert len(calls) == 1
    settings = {kw.arg: ast.unparse(kw.value) for kw in calls[0].keywords}
    assert {k: settings[k] for k in ("byok", "record_dir", "cosmos_url", "host", "port")} == {
        "byok": "True", "record_dir": "None", "cosmos_url": "None", "host": "'0.0.0.0'", "port": "7860"}
    assert "lightning-d70" in settings["replay_runs"] and "ultra-d85" in settings["replay_runs"]


def test_requirements_are_the_projects_pins_without_the_platform_packages():
    lines = [line.strip() for line in (SPACE / "requirements.txt").read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.startswith("#")]
    pins = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]
    assert lines == [pin for pin in pins if not pin.startswith("gradio")]
    assert not {"gradio", "spaces", "huggingface_hub"} & {re.split(r"[=<>\[]", line)[0].lower() for line in lines}


def _front_matter(text: str) -> dict[str, str]:
    block = text.split("---\n")[1]
    return {key: value.strip().strip('"') for key, _, value in (line.partition(":") for line in block.splitlines())}


def test_the_readme_front_matter_configures_the_space():
    import gradio
    text = (SPACE / "README.md").read_text(encoding="utf-8")
    meta = _front_matter(text)
    assert meta["title"] == "vplan: LLMs plan, physics verifies" and meta["emoji"]
    assert (meta["colorFrom"], meta["colorTo"], meta["sdk"], meta["app_file"]) == ("blue", "indigo", "gradio", "app.py")
    assert meta["sdk_version"] == gradio.__version__  # the version the UI is tested with
    assert (meta["python_version"], meta["license"], meta["startup_duration_timeout"]) == ("3.10", "agpl-3.0", "30m")
    assert 0 < len(meta["short_description"]) <= 60, meta["short_description"]


def test_the_readme_states_the_headline_and_speaks_for_one_person():
    body = (SPACE / "README.md").read_text(encoding="utf-8").split("---\n", 2)[2]
    assert HEADLINE in body
    assert "https://github.com/boelnasr/vplan" in body and "https://youtu.be/y_Ig1zOWKPY" in body
    assert "Ultra" in body and "beside" in body  # the second results are reported next to the headline
    assert re.search(r"\bI\b", body) and not re.search(r"\b(we|our|us)\b", body, re.IGNORECASE)


@pytest.mark.skipif(not (ROOT / "README.md").is_file(), reason="README.md is outside the Docker build context")
def test_the_headline_is_the_repo_readmes():
    assert HEADLINE in (ROOT / "README.md").read_text(encoding="utf-8")
