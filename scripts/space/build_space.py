"""Assemble a Hugging Face Space folder, ready to upload: python scripts/space/build_space.py --out DIR [--runs RUNS].

The folder keeps the repository's layout (src/vplan, config/), because vplan/config.py finds config/ two levels above
its package, and adds the Space's own app.py, requirements.txt and README.md, the LICENSE, and replay data under data/:
a curated part of two evaluation runs (the full runs are 280 MB and 368 MB, mostly recorded physics) and the whole test
seed set they were made on, so that ui.sources can check its hash.

Curation is deterministic. Per run, an attempt is kept if it is in ALWAYS (the video's repair and refusal) or is the
first match of a category in CATEGORIES that no kept attempt covers yet. Among matches, scenes with obstacles come
first, then the lowest task index; the file keeps the attempts in their original order and as the original bytes."""
from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SPACE = Path(__file__).resolve().parent
# (folder under data/runs/, run folder under --runs): the D70 headline run and the D85 merged-findings run with Ultra
RUNS = (("lightning-d70", "dryrun-tf/c3-test"), ("ultra-d85", "a1-test/ultra-c3-test"))
SEEDS = "seeds/test"
SEED_FILES = ("tasks.jsonl", "oracle.jsonl", "manifest.json", "exclusions.jsonl")
ALWAYS = {"3:test:1": "shown in the demo video (repair)", "3:test:3": "shown in the demo video (refusal)"}


@dataclass(frozen=True)
class Attempt:
    """What the rules read from one attempts.jsonl line (an evaluate EvalAttempt) and its task in the seed set."""
    attempt_id: str
    line: bytes
    task_index: int
    outcome: str
    rounds: int
    success: bool | None  # the Judge's verdict; None when nothing was judged
    feasible: bool
    impossible_type: int | None
    obstacles: int


def _refusal(kind: int) -> tuple[str, Callable[[Attempt], bool]]:
    return (f"correct refusal, impossible type {kind}",
            lambda a: a.outcome == "refused" and bool(a.success) and a.impossible_type == kind)


def _solved(a: Attempt) -> bool:
    """A solvable task executed and judged a success. In more than one round, earlier rounds were blocked: a repair."""
    return a.feasible and a.outcome == "executed" and bool(a.success)


CATEGORIES: tuple[tuple[str, Callable[[Attempt], bool]], ...] = (
    ("repair in 2 rounds", lambda a: _solved(a) and a.rounds == 2),
    ("repair in 3 rounds", lambda a: _solved(a) and a.rounds == 3),
    ("first-round success", lambda a: _solved(a) and a.rounds == 1),
    *(_refusal(kind) for kind in (1, 2, 3, 4)),
    ("budget exhausted on a solvable task", lambda a: a.feasible and a.outcome == "budget_exhausted"),
    ("executed, but the Judge failed it", lambda a: a.feasible and a.outcome == "executed" and a.success is False),
)


def _lines(path: Path) -> list[bytes]:
    """The non-empty lines of a JSONL file, each as stored (with its newline)."""
    with open(path, "rb") as f:
        return [line for line in f if line.strip()]


def read_attempts(run_dir: Path, seed_dir: Path) -> list[Attempt]:
    """Every attempt that has a record to replay, in file order, joined to its seed-set task and oracle."""
    tasks = [json.loads(line) for line in _lines(seed_dir / "tasks.jsonl")]
    oracles = [json.loads(line)["oracle"] for line in _lines(seed_dir / "oracle.jsonl")]
    out = []
    for line in _lines(run_dir / "attempts.jsonl"):
        row = json.loads(line)
        record = row.get("record")
        if not record:  # a crashed attempt: nothing to replay
            continue
        index, oracle, judge = row["task_index"], oracles[row["task_index"]], record.get("judge")
        feasible = oracle["feasibility"] == "feasible"
        out.append(Attempt(row["attempt_id"], line, index, record["outcome"], len(record["rounds"]),
                           judge["success"] if judge else None, feasible,
                           None if feasible else oracle["proof"]["type"], tasks[index]["obstacles"]))
    return out


def curate(attempts: list[Attempt]) -> dict[str, str]:
    """{attempt id: why it was kept}, in the order the rules chose them."""
    by_id = {a.attempt_id: a for a in attempts}
    missing = [i for i in ALWAYS if i not in by_id]
    if missing:
        raise SystemExit(f"error: no replayable attempt {', '.join(missing)} in this run")
    kept = dict(ALWAYS)
    for name, matches in CATEGORIES:
        if any(matches(by_id[i]) for i in kept):
            continue  # an attempt already kept shows this
        found = min((a for a in attempts if matches(a)), key=lambda a: (a.obstacles == 0, a.task_index), default=None)
        if found is not None:
            kept[found.attempt_id] = name
    return kept


def _copy_run(source: Path, dest: Path, seed_dir: Path) -> dict[str, str]:
    dest.mkdir(parents=True)
    for name in ("manifest.json", "sessions.jsonl"):
        shutil.copyfile(source / name, dest / name)
    attempts = read_attempts(source, seed_dir)
    kept = curate(attempts)
    with open(dest / "attempts.jsonl", "wb") as f:
        for a in attempts:
            if a.attempt_id in kept:
                f.write(a.line if a.line.endswith(b"\n") else a.line + b"\n")
    return kept


def build(out: Path, runs: Path) -> dict[str, dict[str, str]]:
    """Write the Space folder to out (which must be empty or new); returns the kept attempts per run folder."""
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"error: {out} is not empty; choose a new folder")
    out.mkdir(parents=True, exist_ok=True)
    for name in ("app.py", "requirements.txt", "README.md"):
        shutil.copyfile(SPACE / name, out / name)
    shutil.copyfile(REPO / "LICENSE", out / "LICENSE")
    shutil.copytree(REPO / "src" / "vplan", out / "src" / "vplan",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (out / "config").mkdir()
    for path in sorted((REPO / "config").glob("*.json")):
        shutil.copyfile(path, out / "config" / path.name)
    seed_dir = runs / SEEDS
    (out / "data" / "seeds" / "test").mkdir(parents=True)
    for name in SEED_FILES:
        shutil.copyfile(seed_dir / name, out / "data" / "seeds" / "test" / name)
    return {dest: _copy_run(runs / source, out / "data" / "runs" / dest, seed_dir) for dest, source in RUNS}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Hugging Face Space folder.")
    parser.add_argument("--out", type=Path, required=True, help="a new or empty folder")
    parser.add_argument("--runs", type=Path, default=REPO / "runs",
                        help="the folder holding the evaluation runs and seeds/test (default: <repo>/runs)")
    args = parser.parse_args(argv)
    chosen = build(args.out, args.runs)
    for dest, kept in chosen.items():
        print(f"{dest}: {len(kept)} attempts")
        for attempt_id, why in kept.items():
            print(f"  {attempt_id}  {why}")
    size = sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file())
    print(f"{args.out}: {size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
