"""Cached W3.5 UI fixtures shared by the UI test modules: real, offline attempts (heuristic planner or a fake
provider client), each simulated once per test session."""
from __future__ import annotations

import functools
import hashlib
import html as html_lib
import json
import re
from pathlib import Path

import numpy as np

from vplan import records
from vplan.ui import live

SECRET = "sk-or-v1-UI-TEST-SECRET-0123456789abcdef"  # a fake key; it must never reach any UI output
MAGENTA = (255, 0, 255)

# Internal project terms that must not reach the screen (W4.1): spec and decision references, check and predicate
# codes as labels, oracle/witness/heuristic jargon, raw body ids, outcome codes and Judge/plan field names.
JARGON = re.compile(r"spec §|§ ?\d|\bD\d{2}\b|\bV-\d\b|\b[SV]\d\b|\boracle\b|\bwitness\b|heuristic|\bW3\.\d|cube_red|"
                    r"cube_green|wall_1|\bo_[0-9a-f]{6,}|robot:|panda_link|panda_joint|--cosmos-url|"
                    r"\bmalformed|protocol_error|sim_error|stale_approval|service_failure|budget_exhausted|"
                    r"visual_uncertain|\bdwell|\bsigma|target_id|place\.position|release_height_m|transport_via",
                    re.IGNORECASE)


def visible(markup_html: str) -> str:
    """The text a viewer reads: tags (and their tooltip attributes) removed, entities decoded."""
    text = re.sub(r"<pre[^>]*>.*?</pre>", " ", markup_html, flags=re.S)  # recorded plan JSON keeps its values
    return re.sub(r"\s+", " ", html_lib.unescape(re.sub(r"<[^>]+>", " ", text)))


@functools.lru_cache(maxsize=None)
def basic_task() -> live.LiveTask:
    return live.basic_task(live.BASIC_INSTRUCTION, obstacle=True)


@functools.lru_cache(maxsize=None)
def basic_result() -> tuple[live.LiveResult, tuple[live.LiveEvent, ...]]:
    """The heuristic on the basic scene with its wall: one round, executed, Judge success, GIF rendered."""
    events: list[live.LiveEvent] = []
    result = live.run_live(basic_task(), live.heuristic_option(), k=3, on_event=events.append)
    return result, tuple(events)


@functools.lru_cache(maxsize=None)
def enclosed_result() -> tuple[live.LiveResult, tuple[live.LiveEvent, ...]]:
    """An impossible type-3 task (target enclosed): the heuristic plans anyway and the Verifier blocks every round."""
    events: list[live.LiveEvent] = []
    task = live.generated_task(1, "impossible 3: target enclosed")
    result = live.run_live(task, live.heuristic_option(), k=1, on_event=events.append, render_execution=False)
    return result, tuple(events)


def low_transport(plan: records.PickPlacePlan) -> records.PickPlacePlan:
    """The heuristic plan with its transport lowered through the basic scene's wall: V-6 blocks it."""
    return plan.model_copy(update={"transport_via": ((0.5, -0.15, 0.12), (0.45, 0.28, 0.12))})


def magenta_pixels(image: np.ndarray) -> int:
    r, g, b = (image[:, :, i].astype(int) for i in range(3))
    return int(((r > 200) & (g < 80) & (b > 200)).sum())


def write_eval_run(root: Path, task: live.LiveTask, record: records.AttemptRecord, *, config: int = 5,
                   split: str = "dev") -> tuple[Path, Path]:
    """A minimal W3.1 run directory and its seed set holding one task (index 0), in the formats the runner writes."""
    from vplan.evaluate.seedsets import seed_set_hash

    seeds, run = root / "seeds", root / "run"
    seeds.mkdir(parents=True)
    run.mkdir(parents=True)
    task_id = f"{split}_00000000"
    row = records.SeedSetTask(task_id=task_id, index=0, seed=task.seed, split=split,
                              distractors=len(task.task.scene.objects) - 1, obstacles=len(task.task.scene.obstacles),
                              task=task.task)
    (seeds / "tasks.jsonl").write_text(row.model_dump_json() + "\n", encoding="utf-8", newline="\n")
    (seeds / "oracle.jsonl").write_text(records.SeedSetOracle(task_id=task_id, index=0, oracle=task.oracle)
                                        .model_dump_json() + "\n", encoding="utf-8", newline="\n")
    digest = seed_set_hash(*(hashlib.sha256((seeds / n).read_bytes()).hexdigest() for n in ("tasks.jsonl",
                                                                                            "oracle.jsonl")))
    attempt_id = f"{config}:{split}:0"
    manifest = {"schema_version": 1, "kind": "eval_run", "run_id": "run", "config": config, "mode": record.mode,
                "k": record.k, "provider": "heuristic", "model": "template-heuristic", "split": split,
                "seed_set_hash": digest, "seed_set": {"dir": str(seeds), "resolved": str(seeds.resolve())},
                "n": 1, "n_prepared": 1}
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    attempt = records.EvalAttempt(attempt_id=attempt_id, config=config, split=split, task_index=0, task_id=task_id,
                                  session=0, outcome=record.outcome, tries=1, wall_s=1.0,
                                  record=record.model_copy(update={"attempt_id": attempt_id}))
    (run / "attempts.jsonl").write_text(attempt.model_dump_json() + "\n", encoding="utf-8", newline="\n")
    return run, seeds
