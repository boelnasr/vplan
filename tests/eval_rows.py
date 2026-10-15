"""Synthetic evaluation rows for the W3.2 metrics and report tests.

AttemptRecords come from the real run_task with a scripted planner and scripted evaluations, so every round, outcome,
usage total and observable-failure tag is what the runner would record. Tasks are hand-built on the basic scene. No
simulation, network or model calls.
"""
from __future__ import annotations

import random
from functools import lru_cache

from vplan import records
from vplan.config import config_snapshot
from vplan.planners.heuristic import heuristic_geometry
from vplan.repair import run_task
from vplan.tasks.scenes import make_basic_scene

SCENE = make_basic_scene(with_obstacle=True)  # cube_red (the target), cube_green, obstacle wall_1
SPEC = records.TaskSpec(instruction="put the red cube in the bin", scene=SCENE, robot_facts=records.RobotFacts(
    base_pose=SCENE.robot_base, reach_bound_m=0.9489, gripper_opening_m=0.08, table_height_m=0.0))
_COLORS = ("blue", "yellow", "purple", "orange")


@lru_cache(maxsize=None)
def spec_for(distractors: int, obstacles: int) -> records.TaskSpec:
    """SPEC with its scene cut or padded to the given counts (cube_red stays the target; cube_green and wall_1 come
    first), so a row's scene counts are the scene's own, as AttemptRow requires. Nothing is simulated, so the extra
    bodies only need distinct ids."""
    red, green = SCENE.objects
    extra = tuple(green.model_copy(update={"id": f"cube_{i}", "color": _COLORS[i % len(_COLORS)],
                                           "pose": records.Pose(position=(0.35 + 0.05 * i, -0.30, 0.02))})
                  for i in range(max(0, distractors - 1)))
    walls = tuple(records.Obstacle(id=f"box_{i}", half_extents=(0.02, 0.02, 0.05),
                                   pose=records.Pose(position=(0.30 + 0.05 * i, 0.45, 0.05)))
                  for i in range(max(0, obstacles - 1)))
    scene = SCENE.model_copy(update={"objects": ((red, green) + extra)[:1 + distractors],
                                     "obstacles": (SCENE.obstacles + walls)[:obstacles]})
    return SPEC.model_copy(update={"scene": records.Scene.model_validate(scene.model_dump())})
GOOD = heuristic_geometry(SCENE, "cube_red")  # the canonical plan, which the feasible oracle's witness used
FIXED = GOOD.model_copy(update={"transport_via": ((0.5, -0.15, 0.30), (0.45, 0.28, 0.30))})  # a changed repair
WRONG = heuristic_geometry(SCENE, "cube_green")  # a plan for the wrong object

OPENROUTER = ("openrouter", "nvidia/nemotron-3-ultra-550b-a55b:free")
TOKEN_FACTORY = ("api.tokenfactory.nebius.com", "nvidia/Nemotron-3-Ultra-550b-a55b")
HEURISTIC = ("heuristic", "template-heuristic")
FRONTIER = ("openrouter", "frontier/reference-model")
SEED_SET_HASH = "ab" * 32
CFGS = config_snapshot()
PASS = {p: True for p in ("S1", "S2", "V1", "V2", "V3", "V4")}

FEASIBLE = records.TaskOracle(target_id="cube_red", reference_plan=GOOD, reference_result=records.JudgeResult(
    success=True, predicates=PASS, violations=(), details={}))
_REASONS = {1: "target_unreachable", 2: "destination_unreachable", 3: "target_enclosed", 4: "no_matching_object"}
_LID = records.Obstacle(id="lid", half_extents=(0.04, 0.04, 0.01), pose=records.Pose(position=(0.5, -0.15, 0.06)))


def impossible(kind: int) -> records.TaskOracle:
    return records.TaskOracle(
        target_id=None if kind == 4 else "cube_red", feasibility="impossible",
        proof=records.ImpossibilityProof(type=kind, refusal_reason=_REASONS[kind], params={"clearance_m": 1.2},
                                         enclosure=(_LID,) if kind == 3 else ()))


def call(plan=None, outcome=None, *, source=OPENROUTER, tokens=100, latency_s=1.0, status_code=None):
    provider, model = source
    return records.PlannerCall(outcome=outcome or ("proposed" if plan is not None else "malformed"), plan=plan,
                               provider=provider, model=model, protocol="forced", next_protocol="forced",
                               temperature=0.7, latency_s=latency_s, raw_request={}, total_tokens=tokens,
                               status_code=status_code)


def refusal(reason: str = "target_unreachable") -> records.Refusal:
    return records.Refusal(reason=reason, explanation="scripted")


def judged(*failed: str, violations: tuple[str, ...] = ()) -> records.EvaluationResult:
    """An executed plan's evaluation; `failed` lists the Judge predicates that fail."""
    return records.EvaluationResult(outcome="executed", judge=records.JudgeResult(
        success=not failed, predicates={p: p not in failed for p in PASS}, violations=violations or failed,
        details={}))


def blocked(*checks: tuple[str, str]) -> records.EvaluationResult:
    """A gate rejection with one Finding per (check, stage)."""
    return records.EvaluationResult(outcome="blocked", findings=tuple(records.Finding(
        stage=stage, check=check, plan_field="transport_via", measured=0.0, threshold=0.01,
        direction="below margin") for check, stage in checks))


class Scripted:
    def __init__(self, calls):
        self.calls = list(calls)

    def plan(self, task, *, feedback=None, protocol="forced", event_sink=None):
        return self.calls.pop(0)


def as_judged(oracle: records.TaskOracle, evaluation: records.EvaluationResult) -> records.EvaluationResult:
    """The evaluation as the Judge scores it since D48: an executed plan on an impossible task keeps its physics
    predicates and fails R. A custom run_task evaluator owns that Judge call once D48 lands, and run_task adds R itself
    before it, so the scripted rows read the same either way."""
    j = evaluation.judge
    if oracle.feasibility == "feasible" or evaluation.outcome != "executed" or j is None or "R" in j.predicates:
        return evaluation
    return evaluation.model_copy(update={"judge": records.JudgeResult(
        success=False, predicates={**j.predicates, "R": False}, violations=j.violations + ("R",),
        details={**j.details, "refusal": "impossible task attempted"})})


def attempt(oracle, calls, evaluations=(), *, mode="repair", k=3, attempt_id="attempt"):
    """A real AttemptRecord from run_task; `evaluations` answer the scripted plans in order."""
    pending = list(evaluations)
    return run_task(SPEC, oracle, Scripted(calls), k=k, mode=mode,
                    evaluator=lambda plan: as_judged(oracle, pending.pop(0)), attempt_id=attempt_id,
                    config_snapshot=CFGS)


MODES = {1: ("unverified", 0), 3: ("repair", 3), 5: ("repair", 3), 6: ("unverified", 0), 7: ("resample", 3)}
SOURCES = {1: OPENROUTER, 3: OPENROUTER, 5: HEURISTIC, 6: FRONTIER, 7: OPENROUTER}


def row(config, index, oracle, rec=None, *, source=None, outcome=None, distractors=0, obstacles=0,
        error_type=None, wall_s=None, seed_set="test", seed_set_hash=SEED_SET_HASH) -> records.AttemptRow:
    provider, model = source or SOURCES[config]
    return records.AttemptRow(config=config, seed_set=seed_set, seed_set_hash=seed_set_hash, task_index=index,
                              task_id=f"{seed_set}_{index:04d}", task=spec_for(distractors, obstacles), oracle=oracle,
                              distractors=distractors,
                              obstacles=obstacles, outcome=outcome or rec.outcome, attempt=rec, provider=provider,
                              model=model, error_type=error_type, wall_s=wall_s)


def scripted_row(config, index, oracle, calls, evaluations=(), *, source=None, **kwargs) -> records.AttemptRow:
    mode, k = MODES[config]
    source = source or SOURCES[config]
    calls = [c.model_copy(update={"provider": source[0], "model": source[1]}) for c in calls]
    rec = attempt(oracle, calls, evaluations, mode=mode, k=k, attempt_id=f"{config}:test:{index}")
    return row(config, index, oracle, rec, source=source, **kwargs)


# ---- the hand-checked scenario --------------------------------------------------------------------
# Tasks 0-3 and 6 are feasible, 4 is impossible type 1 (target out of reach), 5 is type 4 (referent absent).
STRATA = {0: (0, 0), 1: (1, 2), 2: (3, 3), 3: (0, 0), 4: (2, 0), 5: (1, 1), 6: (2, 1)}
ORACLES = {0: FEASIBLE, 1: FEASIBLE, 2: FEASIBLE, 3: FEASIBLE, 4: impossible(1), 5: impossible(4), 6: FEASIBLE}
COLLISION = ("collision", "transport")
IK = ("ik_accuracy", "approach")
V1_OBSTACLE = "robot:panda_hand<->wall_1:base@transport"
V1_TARGET = "robot:panda_link5<->cube_red:base@transport"
SCRIPTS = {
    3: {0: ([call(GOOD)], [judged()]),
        1: ([call(GOOD), call(FIXED)], [blocked(COLLISION), judged()]),
        2: ([call(refusal("other"))], []),
        3: ([call(), call()], []),
        4: ([call(refusal("target_unreachable"))], []),
        5: ([call(WRONG)], [judged()])},
    7: {0: ([call(GOOD)], [judged()]),
        1: ([call(GOOD)] * 4, [blocked(COLLISION)] * 4),
        2: ([call(GOOD), call(FIXED)], [blocked(COLLISION), judged()]),
        3: ([call(outcome="service_failure", tokens=None, status_code=503)], []),
        4: ([call(refusal("destination_unreachable"))], []),
        5: ([call(WRONG)] * 4, [blocked(IK)] * 4),
        6: ([call(GOOD)], [judged("S1", "V1", violations=(V1_OBSTACLE,))])},
    1: {0: ([call(GOOD)], [judged()]),
        1: ([call(GOOD)], [judged("S1")]),
        2: ([call(WRONG)], [judged("S1", "S2", "V1", violations=(V1_TARGET,))]),
        3: ([call(refusal("other"))], []),
        4: ([call(refusal("target_unreachable"))], []),
        5: ([call(WRONG)], [judged()]),
        6: ([call(GOOD)], [judged("V3")])},
    5: {0: ([call(GOOD, tokens=None)], [judged()]),
        1: ([call(GOOD, tokens=None)] * 4, [blocked(COLLISION)] * 4),
        2: ([call(GOOD, tokens=None)], [judged()]),
        3: ([call(GOOD, tokens=None)], [judged()]),
        4: ([call(refusal("target_unreachable"), tokens=None)], []),
        5: ([call(refusal("no_matching_object"), tokens=None)], []),
        6: ([call(GOOD, tokens=None)], [judged()])},
    6: {0: ([call(GOOD)], [judged()]),
        4: ([call(GOOD)], [judged()])},
}


def scenario_rows(*, source_override: dict | None = None) -> list[records.AttemptRow]:
    """Configs 1, 3, 5, 6 and 7 over the seven tasks; config 3's task 6 is a worker timeout with no record."""
    rows = []
    for config, scripts in SCRIPTS.items():
        source = (source_override or {}).get(config, SOURCES[config])
        for index, (calls, evaluations) in scripts.items():
            d, o = STRATA[index]
            rows.append(scripted_row(config, index, ORACLES[index], calls, evaluations, source=source,
                                     distractors=d, obstacles=o))
    d, o = STRATA[6]
    rows.append(row(3, 6, FEASIBLE, None, source=(source_override or {}).get(3, SOURCES[3]), outcome="sim_error",
                    distractors=d, obstacles=o, error_type="worker_timeout", wall_s=600.0))
    return rows


# ---- a larger random cohort for the demo report and smoke tests -----------------------------------
def random_rows(n_feasible: int = 48, n_impossible: int = 16, seed: int = 0,
                configs: tuple[int, ...] = (1, 3, 5, 6, 7)) -> list[records.AttemptRow]:
    """Seeded, plausible-looking outcomes for every configuration; for rendering checks, not for statistics."""
    rng = random.Random(seed)
    rows = []
    kinds = [None] * n_feasible + [1 + i % 4 for i in range(n_impossible)]
    for index, kind in enumerate(kinds):
        oracle = FEASIBLE if kind is None else impossible(kind)
        d, o = index % 4, (index // 4) % 4
        for config in configs:
            mode, k = MODES[config]
            calls, evaluations = [], []
            refuse_p = {None: 0.03, 1: 0.7, 2: 0.6, 3: 0.3, 4: 0.8}[kind]
            success_p = {1: 0.45, 3: 0.55, 5: 0.6, 6: 0.65, 7: 0.5}[config] - 0.06 * (d + o) / 2
            if config == 5 and kind in (1, 2, 4):  # the heuristic refuses reach and absent-referent tasks (D39)
                calls.append(call(refusal(_REASONS[kind])))
            for _ in range(0 if calls else k + 1):
                roll = rng.random()
                if roll < refuse_p and config != 5:
                    calls.append(call(refusal(_REASONS.get(kind or 1) if rng.random() < .8 else "other")))
                    break
                if roll < refuse_p + 0.03 and config != 5:
                    calls.append(call())  # malformed
                    continue
                if rng.random() < success_p or mode == "unverified":
                    calls.append(call(GOOD if rng.random() < 0.9 else WRONG, latency_s=rng.uniform(2, 9),
                                      tokens=rng.randint(900, 2600)))
                    # without the gate a plan that would have been blocked executes and usually fails the Judge
                    ok = rng.random() < (success_p if mode == "unverified" else 0.85)
                    bad = [] if ok else [rng.choice(["S1", "V1", "V3"])]
                    evaluations.append(judged(*bad, violations=(V1_OBSTACLE,) if "V1" in bad else ()))
                    break
                calls.append(call(FIXED, latency_s=rng.uniform(2, 9), tokens=rng.randint(900, 2600)))
                evaluations.append(blocked(rng.choice([COLLISION, COLLISION, IK, ("release", "descend_place"),
                                                       ("grasp_preconditions", "plan")])))
            if config == 5:  # no LLM, so no token usage
                calls = [c.model_copy(update={"total_tokens": None, "latency_s": 0.01}) for c in calls]
            rows.append(scripted_row(config, index, oracle, calls, evaluations, distractors=d, obstacles=o,
                                     wall_s=rng.uniform(20, 120)))
    return rows


# ---- a W3.1 run manifest with the local paths a real one carries ----------------------------------------
LOCAL_PATHS = ("C:\\Users\\someone", "someone", "/home/", "/mnt/", "\\\\server", "runs/seeds", "Users")


def local_manifest() -> dict:
    """The shape runner._create writes, with absolute paths where a real run has them."""
    return {
        "schema_version": 1, "kind": "eval_run", "run_id": "c3-dev-20261015T120000Z", "config": 3,
        "n": 7, "n_prepared": 7, "limited": False,
        "provider": "openrouter", "model": "nvidia/nemotron-3-ultra-550b-a55b:free", "split": "dev",
        "seed_set_hash": SEED_SET_HASH, "os": "Windows-10-10.0.26100-SP0", "image_digest": "sha256:feedface",
        "seed_set": {"dir": "runs/seeds/dev", "resolved": "C:\\Users\\someone\\repo\\runs\\seeds\\dev",
                     "manifest_sha256": "cd" * 32, "n_tasks": 7},
        "git": {"head": "f" * 40, "dirty": False},
        "manipulapy": {"version": "1.4.2", "location": "C:\\Users\\someone\\ManipulaPy\\ManipulaPy\\__init__.py",
                       "git": {"head": "e" * 40, "dirty": True}},
        "versions": {"numpy": "1.26.4", "ManipulaPy": "1.4.2"},
        "source_hashes": {"src/vplan/cli.py": "12" * 32}, "source_drift": ["src/vplan/cli.py"],
        "extra": {"cache": "/home/someone/.cache/x", "unc": "\\\\server\\share\\x", "wsl": "/mnt/c/Users/x",
                  "forward": "C:/Users/someone/x", "note": "kept", "paths": ["/tmp/a", "kept too"]},
    }


def run_manifest(config: int, n: int, prepared: int) -> dict:
    """The run-coverage fields of a W3.1 manifest: n planned tasks (a prefix with --n) of the seed set's prepared."""
    return {"schema_version": 1, "kind": "eval_run", "run_id": f"c{config}-test", "config": config, "n": n,
            "n_prepared": prepared, "limited": n < prepared, "seed_set_hash": SEED_SET_HASH}
