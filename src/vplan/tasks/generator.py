"""Seeded feasible-only development candidates, selected by independent reference success.

This intentionally covers the feedback pilot, not the full held-out task generator.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable

import numpy as np

from vplan.config import config_hash, config_snapshot
from vplan.records import (Bin, CandidateExclusion, GeneratedTask, Obstacle, Pose, PreparedCohort,
                           RobotFacts, Scene, SceneObject, TaskCandidate, TaskMetadata, TaskOracle,
                           TaskSpec, TaskStratum, WitnessEvaluation)
from vplan.tasks.reach import reach_bound_m
from vplan.tasks.referents import resolve_referent

STRATA: tuple[TaskStratum, ...] = ("clear", "distractor", "obstacle")
COLORS = ("red", "green", "blue", "yellow", "purple", "orange")


def resolve_instruction(instruction: str, scene: Scene) -> str:
    """The single referent of a template instruction, rejecting absent/ambiguous/unknown phrasing."""
    try:
        matches = resolve_referent(instruction, scene)
    except ValueError:
        matches = []
    if len(matches) != 1:
        raise ValueError(f"instruction must resolve to exactly one object; found {len(matches)}")
    return matches[0]


def _metadata(seed: int, stratum: TaskStratum) -> TaskMetadata:
    digest = hashlib.sha256(f"feedback-pilot-v1:{seed}:{stratum}".encode()).hexdigest()[:16]
    return TaskMetadata(task_id=f"dev_{digest}", seed=seed, stratum=stratum)


def generate_candidate(seed: int, stratum: TaskStratum) -> TaskCandidate:
    """Generate geometry without simulation. Candidate.target_id is private truth."""
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if stratum not in STRATA:
        raise ValueError(f"unknown stratum {stratum!r}")
    rng = np.random.default_rng(np.random.SeedSequence([seed, STRATA.index(stratum)]))
    color = str(rng.choice(COLORS))
    # D29 I-6: center top grasps require 3.6–5.9 cm height; 4 cm stays inside.
    target = SceneObject(id=f"o_{rng.integers(0, 2**64, dtype=np.uint64):016x}", shape="cube", color=color,
                         size=(.04, .04, .04),
                         pose=Pose(position=(rng.uniform(.485, .515), rng.uniform(-.17, -.13), .02)))
    objects = [target]
    if stratum != "clear":
        # Both color and shape matter: one same-color cylinder and one other-color cube.
        objects.extend([
            SceneObject(id=f"o_{rng.integers(0, 2**64, dtype=np.uint64):016x}", shape="cylinder", color=color,
                        size=(.04, .04, .04),
                        pose=Pose(position=(rng.uniform(.59, .63), rng.uniform(-.24, -.21), .02))),
            SceneObject(id=f"o_{rng.integers(0, 2**64, dtype=np.uint64):016x}", shape="cube",
                        color=str(rng.choice([c for c in COLORS if c != color])), size=(.04, .04, .04),
                        pose=Pose(position=(rng.uniform(.54, .56), rng.uniform(-.01, .01), .02)))])
    rng.shuffle(objects)  # target index is not a label shortcut
    obstacles = ()
    if stratum == "obstacle":
        half_height = rng.uniform(.065, .08)
        obstacles = (Obstacle(id=f"o_{rng.integers(0, 2**64, dtype=np.uint64):016x}",
                              half_extents=(rng.uniform(.135, .15), .01, half_height),
                              pose=Pose(position=(.50, rng.uniform(.06, .075), half_height))),)
    scene = Scene(table_top_z=0., objects=tuple(objects), obstacles=obstacles,
                  bin=Bin(floor_center=(rng.uniform(.435, .465), rng.uniform(.27, .29), .01),
                          inner_size_xy=(.16, .16), wall_height=.08))
    instruction = f"Pick the {color} cube and place it in the bin."
    # Max TCP distance from the shoulder (tasks.reach); informational here, never a feasibility proof.
    task = TaskSpec(instruction=instruction, scene=scene,
                    robot_facts=RobotFacts(base_pose=scene.robot_base, reach_bound_m=reach_bound_m(),
                                           gripper_opening_m=.08, table_height_m=scene.table_top_z))
    target_id = resolve_instruction(instruction, scene)
    return TaskCandidate(metadata=_metadata(seed, stratum), task=task, target_id=target_id)


def evaluate_witness(candidate: TaskCandidate, *, config_snapshot: dict[str, dict] | None = None) -> WitnessEvaluation:
    """Run the unchanged heuristic, verifier, physics execution and independent Judge."""
    from vplan.pipeline import run_attempt
    from vplan.planners.heuristic import heuristic_geometry

    plan = heuristic_geometry(candidate.task.scene, candidate.target_id)
    result = run_attempt(plan, candidate.task.scene, candidate.target_id,
                         seed=candidate.metadata.seed % (2**32), config_snapshot=config_snapshot)
    return WitnessEvaluation(plan=plan, outcome=result.outcome, findings=tuple(result.findings),
                             result=result.judge)


class CohortPreparationError(RuntimeError):
    """Bounded generation failure; .cohort preserves all accepted tasks and exclusions."""
    def __init__(self, cohort: PreparedCohort):
        self.cohort = cohort
        super().__init__(f"prepared {len(cohort.tasks)}/{cohort.requested} development tasks "
                         f"after {cohort.attempted} candidates (limit {cohort.max_candidates})")


def prepare_cohort(n: int = 20, seed: int = 0, max_candidates: int = 200, *,
                   witness_evaluator: Callable[[TaskCandidate], WitnessEvaluation] | None = None,
                   progress: Callable[[int, int, int], None] | None = None) -> PreparedCohort:
    """Accept only successful witnesses; retain every rejection and bound total attempts.

    Stratum follows accepted-task index (7/7/6 for n=20), so hard strata cannot
    disappear through rejection. progress receives (attempted, accepted, requested).
    The evaluator seam supports offline tests; production defaults to real simulation.
    """
    for name, value, minimum in (("n", n, 1), ("seed", seed, 0), ("max_candidates", max_candidates, 1)):
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    captured_config = config_snapshot()
    evaluator = (lambda candidate: evaluate_witness(candidate, config_snapshot=captured_config)
                 ) if witness_evaluator is None else witness_evaluator
    hashes = {key: config_hash(value) for key, value in captured_config.items()}
    accepted, exclusions = [], []
    for offset in range(max_candidates):
        stratum = STRATA[len(accepted) % len(STRATA)]
        metadata = _metadata(seed + offset, stratum)
        try:
            candidate = generate_candidate(metadata.seed, stratum)
        except Exception as exc:
            exclusions.append(CandidateExclusion(metadata=metadata,
                               reason=f"generation_error: {type(exc).__name__}: {exc}"))
        else:
            try:
                witness = evaluator(candidate)
            except Exception as exc:
                exclusions.append(CandidateExclusion(metadata=metadata,
                                   reason=f"simulation_error: {type(exc).__name__}: {exc}"))
            else:
                reason = None
                if witness.outcome != "executed" or witness.findings:
                    reason = "witness_blocked"
                elif witness.result is None:
                    reason = "missing_judge_result"
                elif not witness.result.success:
                    reason = "judge_failure"
                elif witness.plan.target_id != candidate.target_id:
                    reason = "witness_target_mismatch"
                if reason:
                    exclusions.append(CandidateExclusion(metadata=metadata, reason=reason, witness=witness))
                else:
                    accepted.append(GeneratedTask(metadata=metadata, task=candidate.task,
                                    oracle=TaskOracle(target_id=candidate.target_id,
                                                      reference_plan=witness.plan,
                                                      reference_result=witness.result)))
        if progress is not None:
            progress(offset + 1, len(accepted), n)
        if len(accepted) == n:
            break
    cohort = PreparedCohort(seed=seed, requested=n, max_candidates=max_candidates,
                            attempted=len(accepted) + len(exclusions), tasks=tuple(accepted),
                            exclusions=tuple(exclusions), config_hashes=hashes)
    if hashes != {key: config_hash(value) for key, value in config_snapshot().items()}:
        raise RuntimeError("configuration changed during cohort preparation; discard this cohort")
    if len(accepted) != n:
        raise CohortPreparationError(cohort)
    return cohort
