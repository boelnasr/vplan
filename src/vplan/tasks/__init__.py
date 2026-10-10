"""Task generation: the feasible development pilot and the W2.2 generator. Planners receive only TaskSpec."""
from vplan.tasks.generate import (GenerationRejected, check_impossibility, draft_task, generate_task, run_witness,
                                  scene_issues, witness_rejection)
from vplan.tasks.generator import (STRATA, CohortPreparationError, evaluate_witness, generate_candidate,
                                   prepare_cohort, resolve_instruction)
from vplan.tasks.reach import REACH_MARGIN_M, SHOULDER, reach_bound_m
from vplan.tasks.referents import TEMPLATES, resolve_referent
from vplan.tasks.scenes import make_basic_scene

__all__ = ["STRATA", "CohortPreparationError", "evaluate_witness", "generate_candidate",
           "prepare_cohort", "resolve_instruction", "make_basic_scene", "GenerationRejected", "check_impossibility",
           "draft_task", "generate_task", "run_witness", "scene_issues", "witness_rejection", "REACH_MARGIN_M",
           "SHOULDER", "reach_bound_m", "TEMPLATES", "resolve_referent"]
