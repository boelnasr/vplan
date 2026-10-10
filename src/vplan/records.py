"""All shared records. World frame = robot base frame (base fixed at the origin), z up, meters; quaternions xyzw."""
from __future__ import annotations

import hashlib
import math
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

Vec3 = tuple[float, float, float]
PositiveFloat = Annotated[float, Field(gt=0)]
PositiveVec3 = tuple[PositiveFloat, PositiveFloat, PositiveFloat]
Quat = tuple[float, float, float, float]
IDENTITY_QUAT: Quat = (0.0, 0.0, 0.0, 1.0)
UNIT_QUAT_TOL = 1e-5  # |norm - 1|; PyBullet's float32-normalised quaternions are well inside this
# Must equal the COLORS keys in vplan.sim.scene (an unknown colour would otherwise crash build_scene with a KeyError).
Color = Literal["red", "green", "blue", "yellow", "purple", "orange"]
# Body names build_scene registers itself; scene ids must not overwrite them in SceneHandles.bodies / names.
RESERVED_BODY_IDS = frozenset({"table", "bin_floor", "bin_wall_0", "bin_wall_1", "bin_wall_2", "bin_wall_3", "robot"})


def _finite(xs) -> bool:
    return all(math.isfinite(float(x)) for x in xs)


class Frozen(BaseModel):
    # allow_inf_nan=False: NaN/Inf would pass float fields and scene_hash would serialise NaN as null.
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


# ---- scene -----------------------------------------------------------------------------------
class Pose(Frozen):
    position: Vec3
    orientation: Quat = IDENTITY_QUAT

    @field_validator("orientation")
    @classmethod
    def _unit(cls, q: Quat) -> Quat:
        # quat_to_R / the Judge's _R do not normalise: a non-unit q would silently give a non-rotation
        n = math.sqrt(sum(float(v) * float(v) for v in q))
        if abs(n - 1.0) > UNIT_QUAT_TOL:
            raise ValueError(f"orientation must be a unit quaternion (xyzw); norm is {n:.6g}")
        return q


class SceneObject(Frozen):
    id: str
    shape: Literal["cube", "cylinder"]
    color: Color
    size: PositiveVec3  # full bounding-box extents in the object frame, m
    pose: Pose  # pose of the object's center
    mass_kg: float = Field(default=0.1, ge=0)

    @model_validator(mode="after")
    def _cylinder_diameter(self) -> "SceneObject":
        if self.shape == "cylinder" and self.size[0] != self.size[1]:
            raise ValueError("cylinder x/y dimensions must be the same diameter")
        return self


class Obstacle(Frozen):
    id: str
    half_extents: PositiveVec3
    pose: Pose


class Bin(Frozen):
    floor_center: Vec3  # top surface of the bin floor, m
    inner_size_xy: tuple[PositiveFloat, PositiveFloat]
    wall_height: PositiveFloat
    wall_thickness: PositiveFloat = 0.01


class Scene(Frozen):
    table_top_z: float
    robot_base: Pose = Pose(position=(0.0, 0.0, 0.0))
    objects: tuple[SceneObject, ...]
    obstacles: tuple[Obstacle, ...] = ()
    bin: Bin

    @model_validator(mode="after")
    def _check(self) -> "Scene":
        # robot.py kinematics are base-frame; we define world = base frame instead of transforming (CR#16)
        if self.robot_base != Pose(position=(0.0, 0.0, 0.0)):
            raise ValueError("robot_base must be the identity pose (world frame = robot base frame)")
        ids = [o.id for o in self.objects] + [o.id for o in self.obstacles]
        if len(ids) != len(set(ids)):
            raise ValueError("object and obstacle ids must be unique")
        for i in ids:
            if i in RESERVED_BODY_IDS:
                raise ValueError(f"id {i!r} is reserved for a built-in scene body")
        return self


def scene_hash(scene: Scene) -> str:
    """Snapshot version: sha256 of the canonical JSON of the scene."""
    return hashlib.sha256(scene.model_dump_json().encode("utf-8")).hexdigest()


# ---- plans (spec §7) -------------------------------------------------------------------------
Approach = Literal["top", "side_+x", "side_-x", "side_+y", "side_-y"]


class Grasp(Frozen):
    approach: Approach
    yaw_deg: float = Field(ge=-180.0, le=180.0)
    pregrasp_offset_m: float = Field(ge=0.03, le=0.25)


class Place(Frozen):
    position: Vec3  # desired resting center of the object = TCP target at release minus release height
    yaw_deg: float = Field(ge=-180.0, le=180.0)
    release_height_m: float = Field(ge=0.0, le=0.15)


class PickPlacePlan(Frozen):
    kind: Literal["pick_place"] = "pick_place"
    target_id: str
    grasp: Grasp
    transport_via: tuple[Vec3, ...] = Field(default=(), max_length=3)
    place: Place
    rationale: str = ""


class Refusal(Frozen):
    kind: Literal["refusal"] = "refusal"
    reason: Literal["target_unreachable", "destination_unreachable", "target_enclosed",
                    "no_matching_object", "other"]
    explanation: str


Plan = Annotated[Union[PickPlacePlan, Refusal], Field(discriminator="kind")]
PLAN_ADAPTER: TypeAdapter[Plan] = TypeAdapter(Plan)


class SubmitPlanArgs(Frozen):
    """Tool-call arguments. OpenAI-style tool parameters need an object root, so the union is wrapped (CR#17)."""
    plan: Plan


# ---- verification ----------------------------------------------------------------------------
CheckName = Literal["grounding", "ik_accuracy", "joint_limits", "singularity", "continuity",
                    "collision", "grasp_preconditions", "release", "visual"]


class Finding(Frozen):
    stage: str  # segment label, or "plan" for whole-plan checks
    check: CheckName
    plan_field: str
    measured: float | None
    threshold: float | None
    direction: str  # human-readable violated direction; no computed fixes (D17)
    sample_index: int | None = None
    bodies: tuple[str, ...] = ()


SegmentLabel = Literal["approach", "descend", "lift", "transport", "descend_place", "retreat"]


class Segment(Frozen):
    label: SegmentLabel
    source_field: str
    carrying: bool
    q: tuple[tuple[float, ...], ...]  # densified joint samples, rad, 7 each; q[0] = previous segment's end
    t: tuple[float, ...]  # sample times, s; t[0] = 0, strictly increasing
    ik_idx: tuple[int, ...]  # indices into q of the IK solutions (pre-densification samples)
    ik_targets: tuple[tuple[float, ...], ...]  # Cartesian TCP target per IK sample, 4x4 row-major

    @model_validator(mode="after")
    def _structure(self) -> "Segment":
        if len(self.q) < 2 or len(self.q) != len(self.t):
            raise ValueError("a segment needs >= 2 samples and one time per sample")
        if any(len(qq) != 7 or not _finite(qq) for qq in self.q):
            raise ValueError("every sample needs 7 finite joint values")
        if self.t[0] != 0.0 or not _finite(self.t) or any(b <= a for a, b in zip(self.t, self.t[1:])):
            raise ValueError("times must start at 0 and increase strictly")
        if not self.ik_idx or len(self.ik_idx) != len(self.ik_targets):
            raise ValueError("one Cartesian target per IK sample")
        if (list(self.ik_idx) != sorted(set(self.ik_idx)) or self.ik_idx[0] < 1
                or self.ik_idx[-1] != len(self.q) - 1):
            raise ValueError("ik_idx must increase, start after q[0], and end at the last sample")
        if any(len(T) != 16 or not _finite(T) for T in self.ik_targets):
            raise ValueError("ik_targets must be finite 4x4 row-major matrices")
        return self


class CandidateTrajectory(Frozen):
    q_start: tuple[float, ...]
    segments: tuple[Segment, ...]

    @model_validator(mode="after")
    def _structure(self) -> "CandidateTrajectory":
        if len(self.q_start) != 7 or not _finite(self.q_start):
            raise ValueError("q_start needs 7 finite values")
        labels = [s.label for s in self.segments]
        n_transport = len(labels) - 5
        expected = ["approach", "descend", "lift"] + ["transport"] * n_transport + ["descend_place", "retreat"]
        if not 1 <= n_transport <= 4 or labels != expected:
            raise ValueError(f"segment order must be approach, descend, lift, transport x1-4, descend_place, retreat; got {labels}")
        prev = self.q_start
        for s in self.segments:
            if s.q[0] != prev:
                raise ValueError("segment boundaries must be continuous")
            prev = s.q[-1]
        return self



# ---- visual gate (spec §11) ------------------------------------------------------------------
class VisualVerdict(Frozen):
    """One Cosmos request. A learned visual check, never verification; unparseable replies become `uncertain`."""
    verdict: Literal["pass", "fail", "uncertain"]
    reason: str
    latency_s: float = Field(ge=0)
    raw: str

# ---- execution and judging ---------------------------------------------------------------------
class ContactSummary(Frozen):
    """Contacts aggregated per (bodies, links, stage): per-step rows would be megabytes per attempt."""
    body_a: str
    link_a: str  # robot link name, or "base" for single-body objects
    body_b: str
    link_b: str
    stage: str
    first_step: int
    count: int
    max_normal_force_n: float
    max_penetration_m: float  # positive = penetration depth


class ExecEvent(Frozen):
    step: int
    kind: Literal["attach", "attach_rejected", "release"]
    values: dict[str, float]  # capture-volume measurements at the decision
    rel_pos: Vec3 | None = None  # object pose in the hand COM frame (attach only)
    rel_orn: Quat | None = None
    reason: str = ""  # capture_check rejection reason (attach_rejected only)


class ExecutionRecord(Frozen):
    scene_hash: str
    approval: str
    dt_s: float
    n_physics_steps: int
    events: tuple[ExecEvent, ...]
    min_joint_margin_rad: tuple[float, ...]  # per joint over all physics steps
    max_joint_violation_rad: float  # max amount any joint exceeded its URDF limit (0 if never)
    sigma_min_steps: tuple[float, ...]  # Judge-side sigma_min(J_s) at every physics step
    contacts: tuple[ContactSummary, ...]
    tracking_error_rad: dict[str, float]
    final_poses: dict[str, Pose]
    rest_lin_hist: dict[str, tuple[float, ...]]  # |v| per physics step over the final dwell, per object
    rest_ang_hist: dict[str, tuple[float, ...]]  # |w| per physics step over the final dwell, per object
    rest_pose_hist: dict[str, tuple[Pose, ...]] = Field(default_factory=dict)  # aligned with dwell velocities, 240 Hz
    q_replay: tuple[tuple[float, ...], ...]  # 60 Hz arm joints
    finger_replay: tuple[tuple[float, float], ...]  # 60 Hz finger joints
    object_replay: dict[str, tuple[Pose, ...]]  # 60 Hz object poses: replay is playback, not re-simulation

    @property
    def attached(self) -> bool:
        return any(e.kind == "attach" for e in self.events)


class JudgeResult(Frozen):
    success: bool
    predicates: dict[str, bool]  # S1, S2, V1..V4 (or R for refusals)
    violations: tuple[str, ...]
    details: dict[str, str]


# ---- feasible development pilot tasks (oracle records never go to planners) -------------------
# The first three are the pilot generator's; the dense_* strata come from the hard (W2.2) generator.
TaskStratum = Literal["clear", "distractor", "obstacle", "dense_distractors", "dense_obstacles", "dense_both"]


class RobotFacts(Frozen):
    base_pose: Pose
    reach_bound_m: float = Field(gt=0)
    gripper_opening_m: float = Field(gt=0)
    table_height_m: float


class TaskSpec(Frozen):
    instruction: str
    scene: Scene
    robot_facts: RobotFacts


class TaskMetadata(Frozen):
    task_id: str
    seed: int = Field(ge=0)
    stratum: TaskStratum
    split: Literal["dev"] = "dev"


class TaskCandidate(Frozen):
    """Private generation envelope; only .task may cross the planner boundary."""
    metadata: TaskMetadata
    task: TaskSpec
    target_id: str


class WitnessEvaluation(Frozen):
    plan: PickPlacePlan
    outcome: str
    findings: tuple[Finding, ...] = ()
    result: JudgeResult | None = None


ImpossibleType = Literal[1, 2, 3, 4]  # spec §6 numbering
# The correct refusal per impossibility type; index = type - 1.
IMPOSSIBLE_REFUSALS: tuple[str, ...] = ("target_unreachable", "destination_unreachable", "target_enclosed",
                                        "no_matching_object")


class ImpossibilityProof(Frozen):
    """Construction type and proof parameters; the proof is rechecked from the scene (tasks.check_impossibility)."""
    type: ImpossibleType
    refusal_reason: Literal["target_unreachable", "destination_unreachable", "target_enclosed", "no_matching_object"]
    params: dict[str, float] = Field(default_factory=dict)  # e.g. measured distance, reach bound, margin
    enclosure: tuple[Obstacle, ...] = ()  # exact type-3 geometry (also present in the scene)

    @model_validator(mode="after")
    def _consistent(self) -> "ImpossibilityProof":
        if IMPOSSIBLE_REFUSALS[self.type - 1] != self.refusal_reason:
            raise ValueError(f"impossibility type {self.type} requires refusal {IMPOSSIBLE_REFUSALS[self.type - 1]!r}")
        if bool(self.enclosure) != (self.type == 3):
            raise ValueError("enclosure geometry belongs to type 3 only")
        return self


class TaskOracle(Frozen):
    target_id: str | None  # None only for an absent referent (type 4)
    feasibility: Literal["feasible", "impossible"] = "feasible"
    reference_plan: PickPlacePlan | None = None
    reference_result: JudgeResult | None = None
    proof: ImpossibilityProof | None = None

    @model_validator(mode="after")
    def _label(self) -> "TaskOracle":
        if self.feasibility == "feasible":
            if (self.target_id is None or self.proof is not None or self.reference_plan is None
                    or self.reference_result is None or self.reference_plan.target_id != self.target_id
                    or not self.reference_result.success):
                raise ValueError("a feasible oracle requires a successful reference for its target")
        elif (self.proof is None or self.reference_plan is not None or self.reference_result is not None
              or (self.target_id is None) != (self.proof.type == 4)):
            raise ValueError("an impossible oracle carries a proof, no reference, and a target unless it is absent")
        return self


class GeneratedTask(Frozen):
    metadata: TaskMetadata
    task: TaskSpec
    oracle: TaskOracle


class CandidateExclusion(Frozen):
    metadata: TaskMetadata
    reason: str
    witness: WitnessEvaluation | None = None


class PreparedCohort(Frozen):
    seed: int = Field(ge=0)
    requested: int = Field(gt=0)
    max_candidates: int = Field(gt=0)
    attempted: int = Field(ge=0)
    config_hashes: dict[str, str]
    tasks: tuple[GeneratedTask, ...]
    exclusions: tuple[CandidateExclusion, ...]

    @model_validator(mode="after")
    def _accounted(self) -> "PreparedCohort":
        if self.attempted != len(self.tasks) + len(self.exclusions):
            raise ValueError("every candidate must be accepted or excluded")
        if self.attempted > self.max_candidates or len(self.tasks) > self.requested:
            raise ValueError("cohort exceeds its generation bounds")
        return self


# ---- bounded paired feedback experiment ------------------------------------------------------
PlannerProtocol = Literal["forced", "required", "content"]
AttemptOutcome = Literal["executed", "refused", "budget_exhausted", "malformed_final",
                         "service_failure", "stale_approval", "sim_error"]


class PlannerFeedback(Frozen):
    previous_plan: Plan | None = None
    findings: tuple[Finding, ...] = Field(default=(), max_length=5)
    schema_errors: tuple[dict[str, Any], ...] = ()
    summaries: tuple[str, ...] = ()


class PlannerCall(Frozen):
    outcome: Literal["proposed", "malformed", "protocol_error", "service_failure"]
    plan: Plan | None = None
    provider: str
    model: str
    response_model: str | None = None
    protocol: PlannerProtocol
    next_protocol: PlannerProtocol
    temperature: float
    latency_s: float = Field(ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    raw_request: dict[str, Any]
    raw_response: dict[str, Any] | None = None
    errors: tuple[dict[str, Any], ...] = ()
    error_type: str | None = None
    status_code: int | None = None


class EvaluationResult(Frozen):
    """Serializable attempt result; Judge/oracle evidence never becomes planner feedback."""
    outcome: Literal["blocked", "executed", "refused", "stale_approval", "sim_error"]
    findings: tuple[Finding, ...] = ()
    judge: JudgeResult | None = None
    record: ExecutionRecord | None = None
    error_type: str | None = None
    error_message: str | None = None
    error_traceback: str | None = None


class RoundRecord(Frozen):
    call_id: str
    round_index: int = Field(ge=0, le=3)
    call: PlannerCall
    evaluation: EvaluationResult | None = None
    repeat_plan: bool = False
    observable_failures: tuple[str, ...] = ()


class ArmResult(Frozen):
    arm: Literal["repair", "resample"]
    outcome: AttemptOutcome
    rounds: tuple[RoundRecord, ...] = Field(min_length=1, max_length=4)
    call_count: int = Field(ge=1, le=4)
    total_tokens: int | None
    known_total_tokens: int = Field(ge=0)
    missing_usage_calls: int = Field(ge=0)
    latency_s: float = Field(ge=0)


class PairedResult(Frozen):
    task_id: str
    initial: RoundRecord
    repair: ArmResult
    resample: ArmResult
    unique_call_count: int = Field(ge=1, le=7)
    unique_total_tokens: int | None
    unique_known_total_tokens: int = Field(ge=0)
    unique_missing_usage_calls: int = Field(ge=0)
    unique_latency_s: float = Field(ge=0)
    config_hashes: dict[str, str]
    # Execution order of the two arms after the shared initial call; default matches pre-alternation data.
    arm_order: tuple[Literal["repair", "resample"], Literal["repair", "resample"]] = ("repair", "resample")


AttemptMode = Literal["repair", "resample", "unverified"]


class AttemptRecord(Frozen):
    """One run_task attempt (spec §10): every round's call and evaluation, and exactly one outcome."""
    attempt_id: str
    mode: AttemptMode
    k: int = Field(ge=0, le=3)
    outcome: AttemptOutcome
    rounds: tuple[RoundRecord, ...] = Field(min_length=1, max_length=4)
    call_count: int = Field(ge=1, le=4)
    judge: JudgeResult | None = None  # the final round's Judge result (executed or refused); None otherwise
    total_tokens: int | None
    known_total_tokens: int = Field(ge=0)
    missing_usage_calls: int = Field(ge=0)
    latency_s: float = Field(ge=0)
    config_hashes: dict[str, str]

    @model_validator(mode="after")
    def _budget(self) -> "AttemptRecord":
        if self.call_count != len(self.rounds) or self.call_count > self.k + 1:
            raise ValueError("one round per planner call, at most k + 1 calls")
        return self

    @property
    def success(self) -> bool:
        """Every outcome stays in the denominator; only an executed or refused attempt the Judge accepts counts."""
        return self.outcome in ("executed", "refused") and self.judge is not None and self.judge.success


# ---- R9 supervised worker (W2.8) -------------------------------------------------------------
class WorkerResult(Frozen):
    """What one worker try writes to its result file; the parent never parses worker stdout."""
    attempt_id: str
    job: str
    outcome: AttemptOutcome
    status_code: int | None = None
    retry_after_s: float | None = Field(default=None, ge=0)
    payload: dict[str, Any] = {}
    error_type: str | None = None
    error_message: str | None = None
    error_traceback: str | None = None


class SupervisedAttempt(Frozen):
    """One results.jsonl line, written only by the parent. Crashes and hangs are sim_error (spec §16)."""
    attempt_id: str
    job: str
    outcome: AttemptOutcome
    tries: int = Field(ge=1)
    wall_s: float = Field(ge=0)
    exit_code: int | None = None
    status_code: int | None = None
    payload: dict[str, Any] = {}
    error_type: str | None = None
    error_message: str | None = None
    error_traceback: str | None = None
