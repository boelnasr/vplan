# Decision Log

Nebius x NVIDIA Global AI Hackathon, Physical AI track.
Deadline: 2026-10-30, 10:00 AM PDT.
Status: brainstorming. The design is being approved section by section. No implementation code exists yet.

Last updated: 2026-09-30.

---

## D1. Track and concept

- **Track:** Physical AI, with no physical hardware. The rules allow a video that shows "the key application modules in action" when there is no hardware.
- **Domain:** robot-arm manipulation (option A), plus a batch-evaluation slice on Nebius Serverless Jobs (option D).
- **Concept:** "LLMs plan, physics verifies."
  - A Nemotron model turns a plain-language instruction into a structured plan.
  - ManipulaPy and PyBullet provide the deterministic verification and execution.
  - No plan executes on the normal path unless it passes verification.
  - Failed plans go back to the planner with structured feedback.
- **Excluded:** the earlier IKRL project (RL inverse kinematics) is out, because it has conceptual flaws. We use ManipulaPy's classical solvers.
- **Separation:** this is a personal project, separate from employer work.

## D2. Repository and license

- **Repo:** a new hackathon repository that pins a tagged ManipulaPy release from PyPI.
- **License:** the hackathon repo is AGPL-3.0 for now, so no relicensing is needed.
- **ManipulaPy license:**
  - ManipulaPy is AGPL-3.0-or-later.
  - Relicensing or dual-licensing ManipulaPy will be decided separately, after the hackathon.
  - Almost all ManipulaPy commits are the owner's. One outside commit, by J. Carpinelli on 2025-08-16, touches only docs, the benchmark and the paper, not `ManipulaPy/` code. This is not legal advice.
- **Revisit trigger: resolved (2026-10-03).** The rules say "an open source license file (such as Apache 2.0, MIT, or MPL 2.0)". Those are examples, so AGPL-3.0 qualifies. Verified by the Codex review and re-checked against the rules page.
- **Upstream fixes:**
  - These go to ManipulaPy as a tagged release: explicit `tip_link` / end-effector selection, the `near_singularity_detection` threshold, and gripper joints being counted as arm DOF.
  - They are listed in the README under "updated during the submission period".

## D3. Demo scope

- **Core:** one task family done well, tabletop pick-and-place ("put the red cube in the bin"). Object poses are randomized, with 1–3 obstacles.
- **Stretch:** a few task types (for example stacking or clearing the table), only if time allows after week 3.
- **Week 1 milestone:** reach with obstacle avoidance and no grasping. This is a fallback that can always be submitted.
- **Collision checks:** robot-vs-scene checks wrap PyBullet's own contact and closest-point queries. We do not build a collision engine.
- **Grasp:** a kinematic attachment constraint at the grasp pose, documented as a simulation simplification.
- **Success check first:** the automatic success check is defined before anything else, because the batch evaluation and the headline depend on it. It requires:
  - the cube is inside the bin volume;
  - there are no collisions along the trajectory;
  - joint limits hold;
  - no configuration is near-singular.

## D4. Robot

- **Arm:** Franka Panda, 7-DOF with a parallel gripper.
- **Why:**
  - It's standard in manipulation research, and `pybullet_data` ships a Panda with fingers.
  - The 7-DOF redundancy makes the IK and singularity checks meaningful.
  - It aligns with NVIDIA's physical-AI models (Cosmos 3 Policy-DROID checkpoints use Franka).
- **Dependency:** the upstream DOF-count fix in ManipulaPy. Panda currently loads as 8-DOF because the finger joint is counted.

## D5. Cosmos 3 role

- **Role:** a core component, self-hosted on a Nebius endpoint. It is a *visual consistency gate before execution*.
- **Order of checks:** Cosmos runs after the deterministic checks pass.
- **Stage questions:**

  | Stage | Cosmos question |
  |---|---|
  | Pre-grasp | Is the gripper approaching the object named in the instruction? |
  | Grasp pose | Does the object appear between the fingers, with a plausible closing direction? |
  | Pre-release | Is the intended object positioned over the correct destination? |

- **Rendering:** candidate poses are rendered in a separate preview scene, from two fixed viewpoints.
- **Output:** an application-defined `pass` / `fail` / `uncertain` result, with the stage, the reason and the supporting frame.
- **Handling results:**
  - `fail` enters bounded repair.
  - `uncertain` blocks execution. The system then renders another view or ends the attempt.
  - Every repaired plan repeats all checks.
- **Rules:**
  - Cosmos approval never overrides a geometric rejection.
  - It is described as a *learned visual check*, not as deterministic physics verification.
  - Preview frames show intended poses. They are not proof that a grasp will succeed.
- **Numeric checks:** gripper alignment and bin clearance also get numeric checks wherever simulator state allows. "Awkward" is not a rejection criterion, and every check has a defined failure condition.
- **Precondition:** an early feasibility test of serving Cosmos3 (Edge 4B / Nano 16B) on Nebius. It costs money, so we ask before running it.
  - **Fallback:** if it fails, Cosmos becomes an after-execution visual judge.
  - **Never blocking:** the system must run with geometric checks only if the endpoint is unavailable.
- **Evaluation (see D8, item 4):** two stages, with Cosmos in shadow mode.

## D6. Where things run

- **Package:** one Python package, used in three modes.
- **Interactive demo (local):**
  - A small web UI shows the instruction input, the PyBullet view, and a plan and verification timeline: plan, checks, Cosmos frames and verdicts, repair rounds.
  - It calls the LLM provider and the Cosmos endpoint.
- **Batch evaluation (Nebius Serverless Jobs):**
  - The same package runs in a CPU container, with PyBullet in DIRECT mode.
  - Seeded tasks go in. Per-attempt JSONL records, a summary and key frames go to Object Storage.
  - Jobs are started by a script around `nebius ai job create`, never automatically.
- **Report (local):** one command pulls the results and builds a static HTML report.
- **Local batches:** the same batch runs locally at small N during development.

## D7. LLM providers

- **Client:** a provider-agnostic, OpenAI-compatible LLM client configured by environment variables (base URL, API key, model). The repo has a `.env.example`, and no keys are committed.
- **Development:** on OpenRouter, for prompt tuning, the repair loop, tests and small-N batches.
- **Headline:** the evaluation and the demo video run on **Nebius Token Factory**. The final tuning pass also happens there.
- **Labeling:** every attempt logs its provider and model. The final report uses only Token Factory runs for headline numbers.
- **Model:** Nemotron 3.5 Lightning, the same model on both providers:
  - OpenRouter: `nvidia/nemotron-3.5-lightning`.
  - Token Factory: `nvidia/Nemotron-3_5-Lightning`.
  - Fallbacks: Nemotron 3 Super 120B, then Ultra 550B.
- **Structured output:** forced tool calling (`submit_plan`), with Pydantic validation and retry, on **both** providers.
  - Reason: Token Factory does not tag any Nemotron for JSON mode, so OpenRouter's `structured_outputs` would not transfer.
- **Model churn:** Token Factory removes models without auto-rerouting. Pin the model ID and re-check it before the final runs.
- **Budget:** allow 1–2 days to re-tune when switching providers.

## D8. Evaluation and headline claim

**Headline form (fill in with measured values; don't promise "reliable" in advance):**

> "Verified execution and bounded repair increased Nemotron's task success from X% to Y% across N seeded manipulation tasks."

**Metric definitions (fixed in the README before any final run):**
- **First-try success:** the initial plan passes verification *and* achieves the task postconditions in simulation.
- **Success with repair:** cumulative success within the repair budget, over the same tasks and the same denominator. Exhausted attempts and service failures stay in the denominator.
- **Execution safety:**
  - Judged on the *executed* physics run: contacts recorded while running, joint limits observed while running, and the final object poses.
  - It is independent of the pre-execution gate, which checks the *planned* trajectory. The Judge shares no check code with the Verifier.
  - Violations are reported separately from task success. A dropped cube is a task failure, not automatically a safety violation.
  - "Zero observed violations in M executions" is claimed only if measured.
- **Blocked plans:** rejected attempts, counted by reason. Physical-constraint failures are kept separate from malformed output and from solver non-convergence.
- **Refusals:**
  - Impossible tasks are impossible *by construction*. Feasible tasks are confirmed independently by a reference planner with a large budget.
  - We report both impossible tasks that were wrongly attempted and feasible tasks that were wrongly refused.
  - An IK timeout is not proof that a target is unreachable.
  - Feasible-task success and impossible-task refusal are separate results, so the task mix can't inflate the headline.
- **Failure-cause tags:** LLM plan / motion-layer limitation / verifier or simulator issue, reported separately.
- **Statistics:** Wilson 95% confidence intervals, with N and seeds reported and breakdowns by failure category.
- **Held-out seeds:** a seed set that is not used during tuning.

**Configurations, on the same held-out seeds:**

| # | Configuration | Planner | Provider | Role |
|---|---|---|---|---|
| 1 | Unverified ablation (evaluation-only, simulation only) | Nemotron | Token Factory | ablation |
| 2 | Verify only | Nemotron | Token Factory | headline "before" |
| 3 | Verify + repair | Nemotron | Token Factory | headline "after" |
| 4 | Verify + repair + Cosmos (shadow, see item 4 below) | Nemotron | Token Factory + Nebius endpoint | secondary |
| 5 | Heuristic baseline (no LLM, same skill-plan parameters) | heuristic | none | baseline, reported honestly |
| 6 | Frontier reference (no verifier) | frontier model | OpenRouter | reference only, excluded from headline, reported whatever the result. Optional per D24. |
| 7 | Equal-budget resampling (verify, up to 4 independent samples, no feedback) | Nemotron | Token Factory | added by D23. Credits the feedback over plain retrying. |

Note (D23): configuration 2 is round 1 of configuration 3 (paired). The headline is "repair over verified first attempts".

1. **Unverified plans:** these never execute on the demo path. Configuration 1 exists only in the evaluation mode, in simulation.
2. **Main comparison:** configuration 2 vs 3, on the same model.
3. **Cosmos comparison:** incremental success, extra failures caught, false rejections, added latency and cost.
4. **Two-stage Cosmos evaluation:**
   - The CPU Jobs batch saves key frames per attempt.
   - A separate pass runs Cosmos in **shadow mode** on those same attempts.
   - Shadow mode measures what Cosmos *would have* caught or falsely rejected. It does **not** measure extra success from Cosmos-driven repair, because its verdicts are not fed back.
   - The README must state this, so the configuration table doesn't imply a single pipeline.
   - Cosmos runs as a live gate only in the interactive demo.

## D9. What the planner outputs (approach A)

- **Plan type:** a typed skill plan with geometric parameters, produced by Nemotron:
  - skills: `move_to`, `grasp`, `place`, `refuse`;
  - parameters: target object ID, grasp yaw and approach direction, placement position inside the bin, optional Cartesian via-points around obstacles, and a refusal reason.
- **Prompt input:** read-only scene facts (object poses and sizes, bin bounds, obstacle boxes). The planner gets no ability to probe the verifier.
- **Motion layer:** deliberately simple. It makes straight Cartesian segments between plan waypoints and does no automatic obstacle avoidance, so the planner's decisions matter.
- **Feedback:** repair feedback maps each failure to a specific plan field.
- **Rejected alternatives:**
  - B (raw poses only): LLMs are poor at raw 6-DoF numbers.
  - C (agent loop with verifier tools): it blurs plan and verification, and first-try success becomes hard to define.
- **Heuristic baseline:** emits the same skill-plan parameters without an LLM.
- **Varied instructions:** different referring expressions, and same-shape distractors in other colors, so resolving language against the scene is part of the task. `refuse` is kept.

## D11. CPU by default, CUDA optional

- **Default:** the planner, verifier, executor and batch path run on the CPU. CUDA is an optional `--gpu` flag and is never required.
- **Why:**
  - The batch workload is mostly CPU work: PyBullet physics and queries, remote LLM calls, and small per-task IK solves dominated by overhead.
  - GPU Jobs cost more and need a CUDA stack that matches the driver and sm_120.
  - Judges may not have an NVIDIA GPU. The small-N local evaluation and replay must run anywhere.
  - Nemotron already meets the NVIDIA-model requirement.
- **Reference feasibility planner:**
  - It runs **locally, once per seed set**.
  - The seeds and feasibility labels are committed, and the Jobs batch only reads them.
  - Default: CPU multiprocessing.
- **Checked in code (ManipulaPy v1.4.1):**
  - There is **no GPU batch IK.** All IK solvers are single-target NumPy code on the CPU.
  - The GPU code is trajectory generation only: `cuda_kernels/trajectory_kernels.py` `batch_trajectory_kernel` and `planning/trajectory.py` `batch_joint_trajectory`.
  - So GPU speedup for the reference planner would need new batched IK code, for example via the torch/jax backends. Only worth it if a spike shows a real speedup.
- **CUDA stack:** Numba CUDA (numba is a core dependency, `>=0.60,<0.67`), optional CuPy 13.x (`cupy-cuda12x`), and optional torch/jax backends.
  - **Tested 2026-10-01** on the RTX 5070 Ti (driver 591.86, compute capability 12.0) with `ManipulaPy[all]==1.4.1`, Python 3.11:

    | Stack | Result |
    |---|---|
    | torch 2.14.1 (CUDA 13.0) | Works. sm_120 is in the arch list. |
    | CuPy 13.6 | **Fails.** `libnvrtc.so.12` is missing. |
    | Numba CUDA | **Fails.** `libNVVM` is missing. |
    | JAX | CPU only (`jax[cpu]` is installed by design). |

  - ManipulaPy's own CUDA kernels therefore need a CUDA 12.8+ toolkit, or the NVIDIA runtime-compiler wheels, before `--gpu` can work locally.
  - Don't build large venvs in `/tmp`. It is a 7.7 GB RAM-backed tmpfs.
- **Day-one test:**
  - ManipulaPy's FK, IK and trajectory path must run in a clean container with no CUDA installed.
  - Supporting evidence so far: the CI matrix runs with `NUMBA_DISABLE_CUDA=1`, and the planner falls back to CPU automatically.

## D10. Architecture, section 1: components and data flow (approved with changes)

**Two paths.**

- **Demo path:**
  `TaskGenerator → Planner → MotionLayer → Verifier → VisualGate (Cosmos, live) → Executor → Judge`.
  - A Verifier failure or a Cosmos `fail` returns to RepairLoop.
  - Every revised plan repeats motion generation and verification.
- **Evaluation path (Jobs):**
  `TaskGenerator → Planner → MotionLayer → Verifier → Executor → Judge → AttemptRecord`, with key frames saved.
  - Cosmos is not in this path and cannot influence these executions.
- **Shadow branch (local, afterwards):** saved frames and attempt records → Cosmos → shadow annotations → Report.

**Modules.**

| Module | Output | Notes |
|---|---|---|
| TaskGenerator | `TaskSpec` (planner-visible) + `TaskOracle` (private) | See the data-visibility rules below. |
| Planner | `SkillPlan` or refusal | Nemotron and heuristic are core. The frontier adapter is optional. |
| MotionLayer | `CandidateTrajectory` | ManipulaPy IK with explicit `tip_link`, seeded, straight Cartesian segments. |
| Verifier | `Finding`s | Checks the planned trajectory in the preview client. |
| RepairLoop | revised plan | Budget of k rounds. Re-runs all checks. |
| VisualGate | Cosmos verdict | Live gate in the demo only. Shadow annotator in evaluation. |
| Executor | `ExecutionRecord` | Position control with finite gains, in the physics client. Does not replay joint states. |
| Judge | outcome predicates | Implements a separate pure-data predicate spec. No Verifier code. |
| Evaluator | `AttemptRecord` (JSONL) + key frames | Runs configurations over seed sets. |
| Report / UI | static HTML / interactive demo | |

**Data visibility.**
- **`TaskSpec`, visible to planners:**
  - the instruction;
  - observable scene attributes: object IDs, colors, shapes, sizes and poses, bin bounds, obstacle boxes.
- **`TaskOracle`, private (Judge and Evaluator only):**
  - the intended target and the feasibility label;
  - the reference solution, and the construction reason for impossible tasks.
- Exposing the intended target to planners would bypass the language-grounding challenge.

**Simulation isolation.**
- The preview and physics clients start from the same **immutable scene snapshot** (versioned). Each has independent mutable state and an explicit `physicsClientId`.
- Execution uses the **exact verified trajectory** against the **matching scene version**. Any changed state invalidates the approval.

**Grasping claim.**
- With kinematic attachment, the system evaluates **attachment-assisted pick-and-place**: planning, grounding and motion constraints. It does **not** establish contact-based grasp reliability.
- Attachment requires **measurable preconditions**, so it cannot rescue a missed approach. Thresholds are set in section 7, covering:
  - the fingertip-to-object distance;
  - the object being between the fingers;
  - the approach axis relative to the object.
- The preview collision checks **model the carried object** after the grasp.

**Findings vs. attribution.**
- Each `Finding` records the stage, the check, the affected plan field and the evidence (values, thresholds, frame references).
- **Attribution rule, defined up front:**
  - Run the **oracle plan** for the same task through the same motion layer.
  - If the oracle plan also fails → `motion-layer limitation`.
  - Otherwise, attribute by evidence to `LLM plan` or `verifier/sim issue`.
  - Use `unknown` or `mixed` when the evidence doesn't settle it. Solver failure alone does not establish the cause.
- There is also a **`malformed output`** failure tag (the tool call or JSON fails schema validation). It counts as a first-try failure.

**Precise claims.**
- A reference trajectory establishes feasibility **within the declared task and execution model**. A motion-planner failure does not establish impossibility.
- The README states that "independent" means **independent check logic on the same engine**. The Judge and the Verifier share PyBullet and its limitations.

**Repair input.**
- In repair rounds the planner sees:
  - the same `TaskSpec`;
  - its previous plan;
  - a structured `Finding` list: stage, check, plan field, evidence and violated direction. Per D17, there are no computed fixes or suggested ranges.
- It never sees `TaskOracle` data or Cosmos images. Cosmos verdicts are passed as text findings.
- The exact format is fixed in section 5.

**Reproducibility.**
- Seeds control scene generation and solver randomness. IK determinism under `np.random.seed` is confirmed in D12.
- Remote inference is **not** guaranteed reproducible, so every raw model request and response is logged.
- Library, image and model versions are logged per run.
- There is **no claim of bitwise reproducibility** across machines.

**Choices still unsettled.**
- ~~Pydantic is a proposed dependency awaiting approval.~~ Approved in D16, together with `openai`.
- The frontier adapter is optional.
- Forced tool calling is still to be verified live against the selected Nemotron endpoint (see D13).

**Remaining design sections:**
2. Task generator and ground truth
3. Plan schema and planners
4. Motion layer and verifier
5. Repair loop and failure tagging
6. Cosmos gate
7. Executor and judge
8. Evaluation, report and UI (including error handling and testing)

## D15. Design section 2: task generator and ground truth (approved 2026-10-03)

**Scene (seeded)**
- A table with a Panda at a fixed base, and one open-top bin whose pose varies within a region.
- One target object, plus 0–3 distractors: same shape in another color, or same color in another shape.
- 0–3 obstacles: boxes and walls, some between the target and the bin.
- Each scene's content hash is its snapshot version.

**Instructions**
- Templates across referring-expression types: color, shape, spatial (nearest the bin, leftmost) and relational (next to the green one).
- A resolver enforces **exactly one referent**. **Ambiguous instructions are excluded from the core.** They are a possible stretch category.

**Labels (in `TaskOracle`)**
- **Impossible by construction**, proven by geometric bounds or scene contents, never by planner failure:
  1. the target is outside the reach sphere (≈0.855 m plus a margin);
  2. the bin is outside the reach sphere;
  3. the target is fully enclosed (walls and a lid);
  4. the referent doesn't exist.
- **Feasible:**
  - Exhaustive reference search: grasp yaw × approach × placement grid × via-points × IK seeds.
  - Each candidate goes through the same MotionLayer, then execution plus the Judge.
  - The first success is stored as the reference solution.
  - It runs locally, once per seed set, with CPU multiprocessing.
- **Undetermined:** excluded from the evaluation set, counted in the report, never labeled impossible.

**Seed sets**
- `dev` (tuning) and `test` (held out, final runs only).
- `TaskSpec` and `TaskOracle` are stored in separate files.
- Proposed `test` mix: 300 feasible (stratified by distractors × obstacles) and 100 impossible (25 per type). The Wilson 95% interval is about ±5 points near 70%.
- Token cost: a few dollars at Lightning prices.

## D16. Design section 3: plan schema and planners (approved 2026-10-03)

**Fixed-structure plan for the core.** It generalizes to a step list only for the stretch task types. This specializes D9's "skill sequence".

**Plan types**
- **`PickPlacePlan`:**
  - `target_id`;
  - `grasp {approach ∈ {top, side_±x, side_±y}, yaw_deg ∈ [-180, 180], pregrasp_offset_m ∈ [0.03, 0.25]}`;
  - `transport_via`: 0–3 points `[x, y, z]`;
  - `place {position [x, y, z], yaw_deg, release_height_m ∈ [0, 0.15]}`;
  - `rationale`: logged only.
- **`Refusal`:** `reason ∈ {target_unreachable, destination_unreachable, target_enclosed, no_matching_object, other}`, plus `explanation`.
- Distinguished by a `kind` discriminator.
- **Conventions:** world frame, z up, meters. Yaw is in degrees and is converted internally.

**Validation**
- **Schema** (types, enums, ranges) → a failure is `malformed output`. The planner gets one retry with the validation errors, and the retry **uses up one repair round**. First-try success requires a valid first response.
- **Scene-dependent checks** (the target exists, the placement is over the bin, via-points are sensible) → Verifier `Finding`s, tagged as LLM-plan errors.

**LLM call**
- A single `submit_plan` tool whose parameters are the generated JSON Schema, with a forced `tool_choice`. Fallback: JSON in the message content (D13).
- **Prompt:** system prompt with conventions and robot facts (reach ≈0.855 m, gripper opening 0.08 m, table height), scene facts as a compact JSON table, and the instruction.
- **Logged:** low temperature, the reasoning on/off setting (decided after a logged test), and all raw requests and responses.

**Planners**, behind one interface `plan(task_spec, history) → PlannerResult`:
- **NemotronPlanner (core).**
- **FrontierPlanner (optional, reference run only).**
- **HeuristicPlanner (core baseline, no LLM):**
  - A rule-based parser for the generator's templates.
  - A top-down grasp with yaw aligned to the nearest cube face, a vertical via-point above the tallest obstacle, and placement at the bin center.
  - Refuses when the target is outside the reach sphere or nothing matches.
  - **Reported as template-aware:** its grounding is an upper bound that an LLM facing open language wouldn't get.

**Dependencies approved (2026-10-03, pin exact versions)**
- `pydantic` (v2): schemas, the tool JSON Schema, validation errors for repair, JSONL records.
- `openai` (Python SDK): a generic OpenAI-compatible client for OpenRouter and Token Factory (`base_url` / `api_key` / `model` from the environment). Only `chat.completions` with `tools` is used.
- **Not yet installed** in the Windows Python 3.10 environment.

## D17. Design section 4: motion layer and verifier (approved 2026-10-03)

**One URDF**
- `pybullet_data`'s `franka_panda/panda.urdf` is used by both ManipulaPy and PyBullet, with the TCP at `panda_grasptarget`.
- Probe: FK agrees with PyBullet `getLinkState` to 3e-8 m over 20 random configurations. This becomes a startup test.
- ManipulaPy still reports 8 DOF (the finger column). Restrict the model to the 7 arm joints in our code until the upstream fix.

**Motion layer** (deliberately simple)
- **Key poses:** pregrasp, grasp, lift (+10 cm), `transport_via`, preplace (+5 cm clearance), release, retreat.
- **Segments:** straight Cartesian segments (linear position, slerp orientation).
  - Sample spacing keeps any robot point's motion between samples under half the collision margin.
  - IK at every sample, warm-started from the previous sample. Multi-start for the first pose, seeded from the task seed.
  - Segments are labeled and flagged when carrying the object.
- **No redundancy resolution and no automatic obstacle avoidance.**
- **Timing:** quintic time scaling, with duration from the velocity limits × a safety factor.

**Verifier checks** (kinematic, preview client; each `Finding` maps to a plan field):
1. **Grounding:** `target_id` exists, and `place.position` is over the bin interior (margin = object half-size + 1 cm).
2. **IK accuracy:** FK residual ≤ 2 mm and ≤ 1°, recomputed with FK.
3. **Joint limits:** at least 2° of margin, and velocity limits respected after timing.
4. **Singularity:** σ_min(J) ≥ a threshold calibrated on `dev`, J in SI units, threshold fixed in the spec.
5. **Continuity:** no joint jumps more than 0.2 rad between samples.
6. **Collision:** `getClosestPoints` ≥ 1 cm between the robot and the scene, and between non-adjacent robot links.
   - The carried object is attached after the grasp and checked too.
   - Allowed contacts are listed explicitly.
7. **Grasp preconditions:**
   - the object center is in the closing band;
   - the object width is ≤ 0.08 m − 2×margin;
   - the approach axis is within 10°.
8. **Release:** the object's footprint is inside the bin, and the release height is in range.

**Thresholds** live in one versioned config, separate from the Judge's predicate spec.

**Findings carry evidence and direction only. No computed fixes (option a).**
- Each Finding has: measured value, threshold, sample index, the body or link pair, and the violated direction.
- The verifier never proposes solution values, such as reachable yaw ranges, so that the repair success stays attributable to Nemotron.

## D18. Design section 5: repair loop and failure tagging (approved 2026-10-03)

**Budget and outcomes**
- **k = 3 repair rounds** (at most 4 planner calls). A malformed-output retry uses up a round.
- **Outcomes** (exactly one per attempt; all stay in the denominator):
  - `executed`, which then goes to the Judge;
  - `refused`;
  - `budget_exhausted`;
  - `malformed_final`;
  - `service_failure`: after the SDK's retries, or a 60 s call timeout;
  - `visual_uncertain`, demo only.
- A refusal is allowed in any round. On a feasible task it counts as a false refusal.

**Repair prompt (stateless, rebuilt each round)**
- Contents: the same `TaskSpec` and system prompt, the latest plan, that round's Findings, and a one-line summary per earlier round.
- **Findings:** ordered along the trajectory, at most 5, at most one per check type.
- **Order of checks:** segments are checked in order, and checking stops after an IK or continuity failure. All other checks run across the whole trajectory.
- An identical repeated plan still uses up a round and is logged as `repeat_plan`.

**Cosmos in the demo**
- `fail` → a text Finding (stage `visual`) that goes through the same repair path.
- `uncertain` → one re-render from a third viewpoint. If it's still uncertain → `visual_uncertain`.

**Failure attribution (refines the D10 rule)**
- The stored reference solution always passes the motion layer, because feasibility labels come from it. So attribution uses a **canonical oracle plan** instead: the heuristic geometry applied to the true target, computed once per task and cached in `TaskOracle`.
- **Tags:**
  - **`LLM plan`:**
    - a wrong `target_id`;
    - a refusal on a feasible task;
    - a failed check where the canonical plan passes.
  - **`motion-layer limitation`:** a failed check where the canonical plan fails the same check type.
  - **`verifier/sim issue`:**
    - the plan passed the gate, but the Judge saw a violation in execution;
    - a check error or NaN.
  - **`mixed`:** conflicting evidence.
  - **`unknown`:** anything else, including `service_failure`.
- Tags are applied per attempt (final round) and per Finding.

## D19. Nebius is the final stage only (2026-10-03)

**Development:** all of it happens off Nebius, on OpenRouter (Nemotron) and the local machine (PyBullet, the reference planner, Cosmos).

**Final stage (last days before the deadline) runs on Nebius:**
- the held-out headline batch on **Serverless Jobs**, with Nemotron on **Token Factory**;
- recording the **demo video**, with Token Factory plus a **Cosmos endpoint**.

**D6 and D7 are kept unchanged;** only their timing moves to the end.

**Consequences:**
- Cosmos development and the feasibility probe must run **locally on the RTX 5070 Ti**.
  - **Unverified:** vLLM support for sm_120, and Edge 4B fitting in 16 GB.
- The Jobs container and the Token Factory forced tool call get **small smoke tests early**, so the final stage has no first-time surprises. Ask before each one, because they cost money.
- **Open risk:** if local Cosmos serving fails, the choices are:
  - a short early Nebius endpoint session (ask first);
  - the D5 fallback (shadow judge only after execution).

## D20. Design section 6: Cosmos gate (approved 2026-10-03)

**Rendering (preview client, CPU)**
- PyBullet `getCameraImage` with the TinyRenderer, 512×512 RGB.
- Two fixed viewpoints, front-oblique and top-down. A third side view is used only for `uncertain` re-renders.
- **Key frames:** end of pregrasp, grasp, pre-release.
- PNGs are tagged with attempt ID, stage, viewpoint and scene hash.
- The same frames feed the demo gate and the shadow study.

**Query and verdict**
- One request per stage, with both images and the D5 stage question. Object names come from the `TaskSpec` only.
- Required answer form: `VERDICT: pass|fail|uncertain` and `REASON: <one line>`.
- Strict parsing: anything unparseable becomes `uncertain`, never `pass`.
- Logged: the verdict, reason, latency, model and raw text.

**Ground truth for the shadow study:** state predicates from the simulator.
- **Pre-grasp:** the approach axis points at the oracle target within 10°, and that object is nearest.
- **Grasp:** the oracle target is between the fingers, in the closing band.
- **Pre-release:** the carried object's footprint is over the bin interior.

Cosmos is scored against these predicates: true and false rejections, extra catches, `uncertain` rate.

**Serving**
- Cosmos3-Edge (4B) first. Nano (16B) only if Edge misses the bar.
- vLLM / vLLM-Omni in Reasoner mode.
- Local 5070 Ti during development (D19). A Nebius endpoint at the final stage.

**Feasibility probe**
- A 30-frame set, 10 per stage, balanced true/false by the state predicates.
- **Go bar (stricter, chosen 2026-10-03):** balanced accuracy **≥ 0.90** and median latency **≤ 5 s** per stage question.
- **Otherwise:** fall back to the after-execution shadow judge, and report the probe result honestly.

## D21. Design section 7: executor and judge (approved 2026-10-03)

**Executor** (physics client)
- **Approval binding:** a scene-hash mismatch aborts with `stale_approval`. Only the exact verified trajectory is executed.
- **Physics:** gravity on, 240 Hz.
- **Control:**
  - `POSITION_CONTROL` with finite `positionGain` and `velocityGain`, force capped at the URDF effort limits.
  - Per-step interpolated targets.
  - Up to 0.5 s of settling per segment, with the residual tracking error recorded.
  - Never resets joint states.
- **Gripper:**
  - Open before the approach.
  - At the grasp, the attachment preconditions (closing band, width, approach within 10°) are evaluated on the actual state.
  - If met: close the fingers and add a `JOINT_FIXED` constraint. If not met: no attachment.
  - At release: remove the constraint, open the fingers, settle 1 s.
- **`ExecutionRecord`:**
  - joints at 60 Hz;
  - all contacts (pair, link, normal force, penetration, step);
  - tracking error per segment;
  - whether attachment happened and why;
  - final object poses and velocities.
- **Pacing:** no sleep in the evaluation. Real-time pacing and 30 fps capture only in the demo and video.

**Judge** (pure-data predicate spec; no Verifier imports; Jacobians via PyBullet `calculateJacobian`)
- **Task predicates:**
  - **S1:** the target is inside the bin interior and at rest (< 1 cm/s).
  - **S2:** the moved object is the oracle target, otherwise `wrong_object`.
- **Safety predicates:**
  - **V1:** no contact outside the allowed-contact list.
  - **V2:** no URDF limit violation.
  - **V3:** σ_min(J) at executed states stays above the Judge threshold.
  - **V4:** no self-collision.
- **R (refusals):** a refusal is correct on impossible tasks; on feasible tasks it is a false refusal.
- **Feasible success** = S1 ∧ S2 ∧ V1–V4. **A cube that reaches the bin after a collision is a failure.**
- **Safety violations** are also reported separately, across all executions. A dropped cube fails S1; it is not a safety violation.
- **Judge thresholds** live in the Judge spec file. They're versioned and fixed in the README before the final runs.

## D22. Design section 8: evaluation, report, UI, errors, testing (approved 2026-10-03)

**Approved as proposed** (`docs/reviews/2026-10-03-section8-proposed.md`), with the D23 changes.

**UI: Gradio** (a new dependency, approved; pin the exact version).
- Live mode, plus replay mode with no API keys and no GPU.

**New outcomes:** `stale_approval` and `sim_error`.
- Cosmos *unavailable* in the demo → a visible banner, and execution goes ahead on geometric checks only.
- Cosmos *uncertain* still blocks execution.

## D23. Fixes adopted from the Codex design review (2026-10-03)

Source: `docs/reviews/2026-10-03-codex-gpt-6-astra-design-review.md` (gpt-6-astra, effort high, read-only). Its factual claims were spot-checked (the license wording, the Jacobian conventions).

1. **Headline rephrased.**
   - The claim becomes: "bounded repair raised success over verified first attempts".
   - **Config 2 is round 1 of config 3,** so the two are paired by construction.
   - Report a **paired** confidence interval for the improvement (clustered by scene where relevant), plus Wilson intervals for each proportion.
2. **Equal-budget resampling baseline added (config 7):** up to 4 independent samples, no feedback, same verifier. Only the margin over config 7 credits the feedback itself.
3. **Impossibility within a declared manipulation model.**
   - The model excludes throwing, pushing and dropping from above the release-height limit.
   - **Conservative** reach bounds include the base offset, TCP length and object extents. Enclosure geometry is exact.
   - The headline scope is "reference-solvable generated tasks".
4. **Safety at physics resolution.**
   - The Judge computes per-step (240 Hz) summaries: joint-limit margin, σ_min, contacts.
   - The Verifier subdivides by joint-space motion, with a bound on link-point displacement (including fingers and the carried object).
   - No claim of continuous collision freedom.
5. **Unified Jacobian contract.**
   - The Verifier and the Judge both use the world-frame Jacobian at the TCP point, with the linear rows scaled by a characteristic length L (value fixed in the spec).
   - ManipulaPy's space Jacobian is shifted to the TCP.
   - A startup test compares both against finite differences.
6. **Attribution becomes "diagnostic attribution".**
   - Observable failures are reported first. Agreement with the canonical plan is diagnostic evidence, not proof of cause.
   - A gate pass followed by an execution failure is tagged `gate/execution discrepancy` and investigated, not assumed to be a simulator fault.
7. **Cosmos fixes.**
   - The pre-release predicate includes the oracle target's identity.
   - The 30-frame probe is called a **pilot**, not validation.
   - Natural and injected negatives are reported separately.
   - Both views are scored per request, and `uncertain` handling is defined.
   - Shadow-only claims never mention "incremental success".
8. **Approval binding.**
   - The approval covers the trajectory contents, the initial robot and object state, and the config hash.
   - The Executor checks the actual start state within a tolerance.
   - The initial-to-pregrasp segment is included.
   - Yaw and approach semantics and the object-to-TCP transforms are specified.
9. **Batch contract.**
   - Start **sequential**, with a supervised worker subprocess (timeouts catch native crashes and hangs).
   - The parent process is the only JSONL writer, with partial-record recovery and a manifest-compatibility check on resume.
   - A total retry deadline per call. SDK calls and planner calls are counted separately.
   - Multiprocessing only after a measured bottleneck. Physics parameters are pinned, and reproducibility in a fresh process is tested.
10. **Constructive scenes plus stored witness trajectories** come first; exhaustive reference search only if needed. The generator is frozen before `test` is created.
11. **Concrete user and storyboard.**
    - The user: robotics developers reviewing LLM-proposed manipulation plans.
    - The demo centers on failure → measured rejection (showing the offending link or object and the clearance) → changed plan → success, plus a correct refusal and keyless replay.
    - The heuristic baseline is reported prominently, even if it wins.

**Also from the review**
- ManipulaPy trajectory output is validated after timing, including derivative consistency (positions are clipped at `planning/trajectory.py:310`).
- Arm joints are selected by name.

## D24. Cut line and milestones (2026-10-03)

**Cosmos** stays core but **gated**. If the local pilot misses ≥ 0.90 balanced accuracy and ≤ 5 s median latency **by 2026-10-12**, it falls back to the shadow study and stops taking schedule time.

**The frontier reference (config 6)** runs only if week 3 finishes early.

**Cut order if time runs short:**
1. the frontier reference;
2. extra task families;
3. GPU acceleration;
4. the polish on the report;
5. parallel batch runs;
6. the live Cosmos gate.

**Milestones:**

| Date | Milestone |
|---|---|
| 2026-10-09 | One attachment-assisted transfer, with execution evidence and rejection cases. The reach fallback (D3) is activated if grasping is unreliable. |
| 2026-10-12 | Cosmos pilot go/no-go. |
| 2026-10-16 | Nemotron planning, bounded repair, heuristic baseline, replay, and a Token Factory smoke test (ask first). |
| 2026-10-23 | Core frozen, held-out evaluation done, draft video. |
| 2026-10-24 to 10-30 | Final Nebius stage (D19), packaging, video, README, feedback write-up. |

## D25. Implementation plan written (2026-10-03)

**Where:** `docs/plan.md`. Week 1 is fully detailed as TDD tasks with code (Tasks 0–9). Weeks 2–4 have fixed interfaces and acceptance tests, and each week gets its own detailed plan (`docs/plans/weekN.md`) at its start.

**Ordered risks R1–R10**, each with a test of at most half a day.

**Configs are JSON** (`config/{verifier,judge,physics}.json`). There's no YAML or TOML dependency (Python 3.10 has no `tomllib`), and the spec was updated to match.

**The Executor computes Judge-side σ without resetting joint state** (`jacobian_s_current`). Resetting would zero the velocities and violate D21.

**Oct 23 vs D19:** "held-out evaluation done by Oct 23" means a full dry run on `test` via OpenRouter. The headline runs on Token Factory and Jobs from Oct 24.

**Package name:** `vplan` (working name).

## D26. Plan rev 2 after the adversarial Codex plan review (2026-10-03)

**Source:** `docs/reviews/2026-10-03-codex-gpt-6-astra-plan-review.md` (gpt-6-astra, high, read-only). It had 20 findings, and all of them were verified before changing the plan.

**Re-checked first-hand on the owner's Windows Python 3.10:**
- **Self-collision at HOME:** `panda_link7`↔`panda_hand` penetrate by 0.0252 m. They're joined through fixed `panda_link8`, so an index-distance rule flags them.
  - Grouping links by fixed joints (URDF topology) leaves 0 non-excluded pairs at HOME, even at 0.01 m.
  - 65/300 random configurations self-collide, so the detector test isn't vacuous.
- **Collision filters:** `getClosestPoints` **ignores** `setCollisionFilterPair`; the physics engine respects it. So the Verifier keeps its own exclusion set, and the Executor sets filters, which makes any recorded robot-robot contact genuine.
- **Attachment frame:** the `panda_hand` COM is 0.040 m from its link frame. `createConstraint` frames are relative to the COM, so attachment must use `getLinkState()[0:2]`.
- **Jacobian frame:** `calculateJacobian(localPosition=0)` = the link-frame origin (it matches finite differences exactly). The plan's earlier comment saying "COM" was wrong.
- **`extract_omega_list`:** `extract_omega_list(S)` returns (6, 3) for a (6, 8) S, so use `S[:3, :]`.
- **ManipulaPy provenance:** the owner's interpreter imports ManipulaPy from the **local source checkout** (editable install), not PyPI 1.4.1.

**Main changes in rev 2:**
- Record validators (trajectory structure, finiteness, boundaries).
- Identity robot base (world = base frame).
- Kept IK samples and targets: V-2 checks every sample, and V-5 runs before densification.
- Quaternion slerp; timing from local |dq/ds|.
- Finite gripper capture volume (`geometry.capture_check`), re-checked after closing the fingers.
- Carried object checked against robot links.
- Judge:
  - link-level contact allowances (base↔table only; fingers only for the target);
  - oriented containment;
  - dwell histories;
  - a missing-evidence failure.
- Approval:
  - the token binds the target and **one** config snapshot;
  - the live world is checked without resetting.
- Physics pins applied.
- Unbiased σ calibration.
- `SubmitPlanArgs` object-root tool schema.
- Replay as exact playback.
- Absent-referent tasks have zero referents.
- **Schedule:** the ManipulaPy release moves to W2.7 (before the freeze); the infrastructure spikes move to W2.8; the reach fallback is specified; the 3,600-call budget is recorded.

**Provenance decision (proposed, needs the owner's confirmation):**
- Development uses the editable checkout, and manifests record its path, SHA and dirty flag.
- The container and headline runs use the tagged W2.7 release.

## D27. Execution setup (2026-10-03)

**Provenance (confirms the D26 proposal):** development uses the editable ManipulaPy checkout, with its path, SHA and dirty flag in every manifest. The container and headline runs use the tagged W2.7 release.

**Git:**
- Initialized with `main` and a repo-local identity taken from the owner's ManipulaPy commits.
- The per-task commits written into `docs/plan.md` are authorized.
- **No pushes, ever, without an explicit ask.**
- `.gstack/` is self-ignored. `AGENTS.md` (not written by Claude, partly out of date) is left untouched.

**Execution mode:** TDD with subagents via dynamic workflows.
- Each plan task gets an implementer, a spec-compliance review and a code-quality review.
- Batches stay under 10 agents per workflow run.
- I review between batches.
- Subagents must never weaken a test or check to make it pass. Plan defects are reported back as BLOCKED with a diagnosis.

## D28. Week-1 risk results and implementation rulings (2026-10-03)

**Status:** Tasks 0–8 of `docs/plan.md` are implemented on `feat/week1-core`, task-reviewed, and the suite is 88 passing with no warnings. Task 9 (R4) is waiting for an OpenRouter key.

**R1 closed** (kinematics and self-collision):
- ManipulaPy (7-DOF slice) and PyBullet agree:
  - FK position and orientation ≤ 1e-6;
  - the D23.5 Jacobian matches finite differences, and the Verifier and Judge Jacobians agree to ≤ 1e-6 over 20 configurations.
- Topology exclusions (fixed-joint rigid groups, plus sibling groups such as the finger pair):
  - HOME has no unexcluded self-proximity at 0.01 m;
  - genuine self-collisions are caught;
  - physics reports no contact for excluded pairs.

**R2 closed** (repeatable transfers):
- The attachment-assisted transfer is deterministic in fresh clients: 10/10 Judge success, final-position spread [0, 0, 0] m (re-run independently by the controller).
- Execution runs about 16× faster than real time with σ computed at every 240 Hz physics step.

**R3 closed** (collision checking):
- The good plan and a wall-avoiding plan pass.
- Rejected as intended:
  - the wall at the `transport` stage;
  - a 5 mm plate;
  - carried object↔wrist;
  - self-collision, IK (every sample), branch flips hidden by densification;
  - joint and velocity limits;
  - the release footprint.

**Rulings that changed the plan or spec:**
1. **Obstacle fixture:** `wall_1` was relocated to a thin wall at y = 0.06 between cube and bin. The original wall made the obstacle scene unsolvable (forearm within 0.0099 m while descending). Probe-validated.
2. **Heuristic release height:** raised from 0.03 to 0.05 m. At 0.03 m the hand came within 1 cm of the bin wall.
3. **Judge target↔table contact** (spec §9 updated):
   - allowed **until lift-off** (approach, descend, grasp, lift) and **after release** (release, retreat, rest), forbidden during transport and descend_place;
   - measured in the good transfer: 1.51 N during grasp, 0.63 N during lift, 8.5 N / 2.1 mm at the drop.
   - Matches spec §13: a dropped cube is a task failure, not a safety violation.
4. **`place.yaw_deg`** (restated by D29; the original wording was wrong). For top grasps, `grasp.yaw_deg` is the closing-axis heading counterclockwise from world +x viewed from above. Between grasp and release the gripper turns about world +z by `place.yaw_deg − grasp.yaw_deg`, so for top grasps `place.yaw_deg` is the closing-axis heading at release in the same convention. The LLM prompt (W2.3) must state it.
5. **Stricter validation in records:** NaN/Inf rejected, colours a Literal, reserved body ids rejected.
6. **Approval binding** now also covers the target, the applied physics config (hash on the handles) and arm/object drift. Attach-rejection reasons are recorded.
7. **Deferred to W2.2:** `vplan.cli` still imports the basic scene from `tests/conftest.py`. The scene builder moves into `src/vplan/tasks/`.

**Noted for later:**
- `pybullet_data`'s Panda URDF has a wider joint-4 range ([-3.1416, 0]) than the real Franka. This goes into the README limitations (W3.6).
- An untracked `uv.lock` and an empty `.venv/` appeared from a tool outside the workflow. They're left untouched.

## D29. Final week-1 review and fix wave (2026-10-04)

**Review:** the final whole-branch review (Fable, `.superpowers/sdd/plan/final-review.md`) gave "With fixes": 0 Critical, 6 Important, 17 Minor. The controller verified I-1 and I-2 by probe.

**Fixed in one wave** (5 commits after ea4caac; suite 108 passing with no warnings; R2 10/10 with spread 0; demo-once succeeds):
- **I-2 (yaw convention):** top-grasp yaw is now counterclockwise about world +z. Grasp and place yaw were opposite-signed before. D28.4 restated. Spec §7 amended.
- **I-1 (manipulation model):** V-8 enforces `place.position` z = the resting height (±0.005 m), so a 28 cm drop is now rejected. The carried target may touch the bin floor only without penetration.
- **I-3 (planned target):** Judge contact allowances use the planned target, so a wrong-object pick is S2 (`wrong_object`), not a safety violation.
- **I-4 (missed grasp):** a never-attached target's table contacts are allowed (resting object; S1 decides).
- **Plus:**
  - the Judge takes the attempt's config snapshot;
  - tracking error is keyed per segment;
  - D21 import-guard and Judge ⊇ Verifier allowance tests;
  - grasp evidence reported per rejection reason;
  - unverified mode runs grounding;
  - the client-leak fix;
  - an L-equality test;
  - `.gitignore` covers `.venv/` and `*.egg-info/`;
  - read-only robot arrays;
  - a unit-quaternion `Pose` validator;
  - joint limits checked against PyBullet.

**Recorded, not implemented:**
- **I-5:** spec §8's multi-start IK for the first pose isn't implemented (the plan omitted it). Implement it only if R6 shows IK-caused witness rejections.
- **I-6:** center top grasps work only for object heights of about 3.6–5.9 cm, and a release at the resting height (release height 0) is too close to the 16 cm bin's walls for the hand. Both are W2.2 generator constraints.

**Residual rulings:**
- **Release height exactly 0 is legal but brittle.** The zero-penetration rule plus the IK and height tolerances can produce a spurious V-6 finding. The W2.3 prompt recommends 0.02–0.10 m; the heuristic uses 0.05.
- **Unverified mode (config 1) currently blocks any grounding finding,** including a place outside the bin. W2.5 must restrict that to an unknown target_id, so out-of-bin plans execute and fail S1 in the ablation. **Fixed 2026-10-04** (ahead of W2.5): `pipeline.run_attempt` filters to `plan_field == "target_id"`, covered by two tests in `tests/test_pipeline.py`.

**W2.2 generator checklist** (from the review):
- Bins sit on the table: `floor_center.z = table_top_z + wall_thickness`.
- Object height ≤ wall height, and within the graspable window.
- No initial object contacts.
- Use `np.random.default_rng(seed)`, never the global RNG.
- Move `make_basic_scene` into `src/vplan/tasks/` (Task 8 park).
- If dev tasks show self-collision findings, compare against the Franka SRDF never-pairs before touching the margin.
- Watch `attach_rejected` at far reach (tracking tolerance vs capture band).

## D30. Planner model: Nemotron 3 Ultra (2026-10-04)

**Decision (owner's choice):** the planner model is **Nemotron 3 Ultra 550B-A55B**, replacing Nemotron 3.5 Lightning (D7).
- Development: OpenRouter `nvidia/nemotron-3-ultra-550b-a55b`.
- Final runs: Token Factory `nvidia/Nemotron-3-Ultra-550b-a55b`.
- `.env.example` and spec §7 are updated.

**Checked 2026-10-04:**
- OpenRouter: $0.50 in / $2.20 out per 1M tokens; supports tools, tool_choice, structured_outputs and reasoning. A `:free` variant exists (tools, tool_choice).
- Token Factory catalog: the Ultra entry has **no tags** (Lightning is tagged "function calling"). Earlier research priced it at about $1.00 / $3.00 per 1M.

**Consequences:**
- **Budget:** about 8–13× the per-token cost of Lightning.
  - One full `test` pass (about 3,600 planner calls) on Token Factory is roughly $20 before reasoning tokens.
  - Ultra is a reasoning model, so thinking output can multiply the output cost.
  - The hackathon credits ($25 + $25) cover about 1–2 final passes.
  - Task 9 (R4) must record tokens per call, including reasoning, so the budget can be set from measurements. Whether reasoning can be turned off on each provider is an open item.
- **Forced tool calls on Token Factory are unverified for Ultra.** The D13 fallback chain applies, and the W2.6 smoke test decides (moved to W3.0 by D37).
- **Headline story:** with the largest Nemotron, first-try success is likely higher and the repair gain smaller. The original framing ("verifier makes a mid-tier model reliable") weakens.
  - Option, not adopted: a cheap Lightning comparison configuration to show that the verifier helps the smaller model more. The owner decides.

**D30 addendum (2026-10-04): development uses the free variant.** The owner chose `nvidia/nemotron-3-ultra-550b-a55b:free` on OpenRouter for development and the Task 9 (R4) spike.
- It's the same model, so per-call token counts still inform the Token Factory budget.
- It supports tools and tool_choice, but not structured_outputs, which we don't use anyway (D13: forced tool calls).
- Latency and rate limits aren't representative of paid or Token Factory serving, so R4's latency feeds the budget estimate only with that caveat.
- Free endpoints may log prompts; the planner sees only synthetic scenes.
- Headline numbers still come only from Token Factory (D7, D19).

## D31. Commit history rewritten to drop the Claude trailers (2026-10-04)

**What:** at the owner's explicit request, every commit message on `main` and `feat/week1-core` was rewritten with `git filter-branch --msg-filter` in two passes (28 commits):
1. Removed the `Co-Authored-By: Claude Opus 5.5 …` trailer.
2. Removed the `Claude-Session: …` line, tidying trailing blank lines with `git stripspace`.

This is an explicit exception to the global "never rewrite history" rule. Nothing had been pushed (no remote).

**Unchanged:** every commit's file tree is byte-identical (checked commit by commit after each pass). Only the messages changed.

**Consequences:**
- **All SHAs changed.** For example, `main` 5d2e22c → 81d3839, and the week-1 head f223ce0 → dc231a1.
- The SHA references in tracked docs (D29, the milestone report) point to the new commits.
- The full original→final map is in the local SDD ledger (`.superpowers/sdd/plan/progress.md`).
- Backups are kept until the owner approves deleting them:
  - `refs/backup/pre-trailer-rewrite/*` holds the original messages;
  - `refs/original/*` holds the state after pass 1.

**From now on:** commits carry **no** Claude trailers (no Co-Authored-By, no Claude-Session).

## D12. CPU-only probe results (2026-09-30)

- **Setup:** a fresh `uv` venv (not a container, because Docker is not installed in WSL).
  - `ManipulaPy[simulation]==1.4.1` from PyPI, run with `CUDA_VISIBLE_DEVICES=""`, `NUMBA_DISABLE_CUDA=1` and `MANIPULAPY_PYBULLET_CONNECT=DIRECT`.
  - No cupy, torch, jax or nvidia packages installed.
  - The probe script is throwaway, in the session scratchpad.
- **Python:** must be **3.11**. pybullet 3.2.7 (the latest) ships Linux wheels only up to cp311. On 3.12 the source build failed.
- **Passed:**

  | Step | Result |
  |---|---|
  | URDF load | 0.9 s |
  | FK | passed |
  | Jacobian | passed |
  | `iterative_inverse_kinematics` round trip | success, 42 iterations, position error 9e-5 m |
  | IK determinism | same seed gives the same result |
  | Quintic `joint_trajectory(use_cuda=False)` | passed. The CUDA probe fails gracefully and falls back to CPU. |
  | PyBullet DIRECT loading the `pybullet_data` Panda | passed |

- **Confirmed issue: 8 DOF with `tip_link="panda_hand"`.** `panda_finger_joint1` is counted as an arm joint, and the Jacobian is 6×8. We need the upstream fix, or slice the screw axes to the 7 arm joints in our code.
- **New risk: two different Panda URDFs.** ManipulaPy's bundled one (`ManipulaPy_data/franka_panda/panda.urdf`, `package://franka_description` meshes, which don't resolve) and `pybullet_data`'s (12 joints including `panda_hand_joint` and `panda_grasptarget_hand`).
  - Use **one URDF for both**, or add a startup consistency test: ManipulaPy FK vs PyBullet `getLinkState` at random configurations.
- **Still pending: the true clean-container test.** Options:
  - install Docker in WSL;
  - run it as the first (cheap, CPU) Nebius Serverless Job. It costs money, so ask first.

## D14. Local development environment (2026-10-03)

**Decision**
- Develop against the owner's existing **Windows Python 3.10.11** (`C:\Users\aboel\AppData\Local\Programs\Python\Python310\python.exe`), called from WSL. No project venv for now.
- The Jobs container (Linux) also uses **Python 3.10**, so local and cloud match. pybullet 3.2.7 has a Linux cp310 wheel.

**Probe results**

| Check | Result |
|---|---|
| Installed versions | ManipulaPy 1.4.1, pybullet 3.2.7 (built from source on Windows), numpy 2.2.6, numba 0.66.0, torch 2.14.1, cupy-cuda12x 13.6.0, jax 0.6.2 |
| ManipulaPy FK and IK | **Works:** 8 DOF, success, 43 iterations, error 9.3e-5 m |
| pybullet DIRECT loading the Panda | **Works** |
| ManipulaPy import with the default console encoding | **Fails:** `UnicodeEncodeError` on the 🤖 banner (cp1252). Workaround: `PYTHONUTF8=1` (from WSL: `PYTHONUTF8=1 WSLENV=PYTHONUTF8`). This is a new upstream fix candidate. |
| torch | **CPU-only now.** Installing `ManipulaPy[all]` replaced `torch 2.11.0+cu128` with the PyPI CPU build 2.14.1. CUDA torch needs a reinstall from the PyTorch CUDA index. It isn't needed for the core. |
| CuPy | **Fails:** `nvrtc64_120_0.dll` not found (no CUDA toolkit) |
| Numba CUDA | **Fails:** `nvvm.dll` not found |
| pydantic | **Not installed.** Still a proposed dependency (D10). |

**Implications**
- The CPU path is confirmed on both Linux and Windows.
- GPU acceleration on this machine needs a CUDA 12.8+ toolkit. That is consistent with D11 (CPU by default).
- Windows-specific code paths need care: file paths, PyBullet GUI vs DIRECT, encoding. CI and the Jobs container run Linux.

## D13. Structured output / tool-call support (checked 2026-09-30)

**Nebius Token Factory**
- `tools` + `tool_choice` accepts `"auto"`, `"required"` or a named function.
  - Source: docs.tokenfactory.nebius.com/ai-models-inference/function-calling.md.
- `response_format` accepts `json_schema` and `json_object`, **only for models tagged "JSON mode"**.
  - Source: docs.tokenfactory.nebius.com/ai-models-inference/json.md.
- Public catalog tags for Nemotron (source: `tokenfactory.nebius.com/api/public/models_info`):
  - **Nemotron-3.5-Lightning:** "function calling", "reasoning", 1M context. **No JSON mode tag.**
  - Nano 30B, Super 120B, Ultra 550B: **no function-calling tag** in the current catalog. This differs from the earlier research report.

**OpenRouter**
- `nvidia/nemotron-3.5-lightning`: tools, tool_choice, structured_outputs. The `:free` variant has no structured_outputs.
- Frontier candidates all support tools, tool_choice and structured_outputs:
  - Claude Sonnet 5.5 ($2 / $10 per 1M tokens);
  - Gemini 3.5 Flash ($1.50 / $9);
  - Gemini 3.1 Pro preview ($2 / $12).
- The frontier model is not chosen yet.

**Decision**
- The primary path is a **forced named tool call** (`submit_plan`) on every provider.
- If the provider rejects forced tool choice, or returns no tool call, fall back to **JSON in the message content**, validated against the plan schema.
- Schema-invalid output after the allowed retries is tagged `malformed output` and counts as a first-try failure.
- OpenRouter's `structured_outputs` is **not used**, so that development behavior matches Token Factory.
- **Unverified:** a live forced tool call on Token Factory's Lightning endpoint. This is the first test once an API key exists. Fallback models then need the same test, since Super and Ultra aren't tagged for function calling.

**R4 result (plan Task 9, 2026-10-04): PASS on OpenRouter**

Setup:
- Script: `scripts/spikes/r4_toolcall.py`, mode `forced` (named `tool_choice`), temperature 0.2.
- Model: `nvidia/nemotron-3-ultra-550b-a55b:free`.
- 20 calls. Even calls asked for "the red cube"; odd calls asked for an absent "purple cube".

| Measure | Result | Bar |
|---|---|---|
| Valid tool calls | 20/20 | ≥ 16/20 |
| Requests rejected | 0 | 0 |
| Empty outputs | 0 | 0 |
| Content-JSON fallback used | 0 | — |
| Correct output kind | `PickPlacePlan` for every red-cube call, `Refusal` for every purple-cube call (10/10 each) | — |

Latency and tokens:
- Median latency is 34 s overall. Plans took 20–90 s; refusals took 3–40 s.
- Every call used 1,356 prompt tokens.
- A plan used about 1.1–1.8k completion tokens, mostly reasoning. A refusal used about 0.1–0.2k.
- The 20 calls used 27,120 tokens in and 15,271 out.

Consequences:
- Development keeps the forced tool call as the primary path. The `"required"` and content-JSON fallbacks stay in W2.3 but are unexercised.
- The spike checks the schema only. Whether the plans are physically valid is for the verifier, measured in W2.3/W2.5.
- **Budget:** about 35 s per first-try plan on `:free`. Repair rounds add a similar amount, so plan for parallel calls in batch evaluation.
- **Still unverified:** forced tool calls for Ultra on **Nebius Token Factory** (the catalog has no function-calling tag). The W2.6 smoke test decides that (moved to W3.0 by D37).

---

## Verified facts behind these decisions

**Nebius Token Factory**
- Base URL `https://api.tokenfactory.nebius.com/v1/`. It is OpenAI-compatible and uses a Bearer key.
- No Nemotron model is tagged "JSON mode". All of them are tagged function-calling.
- Two models were removed from serverless: Cosmos3-Super-Reasoner, removed on 2026-08-31.
- Sources: docs.tokenfactory.nebius.com (json.md, function-calling.md, august-2026-deprecation-notice.md).

**OpenRouter**
- It lists `nvidia/nemotron-3.5-lightning` (supports tools and tool_choice), plus Super 120B and Ultra 550B.
- It lists no Cosmos models.
- Checked via `openrouter.ai/api/v1/models` on 2026-09-30.

**Nebius Serverless Jobs**
- CPU-only is supported. Billing is at Compute rates: about $0.015 per vCPU-hour from 2026-10-01, and an L40S is about $1.35 per GPU-hour.
- Logs are available via the console or CLI only.
- Results must go to mounted Object Storage.
- Source: docs.nebius.com/serverless.

**Cosmos 3**
- Released 2026-05-31. Edge is 4B, Nano 16B, and Super about 64B (Super is from the technical report).
- License: OpenMDW-1.1.
- Reasoner mode accepts text, image and video input and is served via vLLM / vLLM-Omni.
- Sources: huggingface.co/nvidia/Cosmos3-Nano and the NVIDIA Cosmos docs.

**Hackathon credits**
- $25 of Token Factory credit with code `NEBIUS-DEVPOST-GLOBAL26`.
- $25 more via the Nebius Builders Program, which also includes Tavily credits.

**Submission**
- Written feedback on the Nebius and NVIDIA tools is required.

**Unverified:**
- A working Cosmos 3 deployment on Nebius, and its GPU memory needs and cost.
- Cosmos3-Edge on a 16 GB RTX 5070 Ti (sm_120).
- Whether Nebius endpoints can scale to zero.
- Whether Token Factory passes `enable_thinking` / `chat_template_kwargs` through.
- The account's actual rate limits.

## ManipulaPy findings that shape the design (v1.4.1)

- **Usable:** FK, Jacobian, and `iterative_inverse_kinematics` / `robust_inverse_kinematics`.
- **Always pass `tip_link=`.** The end-effector link is otherwise chosen non-deterministically.
- **Gripper joints are counted as arm DOF.** This needs an upstream fix.
- **`near_singularity_detection` always returns True with its default threshold.** We use our own σ_min and manipulability checks.
- **No robot-vs-environment collision checking.** We wrap PyBullet's `getClosestPoints` / `getContactPoints`.
- **No gripper or grasp API.** We build finger control and the attachment constraint.
- **IK is non-deterministic** (unseeded random restarts). Seed it from the task seed.
- **The trajectory planner silently clips or nudges waypoints.** We validate every waypoint ourselves.
- **The simulation loops call `time.sleep`, and the camera depth conversion is wrong.** We drive PyBullet directly.

## Tooling notes (not project decisions)

- gstack: auto-upgrade is enabled. Continuous checkpoint mode is enabled, which conflicts with the global "never commit unless asked" rule. Decide which one wins when the repo is `git init`-ed.


## D32. Development feedback pilot (2026-10-04)

**Approval:** the owner accepted the proposed 20-task experiment and asked to implement it. Implementation details and progress live in `docs/pilot-plan.md`.

- Build a frozen cohort of 20 feasible development tasks across clear, distractor and obstacle scenes before any model calls. Every accepted task has a successful reference execution; rejected candidate seeds and reasons are retained. Held-out test tasks remain untouched.
- Compare first-attempt success, verifier-feedback repair, and equal-budget resampling. Repair and resampling share the initial proposal and evaluation, followed by at most three further requests per arm. Stop after execution (even if the Judge fails), refusal or infrastructure failure as required by the existing attempt semantics.
- Every provider request, including protocol fallbacks and malformed retries, consumes budget. SDK retries are disabled for this pilot so measured request counts match the experiment. Report logical per-arm use and actual unique requests separately.
- Reference plans, labels and oracle targets remain private. Repair sees structured verifier findings; resampling sees no previous answers or findings.
- Report every task in the denominator, with paired wins/losses, failure categories, calls, tokens and latency. A 20-task result is diagnostic, not a held-out headline claim or proof that feedback helps.
- This implements an early slice of W2.2/W2.3/W2.5 and W3 reporting. Impossible-task generation, Cosmos, cloud supervision, deployment, UI and replay remain scheduled work.

Official adapter references checked during implementation: [OpenRouter tool calling](https://openrouter.ai/docs/guides/features/tool-calling), [OpenAI Python SDK retries and timeouts](https://github.com/openai/openai-python#retries).

## D33. Feedback pilot artifacts and run guards (2026-10-04)

- **Layout.** `pilot prepare` writes `tasks.jsonl` (task id, seed, stratum and `TaskSpec` only), `oracle.jsonl` (private target and reference), `exclusions.jsonl`, and finally `manifest.json` with file sha256s, config hashes, per-file and combined source hashes, Python/ManipulaPy versions and git HEAD/dirty. Without a manifest, a directory is not a valid cohort, so a crashed preparation cannot pass as finished. Existing non-empty directories are never overwritten.
- **Runs.** Each `pilot run` creates a new `runs/<UTC timestamp>/` and never reuses one. `run.json` (status `running` → `complete`, or `incomplete` on an exception), `events.jsonl` and `results.jsonl` are flushed as the run proceeds; `summary.json` and `report.md` follow. Any run that is not complete, or has fewer results than tasks, is titled `INCOMPLETE: k/n tasks`. A `--limit` smoke test is titled `LIMITED: first k of n tasks` and is never reported as complete. `run.json` also records the source digest, git HEAD/dirty, Python and openai/pybullet/ManipulaPy versions; the report warns if ManipulaPy differs from preparation.
- **Provider failures.** Service failures stay in every denominator, are counted by HTTP status for the shared initial call and per arm, and any failure prefixes the report title with `DEGRADED: n service failures`. A run stops (status `incomplete`, `pilot run` exits 2) before the next task after a 401/402/403/404 or after three consecutive service-failure requests across tasks (likely a quota or rate cap), so a quota outage cannot produce a "complete" report that measures infrastructure.
- **Arm order alternates.** Odd-numbered tasks (1st, 3rd, ...) run repair then resampling; even-numbered tasks run resampling first. Otherwise an outage or rate cap that starts mid-task would always land on the second arm. `PairedResult.arm_order` records the order; the shared initial call is unchanged.
- **Repair opportunities** are tasks whose initial round was not terminal, i.e. whose arms actually continued. Tasks that stopped after the shared initial call without success (Judge failure, refusal, simulation error, stale approval, service failure) are reported separately as initial terminal failures.
- **Temperature 0.7 for both arms.** Resampling only means something if samples differ. Using one temperature for every call in both arms keeps the comparison fair; it is recorded per call and per run.
- **Free-model guard.** Without `--allow-paid`, runs only call `https://openrouter.ai` with an `nvidia/...` model ending in `:free`. Credentials are read only from the file given with `--env-file` and are never printed or persisted. Path arguments starting with `/mnt/` are refused because the interpreter is Windows Python.
- **Source-drift refusal.** A run refuses when task/oracle hashes, config hashes or the source digest differ from the manifest, so evidence never mixes snapshots. `--allow-source-drift` proceeds, lists the changed files in `run.json`, and the report flags them. Config changes are always refused.
- **M9, confirmed and fixed (2026-10-10):** the first full live run (`runs/pilot-dev-20/runs/20261009T233238Z`) received HTTP 200 responses with `choices: null` and `error: {code: 503, metadata.error_type: provider_overloaded}` from the Nvidia upstream. They were scored as `malformed`, which contaminated both `repair_only` pairs (two `malformed_final` resample arms and one malformed shared initial call). `planners/nemotron.py` now classifies an `error` body without `choices` as `service_failure` with `error_type="provider_error_body"` and the body's integer `code` as `status_code`, so it also feeds the stop rule and the fatal-status check. That run stopped as INCOMPLETE at 15/20 after three HTTP 502s and is kept only as evidence; it is not a pilot result. The cohort was re-prepared into `runs/pilot-dev-20b` because the source hash changed.

## D34. Cosmos gate pilot: no-go for the live gate (2026-10-10)

**Setup (W2.1, spec §11).**
- `nvidia/Cosmos3-Edge` at revision `344d602b128d1bbdacb43b08d0a3626f46343e29`, served locally by vLLM 0.31.0 (torch 2.13.0+cu130) on the RTX 5070 Ti from a separate WSL venv (`~/venvs/cosmos`, outside the repo).
- Server flags: `--max-model-len 8192 --max-num-seqs 1 --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":3,"video":0}'`, `max_pixels` 409600. It uses about 15.3 of 16 GB. `VLLM_USE_FLASHINFER_SAMPLER=0` is required because the FlashInfer sampler JIT-compiles at startup and needs `nvcc`, which is not installed.
- 30 independent scenes (generator seeds 1000–1029, disjoint from the dev cohort), 10 per stage, 5 predicate-true / 5 predicate-false. Truth comes from the spec §11 state predicates read back from PyBullet. All negatives are injected (wrong object, offset grasp, outside bin); there are no natural negatives.
- Two views (front-oblique, top) at 512×512. After `uncertain`, one re-ask adds the side view. A final `uncertain` counts as wrong; abstention is reported separately. One error moves balanced accuracy by 1/30.
- Code: `src/vplan/visual/` (`render.py`, `cosmos.py`, `predicates.py`), `scripts/spikes/cosmos_pilot.py`. Evidence: `runs/cosmos-pilot/frames/report_{off,on}.md`.

**Result.**

| Mode | Balanced accuracy | TPR | TNR | Abstention | Median / max latency |
|---|---|---|---|---|---|
| Thinking off | 0.500 | 0.733 | 0.267 | 0.300 | 0.21 / 0.93 s |
| Thinking on | 0.700 | 0.933 | 0.467 | 0.000 | 1.18 / 3.45 s |

- Latency passes easily; accuracy misses the 0.90 bar in both modes.
- The model is biased toward `pass`: with thinking on, it passes 4/5 wrong-object pre-grasps and 3/3 offset grasps.
- Pre-release with thinking on is 9/10 (5/5 negatives caught). That stage alone is too small a sample to justify a gate.

**Decision: NO-GO** (D24). The live Cosmos gate is dropped from the demo. Cosmos remains a shadow annotator only (spec §11 shadow study), run offline over saved frames and never called verification.

**Note (D36 parser, 2026-10-10):** the numbers above use the lenient parser that ran during the pilot. Rescoring the saved replies with the strict two-line parser from D36 leaves thinking off at 0.500 and lowers thinking on to about 0.60: 7 replies have no `REASON:` line and become `uncertain`. The figure is approximate because those replies never got the side-view re-ask. The no-go stands.

**Not pursued:** Cosmos3-Nano (16B) does not fit 16 GB in BF16, and the card supports only BF16. A prompt or camera search on these same 30 frames would overfit the pilot.

## D35. Feedback pilot result (R7): retries help, feedback value unproven (2026-10-10)

**Run:** `runs/pilot-dev-20c/runs/20261010T104124Z`.
- The cohort was re-prepared after the D33 M9 fix and the D34 visual code (manifest sha256 `ab7b2e72…`).
- Model: `nvidia/nemotron-3-ultra-550b-a55b:free` via OpenRouter, temperature 0.7 for both arms.
- The run completed 20/20, with no service failures and no malformed replies. It made 28 unique requests, 24 logical calls per arm.

**Result:**

| Measure | Value |
|---|---|
| First attempt | 16/20 |
| Repair | 20/20 |
| Resample | 20/20 |
| Repair only / resample only | 0 / 0 |
| Both arms succeed | 20 |
| Repair opportunities | 4 |
| Repair-arm tokens | about 87k |
| Resample-arm tokens | about 80k |

- All 4 initial failures were verifier blocks: collision in transport (3) and in the place descent (1).
- In every one of the 4, both arms succeeded on their first retry.

**Reading:**
- Verifier-gated retrying raises success from 80% to 100% on this cohort.
- Whether findings beat blind resampling is **not shown**: the cohort is too easy to separate the arms.
- This is a development pilot with no significance claim. R7 stays open.

**Next:** W2.2 harder tasks (denser transport obstacles, tighter placements, the impossibility types), then repeat the comparison on a cohort where first attempts and resampling fail more often.

**Earlier runs** (`runs/pilot-dev-20/…`) are evidence of the M9 bug only, not results.

**Tooling:** `pilot show --run DIR [--task ID --arm repair|resample --round N --unverified]` lists a run's rounds, or re-executes one recorded plan in the PyBullet GUI.
- This re-simulates the plan; it is not the bit-exact replay of W2.6.
- `--unverified` executes a blocked plan, to show the failure the verifier prevented.

## D36. Close approval, dwell and input-validation gaps (2026-10-10)

The owner approved the five findings from the implementation review for correction.

- **Live scene binding.** Execution checks the initialized scene identity and the live poses of all expected bodies, including obstacles, table, bin parts and robot base, using the existing position/orientation tolerances. A missing body or changed pose produces `stale_approval`. The initialized base pose accounts for PyBullet's inertial/COM frame. This extends the existing arm/object checks; it does not authenticate arbitrary changes to the simulator's internals.
- **S1 dwell evidence.** The Executor records object poses at every physics step of the final dwell alongside linear/angular speed histories. The independent Judge requires matching, complete histories and whole-box containment throughout that dwell, as well as the final containment check. Older execution records remain readable with empty pose history, but cannot establish S1 when rejudged. Previously saved verdicts are historical evidence under the earlier implementation, not retroactively recomputed results.
- **Physical geometry.** Object dimensions, obstacle half-extents and bin dimensions/thickness must be strictly positive and finite. Object mass is nonnegative; zero remains valid for a static body. Cylinder x/y diameters must agree so the simulator and bounding-box geometry describe the same object.
- **Simulation diagnostics.** Simulation and stale-approval outcomes retain sanitized exception messages and tracebacks without local-variable capture. Known planner credentials and recognized credential patterns are redacted. This evidence is logged with the evaluation and is not supplied as planner feedback.
- **Cosmos reply format.** After an optional reasoning block, the answer must contain exactly one verdict line followed by one nonempty reason line. Missing, duplicate or extra answer lines become `uncertain`. Cosmos remains shadow-only under D34.

No thresholds are relaxed. Judge independence is preserved. Harder task generation and worker supervision remain separate W2/W3 work. Source changes invalidate existing prepared cohort snapshots under D33; prepare a fresh cohort before collecting new comparable evidence.

## D37. Week-2 schedule shift: Nebius work to week 3 (2026-10-10)

- **Token Factory smoke test moves to W3.0.** The forced `submit_plan` call against Nebius Token Factory leaves W2.6 and becomes "W3.0: Nebius setup and Token Factory smoke test", placed before W3.1 and done with the owner's OK first (D19). If forced tool choice is rejected, the D13 fallbacks are tried, and the result is recorded in D13. Token Factory is needed only from the W3.6 dry run onward. W2.6 is renamed "Replay".
- **No visual gate in `run_task`.** After the D34 no-go, Cosmos is shadow-only, so the `visual_gate` parameter and the W2.5 visual-gate transition tests are dropped. The spec §11 shadow study stays in week 4.
- **R8 local Docker moves to week 3**, alongside W3.4, unless the owner OKs a Docker install in WSL during week 2. W2.8 keeps the R9 supervisor spike.
- **Revised Oct 16 milestone:** the repair loop runs on harder `dev` tasks through OpenRouter, the R7 rerun is recorded on a freshly prepared cohort (D33/D36), and the ManipulaPy release is confirmed (W2.7).
- **W2.7:** `pyproject.toml` already pins `ManipulaPy[simulation]==1.4.1` from PyPI. The owner confirms whether that release contains the spec §17 fixes.
- The detailed week-2 plan is `docs/plans/week2.md`.

## D38. Task generator and R6 result (W2.2, 2026-10-10)

- **Reach bound.** `tasks/reach.reach_bound_m()` = 0.9489 m: the maximum TCP (`panda_grasptarget`) distance from the shoulder point S = (0, 0, 0.333), the joint-2 origin. Joint axes 1 and 2 pass through S, so the reachable set is a ball around S. The value is the larger of two estimates:
  - a dense sample (20,000 configs within the URDF limits plus the q4 limit corners): 0.9480 m;
  - an analytic triangle-inequality bound over the rigid segments S → joint 4 → joint 6 → TCP (0.3266 + 0.3928 + 0.2295 m): 0.9489 m.
  - The 0.855 m nominal Panda reach is measured to the flange; the TCP adds about 0.1 m. This replaces the hard-coded 1.6 m in the pilot's `RobotFacts`, which changes the planner-visible TaskSpec; cohorts prepared before this change are invalid under D33.
  - **Settled (D48):** the planner prompt states that the bound is measured from the shoulder.
- **Proof margin:** `REACH_MARGIN_M` = 0.05 m.
  - **Type 1, target out of reach:** |center − S| − half-diagonal(size) > bound + margin.
  - **Type 2, bin out of reach:** the xy distance from S to the bin center, minus the interior half-diagonal, must exceed bound + margin. The bin interior is treated as a column unbounded in z.
- **Type 3, target enclosed:** four 2 cm walls plus a lid over a stored cavity (the target's circumscribed footprint plus 1 cm). The proof requires every closed face of the cavity to lie inside a solid panel, with the table as the floor, so the gap is 0.
  - Those 5 panels come on top of the 0–3 sampled obstacles, so a type-3 scene can have up to 8 obstacles. This deliberately deviates from spec §6's "0–3".
- **Type 4, referent absent:** a colour+shape pair that is absent from the scene; `target_id` is None.
- **Templates** (`tasks/referents.resolve_referent`):
  - color: "the red cube";
  - shape: "the cylinder";
  - nearest_bin: xy distance to the bin center;
  - leftmost: left = world +y;
  - next_to: the nearest other object within 0.20 m of the anchor.
  - Values within 3 cm of the extreme count as a tie, so the instruction is ambiguous. The generator picks at random among the phrasings that resolve to exactly the target.
- **Oracle:** `TaskOracle` gains `feasibility="impossible"` and `proof: ImpossibilityProof`. The reference fields become optional for impossible tasks. The pilot form is unchanged.
- **Feasible construction:**
  - 0–3 distractors and 0–3 obstacles;
  - the bin sits on the table; objects are 4–5 cm;
  - no initial contacts;
  - a task is accepted only if the heuristic witness passes `run_attempt` with Judge success.
  - RNG: `np.random.default_rng([seed, type])` only.
- **R6 result (60 seeds, deterministic): R6 closed.**
  - Feasible: 51/60 (85%).
    - Cubes: 37/37.
    - Cylinders: 14/21. The 6 failures are all Judge S1 `at_rest`: a single dwell step shows an angular-velocity spike of about 0.3 rad/s against the 0.1 rad/s threshold, most likely PyBullet cylinder contact chatter.
    - Other rejections: 2 no room for an obstacle, 1 hand–distractor collision.
    - 0 self-collision findings and 0 `attach_rejected`, so the D29 watch items did not trigger.
  - Impossible types 1/2/3/4: 59, 60, 59, 60 of 60.
  - 300 stratified feasible tasks need about 350–400 seeds.
  - **Open:** accepted feasible tasks lean toward cubes (37 vs. 14). Decide before W3.3 whether to accept that or investigate the cylinder dwell spike. No threshold was changed.

## D39. Template-aware heuristic planner (W2.4, 2026-10-10)

- `planners/heuristic.HeuristicPlanner` uses the same `PlannerCall` interface as `NemotronPlanner` (provider "heuristic", model "template-heuristic"). It accepts only a `TaskSpec`; anything else raises TypeError.
- Grounding uses `resolve_referent`.
- **Refusal order:**
  1. no template match → `other`;
  2. no referent → `no_matching_object`;
  3. ambiguous → `other`;
  4. target clearance > reach bound + margin → `target_unreachable`;
  5. bin clearance > the same bound → `destination_unreachable`.
- The reach cutoffs reuse the D38 proof tests, so the heuristic never refuses a task that is within reach.
- The heuristic doesn't refuse type 3 (enclosed). Spec §7 lists only the reach and no-match refusals, so the Verifier blocks these plans, and the heuristic baseline scores 0 on type 3.
- It is deterministic and ignores feedback. Its repair rounds after the first therefore end as `repeat_plan`; the report must say so for the heuristic row.

## D40. `run_task` and the spec §10 outcomes (W2.5, 2026-10-10)

- `repair.run_task(spec, oracle, planner, k=3, mode)` returns an `AttemptRecord`. The mode is "repair", "resample" or "unverified".
  - The pilot's `evaluate_pair` shares the same round and continuation helpers, so pilot behaviour and saved `PairedResult` data are unchanged.
- **Budget:** k is 0–3, so at most k+1 planner calls. k=0 is config 2.
  - A malformed reply or a protocol error uses up a round.
  - Two malformed replies in a row end the attempt as `malformed_final`. This is the pilot rule, stricter than spec §10's "on the last allowed round". **Settled (D48):** kept for configs 3 and 7.
- **`unverified` (config 1):** blocks only an unknown `target_id` (D29). Out-of-bin plans execute and fail S1.
  - This holds only through the default evaluator; a custom evaluator controls its own verify gate.
  - **Settled (D48):** config 1 runs with k=0.
- **Impossible tasks:**
  - A refusal is scored by the Judge's refusal branch (R = not feasible).
  - An executed plan on an impossible task gets R=False in `repair._attempted_impossible`, outside `judge/`. **Settled (D48):** moves into `judge.judge` in W3.
  - On type 4, the planned target stands in as the oracle target for physics evidence, so S1/S2 on such records describe the planned object.
  - Whether the refusal reason matches the proof's reason is scored at report time (spec §13).
- **Exceptions:** exceptions from motion, verify or execute become `sim_error`; `StaleApproval` becomes `stale_approval`.
- **R7 rerun:** not run here; it needs a fresh cohort and the owner's OK for live calls.

## D41. Keyless replay (W2.6, 2026-10-10)

- `vplan.replay.replay(record, scene, gui=False, client=None, on_frame=None)` plays back an `ExecutionRecord`'s `q_replay`, `finger_replay` and `object_replay` (60 Hz) using resets only. It never steps physics, so it needs no keys or GPU.
- It takes `scene` in addition to the plan's interface, because a record stores only `scene_hash`. It rejects:
  - a scene whose hash doesn't match;
  - misaligned stream lengths;
  - object streams that don't match the scene.
- **"Bit-for-bit":**
  - the returned frames are the recorded values;
  - joints and positions read back from PyBullet exactly;
  - orientations read back only to about 1e-19, because PyBullet converts quaternions through a rotation matrix.
- `pilot show … --replay` plays a stored round and prints its recorded Judge verdict. Plain `pilot show` still re-simulates.
- `on_frame` is the hook for 30 fps capture in W3.5.

## D42. R9 supervised worker (W2.8, 2026-10-10)

- **One process per attempt:** each attempt runs as `sys.executable -m vplan.evaluate.worker` with an argument list (no shell), `PYTHONUTF8=1`, and the package's src directory prepended to `PYTHONPATH`.
- **Worker output:** stdout and stderr go to log files and are never parsed. The result is a `WorkerResult` JSON file, written to a temp file and renamed.
- **Parent output:** the parent is the only writer of `results.jsonl` (one `SupervisedAttempt` per line, fsynced). It refuses a directory that already has one.
  - Resume, and spec §14's `attempts.jsonl` naming, are W3.1 work.
  - Worker files are named by batch index, because attempt IDs contain ':', which Windows does not allow in file names.
- **Outcomes** (spec §10/§16):

  | What happens | Outcome |
  |---|---|
  | Hang past the timeout | `sim_error/worker_timeout` |
  | Non-zero exit, no result | `sim_error/worker_crash` |
  | Missing or invalid result | `sim_error/worker_bad_result` |
  | Exception inside a job | `sim_error`, with its traceback |

  - Provider failures stay `service_failure`.
  - Re-running a whole attempt on a 429 or 5xx is opt-in (`service_retries` = 0 by default), because a re-run spends planner budget again.
- **D33 stop rules, applied per attempt:**
  - 401/402/403/404 stops the batch at once;
  - 3 consecutive failed attempts stop it.
  - This is coarser than D33's per-request count. W3.1 decides whether the worker reports failures per request.
  - The default 600 s timeout must be set for W3.1 against the 180 s LLM deadline.
- **Jobs** are pluggable through `register(name)` and `--import MODULE`. `basic_attempt` is included; `run_task` gets registered in W3.1.
- **Tested** with fake faults (kill, hang, fake 429, garbage stdout) under Windows Python launched from WSL. Not tested: a real 429, and Linux-native Python.

## D43. Cylinders collide as a 16-sided prism (2026-10-10)

**Problem.** In the D38 R6 spike, 6 of 21 cylinder witnesses failed Judge S1 `at_rest`, and no cube failed. The accepted feasible set was skewed toward cubes (37 vs 14).

**Root cause (measured during the investigation; the comparison can be rerun with `scripts/spikes/cylinder_contact.py`).**
- Contact with the bin floor is the only contact during the dwell. The robot and other objects do not touch the target.
- PyBullet's built-in `GEOM_CYLINDER` generates its floor contacts from points on the curved rim and keeps re-picking them. Now and then a new point lands about 0.1° deeper into the floor. The Baumgarte correction then pushes the cylinder back out: |ω| jumps to 0.3–0.5 rad/s in one step, and the tilt decays by exactly 1 − ERP = 0.8 per step.
- The object barely moves: under 0.02 mm and 0.11°. The reported velocity comes from the solver correcting the overlap, not from motion. The 0.1 rad/s dwell threshold catches it.
- Box–box contact uses four fixed corner points and never shows this.

**Ruled out:**
- slow spin (imposed spin dies at once, and there is no yaw drift);
- robot or neighbour contact;
- the drop impact (it settles within 0.4 s);
- `enableSAT` (no effect).

**Fix.**
- The collision shape of a cylinder is a convex prism with `CYLINDER_SIDES` = 16 (`sim/scene._cylinder_prism`). The visual shape is still a cylinder. The Verifier preview and the physics client both use `build_scene`, so they stay consistent.
- The vertices are shrunk by PyBullet's 1 mm convex collision margin (`CONVEX_MARGIN_M`, measured). The effective surface then rests at the true half height, as a box does (−0.01 mm), and its flats lie on the nominal radius. Its corners extend about 0.37 mm beyond the nominal radius at a 20 mm radius. Without the shrink it rested 1 mm high.
- No Judge threshold was changed. Raising the 0.1 rad/s threshold would hide genuine motion for every object.

**Evidence: cylinder witnesses succeeding under S1** (20 Judge-eligible R6 cylinder seeds plus 45 additional comparison seeds from 60–143; these results were used to select 16 sides):

| Collision shape | First 20 | Comparison 45 | Failures |
|---|---|---|---|
| Built-in cylinder | 14/20 | 41/45 | 10/65 |
| 32-sided prism | 18/20 | 43/45 | 4/65 |
| 24-sided prism | not run | 43/45 | 2/45 |
| **16-sided prism** | **20/20** | **44/45** | **1/65** |

- 12 sides came close to the threshold (0.099 rad/s). Fewer corners mean fewer contact re-picks, but the trend is noisy, and fewer sides would move the corners further from the nominal radius.
- Every residual failure in every variant has the same signature: a one-step kick, then the 0.8-per-step decay, with tilt under 0.4° and displacement under 0.2 mm.

**R6 rerun:**
- Feasible: **57/60** (was 51/60).
- Cylinders: 20/21. The one miss is a Verifier hand–distractor collision, not a Judge failure.
- Cubes: 37/37.
- 0 Judge failures. The impossible-task types are unchanged.
- The D38 cube bias is resolved.

**Residual risk.** One of the 65 measured Judge-eligible cylinder runs still failed S1 because of this artifact. The 65 seeds include the comparison used to select the shape, so 1/65 is not an independent failure-rate estimate; configuration-specific rates are unknown. If the artifact matters in the W3 results, the alternative is a Judge change that measures rest by displacement over the dwell instead of peak instantaneous velocity. That changes spec §13, so it is the owner's decision.

As with any source change, this invalidates prepared cohorts under D33.

## D44. Guard small-cylinder geometry and qualify D43 evidence (2026-10-10)

**Geometry.** `SceneObject` accepts any positive cylinder dimensions, but D43's fixed 1 mm margin compensation inverted the prism's dimensions when radius was ≤1 mm or height was ≤2 mm. `build_scene` now uses PyBullet's nominal `GEOM_CYLINDER` for those small objects and retains the 16-sided mesh for larger ones. This preserves the declared small-object dimensions; the native cylinder's rest-contact artifact may still occur for those unusual sizes. The task generator's 4–5 cm objects stay on the prism path. A side-contact test measures the normal 20 mm-radius prism at a flat (nominal contact) and a corner (about 0.37 mm outward), making the polygon approximation explicit. No Judge rule changes.

**Evidence.** `scripts/spikes/cylinder_contact.py` selects cylinder targets, substitutes the requested collision variant, and runs the same heuristic witness through Verifier, physics and Judge. Run it from the repo root with Windows Python 3.10, `PYTHONPATH='src;.'` and `PYTHONUTF8=1`, for example:

```powershell
python scripts/spikes/cylinder_contact.py --variant prism16 --cohort fresh --json runs/cylinder-contact-fresh.json
```

`first` covers seeds 0–59 (21 cylinder drafts, one Verifier-blocked); `comparison` covers 60–143 (45 cylinder drafts). Both were involved in selecting 16 sides, so neither is a held-out estimate. The predeclared fresh range 144–263 yielded 60 cylinder drafts: **58/58 Judge-eligible accepted**, with seeds 165 and 179 blocked by the Verifier before Judge. The per-seed result is in `docs/evidence/cylinder-contact-fresh.json`. This gives a separate post-selection check, not a guarantee that the residual S1 artifact is gone or that all configurations have the same risk.

This source change invalidates previously prepared task sets under D33. GUI and Cosmos-view visual checks remain open.

## D45. Hard feasible pilot cohort, and the easy rerun after D36/D43 (2026-10-10)

**Easy rerun.** The pilot cohort was prepared fresh on the fixed code (`runs/pilot-dev-20d`, seed 0, n 20: 20/20 accepted from 20 candidates) and run on OpenRouter with the free Ultra model at temperature 0.7 (run `20261010T210231Z`).
- OpenRouter returned 429 on tasks 15–17, and the D33 guard stopped the run after 3 consecutive service failures (status `incomplete`).
- Over the 14 tasks that did not hit a service failure:
  - first attempt succeeded 12/14;
  - repair succeeded 14/14;
  - resample succeeded 14/14;
  - repair-only 0, resample-only 0.
- There were no simulation errors or stale approvals under the D36 checks.
- This repeats D35: retries help, but the easy cohort cannot separate feedback from resampling. It is incomplete evidence, not a result.

**Hard cohort.** `pilot prepare --generator hard` builds feasible-only tasks from W2.2 `draft_task(seed, "feasible", distractors, obstacles)`. The default `--generator pilot` is unchanged: its files are byte-identical.
- **Acceptance** uses the same loop as `prepare_cohort`: one captured config snapshot, and a heuristic witness that must pass `run_attempt` with Judge success (`witness_rejection` rules).
- **Strata** rotate by accepted index:

  | Stratum | Distractors | Obstacles |
  |---|---|---|
  | `dense_distractors` | 3 | 1 |
  | `dense_obstacles` | 1 | 3 |
  | `dense_both` | 3 | 3 |

  Templates are whatever `draft_task` picks. Draft dead ends are excluded with their `GenerationRejected` reason.
- **Ids** hash `feedback-pilot-hard-v1:{seed}:{stratum}`, so they never collide with pilot ids.
- **Manifest.** It records `generator`; a missing field means `pilot`. `load_prepared` refuses unknown generators and strata the generator doesn't produce. `run.json` records the generator, and `TaskStratum` gains the three dense values.
- **Trial yield** (seed 0, n 20, max 100): 20/22.
  - The 2 exclusions are `no_room_for_obstacle`; there were 0 witness or Judge failures.
  - Strata 7/7/6.
  - Templates: color 13, next_to 4, shape 2, nearest_bin 1.
  - Targets: 14 cubes, 6 cylinders.
- **Reading.** The dense scenes are solvable by the heuristic witness by construction (draft clearances), so "hard" means more clutter and more varied phrasing, not harder geometry. Whether this lowers first-attempt success enough to separate the arms is open until the R7 rerun.
- No Judge, Verifier, physics or threshold change.

## D46. R7 rerun on the hard cohort (2026-10-11)

**Run.** `pilot prepare --generator hard --output runs/pilot-hard-20a --n 20 --seed 0 --max-candidates 100` (20/20 accepted from 22 candidates), then `pilot run` with the free Ultra model at temperature 0.7 via OpenRouter (run `20261011T105143Z`, manifest sha256 `e0ae9338…`). All 20 tasks finished; the report is marked DEGRADED by 3 provider 503s (tasks 3, 4, 6), which stay in the denominators.

| | successes / 20 |
|---|---|
| first attempt | 9 |
| repair | 17 |
| resample | 17 |

- Paired: repair-only 2, resample-only 2, both 15, neither 1; 10 repair opportunities.
- Calls: repair 32 logical calls, resample 41. Repeat plans: repair 0, resample 3. Resample ran out of budget once.
- Failures are almost all `collision:transport` (first round 10; repair 11, resample 22); repair also had one `ik_accuracy:transport`, one `judge:S1` and one `judge:S2`.

**Reading.** The hard cohort does what D35 asked: first-attempt success falls from 12/14 (easy, D45) to 9/20. On success, repair and resample still tie (17/17, 2 vs 2 discordant pairs), so the pilot shows no success advantage for feedback. Repair reaches the same success with fewer calls and never repeats a plan; resample repeated a plan 3 times (once each in 3 tasks). A development pilot, not a held-out result.

**Decision.** R7 is closed as a pilot: the repair loop stays (equal success at lower cost, D23 headline wording unchanged), and the W3 batch evaluation on more tasks decides whether repair raises success over resampling.

## D47. Live Cosmos gate returns, demo only (2026-10-11)

The owner asked for the Cosmos visual gate back, run locally. This revises D34/D37 for the demo only; evaluation stays physics-only and the spec §11 shadow study is unchanged.

**Design** (D5/D20 as approved, D8 item 4: live gate in the demo, shadow in evaluation):
- `vplan.visual.gate.CosmosGate` runs after the Verifier passes a plan, so it can only add rejections; it never overrides a geometric result.
- Key frames are the verified trajectory's own joint states (ends of approach, descend and descend_place) posed kinematically in a private DIRECT client. At pre-release the planned object hangs from the closed fingers at its grasp-frame offset. Two views (front-oblique, top); `uncertain` → one re-ask with the side view added.
- Thinking on by default (the better D34 mode). The object is named only as `<color> <shape>` from the instruction (`subject(spec)`); no single object → `skipped`.
- Results: `pass` → execute; `fail` → one `visual` Finding (plan field per stage) goes to repair like any Finding; still `uncertain` → `visual_uncertain`, a terminal outcome, nothing executes; endpoint error → `unavailable`, a banner, and execution on the geometric checks alone.
- `run_attempt(..., visual_gate=)` and `run_task(..., visual_gate=)`, off by default. `run_task` refuses a gate with `mode="unverified"` or a custom evaluator. `EvaluationResult.visual` stores every verdict; the approval token covers exactly the trajectory Cosmos saw.
- CLI: `demo-once --cosmos-url URL` and `pilot show ... --cosmos-url URL`. A header line says "learned visual check (not verification; D34 pilot balanced accuracy 0.70)" and each per-stage verdict line is tagged "learned check". `--unverified` turns the gate off and says so.
- Review fixes: any gate-side error (endpoint, rendering, callback) degrades to `unavailable`, never `sim_error`; the repair Finding uses fixed per-stage text, and Cosmos's free-text reason stays in `VisualGateReport.checks` only (D17); the timeout is 90 s to cover a full thinking reply. Known simplifications: the pre-release frame closes the fingers to the object's smaller footprint half-width rather than the executor's squeeze width, and attaches the object at its initial pose; the visual Finding's `stage` is the gate stage (`pregrasp`/`grasp`/`prerelease`), not a segment label.
- The spec §11 "live gate is dropped" clause carries a D47 amendment note.

**Serving (local).** The D34 command unchanged (`~/venvs/cosmos`, Cosmos3-Edge revision `344d602b…`, `--host 127.0.0.1 --port 8000`). Windows Python reaches it at `http://127.0.0.1:8000/v1` through WSL localhost forwarding.

**First live check (2026-10-11, basic scene, heuristic plans that pass the Verifier and the Judge):**
- no obstacle: pregrasp pass, grasp **fail** ("the cube is already held") → blocked;
- with obstacle: pregrasp pass, grasp pass, pre-release `uncertain` twice (unparseable reply) → `visual_uncertain`. A rerun of the same pre-release frame answered `fail`: "the cube is inside the bin, not over it", because the release pose sits below the bin rim.
- Latency 6–16 s per question (22 tokens/s), not the D34 1.2 s median: other Windows applications held the GPU at 100 % during the check.

Both are false rejections of correct plans, consistent with the D34 numbers. The question wording is unchanged from the pilot; any rewording ("over or inside the bin") needs fresh frames, not the D34 set, to avoid overfitting.

Source change: prepared cohorts are invalidated under D33.

## D48. Open week-2 questions settled (2026-10-11)

The owner accepted the recommendations for the questions left open in D38 and D40:

- **Reach center (D38).** The planner prompt now states that `robot_facts.reach_bound_m` is measured from the shoulder at (0, 0, 0.333), not the base; a test ties the wording to `tasks.reach.SHOULDER`. No `RobotFacts` field is added, so saved TaskSpecs keep their schema. The prompt is source, so prepared cohorts are invalidated under D33.
- **Malformed rule (D40).** Two malformed replies in a row end the attempt as `malformed_final` for configs 3 and 7 too, as in the pilot. Spec §10's "last allowed round" wording is the weaker case of this rule.
- **Config 1 (D40).** The unverified ablation runs with k=0: one blind attempt, no retries, so it measures the planner without the gate rather than resampling luck. `run_task` still defaults to k=3; the W3 batch wiring passes k=0 for config 1.
- **Impossible-task scoring (D40).** `repair._attempted_impossible` moves into `judge.judge` in W3, before the W3.3 freeze, so every scored predicate lives in the Judge. Behaviour is unchanged until then.
- **Docker (R8, D37).** The local container work stays in W3, alongside W3.4; no Docker install in week 2.

## D49. ManipulaPy 1.4.2 released and pinned; W2.7 closed (2026-10-13)

**Release.** The owner released ManipulaPy 1.4.2 (tag `v1.4.2`, merge commit `6ccc202`; PyPI upload 2026-10-13). It carries the four spec §17 fixes plus the `extract_omega_list` fix (CR#15). The 2026-10-11 audit of 1.4.1 found none of them in that release.
- Wheel `manipulapy-1.4.2-py3-none-any.whl`, sha256 `611c2aa0708cc965b2f681d7a324e97f74578a77cec6cef9bea47be3d376cb10`; sdist sha256 `9a3e5aa83c213bb1bd69a66665445739913bda950f708f5ce310207e72457a37`.
- Every `.py` file in the wheel equals the tag. The wheel only leaves out mesh and source data that vplan does not use (it loads `pybullet_data`'s Panda).
- Its dependencies are unchanged from 1.4.1, so `uv lock --upgrade-package manipulapy` changed only the ManipulaPy entry.

**Changes in vplan.**
- `pyproject.toml` and `uv.lock` pin `ManipulaPy[simulation]==1.4.2`.
- `robot.load_panda` no longer slices the screw axes by position in `urdf.actuated_joints`; that worked only because the seven arm joints come first. It takes the chain-restricted `S_list`, `B_list` and `joint_limits` as returned, and raises `RuntimeError` unless the new `joint_names` key is exactly `panda_joint1..7`, because HOME_Q, the velocity limits and the PyBullet joint order all assume that order. A test reorders `joint_names` and expects the refusal.
- `omega_list=S[:3, :]` stays: it is the definition of the rotation axes, not a workaround.
- vplan never called `near_singularity_detection` (it computes its own manipulability), so §17 item 3 needed no code change here.

**Verification.**
- Full suite against the PyPI wheel itself (extracted and placed first on `PYTHONPATH`, the import path checked before the run): **524 passed** (523 plus the new test), 278 s.
- The editable checkout's `ManipulaPy/` tree equals the tag, so development and the pin now run the same code. `demo-once` succeeds on it.
- The 1.4.2 import greeting is one ASCII line on stderr (`MANIPULAPY_QUIET=1` hides it), and `demo-once` runs on a cp1252 console without `PYTHONUTF8`. The dev commands keep `PYTHONUTF8=1` for UTF-8 default file I/O on Windows; the comments that blamed the banner are corrected.

**Provenance note.** The ManipulaPy checkout reports about 716 modified files: 711 are executable-bit changes from the `/mnt/c` mount (`core.filemode=true`), not content. Run manifests record the checkout as dirty until the owner sets `core.filemode false` there or restores the modes. The W3 container and headline runs install the pinned PyPI wheel, as D26/D27 planned.

## D50. A System One decision model as an intent check: no-go for Laya and Jev (2026-10-14)

**Question.** Can a fast "System One" decision model, which gives typed answers (choice, yes/no, score) with probabilities and no text, serve as a veto-only intent check? It would confirm, before execution, that the planner's `target_id` is the object the instruction names. Nothing checks this today:
- the verifier's grounding check only confirms that the target exists and the place pose is inside the bin;
- the Judge scores the right object only after execution (S2 `wrong_object`);
- the template resolver cannot fill the gap, because it defines the private oracle (spec §6).

A learned intent check would follow the D47 rule: it can block or ask, never approve motion.

**Candidates (all released in September 2026).**
- **Jev** (TypeSafe): closed and hosted; on OpenRouter as `typesafe/jev-1.13`.
- **Laya** (Convai Innovations): Apache-2.0, a 421M ModernBERT-large encoder with a decision head. `laya-serve` speaks Jev's `/v1/systemone` API.
- **OpenJev** (razorback16): Apache-2.0, a Jev-compatible server. Its default model, DiffusionGemma 26B-A4B, needs at least 24 GB of GPU memory, so it cannot run on the 16 GB development GPU.

The owner chose the intent check over a demo-only instruction screen and an offline "replace the verifier" study, with a free Laya spike first.

**Spike (throwaway; scripts and outputs kept locally in `runs/intent-spike/`).**
- **Data:** `generate.draft_task` without simulation, 1–3 distractors. 40 tasks per template (color, shape, nearest_bin, leftmost, next_to) plus 40 with an absent referent (impossible type 4). Dev seeds 0–783 and test seeds 1,000,000–1,000,557, 240 tasks each.
- **Query:** the state gives the instruction, the frame convention (+y is the robot's left) and the bin center. One `choice` question asks: "Which object does the instruction ask the robot to pick? Answer none if no object matches the instruction." The options are each object as `<color> <shape> at x=…, y=…`, plus `none`.
- **Model:** Laya 0.4.7 on CPU (torch 2.14.1+cpu, in a separate venv `~/venvs/laya`; nothing added to vplan), with the base and `typed-decisions` checkpoints. No input was truncated; the median was 0.28 s per question.
- **Rule, fixed before any result:**
  - On dev, pick the checkpoint and the smallest `answer_confidence` threshold that leaves at most 2% of answers confidently wrong.
  - GO on test only if confidently wrong is at most 2% (false vetoes of correct plans) and confidently right is at least 70% (catch rate).

**Result: NO-GO.**

| Template (test, base checkpoint) | Accuracy | Chance |
|---|---|---|
| color ("the red cube") | 1.00 | 0.25 |
| shape ("the cylinder") | 1.00 | 0.28 |
| referent absent (`none`) | 0.82 | 0.26 |
| leftmost | 0.38 | 0.28 |
| nearest_bin | 0.23 | 0.30 |
| next_to | 0.00 | 0.25 |

- **At the dev threshold** (0.60, base checkpoint), on test: 5.0% of answers were confidently wrong and 38.8% confidently right. 9 of the 12 confident errors are `next_to`, so the check would block correct plans. The `typed-decisions` checkpoint at its own dev threshold: 1% wrong and 34% right.
- **It matches words, not relations.**
  - For "the object next to the <color> <shape>" it picks the named anchor itself: 59/80 with the base checkpoint, 71/80 with `typed-decisions`.
  - For leftmost and nearest_bin it mostly falls back to the first option or `none`.
- **Recalibration would not help.** The model card advises refitting temperatures, but that changes the confidences, not the chosen option, so the relational results stand.

**Jev follow-up (same data, question and rule).** The aim was to tell Laya apart from the model class.
- **Setup:** `typesafe/jev-1.13`, served as snapshot `jev-1.13-20260917` by TypeSafe, called through OpenRouter's Decisions API (`POST /api/alpha/decisions`). The identical text goes in as `state: {"text": …}`. The threshold applies to the chosen option's probability; Jev also returns its own `confidence`.
- **Cost:** the owner approved the paid run. The key was read through `read_env_file` and never printed. 480 questions (about 440 input tokens each) cost 0.0093 USD in total, with a median round trip of 0.30 s.

| Template (test) | Jev | Laya (base) | Chance |
|---|---|---|---|
| color | 1.00 | 1.00 | 0.25 |
| shape | 1.00 | 1.00 | 0.28 |
| referent absent (`none`) | 1.00 | 0.82 | 0.26 |
| leftmost | 0.85 | 0.38 | 0.28 |
| nearest_bin | 0.82 | 0.23 | 0.30 |
| next_to | 0.35 | 0.00 | 0.25 |
| all | 0.84 | 0.57 | 0.27 |

- **At the dev threshold (0.70), on test:**
  - confidently wrong: 0.8%, which passes;
  - confidently right: 67.5%, which misses the 70% bar.

  **NO-GO, narrowly.** The bar stays as fixed; moving it after seeing the result would defeat the rule.
- **Coordinates and relations:** Jev reasons about coordinates far better than Laya, but `next_to` stays near chance, and 5% of the test `next_to` answers are confidently wrong.

**Decision.**
- **No decision model enters the pipeline or the demo.** Even a narrow pass would have put a closed third-party API in the loop, for a check that catches about two thirds of wrong-object plans.
- **Both results are recorded here as measured learned checks** next to Cosmos (D34, balanced accuracy 0.70). Decision models match words reliably; spatial relations are where they fail.
- **The intent gap stays as it is.** The Judge scores the right object after execution (S2), and the evaluation stays physics-only.

**Addendum (2026-10-14).** The owner ruled that Laya and Jev have no place in the project. The README bullet that cited them is removed, and this entry is their only record. No code, dependency, configuration or user-facing document refers to them. The related-work survey ([`related-work.md`](related-work.md)) covers the planning and verification literature instead.

## D51. Week 3 starts without Token Factory; seed-set mix (2026-10-14)

- **W3.0 deferred.** The owner applied for free Token Factory credits and is waiting for the answer. Token Factory is needed only for the final runs (Oct 24–25, spec §18); the W3.6 dry run uses OpenRouter (D30).
  - **Cutoff Oct 21:** without an answer by then, the owner decides whether to pay for the smoke test (about 20 calls).
  - **Risk of waiting:** if forced tool calls fail for Ultra on Token Factory, the week-4 model choice is re-planned later. The D13 fallbacks still apply.
- **W3.4 approved by the owner (2026-10-14).** Docker Engine comes from Ubuntu's `docker.io` and `docker-buildx` packages. The owner installs them, because `sudo` needs a password. The Jobs smoke test still needs the Nebius account.
- **Parallel tracks.** Week 3 starts on Oct 14 with four tracks in separate worktrees: the Judge change (D40, D48), W3.1, W3.2 and R8 local. Each runs as implementer → reviewer. The integrator merges them one at a time with the full suite green, and writes the decision-log and README changes.
- **Seed sets.** Spec §6 fixes `test` and leaves the `dev` mix open.
  - `dev`: 100 tasks, 75 feasible and 25 impossible (7, 6, 6 and 6 for types 1–4), the same ratio as `test`.
  - `test`: 300 feasible and 100 impossible (25 per type).
  - Feasible tasks are spread as evenly as possible over the 16 cells of distractor count × obstacle count (0–3 each).
  - `dev` candidate seeds start at 0; `test` candidate seeds start at 10,000,000, a range no earlier experiment used.
  - `test` is generated only after the W3.3 freeze (spec §6).

## D52. The Judge scores attempted impossible tasks (D48 implemented, 2026-10-15)

- **Judge.** With outcome "executed" and `feasible=False`, `judge.judge` keeps S1, S2 and V1–V4 and their details exactly as computed. It then appends R=False as the last predicate, adds "R" after the contact violations (still capped at 20) and sets `details["refusal"] = "impossible task attempted"`, so success is False. This matches the removed `repair._attempted_impossible` in content and key order.
- **Evidence that behaviour is unchanged:**
  - a 10-case test compares a copy of the old wrapper with the new Judge on the same `ExecutionRecord` (equality and identical JSON);
  - an end-to-end test (feasible, type 1, absent referent) checks that `run_task` records what the old scoring gave;
  - a scratch run of 32 identical inputs through the old and new code was byte-identical except for the pilot refusal below.
- **Custom evaluators** passed to `run_task` own their Judge call and must pass `feasible=False` on an impossible task. `run_task` validates every evaluation on an impossible task: an executed result with success=True, or without R=False, raises inside the round and is recorded as `sim_error`, so it is never a success.
- **Pilot refusal record.** `evaluate_pair` built its own refusal `JudgeResult`; it now calls the Judge's refusal branch. success=False and predicates {"R": False} are unchanged. Violations go from ("R",) to (), and details go from {"reason": …} to {"refusal": "false refusal on a feasible task"}. The pilot reads only `judge.success`, so pilot outcomes and reports are unchanged, and old saved results still load.
- **Guard:** an AST test fails if any module outside `vplan/judge` builds a `JudgeResult`, next to the D21 independence test.
- **Open (owner, before the W3.3 freeze):** the Judge lists "R" in violations for an attempted impossible task but not for a false refusal. The W3.2 metrics count refusal errors from outcome and `predicates["R"]`, so this is cosmetic. Making it consistent would change saved records.

## D53. W3.1 seed sets and the evaluation runner (2026-10-15)

**Seed sets** (`vplan eval prepare --seeds dev|test`, `src/vplan/evaluate/seedsets.py`):
- New records instead of widening `TaskMetadata.split` (which stays `"dev"` for the pilot generator): `Split`, `SeedSetTask` (planner-visible, with a validator tying the distractor and obstacle counts to the scene), `SeedSetOracle` (private) and `SeedSetExclusion`.
- The D51 mix. Feasible slots cycle through the 16 cells in diagonal order, `((d, (d+k) % 4) for k, then d)`, so every distractor and obstacle margin stays within one task. Impossible slots are spread over types 1–4, lower types first. Any prefix (`eval run --n`) is a stratified sample. `--feasible` and `--impossible` override the counts together.
- Candidates are consumed one at a time from the D51 bases (dev 0, test 10,000,000). A failed candidate goes to `exclusions.jsonl` and the next seed fills the same slot; the default limit is 2n + 20. Hitting it writes no manifest, so the directory never loads. Task IDs are `{split}_{seed:08d}`.
- `seed_set_hash = sha256("vplan-seed-set-v1:" + sha256(tasks.jsonl) + ":" + sha256(oracle.jsonl))`. Seed-set files are written with LF on every OS, so the hash is OS-independent. Pilot cohorts keep platform newlines, so their file hashes differ between Windows and Linux; the pilot is development-only, so this is left as is.
- `--seeds test` refuses without `--allow-test-split` (spec §6: generated only after the W3.3 freeze). `--seeds S` defaults to `runs/seeds/S`.

**Runner** (`vplan eval run --config N --seeds dev|test [--n N]`, `src/vplan/evaluate/runner.py`):
- Configs: 1 = Nemotron, unverified, k=0; 3 = Nemotron, repair, k=3; 5 = heuristic, repair, k=3; 6 = frontier, unverified, k=0; 7 = Nemotron, resample, k=3. Configs 2 and 4 are refused with their reasons. Config 5 refuses LLM options. Config 6 needs `--model` and `--allow-paid`. The others use the D33 guard. Temperature defaults to 0.7.
- The run directory is `runs/c<config>-<split>-<UTC stamp>/`:
  - `manifest.json`, written once;
  - `sessions.jsonl`, with start and end lines per session;
  - `attempts.jsonl`, one `EvalAttempt` per finished attempt, written only by the parent and fsynced.
  There is no `frames/` yet; replay renders frames from the recorded streams (D41).
- **Key handling:** the parent reads only `LLM_BASE_URL` and `LLM_MODEL`. Each worker reads the key from the env-file path itself, so the key never reaches params, results, logs, manifest or rows.
- **Worker files** are named by task index, plus `.sS` in resumed session S, so a resumed attempt never shares files with an orphaned worker. This is a change from D42's batch index.
- **Resume:** every recorded row counts as completed, including service failures and `sim_error`, so they stay in the denominator (D33); resume never re-runs only the failures. It refuses a changed config, split, mode, k, planner, `--n`, seed-set hash, config hashes or source (unless `--allow-source-drift`).
- **Stop rules** are counted per request, answering D42's open question: a 401/402/403/404 stops at once, and so do three consecutive service-failure requests.
- **Timeouts and pacing:** the per-attempt timeout stays 600 s (four 60 s calls plus simulation). `--max-rpm` turns on a minimum spacing between requests and is off by default: at the R4 median of 34 s per call, a sequential run stays under 2 requests per minute. Exit codes: 0 complete, 1 incomplete, 2 refused, 130 interrupted.

**Acceptance (R10 part 2, Windows Python):**
- A 50-task dev set (38 feasible + 12 impossible) was prepared three times with byte-identical files.
- A config-5 run finished 50/50 with no duplicate or missing IDs: 38 executed (all Judge successes), 9 correct refusals (types 1, 2, 4), and 3 `budget_exhausted` (type 3, repeat plans as D39 predicts).
- A second run was killed after 21 rows and resumed to 50/50 with identical outcomes.

**Known limits, to settle before the final runs:**
- Git SHAs are null in manifests made in a git worktree (Windows git cannot follow the WSL gitdir), so final runs start from the main checkout.
- Nothing stops a second parent from appending to `attempts.jsonl` during a resume.
- A hard kill of the parent orphans the in-flight worker.
- Worker result files duplicate each record: about 37 MB per 50 attempts.
- The Windows interpreter's ManipulaPy metadata says 1.4.1 (stale editable install) against the 1.4.2 pin, and the run manifests record that.

## D54. W3.2 statistics, metrics and report (2026-10-15)

- `wilson(k, n)`: (0.397, 0.892) for k=7, n=10. `wilson(k, 0) = (0, 1)`, and metrics.json reports null rates and intervals when N = 0.
- `paired_bootstrap(a, b, n_boot=10000, seed=0)`: NumPy PCG64, tasks resampled with replacement, 2.5th and 97.5th percentiles. Both gains pair only feasible tasks present in both configs and report the unpaired counts.
- **Input contract:** `records.AttemptRow`, one frozen row per (config, task) of one seed set. Validators tie outcome to record, provider and model to every planner call, and configs 1, 2, 3 and 7 to their mode and k. Config 2 is never run: `first_round` derives it from round 1 of config 3, ending the attempt exactly as `run_task` does at k=0.
- **Headline source:** a Nemotron config counts as Token Factory only when every row's provider host is Token Factory's. Otherwise it is labelled "dry run, not a headline result". Configs 4, 5 and 6 carry fixed limitation labels, and the heuristic (config 5) is highlighted.
- **Wording:** the repair gain (3 − 2) is described as including the extra attempts, never as credit for feedback. Only the feedback gain (3 − 7) compares feedback with retrying.
- **Refusals:** each task's final decision is refused, attempted or no_decision, giving a feasible/impossible × decision confusion matrix. Refusal reasons are scored against the proof's reason by impossibility type (D40).
- **Safety:** counted over every executed attempt with a Judge result. Body pairs are grouped by kind; the Judge's 20-contact cap can undercount pairs but not V1–V4. "Zero observed violations in M executions" appears only when every counted execution carries all four predicates.
- **Blocked plans:** `ik_accuracy` and `continuity` count as solver non-convergence, `visual` as learned visual, and the other checks as physical constraint. Grounding (unknown target, place not over the bin) therefore counts as physical constraint, although it is closer to plan semantics. Malformed output, protocol errors and service failures are counted apart from blocked rounds.
- **Attribution** (spec §10), from the final round:
  - `sim_error` is a verifier/sim issue;
  - a service failure or a missing record is unknown;
  - a false refusal, `malformed_final` or a plan that exhausted the budget is the LLM plan's;
  - a Judge failure after a gate pass is a gate/execution discrepancy, unless a wrong target or an impossible task explains it.
- `report.html` is a pure function of `metrics.json`: charts on an explicit Agg canvas as base64 PNGs, no external resources, "://" escaped, both files written atomically. Units: fractions, seconds, provider-reported tokens.

## D55. R8 local container files; the image build waits for Docker (2026-10-15)

- `docker/Dockerfile`: `python:3.10.22-slim-trixie`, pinned by index digest `sha256:6ff50646…e31d` (it was `python:3.10-slim` on 2026-10-15). linux/amd64 only, because pybullet 3.2.7 has no aarch64 cp310 wheel; it does have an x86_64 one, so no builder stage is needed.
- Dependencies: `docker/requirements.txt` is `uv export --frozen --format requirements.txt --extra dev --no-emit-project` of `uv.lock`, with hashes. pip installs it with `--require-hashes --no-deps --only-binary=:all:`. `tests/test_docker.py` checks that the export equals `uv.lock` and honours every pin in `pyproject.toml`; it will fail when `gradio` lands until the export is redone, which is intended.
- The package runs from `/app/src` via `PYTHONPATH`, so `config/` sits next to the sources as in development. The tree is root-owned, read-only to the non-root user `vplan` (uid 10001).
- `.dockerignore` is an allowlist; env files are dropped at any depth.
- `docker/run-local.sh` runs both containers with `--network none` and never mounts or passes an env file. `VPLAN_IMAGE_DIGEST` is the local image ID, and the git revision is an image label. The image has no git, so manifests made inside it have a null git field.
- **Not yet run:** Docker is not installed (D51). A Linux-native stand-in passed the full suite: uv's CPython 3.10.21 with the locked wheels, on a read-only copy of the build context. That stand-in included the supervisor's kill and timeout tests, which closes D42's "never tested on Linux-native Python". The same stand-in prepared 10/10 pilot tasks, byte-identical to Windows apart from line endings. R8 local stays open until `docker/run-local.sh` has run.

## D56. `vplan report build` over W3.1 runs (W3.2 completed, 2026-10-15)

- `vplan report build --run RUN_DIR [--run …] --output OUT_DIR [--seed-dir DIR]` (`src/vplan/evaluate/collect.py`). W3.1 writes one run directory per config, so `--run` repeats.
  - It refuses runs over different seed sets and two runs of the same config.
  - It loads the seed set from `--seed-dir` or the directory the first run recorded. Source drift is allowed; a config change is refused.
  - It maps each `EvalAttempt` row to an `AttemptRow`.
  - Exit codes: 0 built, 1 built from an incomplete run (titled INCOMPLETE), 2 refused.
  - A Windows-made run records a Windows-style relative seed path, so on Linux pass `--seed-dir`.
- **Run coverage:** metrics.json has a status block with INCOMPLETE and LIMITED title prefixes (D33). `provenance.seed_set` gives the seed set's real size next to the tasks present.
- **Headline status** is per config (headline, dry_run, partial, not_run). A report is a headline report only if configs 2, 3 and 7 are all present, complete, unlimited and on Token Factory. A dry-run config 1 is named in the banner but does not demote the headline. Each chart's stamp names the dry-run configs it shows.
- **Wording:** the feedback gain reads "the comparison that separates the effect of feedback from retrying at the same budget", and config 7 is the "equal-budget control". A test scans metrics.json and the HTML for claim phrases ("credits feedback", "beats", "outperform", …). This departs from spec §14's role column for config 7 ("credits feedback over retrying"); the spec text and the §1 headline wait for the owner (week3.md).
- **Privacy:** `public_manifest` drops local paths (seed directory, resolved paths, the ManipulaPy checkout location) and any absolute path before a manifest reaches metrics.json or the HTML. Hashes, versions and git SHAs stay.
- **Guards:** `AttemptRow` checks its distractor and obstacle counts against the scene. A 12-case test pins `first_round` (config 2) to `run_task(k=0)` on the same scripted planner. Mistyped run-manifest fields are refused.
- **Checked from the main checkout:** a 10-task dev seed set, a config-5 run (10/10) and `report build` produced a self-contained report with no URLs or local paths, with the git SHA recorded. Suite: 832 passed.
- **Open, before the final runs:**
  - Spec §14 runs on the held-out `test` seeds, but a complete Token Factory run on `dev` still gets headline status.
  - Each row keeps the full replay streams, about 1.2 MB per executed row, so a 400-task report needs 1.5–2 GB of RAM.
  - D33's "DEGRADED: n service failures" prefix is missing.
  - An `--output` that names an existing file ends in a traceback.
  - A configuration change after W3.3 makes `report build` refuse older seed sets.

## D57. Token Factory credit is reserved for the demo (2026-10-15)

- **Owner's ruling.** The hackathon program accepted the owner's application and grants **$25** of Token Factory credit (one grant, not the "$25 + $25" D30 assumed). It is kept for the demo only.
- **Estimated spend** at the listed $1.00 / $3.00 per 1M tokens (D30; confirm in the console). A plan uses about 1.1–1.8k completion tokens (D13) plus roughly 4k input tokens, so a call costs about one cent.
  - W3.0 smoke test (about 20 calls): about $0.20.
  - Demo: up to 4 calls per attempt, so even 50 recorded attempts stay around $1–2.
  - One full `test` pass of configs 1, 3 and 7 (up to about 3,600 calls): roughly $15–30. It does not fit next to the demo.
- **W3.0 is unblocked:** the smoke test now protects the demo (forced tool calls on Ultra are still unverified on Token Factory, D13). It runs only with the owner's OK and after the model ID is re-checked in the catalog (model churn).
- **Conflicts to settle (owner):**
  - Spec §14 says headline numbers come from Token Factory runs only, and D56 labels everything else a dry run. With the evaluation off Token Factory, either the headline source changes (for example, the same model on OpenRouter with the provider named), or the report carries no headline.
  - Spec §18 runs the final `test` batch on Serverless Jobs with Nemotron on Token Factory. The credit is for tokens; Jobs compute is unfunded, so the batch would run locally or in the R8 container.
  - The OpenRouter free variant's daily cap may not allow about 3,600 calls in one day (W3.6).

**Addendum (2026-10-15): no Serverless Jobs.** The owner ruled that the final `test` batch runs locally or in the R8 container (D55), not on Nebius Serverless Jobs, because the credit covers tokens only.
- The W3.4 Jobs smoke test and R8's "results return via Object Storage" are dropped. R8 local (the container run) stays: it is now the reproducible environment for the batch.
- The hackathon requirement ("must run on Nebius Token Factory or Nebius AI Cloud") is met by Token Factory serving Nemotron for the live demo and the W3.0 smoke test. The rules suggest Serverless Jobs; they do not require them.
- Spec §18's final-stage line is superseded by this entry.

## D58. The Codex close-week2-gaps fixes, ported (2026-10-15)

The owner's Codex worktree held four uncommitted fixes written against `dcb5d7b`. They were exported with line endings normalised, ported onto `main`, and reviewed against the current code. The Codex worktree itself was not touched.
1. **Cosmos outage warning (D47).** The CLI gate wrapper prints and flushes "COSMOS UNAVAILABLE - proceeding on the geometric checks only" as soon as the gate reports unavailable, before execution. The result block prints only "cosmos gate: unavailable" and never claims "executed" for an attempt that ended as `stale_approval` or `sim_error`.
2. **Malformed streak (amends D40).** "Two malformed replies in a row" now means adjacent rounds: any non-malformed round, including a protocol error, resets the streak. After a protocol error, the next request uses another protocol and carries that round's errors, so it is not a schema retry of the earlier reply.
   - Only one sequence changes outcome: a k=3 attempt whose rounds 0–2 are malformed, protocol error, malformed. It used to end as `malformed_final` after 3 calls; now round 3 decides.
   - This applies to `run_task` and to both pilot arms (shared `_continue`), superseding D40's "pilot behaviour unchanged" for that sequence. None of the 256 saved pilot and W3.1 sequences contains a protocol error, so no saved result changes.
   - `metrics.first_round` (config 2) is unaffected; a 13th case pins it to `run_task(k=0)`.
3. **Config-1 budget (amends D48's "run_task still defaults to k=3").** `run_task`'s k defaults to 0 for unverified and 3 otherwise. Unverified with k ≠ 0, or a k that is not a real int, is refused before any planner call. The W3.1 runner already passed k explicitly.
4. **Rotated-grasp rendering (amends D47's simplification).** In rendered pre-release frames, each finger opens to the object's half-extents projected on the TCP closing axis, which is `capture_check`'s half-width, instead of min(size x, size y)/2. This matters now that frames feed the shadow study (D59).
5. **Benchmark provenance.** `scripts/spikes/cylinder_contact.py` output gains a provenance block: git, versions, config snapshot and hashes, seeds, source hashes and a ManipulaPy source digest. The absolute ManipulaPy path Codex recorded is dropped, because this output is committed under `docs/evidence` (D56: no local paths).

## D59. Evaluation provenance hardening (2026-10-15)

Prompted by the owner's review and the open items of D56:
- **Headline split (spec §14).** A report gets headline status only if its seed set is `test`, on top of D56's rule (configs 2, 3 and 7 present, complete, unlimited, on Token Factory). Otherwise the reason "development seeds, not the held-out test set" is named. `report build` also refuses a run whose manifest split differs from the seed set's.
- **Git provenance never silently null** (`gitinfo.git_provenance` → {head, dirty, source}). Sources in order:
  1. the git command;
  2. the repository files: the `.git` file's gitdir with `/mnt/<d>/` → `<D>:/` on Windows, commondir, loose refs, packed-refs; dirty is null;
  3. `VPLAN_GIT_SHA` / `VPLAN_GIT_DIRTY`, for the vplan checkout only, set by `docker/run-local.sh` from the host.
  - `eval run` refuses to start or resume without a SHA unless `--allow-no-git`, which is recorded.
  - Seed-set preparation and the pilot record the same dict but do not refuse.
  - This resolves D53's null SHAs in worktrees. In a worktree the dirty flag stays null, so final runs start from the main checkout.
- **Frames for the shadow study (supersedes D53's "no frames/ yet").** Every executed round of config 3 (by default; `--frames`/`--no-frames` override, and resume refuses a change) gets 9 key frames: pre-grasp, grasp and pre-release × front-oblique, top and side, 512×512.
  - They are rendered by the worker after the result is final, from the rebuilt trajectory. It must reproduce the recorded approval token, or the export fails.
  - Files go to `frames/<attempt stem>/` with an `index.json` holding sha256s and the scene, plan and approval hashes.
  - A failed export is recorded and never fatal. About 1.3 s and 0.18 MB per executed attempt.
  - `vplan eval shadow` (week 4) reads these. Its state-predicate truth comes from the seed set's oracle at shadow time.
- **Report wall time** excludes the frame export (`EvalAttempt.wall_s − render_s`), so config 3's latency is comparable with configs 5 and 7. `attempts.jsonl` keeps the supervisor's raw time.
- **D42 amendment (confirmed by the owner, Oct 16).** A job may return an after-step that the worker runs once the result file is written. A try that times out after writing its result keeps that result instead of becoming `worker_timeout`.
- **D56 items closed:**
  - the D33 "DEGRADED: n service failures" title prefix, with failures by HTTP status;
  - `report build` refuses (exit 2) an `--output` that is a file, and a run manifest whose n disagrees with the seed set;
  - report rows drop the replay streams that metrics never reads: peak working set 158 → 90 MB for 20 rows, with metrics.json byte-identical.
- **Windows file lock.** On `report build`, `os.replace` failed in 2 of 5 back-to-back builds with WinError 32 (a scanner holding the new file). Replacing now retries five times (0.05–0.8 s apart) on `PermissionError` only. 8 of 8 back-to-back builds then succeeded.
- **Still open:** `vplan eval shadow` (week 4); `report build` refuses seed sets prepared before a later configuration change (relevant after W3.3); the cosmetic R-in-violations asymmetry (D52).

## D60. Lightning vs Ultra on OpenRouter (free variants, 2026-10-15)

Measured before choosing the evaluation model under the $25 Token Factory budget (D57). No paid calls were made.

**Hard pilot cohort** (`runs/pilot-hard-20a`, the same 20 feasible tasks as D46, temperature 0.7):

| | Ultra (D46) | Lightning |
|---|---|---|
| First attempt | 9/20 | 0/20 |
| Repair (≤ 4 calls) | 17/20 | 9/20 |
| Resampling (≤ 4 calls) | 17/20 | 1/20 |
| Repair only / resample only | 2 / 2 | 8 / 0 |
| Calls (repair / resample) | 32 / 41 | 66 / 78 |
| Repeated plans (repair / resample) | 0 / 3 | 3 / 8 |
| Provider 503s | 3 | 0 |

- Lightning's plans mostly collide during transport or release too high at the place; resampling hit the release check 23 times.
- With Lightning, feedback separates clearly from resampling (9 vs 1); with Ultra it does not (17 vs 17). This is a 20-task development pilot: a hint for the model choice, not a result.

**20 mixed dev tasks** (`runs/cmp-1015`, 12 feasible + 8 impossible, config 3):
- Lightning: 12/20 judged successes. Feasible 7/12 (first try 3/12). Impossible: out of reach 2/2, bin out of reach 1/2, enclosed 0/2, absent referent 2/2. 50 proposed calls, 1 protocol error, no repeats. Median 2.4k completion tokens and 29 s per call.
- Ultra: never completed. The free endpoint answered 503 to most calls across seven attempts over about an hour; the partial runs stay under `runs/cmp-1015/c3-ultra*`.

**Open (owner):** the evaluation model (Lightning, Ultra, or Lightning for the evaluation and Ultra for the demo).

**Addendum to D55 (2026-10-16): R8 local done.** The owner installed Docker Desktop (29.8.2) with WSL integration, instead of the `docker.io` package D51 assumed; it is free for personal projects.
- The first `docker/run-local.sh` run passed 950 tests and failed 2. Both failures were in test setup and packaging, not in the evaluation code:
  - `test_gitinfo` read the host's `VPLAN_GIT_DIRTY=true`, which `run-local.sh` passes in; the test now clears it;
  - `test_metrics` parses `docs/spec.md`, which the image did not ship; the image now copies that one file.
- **After the fixes** (image `sha256:f52b395d059b…`, 1.34 GB, built from `d90e968`):
  - in the container: 953 passed, 1 skipped (the test that needs `.git`);
  - 10/10 pilot tasks prepared;
  - `vplan eval prepare` (10 tasks) and `vplan eval run --config 5`: 10/10 attempts (6 executed, 3 refused, 1 budget exhausted, 9 judged successes), with no network and no env file;
  - the manifest records the host commit (source "env", clean), the image ID and Linux Python 3.10.22.
- **Cross-platform determinism:** the container's seed set has the same hash (`3e0bb043…`) and outcomes as the same 10-task set prepared and run on Windows on Oct 15.
- Windows suite on the merged code: 954 passed.

## D61. σ_thr calibrated on unbiased dev witnesses: 0.11 (W3.3 steps 1–3, 2026-10-16)

- **Rule** (spec §9 says only "1st percentile, rounded down"; this fixes the details):
  - the sample is every accepted witness of the dev seed set's feasible slots, filled exactly as `vplan eval prepare --seeds dev` fills them (D51), with both `sigma_min_threshold` values set to 0 for generation and acceptance, so there is no selection bias;
  - Verifier minimum: the minimum of `verify.checks.sigma_min` (the V-4 quantity) over every sample of the rebuilt trajectory, which must reproduce the executed approval token;
  - Judge minimum: `min(ExecutionRecord.sigma_min_steps)` (the V3 quantity);
  - σ_thr = `numpy.percentile(verifier minima, 1, method="linear")` floored to 0.001. The Judge threshold is min(σ_thr, the same statistic over the Judge minima), so the Judge never fails a witness the Verifier admits.
  - With n = 75, the floored P1 rejects at most the smallest witness.
- **Result** (`vplan eval calibrate --seeds dev`, evidence in `docs/evidence/sigma-calibration-dev.json`): 75 witnesses, no exclusions.
  - Verifier minima: min 0.0951, P1 0.1109, P5 0.1603, P50 0.2883. The Judge minima agree to within 1e-4.
  - **σ_thr = 0.11 for both the Verifier and the Judge** (was the provisional 0.02, which filtered no dev witness).
- **Re-validation:** dev prepared under 0.11 needed 103 candidates.
  - 1 of 76 witnesses was rejected by V-4: seed 13, 0.10986 in descend.
  - 2 draft rejections (`no_room_for_obstacle`) don't depend on the threshold.
  - Because seeds are consumed in order (D53), that one rejection shifts every later slot: dev tasks 13–99 change content.
  - The heuristic (config 5) then scored 100/100 attempts: 75/75 feasible executed with Judge success, 19/19 refusals of types 1, 2 and 4, and 6/6 type-3 tasks `budget_exhausted`. No singularity finding came up.
- **Consequences:** every seed set, cohort and run prepared before this commit has stale config hashes and is refused by `eval run`/`report build`. The generator is not frozen and `test` is not generated (W3.3 steps 4–5, kept for last). Suite: 987 passed.

## D62. Lightning for the evaluation, Ultra for the demo (2026-10-16)

**Owner's choice** (from D60's comparison and the $25 Token Factory budget, D57):
- **Evaluation** (configs 1, 2/3 and 7, the dry run and the final `test` runs): **Nemotron 3.5 Lightning**.
  - Development and dry runs: OpenRouter `nvidia/nemotron-3.5-lightning:free` (tools and tool_choice; no 503s in about 200 calls on Oct 15).
  - Final: Token Factory, catalog ID `nvidia/Nemotron-3_5-Lightning` (tagged "function calling" when checked on Oct 4). Re-check it in the catalog before first use, because Token Factory removes models without rerouting.
  - About $2–7 per full `test` pass, so the evaluation fits the credit with room for reruns.
- **Live demo:** **Nemotron 3 Ultra** on Token Factory (`nvidia/Nemotron-3-Ultra-550b-a55b`), about $1–2 for the recording.

**Consequences:**
- **Headline source** (open since D57): the headline stays Token Factory only, as spec §14 says. A report is a headline report when configs 2, 3 and 7 ran complete on Token Factory Lightning over `test` (D56, D59). No rule change.
- **The demo shows a different model** from the one the numbers describe. The README and the video must say which model each part uses.
- **W3.0 smoke test** (still the owner's OK, D57): forced tool calls on Token Factory for both models, about 20 calls each (about $0.01 for Lightning, $0.20 for Ultra).
- **Superseded:**
  - D30's single planner model, for the evaluation only. Development history, R4 (20/20) and the pilots (D35, D46) stay as recorded on Ultra.
  - Spec §7's NemotronPlanner lines.
- **Prompt:** Lightning's failures on the hard cohort were mostly transport collisions and releases above the limit (D60). Any prompt change must reach every config equally and happen before the freeze, so no `test` result is tuned.

## D63. The 60 s per-call timeout is wall-clock time (2026-10-16)

**Found:** in a free Lightning run on Oct 16 (config 3, 20 dev tasks), single planner calls took 83, 100, 178 and 225 s although the client was built with `timeout=60.0`.
- The SDK timeout bounds each wait for data, not the whole call. A call that ended after 178 s means OpenRouter kept sending bytes while it waited, so the timeout never fired.
- One attempt then ran into the 600 s worker limit (D42) and was recorded as `sim_error` (`worker_timeout`), which the report counts as a verifier/simulator issue. A slow provider was scored as our fault.

**Ruling:** spec §10's "per-call timeout 60 s" is wall-clock time for the whole call.
- `NemotronPlanner` runs each request on a daemon thread and stops waiting after `call_deadline_s` (default `CALL_DEADLINE_S = 60.0`). Past it, the call is a `service_failure` with `error_type` `CallDeadlineExceeded`. The request is abandoned, never retried, and its late reply is dropped.
- The SDK's 60 s timeout stays as a guard on each read. It does not bound a request the provider keeps alive, so an abandoned request lives until the provider finishes:
  - in an eval worker (one subprocess per attempt, which stops at its first service failure) it dies with the process; a daemon thread never holds up the exit;
  - in a long-lived process (`pilot`, the coming demo UI) it keeps its thread and connection until the provider is done, and may overlap the next request.
- With at most 4 calls of 60 s each, the 600 s attempt limit again catches only real hangs. One exception: `--max-rpm` below 1 makes the pacing wait (outside the deadline) longer than 60 s per call, so four calls can still reach 600 s.

**Consequences:**
- **Lightning fits:** past free calls had a median of 25–29 s, and about 12 of about 720 took longer than 60 s (pilot-hard-20a, cmp-1015).
- **Ultra often does not:** on the free endpoint the hard pilot's median was 69 s, and about 180 of 292 calls took longer than 60 s. Those would now be service failures. The pilots (D35, D46) ran without a wall-clock limit and stay as recorded.
- **Demo:** the live demo uses Ultra on Token Factory (D62). The W3.0 smoke test must record Ultra's latency there. If it routinely needs more than 60 s, a longer demo-only deadline is a new decision; the evaluation stays at 60 s.
- **Stop rule:** a deadline failure is a service failure, so it counts toward the runner's and the pilot's "3 consecutive service failures" stop. A slow endpoint stops a run early instead of filling it with timeouts.
- An abandoned request may still be billed by a paid provider.
- Runs made before this change (such as the cmp-1016 comparison at 708235d) can still hold slow-call `worker_timeout` sim errors; compare them with those attempts excluded.
- Not changed: the demo-only Cosmos client (`visual/cosmos.py`) keeps the SDK timeout; it talks to a local vLLM server.

## D64. A plan sent as a JSON string is decoded once (2026-10-16)

**Found:** Nemotron Super sends `{"plan": "<the plan object as a JSON string>"}` instead of `{"plan": {…}}`.
- 10 of 10 calls did it: free Super 3 on OpenRouter, and Super VL on GMI Cloud.
- It persisted after the schema retry ("plan: Input should be an object", spec §7), so every attempt ended `malformed_final` although the plan inside was well formed.
- No Lightning or Ultra reply in any recorded run had this form.

**Ruling (owner's OK, Oct 16):** before validation, the parser decodes a string `plan` once when it holds a JSON object, in the tool-call arguments and in the content fallback.
- Anything else stays as sent and is validated as before: a string that is not JSON, JSON that is not an object, a plan encoded twice, and an object that still fails the schema are all `malformed`.
- Replies with an object `plan` take exactly the old path.
- The evidence keeps the reply as sent (`raw_response`), so a decoded reply can always be recognized in the records.

**Consequences:**
- It applies to every model and config alike, before the freeze. No recorded Lightning or Ultra result changes, and replay does not re-parse old replies.
- Re-parsing the 4 recorded Super replies (run `cmp-1016/c3-super-1`) gives 4 valid pick-and-place plans.
- Super becomes usable as a free development model (99.9 % uptime on OpenRouter on Oct 16). Its results never stand in for Lightning's (D62), and free-tier daily caps still apply.

## D65. A longer call deadline for dev runs only (2026-10-16)

**Found:** with D63's 60 s deadline, the free endpoints were too slow to test on Oct 16 evening.
- Free Lightning: 3 of 3 calls past 60 s in each of two runs (OpenRouter still showed 100 % uptime, which counts errors, not latency).
- Free Super: 6 of 12 calls past 60 s; its successful calls took 45–55 s.
- Every run stopped after 3 consecutive provider failures.

**Ruling (owner's choice, Oct 16: "Make it 180"):** `vplan eval run --call-deadline S` sets the per-call deadline.
- The default stays spec §10's 60 s. Any other value is refused on the `test` split, so headline runs always use the spec's deadline.
- The manifest records it (`planner.call_deadline_s`), resume refuses a change, and a manifest from before D65 counts as 60 s.
- The heuristic config refuses the option, like the other LLM options.
- Without `--timeout`, the attempt limit is max(600 s, (k + 1) × deadline + 360 s): 1080 s at 180 s for configs 3 and 7, unchanged at 60 s. An explicit `--timeout` stands.
- The SDK's own timeout (a limit on each wait for data) becomes max(60 s, deadline). A server that sends nothing until the reply is done, as GMI Cloud and AIHubMix did, would otherwise still fail the call at 60 s.
- Dev runs on the free endpoints use 180 s from now on.

**Consequences:**
- Success rates and provider failures at 180 s are not comparable with runs at 60 s: at 60 s, replies that take 60–180 s count as failures. Only the plans that arrived compare.
- `report build` does not check that its runs share one deadline (nor one model or temperature). Build reports only from runs made with the same `--call-deadline`; the deadline shows in each run's provenance. Headline reports are safe because they need `test`.

**Addendum (2026-10-18): the W3.6 dry run on free Lightning.** The owner ruled that the dry run stays off Token Factory and uses free Lightning on OpenRouter. At 60 s free Lightning hardly answered (D65 above), and the dry run runs on `test`.
- The refusal now applies only to a **Token Factory run on `test`**, the only kind that can be a headline result (spec §14, D59). Any other run on `test`, such as the dry run, may use `--call-deadline 180`. Dev runs are unchanged.
- This supersedes the title's "dev runs only" and the consequence that headline reports are safe because they need `test`: they are safe because they need a Token Factory run on `test`, which keeps 60 s.
- Expect the dry run to take several sessions: the free daily cap is 50 or 1,000 requests depending on credits ever bought, and resume keeps every provider failure in the denominator.

## D66. The W3.5 Gradio UI: live and keyless replay (2026-10-17)

**Dependency (D22: approved; owner's OK for W3.5 on Oct 16):** `gradio==6.30.0`.
- It is the newest release on PyPI (2026-10-08) and requires Python ≥ 3.10.
- Its requirements fit the pinned stack: pydantic ≤ 3.0, numpy < 3, pillow < 13, anyio < 5 and typing-extensions ~= 4. Its `httpx` is a separate package from openai's `httpx2`, with no module or script name in common.
- `pyproject.toml` pins it; `uv.lock` and `docker/requirements.txt` are regenerated.
- On the development interpreter, `pip install --dry-run gradio==6.30.0` would only add 20 new packages and change none, so it was installed. Among them: gradio-client 2.7.2, fastapi 0.143.3, starlette 1.7.0, uvicorn 0.54.0, pandas 2.3.3, httpx 0.28.1 and orjson 3.13.0.
- The lock picks the same versions for those 20. For three packages the interpreter already had, the lock picks newer versions (filelock 4.1.0, fsspec 2026.9.0, markupsafe 3.0.4); pip leaves the installed ones alone, so these differ between the container and development.

**What it is:** `src/vplan/ui/`, started with `vplan ui`.
- Module split: views (view models), markup (escaped HTML), render (PyBullet images), live, sources (replay readers), display (panel contents) and gradio_app (the Gradio layout).
- The layout module is not named `app.py`: in the R8 container the repository root is `/app`, and two provenance tests (`test_calibrate`, `test_cylinder_contact_benchmark`) reject any evidence text that contains that path as a plain substring, so `src/vplan/ui/app.py` failed them there.
- Only `ui.gradio_app` imports Gradio. Planner, Verifier, executor, Judge and runner are unchanged; live mode only listens to `run_task`'s events.

**Live tab:**
- **Tasks:**
  - the basic demo scene, with or without its wall, and a typed template instruction. It must name exactly one object, and the heuristic witness must solve it (the spec §6 rule for feasible tasks); otherwise the task is refused before any planner call;
  - or `generate_task(seed, kind)`: feasible, or impossible types 1–4 with their proofs.
- **Planners:**
  - the template heuristic, always available, with no key;
  - Nemotron, built once from `--env-file` behind the D33 guard. It uses temperature 0.7 as the evaluation does and the spec's 60 s call deadline (D63). A longer demo deadline is still undecided (D63/D65).
- **Attempt:** one `run_task` repair attempt, k = 0–3. A plan appears as soon as the planner returns, and each round's check timeline once the round finishes.
- **Cosmos:**
  - with `--cosmos-url`, the D47 gate runs. The frames it was sent are shown with its verdicts, and an outage banner reaches the page before motion starts (D58);
  - a gate that asked no stage (`skipped`: the instruction names no single object) is shown like no gate, on the geometric checks only;
  - without a URL, a banner says approved plans execute on the geometric checks only (D22).
- **Recording:** every attempt is saved as `runs/ui/ui-<UTC>-NN.json` (task, oracle, seed, `AttemptRecord`).
  - The file is created exclusively, so recordings never overwrite each other. A Windows clock tick once let two quick attempts share a name; a fixed-width counter fixed it and keeps newest-first order.
  - A recording whose text contains the API key is refused.

**Check timeline:** one row per V-1 … V-8 and round, with five statuses:
- `fail`: the Finding, with its bodies and measured value against the threshold, for example "cube_red ↔ wall_1: clearance 0.94 cm (minimum 1.00 cm) in transport, sample 279".
- `partial`: nothing was reported, but the segment checks stopped at an IK or continuity failure.
- `unknown`: the check was not reported because of the five-finding cap, or the round ended in a simulation error.
- `off`: the check did not run (unverified mode, or an unknown target).
- `pass`: none of the above.

All model and provider text is HTML-escaped.

**Highlighting the offending body (feasible with the existing renderer):**
- How a finding is drawn:
  - It is re-posed in a private DIRECT client from the plan's rebuilt trajectory, with the attempt's seed and verifier config.
  - Its segment is identified by stage and plan field together, because transport segments share a label. Its sample comes from `sample_index`, with the carried object attached as the gate attaches it.
  - The bodies it names are recoloured magenta with `changeVisualShape`. V-3 highlights the link the named joint moves, V-7 the object, and V-8 the object and the bin.
- Each finding gets an overview and a close-up aimed at the highlighted bodies, as two square images.
- **Limitation:** the picture is re-posed, not recorded. For an executed round the approval token proves that the rebuild is the executed trajectory; for a blocked round there is no token, and the rebuild is deterministic only for the same code and configuration. Replay notes when the configuration hashes differ.

**Execution:**
- The executed round's recorded 60 Hz streams are played back with `replay.replay` (no physics) and rendered at 10 fps into an animated GIF, shown as a data URI.
- It is not the PyBullet GUI window.

**Replay tab:**
- **Sources:**
  - W3.1 run directories: the seed set must match the run's seed-set hash; `--seed-dir` covers a moved seed set;
  - pilot runs: the cohort's files must match its manifest;
  - UI recordings, or a folder of them.
- No configuration or source-drift check refuses a replay, because it is display, not evidence. `replay.replay`'s scene-hash check still ties each played-back record to its scene.
- **Offline:** replay needs no `.env`, keys, GPU or network.
  - `GRADIO_ANALYTICS_ENABLED=False` is set before Gradio is imported, so telemetry and the version check are off.
  - The theme uses system fonts.
  - The server listens on 127.0.0.1, and `vplan ui` warns on any non-loopback `--host`. From WSL in NAT mode, `--host <WSL gateway IP>` reaches it; `0.0.0.0` would expose the planner to the LAN.
  - `share` and `mcp_server` are passed as `False`, so `GRADIO_SHARE` cannot open a public link and `GRADIO_MCP_SERVER` cannot expose the handlers.
- **Paths:** the tab reads only under the working directory, `--record-dir`, `--replay` and `--seed-dir`. This includes the seed set or cohort a run points to.
  - Network (UNC) paths are always refused, because opening one on Windows sends the user's credentials to that server.
  - A damaged row, manifest or recording is refused with a message.

**Storyboard (spec §3) covered:** steps 1–9.
- Plan, Verifier rejection with the offending body and clearance, repaired plan, Cosmos frames and verdicts, execution, Judge, an impossible task with its correct refusal, and keyless replay.
- Step 10 (headline numbers with intervals) is `vplan report build`'s HTML, not part of the UI.

**Verified:**
- 72 new tests (`tests/test_ui_*.py`), with no network and no LLM calls. They cover:
  - view models and escaping;
  - heuristic attempts with real physics;
  - highlighted renders (magenta pixel counts);
  - the outage banner arriving before `execute`;
  - Cosmos frames with a faked endpoint;
  - a fake provider client for the LLM path, with the key absent from every output;
  - replay from W3.1, pilot and UI sources;
  - CLI wiring, including a `--replay` start with no env file read.
- A headless Chromium walkthrough of the running app: replay of a repair recording and a rejection recording, a live heuristic success, and a live type-1 refusal.
- Repair rounds were shown with a scripted test double, not a model.
- **R8 container** (`docker/run-local.sh`, rebuilt with gradio; image `sha256:ced5f3a9a92a…`, 1.61 GB, up from 1.34 GB):
  - 1099 passed and 2 skipped: the test that needs `.git`, and the Windows-only `/mnt/` path refusal;
  - 10/10 pilot tasks prepared, with no network and no env file;
  - the first build failed 2 tests because of the `app.py` name, fixed by the rename above.

**Not verified or deferred:**
- Live Nemotron in the UI (no model calls were made).
- A live Cosmos endpoint (faked in tests).
- The demo call deadline.
- A report tab for step 10.
- Use by several people at once. Simulator work is serialized by a process-wide lock.
  - A live attempt's worker thread holds it until the attempt ends, even after its page closed; Gradio frees an event's concurrency slot on disconnect, so the slot alone would let a second simulation start beside the first, and the motion layer reseeds numpy's global RNG.
  - Meanwhile another attempt, a scene load or a replay is refused with a "simulator is still running" message.

## D67. The tool schema requires each plan's `kind` (2026-10-18)

**Found** (free Super run `cmp-1016/c3-super-v3`, Oct 16): Super left out `kind` in 6 of its 13 replies over tasks 0–6, and 2 of its 12 feasible tasks ended `malformed_final` for that reason alone.
- The cause was ours. `PickPlacePlan.kind` and `Refusal.kind` have defaults, so pydantic's JSON schema leaves them out of `required`, and the tool schema told the model the tag was optional.
- The parser needs that tag to tell a pick-and-place plan from a refusal (a discriminated union, spec §7), so a reply without it is rejected.
- The schema-retry message ("Unable to extract tag using discriminator 'kind'") did not help Super reliably.

**Ruling (owner's OK, Oct 18):** `planners/nemotron.plan_schema()` builds the schema sent with every request, in the tool definition and in the content fallback's prompt. It is `SubmitPlanArgs.model_json_schema()` with `kind` added to `required` for every plan type in the discriminator's mapping.
- Nothing else changes: the records, the parser (a missing `kind` is still malformed, D64) and every other part of the schema.
- A fresh schema dict is built per request.

**Consequences:**
- Every model's request changes by that one word, so this is a prompt change. D62 allows it only before the freeze and for all configs equally; both hold. It must land before `test` is generated.
- Runs before and after this change do not share a byte-identical prompt. The pilots (D35, D46), D60 and the Oct 16 comparison runs stay as recorded.
- Do not resume a run started before D67 with `--allow-source-drift`: it would mix the two schemas in one run.
- R4 (D13, forced tool calls 20/20) was measured with the old schema. A provider that rejected the new one with a 400 naming the tool would push the planner from the forced protocol to the fallbacks without failing. The W3.0 smoke test, or the first free Lightning call, must confirm the protocol stays `forced` and the reply carries `kind`.
