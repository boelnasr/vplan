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
4. **`place.yaw_deg`** is the gripper's closing-axis yaw at release, the same convention as `grasp.yaw_deg`. The LLM prompt (W2.3) must state it.
5. **Stricter validation in records:** NaN/Inf rejected, colours a Literal, reserved body ids rejected.
6. **Approval binding** now also covers the target, the applied physics config (hash on the handles) and arm/object drift. Attach-rejection reasons are recorded.
7. **Deferred to W2.2:** `vplan.cli` still imports the basic scene from `tests/conftest.py`. The scene builder moves into `src/vplan/tasks/`.

**Noted for later:**
- `pybullet_data`'s Panda URDF has a wider joint-4 range ([-3.1416, 0]) than the real Franka. This goes into the README limitations (W3.6).
- An untracked `uv.lock` and an empty `.venv/` appeared from a tool outside the workflow. They're left untouched.

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
