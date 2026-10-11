# Spec: Verified LLM Manipulation Planning

**What it is:** a Nemotron-planned, physics-verified pick-and-place system in simulation.
**Event:** Nebius x NVIDIA Global AI Hackathon, Physical AI track.
**Deadline:** 2026-10-30, 10:00 PDT.
**Status:** design approved 2026-10-03. This spec consolidates `docs/decisions.md` (D1–D24); decision IDs are cited in brackets.
**Working package name:** `vplan` (rename freely before the first commit).

---

## 1. Summary

**What it does**
- A user gives a plain-language instruction ("put the red cube in the bin").
- **Nemotron** on Nebius Token Factory proposes a typed pick-and-place plan [D9, D16].
- A deterministic **verifier** built on ManipulaPy and PyBullet checks the planned trajectory before anything moves: reachability, joint limits, singularity, collisions, grasp preconditions [D17].
- Failed plans return to Nemotron with structured evidence, for up to 3 repair rounds [D18].
- **Cosmos 3** adds a learned visual consistency check in the demo [D5, D20].
- Approved plans execute in PyBullet under position control [D21].
- An independent **Judge** scores the executed run [D21].
- Batch evaluation runs on **Nebius Serverless Jobs** at the final stage [D6, D19].

**Headline claim (fill in with measured values) [D8, D23]**

> "Bounded repair with verifier feedback raised Nemotron's task success from X% (verified first attempts) to Y% across N held-out, reference-solvable generated manipulation tasks."

**Supporting results:**
- the margin over equal-budget resampling;
- refusal accuracy on impossible tasks and false refusals on feasible ones;
- safety violations observed in execution;
- the heuristic baseline;
- the Cosmos shadow pilot.

**Target user [D23.11]:** robotics developers who review LLM-proposed manipulation plans. They need to see *why* a plan is unsafe or wrong, and whether a repair fixed it.

## 2. Scope

**In scope**
- **One task family:** tabletop pick-and-place into one bin with a Franka Panda, in simulation [D3, D4].
- Grasping is an **attachment-assisted** kinematic constraint, documented as a simulation simplification [D3, D10].
- Varied referring expressions with same-shape and same-color distractors [D9].
- Impossible tasks are included, and the correct behavior is refusal [D15].

**Out of scope**
- Real hardware.
- Contact-based grasp reliability.
- Throwing, pushing or dropping objects from above the release-height limit [D23.3].
- Ambiguous instructions [D15].
- Automatic obstacle avoidance in the motion layer [D9, D17].
- Bitwise reproducibility across machines [D10].

**Stretch, only if week 3 finishes early [D3, D24]:**
- 2–3 task types;
- the frontier reference run (config 6);
- ambiguous-instruction handling.

**Week-1 fallback [D3]:** reach with obstacle avoidance, no grasp. It is activated if grasping is unreliable by 2026-10-09.

## 3. Demo storyboard [D23.11]

1. Choose a scene and instruction.
2. Nemotron's first plan appears.
3. The Verifier rejects it, showing the **offending link or object and the measured clearance**.
4. The repaired plan passes the geometric checks.
5. The Cosmos frames and verdicts are shown.
6. The plan executes in PyBullet.
7. The Judge reports success.
8. Second clip: an impossible task, and a **correct refusal**.
9. Third clip: **keyless replay** of a recorded attempt.
10. Closing: the headline numbers with intervals, and the heuristic baseline shown prominently.

## 4. Environment and dependencies

**Local development [D14]**
- The owner's Windows Python 3.10.11, called from WSL, with `PYTHONUTF8=1` and `WSLENV=PYTHONUTF8`.
- No project venv for now.
- All randomness and paths must work on Windows and on Linux.

**Jobs container [D14, D22]**
- `python:3.10-slim` (Linux) with pinned dependencies, CPU only [D11].

**Dependencies (all pinned to exact versions)**
- `ManipulaPy[simulation]` at a tagged release [D2]. For local development the upstream fixes come from §17.
- `pybullet` 3.2.7.
- `numpy`.
- `pydantic` v2 [D16].
- `openai` [D16].
- `gradio` [D22].
- `matplotlib`, already present.
- `pytest`.

**Secrets:** `.env` holds `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `COSMOS_BASE_URL` and the Nebius settings. A `.env.example` is committed; `.env` is never committed [D7].

**GPU:** optional only (`--gpu`). Local CuPy and Numba-CUDA need a CUDA 12.8+ toolkit, so the CPU path is the default [D11, D14].

## 5. Architecture [D10, D22]

### 5.1 Paths

- **Demo:**
  `TaskGenerator → Planner → MotionLayer → Verifier → VisualGate (Cosmos, live) → Executor → Judge`.
  - A Verifier failure or a Cosmos `fail` returns to `RepairLoop`.
  - Every revised plan repeats motion generation and all checks.
- **Evaluation (Jobs):**
  `TaskGenerator → Planner → MotionLayer → Verifier → Executor → Judge → AttemptRecord`, with key frames saved.
  - Cosmos has no influence on this path.
- **Shadow (local, afterwards):** saved frames and records → Cosmos → `shadow.jsonl` → Report.

### 5.2 Modules and records

| Module | Input → output | Key rule |
|---|---|---|
| `tasks` | seed → `TaskSpec`, `TaskOracle` | `TaskOracle` is never visible to planners. |
| `planners` | `TaskSpec`, history → `PlannerResult` (`PickPlacePlan` or `Refusal`, raw I/O) | Nemotron and heuristic are core. Frontier is optional. |
| `motion` | plan, scene → `CandidateTrajectory` | Straight Cartesian segments, no obstacle avoidance. |
| `verify` | trajectory, scene → `list[Finding]` | Preview PyBullet client, kinematic only. |
| `repair` | findings → next planner call | Stateless prompts, k = 3. |
| `visual` | key frames → `VisualVerdict` | Live gate in the demo, shadow annotator in evaluation. |
| `execute` | approved trajectory → `ExecutionRecord` | Physics client, position control. |
| `judge` | `ExecutionRecord`, `TaskOracle` → `JudgeResult` | Separate predicate spec. No `verify` imports. |
| `evaluate` | config, seed set → `AttemptRecord` JSONL, frames, manifest | Sequential first. The parent process is the only writer. |
| `report` | run(s) → HTML, `metrics.json` | Paired intervals for the headline. |
| `ui` | Gradio live and replay | Replay needs no keys and no GPU. |

All records are pydantic models, and JSONL lines are `model_dump_json()` output.

### 5.3 Simulation isolation [D10, D23.8]

- The preview and physics clients start from the same immutable scene snapshot. Each has independent mutable state and an explicit `physicsClientId`.
- **Approval binding:** the approval token is a hash of:
  - the trajectory contents;
  - the initial robot and object state;
  - the scene snapshot;
  - the config (thresholds and physics parameters).
- The Executor recomputes the hash and checks the actual start state within a tolerance (joint 1e-4 rad, object pose 1e-4 m). A mismatch aborts with `stale_approval`.

## 6. Tasks and ground truth [D15, D23.3, D23.10]

**Scene**
- A table with a Panda at a fixed base, and one open-top bin whose pose varies within a region.
- 1 target and 0–3 distractors (same shape in another color, or same color in another shape).
- 0–3 box or wall obstacles.
- The scene content hash is its snapshot version.

**Instruction**
- Templates: color, shape, spatial (nearest the bin, leftmost) and relational (next to X).
- A resolver enforces exactly one referent. Ambiguous instructions are discarded.

**Planner-visible `TaskSpec`**
- the instruction;
- object IDs, colors, shapes, sizes and poses;
- bin interior bounds;
- obstacle boxes;
- robot facts (base pose, reach bound, gripper opening 0.08 m, table height).

**Private `TaskOracle`**
- the intended target ID;
- the label;
- the witness or reference trajectory;
- the canonical oracle plan and its result;
- for impossible tasks, the construction type and proof parameters.

**Manipulation model:** the object must be carried in the gripper and released from at most 0.15 m above its resting place. No throwing, pushing or dragging.

**Labels**
- **Impossible by construction.** Conservative proofs within the manipulation model:
  1. **Target out of reach:** the distance from shoulder axis 2 to the nearest point of the object > (maximum TCP distance from the shoulder over the joint-limit box) + margin. The bound is computed once by sampling joint space densely plus an analytic upper bound (sum of link lengths), using the larger.
  2. **Bin out of reach:** the same bound applied to the nearest point of the bin's interior volume.
  3. **Target enclosed:** four walls and a lid, with gaps smaller than the minimum finger thickness. The exact geometry is stored.
  4. **Referent absent:** no object matches the instruction.
- **Feasible:**
  - Scenes are **constructive**: the generator places objects so that a known witness plan (canonical-style geometry with variations) passes the motion layer, the Verifier and execution plus the Judge.
  - The witness is stored. Exhaustive reference search is used only if constructive generation can't produce enough difficulty variety.
- **Undetermined:** neither a witness nor a proof. Excluded and counted in the report.

**Seed sets**
- `dev`: 100 tasks, used for tuning and threshold calibration.
- `test`: 300 feasible (stratified by distractor count × obstacle count) and 100 impossible (25 per type).
- `test` is generated **after the generator is frozen**. `TaskSpec` and `TaskOracle` are stored in separate files.

## 7. Plan schema and planners [D16]

**`PickPlacePlan`** (`kind = "pick_place"`)
- `target_id: str`
- `grasp`:
  - `approach ∈ {top, side_+x, side_-x, side_+y, side_-y}`;
  - `yaw_deg ∈ [-180, 180]`: for `top`, the heading of the gripper's closing axis, counterclockwise from world +x viewed from above (rotation about world +z). For side approaches, the rotation of the closing axis about the approach axis, measured from world +z. (The top convention was corrected after the final week-1 review; the earlier text gave a clockwise yaw that contradicted place yaw.)
  - `pregrasp_offset_m ∈ [0.03, 0.25]`.
- `transport_via`: list of 0–3 points `[x, y, z]`, world frame, meters.
- `place`: `position [x, y, z]`, `yaw_deg`, `release_height_m ∈ [0.0, 0.15]`.
  - `place.position` is the object's resting center in the bin, so its z must equal the bin-floor height plus half the object's height (V-8 enforces this, ±0.005 m). The drop is then exactly `release_height_m` (manipulation model, §6).
  - `place.yaw_deg`: between grasp and release, the gripper turns about world +z by `place.yaw_deg − grasp.yaw_deg`. For top grasps this makes `place.yaw_deg` the closing-axis heading at release, in the same convention as `grasp.yaw_deg`.
- `rationale: str`, logged only.

**`Refusal`** (`kind = "refusal"`)
- `reason ∈ {target_unreachable, destination_unreachable, target_enclosed, no_matching_object, other}`.
- `explanation: str`.

**Frames:** world frame, z up, meters. Yaw is in degrees in the schema and in radians internally.

**Validation**
- **Schema failure** → `malformed output`. The planner gets one retry with the pydantic error list, and the retry uses up one repair round.
- **Scene-dependent problems** → Verifier `Finding`s.

**LLM call**
- The `openai` SDK, configured by `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL`.
- A single tool `submit_plan`, with parameters = the JSON Schema of `PickPlacePlan | Refusal` and a forced `tool_choice`.
- **Fallback:** JSON in the message content, validated against the same schema [D13].
- **Never used:** OpenRouter `structured_outputs`.
- Low temperature, logged. The reasoning on/off setting is chosen after a logged dev test.
- Every request and response is logged raw.

**Planners**
- **`NemotronPlanner`:**
  - development: OpenRouter `nvidia/nemotron-3-ultra-550b-a55b` (owner's choice, D30);
  - final: Token Factory `nvidia/Nemotron-3-Ultra-550b-a55b`. Its catalog entry has no "function calling" tag, so the D13 fallback chain (forced tool call → `required` → validated content JSON) decides, verified in the W3.0 smoke test (moved from W2.6 by D37).
- **`HeuristicPlanner`:**
  - a template-aware parser;
  - a top-down grasp with yaw aligned to the nearest cube face, a vertical via-point at the tallest obstacle + 0.10 m, and placement at the bin center;
  - refuses outside the reach bound, or when nothing matches.
  - Reported as template-aware.
- **`FrontierPlanner`:** optional [D24].

## 8. Motion layer [D17, D23.4, D23.8]

**Model**
- `pybullet_data/franka_panda/panda.urdf` for both ManipulaPy and PyBullet.
- TCP at `panda_grasptarget`.
- ManipulaPy is restricted to `panda_joint1..7`, selected by name. The finger column is removed.
- **Startup test:** ManipulaPy FK vs PyBullet `getLinkState` at 20 random configurations, maximum error ≤ 1e-6 m.

**Key poses**
- current → pregrasp → grasp → lift (+0.10 m) → `transport_via`… → preplace (place position + `release_height_m` + 0.05 m) → release → retreat (+0.10 m).
- The initial-to-pregrasp segment is included.

**Segments**
- Linear position and slerp orientation.
- Samples are subdivided until **the joint-space motion between samples moves no link, finger or carried-object point by more than 0.005 m**. This is half the collision margin, bounded using Jacobian norms.
- IK: `iterative_inverse_kinematics` warm-started from the previous sample. Multi-start for the first pose, from seeds derived from the task seed (via `np.random.seed`).
- Accuracy is recomputed with FK, never taken from the solver's flag.

**Timing**
- Quintic time scaling per segment, with duration = the maximum over joints of (displacement ÷ (0.5 × velocity limit)).
- **ManipulaPy trajectory output is re-validated after timing** (positions, velocities and accelerations are consistent), because it clips and nudges silently.

## 9. Verifier [D17, D23.4, D23.5]

**How it works**
- Kinematic only: it sets joint states in the preview client and calls `getClosestPoints`. No physics stepping.
- Segments are checked in order. After an IK or continuity failure, later segments aren't checked; every other check runs across the whole trajectory.

| # | Check | Fails when | Plan field |
|---|---|---|---|
| V-1 | Grounding | `target_id` is unknown, or `place.position` is not over the bin interior with margin = object half-size + 0.01 m | `target_id`, `place.position` |
| V-2 | IK accuracy | FK residual > 0.002 m or > 1° | the segment's source pose |
| V-3 | Joint limits | a joint is within 2° of a limit, or over 100% of its velocity limit after timing | the segment |
| V-4 | Singularity | σ_min(J_s) < σ_thr (definition below) | the segment |
| V-5 | Continuity | a joint step > 0.2 rad between samples | the segment |
| V-6 | Collision | closest distance < 0.01 m between the robot or carried object and the table, obstacles, bin or non-target objects, or between non-adjacent robot links, outside the allowed-contact list | the segment, `transport_via[i]` |
| V-7 | Grasp preconditions | the object center is outside the closing band (\|offset\| > 0.01 m), or the object width along the closing axis > 0.08 − 2×0.005 m, or the approach axis is off by > 10° | `grasp.*` |
| V-8 | Release | the carried object's footprint is not fully inside the bin interior, the release height is out of range, or `place.position` z is not the resting height (bin floor + half the object height, ±0.005 m) | `place.*` |

**Allowed contacts (by stage)**
- Fingers ↔ target during `descend` and `grasp`.
- Carried target ↔ bin interior during `descend_place` and `release` (in the Verifier, only without penetration).
- Target ↔ table until lift-off (`approach`, `descend`, `grasp`, `lift`). This is residual support contact while the straight, vertical lift segment raises the object, so no dragging or pressing hazard can occur in it. The Verifier's V-6 allows the same. Measured in the good transfer: at most 0.63 N during `lift` (below the 1 N tolerance) and 1.51 N during `grasp`.
- Target ↔ table after release (`release`, `retreat`, `rest`), Judge only. Once released, a resting or dropped object is a task outcome decided by S1, not a safety violation (§13). This was found by R2: the fixture's bin floor is flush with the table, so a released cube touches both.
- Robot base ↔ table, always.

**Jacobian contract [D23.5]**
- J_s = [J_v / L ; J_ω]:
  - J_v is the linear velocity of the TCP point in the world frame;
  - J_ω is the angular velocity in the world frame;
  - **L = 0.5 m**.
- ManipulaPy's space Jacobian is shifted to the TCP point with J_v = J_v,space − [p_tcp]× J_ω.
- The Judge builds the same J_s from PyBullet's `calculateJacobian`.
- **Startup test:** both agree with each other and with a finite-difference Jacobian to ≤ 1e-6, at 20 configurations.
- **σ_thr** is the 1st percentile of σ_min over the witness trajectories of the `dev` set, rounded down. It is fixed in `config/verifier.json` before `test` is created.

**`Finding` fields**
- `stage`, `check`, `plan_field`;
- `measured`, `threshold`, `direction` (e.g. "below obstacle top by 0.03 m");
- `sample_index`, `bodies` (body names; robot links appear as `robot:<link>`). A `frame_ref` for visual findings arrives with W2.1.
- **No computed fixes [D17].**

**Thresholds** live in `config/verifier.json` (versioned and hashed into the approval token).

## 10. Repair loop [D18]

**Budget:** k = 3 repair rounds (at most 4 planner calls). A malformed-output retry uses up a round.

**Repair prompt (stateless, rebuilt each round)**
- the system prompt and the `TaskSpec`;
- the most recent plan;
- that round's Findings: ordered along the trajectory, at most 5, at most one per check;
- one summary line per earlier round.

The planner never sees `TaskOracle` data or images. Cosmos verdicts arrive as text Findings.

**`repeat_plan`:** a plan that is identical to the previous one still uses up a round and is logged. Plans are compared after rounding positions to 0.001 m and angles to 1°.

**Outcomes (exactly one per attempt; all stay in the denominator)**

| Outcome | Meaning |
|---|---|
| `executed` | Went to the Executor; then judged. |
| `refused` | The planner returned a `Refusal`. |
| `budget_exhausted` | All rounds used without passing the gate. |
| `malformed_final` | Schema-invalid output on the last allowed round. |
| `service_failure` | After the SDK's retries; per-call timeout 60 s, total retry deadline 180 s. |
| `stale_approval` | The approval binding didn't match at execution. |
| `sim_error` | An exception or NaN in the motion layer, verifier or simulator. |
| `visual_uncertain` | Demo only: Cosmos stayed `uncertain` after the third view. |

**Diagnostic attribution [D18, D23.6]**

Observable failure categories are reported first: the check and the stage. Tags are diagnostic evidence, not proof of cause:

| Tag | When it applies |
|---|---|
| `LLM plan` | A wrong `target_id` vs the oracle; a false refusal; a failed check where the canonical oracle plan passes. |
| `motion-layer limitation (diagnostic)` | The plan and the canonical oracle plan fail the same check type. |
| `gate/execution discrepancy` | The plan passed the gate but the Judge found a violation. These are investigated case by case. |
| `verifier/sim issue` | A `sim_error`, or a discrepancy confirmed by investigation as a simulator or checker fault. |
| `mixed` | Conflicting evidence. |
| `unknown` | Anything else, including `service_failure`. |

## 11. Cosmos gate [D5, D19, D20, D23.7, D24]

**Rendering**
- PyBullet TinyRenderer, 512×512 RGB.
- Two viewpoints, front-oblique and top-down. A third side view for re-renders after `uncertain`.
- **Key frames:** end of pregrasp, grasp, pre-release.

**Stage questions**

| Stage | Question |
|---|---|
| Pre-grasp | Is the gripper approaching the object named in the instruction? |
| Grasp | Does the object appear between the fingers, with a plausible closing direction? |
| Pre-release | Is the named object positioned over the bin? |

Object names come only from `TaskSpec` attributes (color and shape).

**Request and verdict**
- One request per stage, with both views.
- Required reply: `VERDICT: pass|fail|uncertain` and `REASON: <line>`.
- Strict parsing: anything unparseable becomes `uncertain`.

**Demo semantics**
- Cosmos runs only after the geometric checks pass, and never overrides a rejection.
- `fail` → repair.
- `uncertain` → render the third view; if it's still uncertain → `visual_uncertain`.
- Endpoint *unavailable* → a visible banner, and execution goes ahead on the geometric checks only.
- Described as a **learned visual check**, never as verification.

**State-predicate truth (shadow study)**
- **Pre-grasp:** the approach axis is within 10° of the vector to the **oracle target**, and the oracle target is the nearest object.
- **Grasp:** the **oracle target** is in the closing band.
- **Pre-release:** the carried object **is the oracle target** and its footprint is over the bin interior.

**Shadow study**
- Run locally over saved frames.
- Reports: true and false rejections, catches the gate missed, the `uncertain` rate, latency.
- **Natural and injected negatives are reported separately.** Injected negatives are wrong-object and misaligned-grasp renders.
- No incremental-success claim from shadow data.

**Serving**
- Cosmos3-Edge (4B) with vLLM / vLLM-Omni in Reasoner mode, on the local RTX 5070 Ti.
- Nano (16B) only if Edge misses the bar.
- A Nebius endpoint at the final stage only.

**Pilot (gate by 2026-10-12)**
- 30 frames, 10 per stage, balanced by the state predicates.
- **Go if** balanced accuracy ≥ 0.90 and median latency ≤ 5 s per stage request.
- **Otherwise:** shadow study only, and the live gate is dropped from the demo. *Amended by D47 (2026-10-11): after the D34 no-go the owner restored the live gate in the demo only, as an opt-in learned visual check with the pilot numbers shown; evaluation stays physics-only.*
- It is called a pilot, not validation.

## 12. Executor [D21, D23.4, D23.8]

**Start**
- The approval check (§5.3), then gravity and a 240 Hz step.
- Physics parameters are pinned in `config/physics.json`: solver iterations, friction, restitution and contact ERP.

**Arm control**
- `POSITION_CONTROL` per arm joint, with finite `positionGain` and `velocityGain` (in config), force = the URDF effort limits.
- Targets are interpolated per step from the timed trajectory.
- Settles for up to 0.5 s per segment, with the residual recorded.
- Never resets joint states.

**Gripper**
1. Open before the approach.
2. At the grasp pose, after settling, evaluate the attachment preconditions on the **actual** state (the same numeric conditions as V-7).
3. If met: close the fingers to the object width, then add a `JOINT_FIXED` constraint between `panda_hand` and the object at the current relative transform, with `maxForce` from config.
4. If not met: no constraint, and the run continues. The object stays put and the Judge sees it.
5. At release: remove the constraint, open the fingers, settle 1.0 s.

**`ExecutionRecord`**
- Per-step (240 Hz) summaries: minimum joint-limit margin per joint, σ_min(J_s) from PyBullet, contacts (pair, links, normal force, penetration, step).
- Joint states at 60 Hz for replay.
- Tracking error per segment.
- Whether attachment happened, with its precondition values.
- Final object poses and linear and angular velocities after settling.

**Pacing:** no `time.sleep` in the evaluation. Real-time pacing and 30 fps frames only in the demo and video.

## 13. Judge [D21, D23.4]

- Implements `config/judge.json`, which is pure data. Imports nothing from `verify`.
- Jacobians come from PyBullet `calculateJacobian`, using the §9 contract.

| ID | Predicate (on executed state) | Kind |
|---|---|---|
| S1 | The oracle target's whole bounding box is inside the bin interior volume, with \|v\| < 0.01 m/s and \|ω\| < 0.1 rad/s for a 0.5 s dwell | task |
| S2 | The object in the bin is the oracle target; otherwise `wrong_object` | task |
| V1 | No contact with penetration > 0.001 m or normal force > 1 N outside the stage-specific allowed-contact list (§9) | safety |
| V2 | No joint beyond its URDF limit at any physics step | safety |
| V3 | σ_min(J_s) ≥ σ_thr_judge at every physics step (default = σ_thr; set separately in `judge.json`) | safety |
| V4 | No self-collision contact | safety |
| R | Impossible task: correct only if `refused`. Feasible task: a refusal is a false refusal. The refusal reason is scored separately. | refusal |

- **Contact allowances use the planned target** (the object the executed plan grasped); S1/S2 use the oracle target. Picking the wrong object is therefore S2 `wrong_object`, a task failure, not a safety violation.
- **A target that was never attached** (missed grasp) is a resting object: its table contacts are allowed in every stage, and S1 decides.
- **Feasible success** = S1 ∧ S2 ∧ V1 ∧ V2 ∧ V3 ∧ V4.
- **Safety violations** (V1–V4) are also reported separately over all executions. A dropped cube fails S1 and is not a safety violation.
- **V4 significance:** V4 counts robot-robot contacts that exceed the V1 significance thresholds (penetration > 0.001 m or force > 1 N), as a noise filter. Topology-excluded pairs are filtered in physics, so any remaining contact is genuine.
- **"Independent"** means independent check logic on the same engine. The README says this.

## 14. Evaluation [D8, D22, D23, D24]

**Configurations (same held-out `test` seeds)**

| # | Configuration | Planner / provider | Role |
|---|---|---|---|
| 1 | Unverified (evaluation-only, simulation) | Nemotron / Token Factory | ablation: the share of raw plans that fail or collide |
| 2 | Verify only = **round 1 of config 3** | Nemotron / Token Factory | headline "before" (paired) |
| 3 | Verify + repair (k = 3) | Nemotron / Token Factory | headline "after" |
| 4 | Cosmos shadow study on config 3's frames | Cosmos (local / Nebius endpoint) | secondary, shadow only |
| 5 | Heuristic baseline (verify, no LLM) | none | baseline |
| 6 | Frontier, no verifier | frontier / OpenRouter | optional reference; excluded from the headline |
| 7 | Equal-budget resampling (verify, ≤ 4 independent samples, no feedback) | Nemotron / Token Factory | credits feedback over retrying |

**Metric definitions (fixed in the README before any final run)**
- **First-try success:** the round-1 plan passes the gate and the Judge reports success.
- **Success with repair:** cumulative success within k = 3. Same tasks and denominator, including every outcome in §10.
- **Repair gain:** paired difference (3 − 2) on the same tasks. **95% interval:** paired bootstrap over tasks, 10,000 resamples. Each task is one scene with one attempt per configuration, so no further clustering is needed.
- **Feedback gain:** paired difference (3 − 7), computed the same way.
- **Proportions:** Wilson 95% intervals, with N and seed-set hashes.
- **Refusals:** the confusion matrix over feasible vs impossible.
  - Wrongly attempted impossible tasks.
  - Wrongly refused feasible tasks.
  - Reported separately from feasible success.
- **Safety:** V1–V4 counts per execution, by body pair. "Zero observed violations in M executions" only if measured.
- **Blocked plans:** counted by check and by outcome. Physical-constraint failures are kept separate from malformed output and from solver non-convergence.
- **Breakdowns:** by stratum (distractors × obstacles), by diagnostic attribution, and by cost and latency per configuration.

**Headline sources:** Token Factory runs only. Provider and model are logged per attempt.

**Runner**
- `vplan eval run --config N --seeds dev|test [--n N]`.
- **Sequential first:** one supervised worker subprocess per attempt batch, with a wall-clock timeout per attempt. The parent is the only JSONL writer.
- Attempt ID = config + seed-set + task index. Resume skips completed IDs and refuses an incompatible manifest. A partial final line is discarded.
- A process-shared rate limiter for LLM calls. Multiprocessing only after a measured bottleneck.

**Run directory:** `runs/<run_id>/{attempts.jsonl, frames/, manifest.json}`. The manifest records:
- git SHA;
- package and Python versions;
- OS;
- image digest;
- provider and model;
- config hashes;
- seed-set hash.

**Shadow study:** `vplan eval shadow --run <id>` → `shadow.jsonl`.

## 15. Report and UI [D22]

**Report**
- `vplan report build --run ...` → a single static HTML file plus `metrics.json`.
- Covers every metric in §14, with matplotlib charts.
- The heuristic row is shown prominently. The frontier and shadow rows carry labels and limitations.

**UI (Gradio)**
- **Live mode:**
  - scene or seed picker plus an instruction box;
  - a plan panel showing pretty JSON;
  - a check timeline with pass/fail, evidence and the offending body highlighted in the render;
  - the Cosmos frames and verdicts;
  - the repair rounds;
  - the execution video;
  - the Judge's verdict.
- **Replay mode:** loads an `AttemptRecord` and its frames or joint states. Needs no keys and no GPU.

## 16. Error handling and logging

- **LLM:** SDK retries and backoff up to the 180 s deadline → `service_failure`.
- **Exceptions:** caught explicitly as `Exception`, never bare. Each becomes an attempt outcome with the traceback logged.
- **Supervision:** the worker subprocess timeout catches native crashes and hangs. One bad attempt never stops a batch.
- **Logged per attempt:** raw LLM requests and responses, every Finding, verdicts, the `ExecutionRecord` summary and the Judge result.
- **Logged per run:** versions.
- **Reproducibility:** seeds control scene generation and solver randomness; remote inference is not reproducible [D10]. Reproducibility in a fresh process is tested for the simulator parts.

## 17. Upstream ManipulaPy changes

These land in a tagged release, listed in the README under "updated during the submission period" [D2]:
1. Explicit, deterministic end-effector selection, with `tip_link` honored and validated.
2. Screw axes restricted to the chain from the base to `tip_link`. This fixes the 8-DOF Panda.
3. A correct `near_singularity_detection` threshold semantic.
4. ASCII-safe import banner (the Windows cp1252 crash) [D14].

Until the release, the hackathon code works around all four (§8) and pins 1.4.1.

## 18. Nebius usage and submission artifacts [D7, D19]

**Development:** OpenRouter plus local.

**Final stage, from 2026-10-24 on:**
- the `test` batch for configs 1, 2/3, 5 and 7 (and 6 if time allows) on Serverless Jobs, with Nemotron on Token Factory;
- the Cosmos endpoint, if the pilot passed;
- recording the demo video.

**Early smoke tests (each needs the owner's OK; they cost money):**
- one Token Factory forced tool call;
- one 10-task Jobs run with recovery from Object Storage.

**Submission**
- a public AGPL-3.0 repo with a license file [D2];
- a README covering setup, run, replay, the metric definitions, the configuration table, limitations ("attachment-assisted", "independent logic on the same engine", the Cosmos shadow scope), how Nemotron, Cosmos and Nebius were used, and the ManipulaPy updates;
- a video of at most 3 minutes, with at least 1 minute of the modules in action;
- a project description;
- the feedback write-up on the Nebius and NVIDIA tools (required).

## 19. Verification items (each with a scheduled test in `docs/plan.md`)

These are facts the design depends on that haven't been confirmed yet:
1. A forced named `tool_choice` works on Token Factory's `Nemotron-3-Ultra-550b-a55b` (no "function calling" catalog tag; D13 fallback otherwise) [D13, D30].
2. Cosmos3-Edge serves on the RTX 5070 Ti (vLLM on sm_120, fits in 16 GB) and meets the pilot bar [D20].
3. The Linux Jobs container runs the evaluation and returns results via Object Storage [D12].
4. Constructive generation produces enough feasible variety for 300 stratified `test` tasks [D23.10].
5. Position control plus attachment gives repeatable transfers [Codex review, risk 2].


### Development feedback pilot [D32]

Before the full evaluation, run the approved 20-task feasible development pilot in `docs/pilot-plan.md`. Freeze witness-validated tasks before model calls. Compare the shared first attempt with up to three further feedback-driven revisions or independent samples. Count every provider request against the four-request per-arm limit, including protocol fallbacks. Keep every task/outcome in the denominator and report actual unique calls separately from each arm's logical call count. This pilot diagnoses whether feedback adds value; it does not replace the held-out evaluation or establish its headline claim. Artifact layout, the free-model guard and snapshot checks are recorded in [D33]; commands are in the README.
