# Feedback pilot implementation plan

**Goal:** Implement the approved 20-task development experiment comparing first-attempt success, verifier-feedback repair, and equal-budget resampling.

**Spec:** `docs/spec.md` sections 6, 7, 10, 13 and 14, plus the user's approved pilot proposal. This is an early slice of W2.2/W2.3/W2.5 and W3 evaluation, not completion of all week-two requirements.

**Architecture:** Freeze a witness-validated development cohort before model calls. Give repair and resampling the same initial proposal and at most three further proposals each. Planners receive only public task records; independent simulation results and private witnesses remain in evaluation code. Persist every round and report paired outcomes, call counts, token use and latency.

## Constraints and scope

- Use existing dependencies and Windows Python 3.10; no installations, commits, pushes or paid calls in this implementation task.
- Preserve all pre-existing local edits, including the unverified-baseline correction.
- Keep Judge independence, immutable records, config thresholds and execution gating intact.
- This pilot contains feasible development tasks only: 20 by default, with clear scenes, distractors and obstacles. Do not generate held-out test tasks or claim the 300/100-task generator is finished.
- Use seeded local generators, opaque object IDs, varied target attributes/positions/bin positions and difficulty strata. Accept tasks only using reference success, never using model outcomes. Record all rejected candidate seeds and reasons; fail clearly if the requested cohort cannot be filled.
- Reference plans and oracle labels are stored separately from planner-visible tasks. A typed planner boundary and request-content tests enforce this.
- No automatic `.env` inspection in development tooling. Live commands may consume credentials internally, never echo or persist them. Default offline tests make zero network calls.
- Use the configured free OpenRouter model for any live development run; Nebius and paid models remain subject to explicit approval.
- Full impossible-task construction, visual gating, cloud supervision, deployment, UI and exact replay remain separate scheduled work.

## Task 1: Public/private records and witness-validated development tasks

Files: `src/vplan/records.py`, `src/vplan/tasks/{__init__,scenes,generator}.py`, `tests/conftest.py`, `src/vplan/cli.py`, `tests/test_tasks.py`.

- [ ] Add frozen `TaskSpec` (instruction, scene, robot facts; no target label or witness), `TaskOracle` (target, feasibility, reference plan/result), and generated-task metadata with seed/stratum.
- [ ] Move the unchanged basic-scene builder from tests into the package; tests and CLI import the production helper.
- [ ] Build deterministic candidates across clear/distractor/obstacle strata. Bins sit on the table, objects do not overlap, center top grasps use the documented height window. Resolve each instruction to exactly one object.
- [ ] Build a cohort using the existing heuristic witness and real `run_attempt`; reject failed witnesses without changing thresholds. Use bounded candidate attempts and record exclusions. Make generation and witness evaluation separately testable.
- [ ] Test determinism, public/private separation, instruction uniqueness, geometry constraints, exclusion accounting, bounded failure, and real successful witnesses across the pilot strata. Run a real 20-task preparation probe.

## Task 2: Model adapter and bounded paired experiment

Files: `src/vplan/planners/nemotron.py`, `src/vplan/repair.py`, records additions, `tests/test_planner.py`, `tests/test_repair.py`.

- [ ] Implement `NemotronPlanner.plan(TaskSpec, ...)` with an injectable OpenAI-compatible client, forced `submit_plan`, required-tool/content fallback, schema validation, explicit provider/model/usage/latency metadata and stateless feedback prompts. Verify SDK/provider APIs using official documentation before implementation.
- [ ] Every provider request consumes budget, including fallback or malformed retries. No hidden SDK retries. Total max four requests per arm; retain each request's outcome and token accounting. Transport errors end as service failure rather than silently increasing budget.
- [ ] Explain coordinate/yaw/resting-height conventions in the system prompt. Feedback carries up to five findings, the latest proposal and compact prior-round summaries; never oracle data.
- [ ] Implement paired task evaluation: share the initial proposal and result; repair receives findings, resampling receives no prior plan/findings. Allow up to three additional calls in each arm. Stop an arm on execution, refusal, service/simulation error, or exhaustion per spec; Judge failures remain failures and do not supply hidden oracle feedback.
- [ ] Record repeated plans using rounded geometry (1 mm / 1 degree) excluding rationale; budget is consumed. Retain exact outcomes, first-round result, calls/tokens/latency and observable failure categories.
- [ ] Test successful repair, independent resampling, shared initial attempt, no feedback/oracle leakage, malformed and provider-fallback budget use, repeated plans, refusal, stale approval, simulation errors, service failure and final exhaustion with injected clients/evaluators.

## Task 3: Runnable pilot and evidence

Files: `src/vplan/evaluate/{__init__,pilot}.py`, `src/vplan/cli.py`, `tests/test_pilot.py`, README/status/spec/decision documentation.

- [x] Add `pilot prepare --output DIR --n 20 --seed 0` and `pilot run --input DIR` commands (exact parser structure may use nested subcommands). Preparation makes no model calls and writes public tasks, separate private witnesses, exclusion log and manifest with hashes/provenance. Existing directories must not be silently overwritten.
- [x] Live run validates the prepared manifest/task hashes, uses only the configured free OpenRouter development endpoint/model, writes incremental per-task evidence, and refuses reruns that would overwrite evidence. No cloud or paid calls.
- [x] Report first-attempt/repair/resampling successes over all tasks, paired wins/losses, repair opportunities, outcome/check counts, logical per-arm cost plus actual unique requests, token/latency totals and missing-usage counts. Label it a development pilot, never a held-out headline result. Incomplete runs must be conspicuous and cannot look complete.
- [x] Test artifact integrity, denominator preservation, shared-call accounting, no credential persistence, CLI offline preparation and clear error paths; use fake providers in tests.
- [x] Update docs with exact commands, implemented scope, remaining work and recorded validation. Run the full offline suite, independently review the diff and address important findings.

Recorded validation (2026-10-04): `tests/test_pilot.py` (34 offline tests, including the review fixes for degraded runs, arm alternation and guards) and the full offline suite pass; a real `pilot prepare --n 3` (local simulation, no model calls) wrote all four artifacts, and a second prepare into the same directory and a non-free `pilot run` were both refused with exit 2. No live `pilot run` has been made yet. Remaining: the live 20-task development run and its report.

## Review focus

Budget fairness includes fallback requests; task selection precedes model outcomes; request payloads exclude private truth; service failures remain in denominators; saved evidence cannot be mixed with changed task/config snapshots.
