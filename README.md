# vplan: LLMs plan, physics verifies

A language model proposes a pick-and-place plan for a Franka Panda arm. Deterministic kinematic and collision checks then approve or reject the plan **before anything moves**. A rejected plan comes back with structured findings the model can use to repair it. An independent Judge scores every executed run in physics simulation.

Built for the [Nebius x NVIDIA Global AI Hackathon](https://nebiusglobalaihackathon.devpost.com/) (Physical AI track, simulation only). The planner is NVIDIA **Nemotron**: **3.5 Lightning** for the evaluation and **3 Ultra** for the live demo, both served by **Nebius Token Factory** in the final runs (D62).

> **Status: week 4 of 4.** The final held-out evaluation ran on Nebius Token Factory on Oct 18 ([results below](#results), D70), and the Cosmos shadow study on Oct 19 (D73). The plan → verify → execute → judge loop runs with both a rule-based planner and Nemotron (3 Ultra and 3.5 Lightning). Also in place: a task generator with provably impossible tasks, verifier-feedback repair, keyless replay of recorded runs, a demo-only Cosmos visual gate, the evaluation pipeline (frozen seed sets, a supervised resumable runner and a self-contained HTML report) and a Gradio UI that shows live attempts and replays recorded ones without keys. ManipulaPy 1.4.2 carries the upstream fixes this project needed ([below](#manipulapy-updates-during-the-submission-period)). Still to come in week 4: planner-prompt work on the `dev` set, the demo video and the submission write-ups.

## How it works

```
plan ──▶ motion layer ──▶ verifier ──(findings → blocked / repair)
         (IK, timing)     8 checks in a
                          preview sim
                              │ pass → approval token
                              ▼
                          executor ──▶ judge
                          (PyBullet,   (task success +
                           240 Hz)      safety predicates)
```

- **Plan:** a typed `PickPlacePlan` (target, grasp, via-points, place pose) or a `Refusal`.
- **Motion:** key poses → Cartesian paths → per-sample IK (ManipulaPy) → densified, time-scaled joint trajectory.
- **Verifier:** checks grounding, IK accuracy, continuity, joint and velocity limits, singularity, collisions (robot, carried object, self), grasp preconditions and release height. Each failure becomes a `Finding` with the measured value, threshold and the plan field to change.
- **Executor:** runs only with an approval token that binds the exact trajectory, scene, target and config. It records contacts, joint margins and manipulability at every physics step.
- **Judge:** scores the executed run on its own. It shares no code with the verifier, so the safety numbers are not the verifier grading itself.
- **Repair:** when the verifier blocks a plan, `run_task` sends its findings back to the planner (fixed-text directions, never a ready-made fix) and tries again within a call budget.
- **Cosmos gate (demo only, optional):** after the verifier passes a plan, NVIDIA **Cosmos3-Edge** looks at rendered key frames (pre-grasp, grasp, pre-release) and can still block it. It is a learned visual check, not verification: it can add rejections but never approves a geometric failure, and the evaluation uses physics only (D34, D47).

## Results

The held-out `test` seed set holds 300 feasible and 100 provably impossible generated tasks (seed-set hash `834d1931…`). Every Nemotron configuration used **Nemotron 3.5 Lightning on Nebius Token Factory**, with a 60 s deadline per call ([D70](docs/decisions.md)). The full report is [`docs/results/report.html`](docs/results/report.html), a self-contained file to download and open in a browser; its numbers are in [`metrics.json`](docs/results/metrics.json).

> **Bounded repair with verifier feedback raised Nemotron 3.5 Lightning's task success from 24.7% (verified first attempts) to 65.0% across 300 held-out, reference-solvable generated tasks. At equal budget, feedback beat resampling by 29.3 points (95% CI 23–36).**

| Config | Planner and gate | Feasible success (n = 300) | 95% CI | Impossible tasks refused (n = 100) | Executions with a safety violation |
|---|---|---|---|---|---|
| 3 | Nemotron, verifier, repair from its findings (≤ 4 calls) | **195 (65.0%)** | 59.4–70.2% | 57 | 0 of 215 |
| 7 | Nemotron, verifier, independent resamples (≤ 4 calls) | 107 (35.7%) | 30.5–41.2% | 52 | 0 of 123 |
| 2 | Nemotron, verifier, first attempt only | 74 (24.7%) | 20.1–29.8% | 42 | 0 of 83 |
| 1 | Nemotron, no verifier | 130 (43.3%) | 37.8–49.0% | 40 | 169 of 346 |
| 5 | rule-based heuristic, verifier | 300 (100%) | 98.7–100% | 75 | 0 of 300 |

- **Feedback, not just retries.** Repair beats equal-budget resampling by +29.3 points (23–36, paired bootstrap over the same tasks). Its gain over the verified first attempt, +40.3 points (34.7–46.0), also includes the extra attempts.
- **Verification keeps execution safe.** Without the verifier, 169 of 346 executions broke a safety predicate, 148 of them with a collision (V1). No verified execution did.
- **The verifier is conservative.** Unverified first attempts succeed more often (43.3%) than verified ones (24.7%), because the verifier also blocks plans the Judge would pass, for example clearances under its 1 cm threshold. Repair more than recovers the gap.
- **Where Lightning fails.** It almost never gives transport waypoints (212 of 218 unverified plans for feasible tasks with obstacles), and the motion layer does not avoid obstacles by design (spec §2), so the carried object hits them. Unverified success falls from 58 of 75 obstacle-free tasks to 20 of 75 with three obstacles. With repair it stays between 45 and 51 of 75 at every obstacle count.
- **Refusals are the weak spot.** Lightning refuses 40–57 of the 100 impossible tasks, and no planner refuses an enclosed target (type 3, 0 of 25 in every config). The verifier blocks most of the rest before they move. One wrongly refused feasible task in 900 Lightning attempts on feasible tasks.
- **Cost and reliability.** 2,510 Token Factory calls cost about $1.61 (median 7.5–8.7 s per call). 35 calls (1.4%) failed at the provider and count as failures, which is why the report title says "DEGRADED".
- These findings were made on `test`, so the frozen prompt stays as it is for these numbers. Prompt work moves to week 4 on `dev` (D70).

**Cosmos shadow study (config 4, [D73](docs/decisions.md)).** NVIDIA **Cosmos3-Edge**, served locally, answered the live gate's three stage questions about the key frames of all 215 executed config 3 attempts, plus 692 injected wrong-object and misaligned-grasp renders: 1,337 questions, scored against simulator state predicates. It never influenced the evaluation, so this measures only what a visual gate would have caught, not added success. It is not a usable gate: it falsely rejected 27% of correct pre-release frames (it reads the object as inside the bin, not over it), caught 1 of 159 wrong-object pre-grasps and 25 of 196 misaligned grasps, and reached balanced accuracy 0.61 against natural negatives and 0.52 against injected ones, at 1.3 s per request. The demo's Cosmos gate stays an advisory, learned check. Summary: [`docs/results/shadow/summary.md`](docs/results/shadow/summary.md).

## Setup

Requires **Python 3.10**.

```bash
pip install "ManipulaPy[simulation]==1.4.2" pybullet==3.2.7 numpy==2.2.6 pydantic==2.13.5 openai==3.24.0 matplotlib==3.10.9 gradio==6.30.0 pytest==9.1.1
cp .env.example .env   # LLM endpoint and key for the Nemotron planner; never commit .env
```

The pinned versions are listed in [`pyproject.toml`](pyproject.toml).

On Windows, `pip` may build pybullet from source, which needs the MSVC build tools.

## ManipulaPy updates during the submission period

The Verifier's kinematics come from [ManipulaPy](https://github.com/boelnasr/ManipulaPy), the author's open-source robotics library. [Version 1.4.2](https://github.com/boelnasr/ManipulaPy/tree/v1.4.2) (PyPI, 2026-10-13) adds the fixes this project needed:

- an explicit, validated `tip_link` and a deterministic default end effector;
- screw axes restricted to the chain from the base to `tip_link` (the Panda is now 7-DOF, not 8 with a finger joint), with the chain's `joint_names`;
- a meaningful default threshold for `near_singularity_detection`;
- an ASCII import message on stderr instead of an emoji banner that crashed Windows cp1252 consoles;
- `extract_omega_list` returning the 3 × n rotation axes.

Details and verification are in [`docs/decisions.md`](docs/decisions.md) (D49).

## Run

On Windows (PowerShell), from the repo root:

```powershell
$env:PYTHONUTF8 = "1"        # UTF-8 file and console I/O on Windows
$env:PYTHONPATH = "src;."    # use "src:." on Linux/macOS
python -m vplan demo-once                     # one attempt, headless
python -m vplan demo-once --obstacle --gui    # with a wall in the way, watched in the PyBullet GUI
python -m pytest -q                           # test suite (~100 s)
```

`demo-once` prints the outcome, any verifier findings and the Judge result. It exits `0` when the Judge reports success.

### Live Cosmos gate (optional, demo only)

Serve `nvidia/Cosmos3-Edge` with an OpenAI-compatible endpoint (we use vLLM on a local 16 GB GPU), then pass its URL:

```powershell
python -m vplan demo-once --cosmos-url http://127.0.0.1:8000/v1
```

The gate prints one line per checked stage. A Cosmos `fail` blocks the plan with one visual finding; `uncertain` is re-asked once with a side view and otherwise stops the attempt. If the endpoint is down or errors, the run says so and executes on the physics checks alone. The first live check showed false rejections and 6–16 s per question, so treat its verdicts as advisory.

### Gradio UI (live and replay)

```powershell
python -m vplan ui                                  # http://127.0.0.1:7860; the template heuristic plans, no key needed
python -m vplan ui --replay runs/ui                 # open the recorded attempts in the Replay tab
python -m vplan ui --cosmos-url http://127.0.0.1:8000/v1                      # add the live Cosmos gate
python -m vplan ui --env-file path\to\llm.env --model <model id> [--allow-paid]  # add the Nemotron planner
```

- **Live tab.** Pick the basic scene (with or without its wall) and type a template instruction such as "Pick the red cube and place it in the bin.", or pick a generated task by seed and kind (feasible, or impossible types 1–4). One attempt runs with verifier-feedback repair (k ≤ 3), and the panels fill in as it runs:
  - the plan as JSON;
  - a check timeline per round (V-1 … V-8), with the offending link or object and the measured clearance;
  - each finding rendered with the offending bodies in magenta (overview and close-up);
  - the Cosmos key frames and verdicts, or a banner saying the plan executes on the geometric checks only;
  - the execution, played back as an animated image;
  - the Judge's verdict.
- **Recording.** Every live attempt is saved under `runs/ui/` (`--record-dir`, `--no-record`).
- **Replay tab.** Opens any of these:
  - a W3.1 run directory (pass its seed-set directory if the recorded path does not resolve on this machine);
  - a feedback-pilot run (`<cohort>/runs/<run_id>`);
  - a UI recording, or a folder of recordings.

  Replay needs no `.env`, no keys, no GPU and no network: Gradio's telemetry is off, and the page loads nothing from the internet.
  It opens paths only under the working directory, `--record-dir`, `--replay` and `--seed-dir`, and never network (UNC) paths.
- **Credentials.** They come only from the file named by `--env-file`. Without `--allow-paid`, only free `nvidia/...:free` models on OpenRouter run (D33). Every planner call has the spec's 60 s deadline (D63).
- **Serving.** The server listens on 127.0.0.1. With Windows Python under WSL2 (NAT networking), a browser inside WSL cannot reach that address. Pass `--host` with the Windows address of the WSL network (the default gateway in WSL's `ip route`) instead of `0.0.0.0`, which would expose the planner to the LAN. `vplan ui` warns whenever `--host` is not a loopback address.

## Feedback pilot

A 20-task, feasible-only development experiment comparing the first attempt, verifier-feedback repair and equal-budget resampling ([`docs/pilot-plan.md`](docs/pilot-plan.md), D32, D33). It is diagnostic, not a held-out result.

Latest run: hard cohort (`--generator hard`), free Nemotron 3 Ultra on OpenRouter, temperature 0.7 ([D46](docs/decisions.md)):

| | successes / 20 |
|---|---|
| first attempt | 9 |
| verifier-feedback repair | 17 |
| equal-budget resampling | 17 |

Both retry strategies roughly double the success rate. Repair used fewer calls (32 vs 41) and never repeated a plan (resampling repeated 3), but the pilot does not separate the two on success. Three provider 503s stay in the denominators. The held-out evaluation settles it: repair beats equal-budget resampling by 29.3 points ([Results](#results)).

```powershell
# 1. Freeze the cohort: local simulation only, no model calls (a few minutes). Refuses a non-empty directory.
#    --generator hard builds dense tasks with distractors and obstacles (default: pilot, the original easy cohort).
python -m vplan pilot prepare --output runs/pilot-dev --n 20 --seed 0 --max-candidates 100

# 2. Smoke test on 2 tasks, then the full run. Credentials come only from the file passed with --env-file.
python -m vplan pilot run --input runs/pilot-dev --env-file .env --limit 2
python -m vplan pilot run --input runs/pilot-dev --env-file .env

# 3. Rebuild summary.json/report.md for a run that was killed before it could write them.
python -m vplan pilot report --run runs/pilot-dev/runs/<run_id>

# 4. List every task's rounds, then watch one recorded plan in the PyBullet GUI.
#    --round 0 is the shared first attempt; --unverified executes it even if the verifier blocked it.
python -m vplan pilot show --run runs/pilot-dev/runs/<run_id>
python -m vplan pilot show --run runs/pilot-dev/runs/<run_id> --task <task_id> --round 0 --unverified
#    --replay plays back the stored execution record (no API key needed); --cosmos-url adds the live Cosmos gate.
```

- `prepare` writes `tasks.jsonl` (planner-visible), `oracle.jsonl` (private targets and reference plans), `exclusions.jsonl` and, last, `manifest.json` with file, config and source hashes.
- `run` refuses before any request if the manifest is missing, a task/oracle file or config changed, or any `src/vplan/**/*.py` / `config/*.json` file changed (`--allow-source-drift` records the drift instead). Without `--allow-paid` it only calls `https://openrouter.ai` models named `nvidia/...:free`; `--model` overrides `LLM_MODEL`. Every call in both arms uses `--temperature` (default 0.7), and the arm that runs first alternates per task. Pass relative or Windows paths, not `/mnt/...`.
- Each run gets a new `runs/<run_id>/` with `run.json`, `events.jsonl` and `results.jsonl` (flushed per event / per task), then `summary.json` and `report.md`. A run that did not finish is titled `INCOMPLETE: k/n tasks`, a `--limit` run `LIMITED: first k of n tasks`, and any provider service failure adds `DEGRADED: n service failures`. A 401/402/403/404, or three consecutive failed requests, stops the run as incomplete. `pilot report` exits 1 for an incomplete run.

After the [D36 validation fixes](docs/decisions.md#d36-close-approval-dwell-and-input-validation-gaps-2026-10-10), prepare a fresh cohort before collecting new results: source hashes have changed. Existing records still load, but rejudging an execution without the new per-step dwell pose history cannot establish S1 success. Saved historical verdicts remain readable.

## Evaluation

The held-out evaluation compares the configurations in spec §14 on the same frozen seed set. No model call happens before the seed set is frozen.

| Config | Planner | Gate | Budget |
|---|---|---|---|
| 1 | Nemotron | none (only an unknown target is blocked) | 1 attempt |
| 2 | Nemotron | verifier | round 1 of config 3 (not run separately) |
| 3 | Nemotron | verifier, with repair from its findings | up to 4 attempts |
| 5 | rule-based heuristic | verifier | up to 4 attempts |
| 6 | a frontier model (optional, paid) | none | 1 attempt |
| 7 | Nemotron | verifier, independent resamples, no feedback | up to 4 attempts |

```powershell
# 1. Freeze a seed set: local simulation only. dev = 75 feasible + 25 impossible tasks (D51);
#    test is generated only after the W3.3 freeze and needs --allow-test-split.
python -m vplan eval prepare --seeds dev

# 2. One run per config. Config 5 needs no keys; LLM configs read credentials only from --env-file.
python -m vplan eval run --config 5 --seeds dev
python -m vplan eval run --config 3 --seeds dev --env-file .env
python -m vplan eval run --config 3 --seeds dev --env-file .env --resume runs/<run_dir>   # continue a stopped run

# 3. Build metrics.json and a single self-contained report.html from several runs over one seed set.
python -m vplan report build --run runs/<c3_run> --run runs/<c7_run> --run runs/<c5_run> --output runs/report

# 4. Config 4: Cosmos answers the gate's questions about config 3's saved frames (local vLLM server; resumable).
python -m vplan eval shadow --run runs/<c3_run> --cosmos-url http://127.0.0.1:8000/v1   # -> runs/shadow-<c3_run>/summary.md
```

- Each run directory holds `manifest.json` (code, config and seed-set hashes, versions, provider and model), `sessions.jsonl` and `attempts.jsonl`, written only by the supervising parent. The API key never reaches any of them.
- **Resume** counts every recorded attempt, including provider failures, so failures stay in the denominator. It refuses a changed config, model or seed set.
- **Call deadline:** each planner call gets 60 s of wall-clock time (spec §10, D63); a slower reply is a provider failure. For dev runs on slow free endpoints, `--call-deadline 180` raises it (recorded in the manifest; refused only for Token Factory runs on `test`, the possible headline results), and the attempt limit grows to match (D65).
- **Provenance:** every manifest records the git commit (read from `.git` directly when the git command is unavailable); `eval run` refuses to start without one unless `--allow-no-git`.
- **Frames:** config 3 saves 9 key frames per executed attempt under `frames/` for the Cosmos shadow study (`--frames`/`--no-frames` to change). Report wall times exclude the rendering.
- **The report labels its own status:** INCOMPLETE, LIMITED and DEGRADED (provider failures) runs, and a "not a headline result" stamp unless configs 2, 3 and 7 all ran complete on Nebius Token Factory over the held-out `test` seeds. The repair gain (3 − 2) includes the extra attempts. Only the feedback gain (3 − 7) separates feedback from retrying at the same budget.
- **Container (R8):** `docker/run-local.sh` builds the locked Linux image and runs the test suite and 10 tasks inside it, with no network and no env file.

### Metric definitions

These are fixed before any final run (spec §14). `metrics.json` reports each of them per configuration.

- **Success:**
  - A feasible task succeeds when the Judge reports S1 ∧ S2 ∧ V1–V4: the right object rests in the bin, with no collision, joint-limit, singularity or self-collision violation.
  - An impossible task succeeds only when it is refused.
- **Denominator:** every task in the seed set. Provider failures, malformed output, exhausted budgets and simulator errors all count as failures.
- **First-try success (config 2):** round 1 of each config-3 attempt, scored as a one-attempt run. The round-1 plan passes the verifier and the Judge reports success, or the model correctly refuses an impossible task.
- **Success with repair (config 3):** success within k = 3 repairs, so at most 4 planner calls, on the same tasks and denominator.
- **Repair gain (3 − 2):** the paired difference on the feasible tasks present in both. It includes the effect of the extra attempts, so on its own it does not credit feedback.
- **Feedback gain (3 − 7):** the paired difference against equal-budget resampling (config 7). It separates the effect of feedback from retrying.
- **Intervals:**
  - gains: a paired bootstrap over tasks (10,000 resamples, seed 0), 95% percentile interval;
  - proportions: Wilson 95% intervals, always with N and the seed-set hash.
- **Refusals:** a confusion matrix of feasible vs impossible by refused, attempted or no decision. It counts wrongly attempted impossible tasks and wrongly refused feasible tasks, apart from feasible success. A refusal's reason is also checked against the impossibility proof, per type.
- **Safety:** V1–V4 counts per execution, with the bodies involved grouped by kind. "Zero observed violations in M executions" appears only when every counted execution was measured.
- **Blocked plans:** counted by verifier check and by outcome. Physical-constraint failures (collision, grounding, joint limits, singularity, grasp, release) are kept apart from solver non-convergence (IK accuracy, continuity) and from malformed or failed model output.
- **Breakdowns:** by stratum (distractors × obstacles, feasible tasks), by diagnostic attribution (spec §10), and by tokens and latency per configuration. Latency excludes the key-frame rendering.
- **Headline:** a report is a headline result only when configs 2, 3 and 7 all ran complete on Nebius Token Factory over the held-out `test` seeds. Everything else is labelled a dry run or a development result. The final runs meet these conditions (D70).

## Repository layout

| Path | Contents |
|---|---|
| `src/vplan/records.py` | All data types (frozen pydantic models) |
| `src/vplan/robot.py` | Panda kinematics through ManipulaPy (7-DOF arm, TCP at `panda_grasptarget`) |
| `src/vplan/motion/` | Key poses and trajectory generation |
| `src/vplan/verify/` | Pre-execution checks and finding ordering |
| `src/vplan/execute/` | Approval token, physics execution, grasp attachment |
| `src/vplan/judge/` | Independent scoring and PyBullet Jacobian |
| `src/vplan/planners/` | Heuristic baseline and the Nemotron adapter (forced tool calls) |
| `src/vplan/pipeline.py` | One attempt end to end |
| `src/vplan/repair.py` | `run_task`: attempts, verifier-feedback repair and resampling within a call budget |
| `src/vplan/replay.py` | Keyless playback of recorded executions |
| `src/vplan/ui/` | The Gradio UI: live attempts and keyless replay (view models, renders, recordings, layout) |
| `src/vplan/tasks/` | Scenes, the task generator (feasible and provably impossible tasks), instruction resolution |
| `src/vplan/visual/` | Cosmos client, key-frame rendering and the demo-only visual gate |
| `src/vplan/sim/` | PyBullet scene building and self-collision exclusions |
| `src/vplan/evaluate/` | Seed sets, the supervised evaluation runner, the feedback pilot, statistics, metrics and the HTML report |
| `docker/` | The locked Linux image and the R8 local check |
| `config/` | Verifier, judge and physics thresholds (JSON) |
| `docs/` | [Spec](docs/spec.md), [plan](docs/plan.md), [decisions](docs/decisions.md), [results](docs/results/report.html), [related work](docs/related-work.md), reviews |

## Conventions

- **World frame:** the robot base frame, in metres and radians.
- **Top-grasp yaw:** counter-clockwise about world +z.
- **Jacobian:** space Jacobian at the TCP, `[J_v / L; J_ω]` with L = 0.5 m.

## License

[AGPL-3.0-or-later](LICENSE). The reasons are in `docs/decisions.md` (D2).
