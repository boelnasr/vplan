# vplan: LLMs plan, physics verifies

A language model proposes a pick-and-place plan for a Franka Panda arm. Deterministic kinematic and collision checks then approve or reject the plan **before anything moves**. A rejected plan comes back with structured findings the model can use to repair it. An independent Judge scores every executed run in physics simulation.

Built for the [Nebius x NVIDIA Global AI Hackathon](https://nebiusglobalaihackathon.devpost.com/) (Physical AI track, simulation only). The planner is NVIDIA **Nemotron 3 Ultra**. Final runs use **Nebius Token Factory** and **Nebius Serverless Jobs**.

> **Status: weeks 1–2 of 4 done.** The plan → verify → execute → judge loop runs with both a rule-based planner and Nemotron 3 Ultra. Also in place: a task generator with provably impossible tasks, verifier-feedback repair, keyless replay of recorded runs, a supervised pilot runner, and a demo-only Cosmos visual gate. The feedback pilot has run against a live model (results below). ManipulaPy 1.4.2 carries the upstream fixes this project needed ([below](#manipulapy-updates-during-the-submission-period)). Week 3 covers the Nebius Token Factory runs, batch evaluation, report and Gradio UI (see [`docs/plan.md`](docs/plan.md)).

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
- **Intent check (tested, not used):** we tried fast "System One" decision models as a pre-execution check that the planner picked the instructed object, on 240 held-out generated tasks.
  - Laya (open, 421M) was perfect on color and shape phrasing but at or below chance on spatial relations ("leftmost", "next to").
  - Jev (closed, via OpenRouter) reached 84% overall but missed the bar we fixed in advance, and was near chance on "next to".
  - The right object is therefore scored only by the Judge, after execution (D50).

## Setup

Requires **Python 3.10**.

```bash
pip install "ManipulaPy[simulation]==1.4.2" pybullet==3.2.7 numpy==2.2.6 pydantic==2.13.5 openai==3.24.0 matplotlib==3.10.9 pytest==9.1.1
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

## Feedback pilot

A 20-task, feasible-only development experiment comparing the first attempt, verifier-feedback repair and equal-budget resampling ([`docs/pilot-plan.md`](docs/pilot-plan.md), D32, D33). It is diagnostic, not a held-out result.

Latest run: hard cohort (`--generator hard`), free Nemotron 3 Ultra on OpenRouter, temperature 0.7 ([D46](docs/decisions.md)):

| | successes / 20 |
|---|---|
| first attempt | 9 |
| verifier-feedback repair | 17 |
| equal-budget resampling | 17 |

Both retry strategies roughly double the success rate. Repair used fewer calls (32 vs 41) and never repeated a plan (resampling repeated 3), but the pilot does not separate the two on success. Three provider 503s stay in the denominators. The week-3 batch evaluation decides the comparison.

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

## Repository layout

| Path | Contents |
|---|---|
| `src/vplan/records.py` | All data types (frozen pydantic models) |
| `src/vplan/robot.py` | Panda kinematics through ManipulaPy (7-DOF arm, TCP at `panda_grasptarget`) |
| `src/vplan/motion/` | Key poses and trajectory generation |
| `src/vplan/verify/` | Pre-execution checks and finding ordering |
| `src/vplan/execute/` | Approval token, physics execution, grasp attachment |
| `src/vplan/judge/` | Independent scoring and PyBullet Jacobian |
| `src/vplan/planners/` | Heuristic baseline and the Nemotron 3 Ultra adapter (forced tool calls) |
| `src/vplan/pipeline.py` | One attempt end to end |
| `src/vplan/repair.py` | `run_task`: attempts, verifier-feedback repair and resampling within a call budget |
| `src/vplan/replay.py` | Keyless playback of recorded executions |
| `src/vplan/tasks/` | Scenes, the task generator (feasible and provably impossible tasks), instruction resolution |
| `src/vplan/visual/` | Cosmos client, key-frame rendering and the demo-only visual gate |
| `src/vplan/sim/` | PyBullet scene building and self-collision exclusions |
| `src/vplan/evaluate/` | Feedback pilot preparation, supervised runs and reports |
| `config/` | Verifier, judge and physics thresholds (JSON) |
| `docs/` | [Spec](docs/spec.md), [plan](docs/plan.md), [decisions](docs/decisions.md), reviews |

## Conventions

- **World frame:** the robot base frame, in metres and radians.
- **Top-grasp yaw:** counter-clockwise about world +z.
- **Jacobian:** space Jacobian at the TCP, `[J_v / L; J_ω]` with L = 0.5 m.

## License

[AGPL-3.0-or-later](LICENSE). The reasons are in `docs/decisions.md` (D2).
