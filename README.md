# vplan: LLMs plan, physics verifies

A language model proposes a pick-and-place plan for a Franka Panda arm. Deterministic kinematic and collision checks then approve or reject the plan **before anything moves**. A rejected plan comes back with structured findings the model can use to repair it. An independent Judge scores every executed run in physics simulation.

Built for the [Nebius x NVIDIA Global AI Hackathon](https://nebiusglobalaihackathon.devpost.com/) (Physical AI track, simulation only). The planner is NVIDIA **Nemotron 3 Ultra**. Final runs use **Nebius Token Factory** and **Nebius Serverless Jobs**.

> **Status: week 1 of 4.** The full plan → verify → execute → judge loop works with a rule-based planner. The Nemotron planner adapter, the repair-versus-resampling loop and the 20-task development feedback pilot (below) are implemented and tested offline; the pilot has not yet been run against a live model. The full task generator, batch evaluation, Cosmos visual gate and Gradio UI are planned next (see [`docs/plan.md`](docs/plan.md)).

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

## Setup

Requires **Python 3.10**.

```bash
pip install "ManipulaPy[simulation]==1.4.1" pybullet==3.2.7 numpy==2.2.6 pydantic==2.13.5 openai==3.24.0 matplotlib==3.10.9 pytest==9.1.1
cp .env.example .env   # LLM settings, needed from week 2; never commit .env
```

The pinned versions are listed in [`pyproject.toml`](pyproject.toml).

On Windows, `pip` may build pybullet from source, which needs the MSVC build tools.

## Run

On Windows (PowerShell), from the repo root:

```powershell
$env:PYTHONUTF8 = "1"        # ManipulaPy prints an emoji banner that breaks the cp1252 console
$env:PYTHONPATH = "src;."    # use "src:." on Linux/macOS
python -m vplan demo-once                     # one attempt, headless
python -m vplan demo-once --obstacle --gui    # with a wall in the way, watched in the PyBullet GUI
python -m pytest -q                           # test suite (~100 s)
```

`demo-once` prints the outcome, any verifier findings and the Judge result. It exits `0` when the Judge reports success.

## Feedback pilot

A 20-task, feasible-only development experiment comparing the first attempt, verifier-feedback repair and equal-budget resampling ([`docs/pilot-plan.md`](docs/pilot-plan.md), D32, D33). It is diagnostic, not a held-out result.

```powershell
# 1. Freeze the cohort: local simulation only, no model calls (a few minutes). Refuses a non-empty directory.
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
```

- `prepare` writes `tasks.jsonl` (planner-visible), `oracle.jsonl` (private targets and reference plans), `exclusions.jsonl` and, last, `manifest.json` with file, config and source hashes.
- `run` refuses before any request if the manifest is missing, a task/oracle file or config changed, or any `src/vplan/**/*.py` / `config/*.json` file changed (`--allow-source-drift` records the drift instead). Without `--allow-paid` it only calls `https://openrouter.ai` models named `nvidia/...:free`; `--model` overrides `LLM_MODEL`. Every call in both arms uses `--temperature` (default 0.7), and the arm that runs first alternates per task. Pass relative or Windows paths, not `/mnt/...`.
- Each run gets a new `runs/<run_id>/` with `run.json`, `events.jsonl` and `results.jsonl` (flushed per event / per task), then `summary.json` and `report.md`. A run that did not finish is titled `INCOMPLETE: k/n tasks`, a `--limit` run `LIMITED: first k of n tasks`, and any provider service failure adds `DEGRADED: n service failures`. A 401/402/403/404, or three consecutive failed requests, stops the run as incomplete. `pilot report` exits 1 for an incomplete run.

## Repository layout

| Path | Contents |
|---|---|
| `src/vplan/records.py` | All data types (frozen pydantic models) |
| `src/vplan/robot.py` | Panda kinematics through ManipulaPy (7-DOF arm, TCP at `panda_grasptarget`) |
| `src/vplan/motion/` | Key poses and trajectory generation |
| `src/vplan/verify/` | Pre-execution checks and finding ordering |
| `src/vplan/execute/` | Approval token, physics execution, grasp attachment |
| `src/vplan/judge/` | Independent scoring and PyBullet Jacobian |
| `src/vplan/planners/` | Planners (week 1: heuristic baseline) |
| `src/vplan/pipeline.py` | One attempt end to end |
| `src/vplan/evaluate/` | Feedback pilot preparation, runs and report |
| `config/` | Verifier, judge and physics thresholds (JSON) |
| `docs/` | [Spec](docs/spec.md), [plan](docs/plan.md), [decisions](docs/decisions.md), reviews |

## Conventions

- **World frame:** the robot base frame, in metres and radians.
- **Top-grasp yaw:** counter-clockwise about world +z.
- **Jacobian:** space Jacobian at the TCP, `[J_v / L; J_ω]` with L = 0.5 m.

## License

[AGPL-3.0-or-later](LICENSE). The reasons are in `docs/decisions.md` (D2).
