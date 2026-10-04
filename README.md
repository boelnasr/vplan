# vplan: LLMs plan, physics verifies

A language model proposes a pick-and-place plan for a Franka Panda arm. Deterministic kinematic and collision checks then approve or reject the plan **before anything moves**. A rejected plan comes back with structured findings the model can use to repair it. An independent Judge scores every executed run in physics simulation.

Built for the [Nebius x NVIDIA Global AI Hackathon](https://nebiusglobalaihackathon.devpost.com/) (Physical AI track, simulation only). The planner is NVIDIA **Nemotron 3 Ultra**. Final runs use **Nebius Token Factory** and **Nebius Serverless Jobs**.

> **Status: week 1 of 4.** The full plan → verify → execute → judge loop works with a rule-based planner. The Nemotron planner, repair loop, task generator, batch evaluation, Cosmos visual gate and Gradio UI are planned next (see [`docs/plan.md`](docs/plan.md)).

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
| `config/` | Verifier, judge and physics thresholds (JSON) |
| `docs/` | [Spec](docs/spec.md), [plan](docs/plan.md), [decisions](docs/decisions.md), reviews |

## Conventions

- **World frame:** the robot base frame, in metres and radians.
- **Top-grasp yaw:** counter-clockwise about world +z.
- **Jacobian:** space Jacobian at the TCP, `[J_v / L; J_ω]` with L = 0.5 m.

## License

[AGPL-3.0-or-later](LICENSE). The reasons are in `docs/decisions.md` (D2).
