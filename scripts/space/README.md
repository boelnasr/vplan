---
title: "vplan: LLMs plan, physics verifies"
emoji: 🦾
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
python_version: "3.10"
license: agpl-3.0
short_description: An LLM plans a robot pick-and-place, physics verifies it
startup_duration_timeout: 30m
---

# vplan: LLMs plan, physics verifies

I built vplan for the [Nebius x NVIDIA Global AI Hackathon](https://nebiusglobalaihackathon.devpost.com/) (Physical AI track, simulation only). An NVIDIA **Nemotron** model proposes a pick-and-place plan for a simulated Franka Panda arm, and deterministic kinematic and collision checks approve or reject it **before anything moves**. Rejected plans go back to the model with structured findings for repair, and an independent Judge scores every executed run in PyBullet physics.

## The result

> **Bounded repair with verifier feedback raised Nemotron 3.5 Lightning's task success from 24.7% (verified first attempts) to 65.0% across 300 held-out, reference-solvable generated tasks. At equal budget, feedback beat resampling by 29.3 points (95% CI 23–36).**

That is Nemotron 3.5 Lightning on Nebius Token Factory, on the held-out `test` set, from Oct 18. Two later results are reported beside it, not instead of it:

- **Nemotron 3 Ultra**, run on the same tasks, reaches 92.7% with repair, but retrying at equal budget does as well (94.7%; difference −2.0 points, CI −5.7 to +1.7). Feedback is what rescues the weaker model.
- **After the evaluation**, I merged richer repair findings. On the same tasks Lightning's repair rose to 76.0% (paired +11.0 points, CI +4.7 to +17.3) and Ultra's to 96.7% (+4.0, CI +0.7 to +7.3), though Ultra's gain came mostly from better first plans, which that change does not touch.

No verified execution of either model broke a safety predicate in any of these runs. All intervals and every configuration are in the [results section of the repository README](https://github.com/boelnasr/vplan#results).

## What you can do here

- **Live tab.** Choose a scene (the basic scene, the basic scene with a wall, or a generated task by seed and type), choose a planner and press *Plan, verify, execute*. You see the plan, the checks round by round, the execution in the physics simulation and the Judge's verdict.
  - *Rule-based baseline* needs no key.
  - *Your model (your API key)* plans with a model of your choice. Pick Nebius Token Factory or OpenRouter, or a custom OpenAI-compatible endpoint, then choose or type a model ID and paste your API key. The model must support forced tool calls (a `tool_choice` that names a function), because the plan comes back as a tool call. A run makes at most four planner calls, each with a 60 s deadline.
- **Replay tab.** Pick one of the bundled runs, press *Open*, choose an attempt and press *Replay*. These are real attempts from the held-out evaluation (Lightning, the headline run, and Ultra with the richer findings): repairs that needed two and three rounds, a first-round success, correct refusals, a task whose repair budget ran out, and a run the Judge failed. Replay needs no key and calls no model.

## How your key is handled

- You type it into a password box. It is sent with the *Plan, verify, execute* request, builds the planner for that one run, and is dropped when the run ends. It is not kept in the page state, written to disk or logged, and it never appears in the results.
- The Space records nothing, and the Cosmos visual check is off.
- Calls are billed to your provider account at its prices. Use a key with a spending limit, or a throwaway key.
- The Space only calls `https` endpoints, refuses a host name that resolves to a private or local address, and does not follow redirects.

The Space runs one simulation at a time. If it says the simulator is busy, wait a moment and try again.

## Links

- Code, setup and the full results: https://github.com/boelnasr/vplan (AGPL-3.0)
- Demo video (2:57): https://youtu.be/y_Ig1zOWKPY
