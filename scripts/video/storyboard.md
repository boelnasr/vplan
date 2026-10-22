# vplan demo video: storyboard

The demo video for the Nebius x NVIDIA Global AI Hackathon. It must stay under 3 minutes, with at least 1 minute of
the modules in action. It follows the storyboard in `docs/spec.md` §3. Live Nemotron calls cost money, so the Nemotron
chapters replay attempts recorded in the held-out evaluation (run `c3-test`, Nemotron 3.5 Lightning on Nebius Token
Factory). The live chapter uses the rule-based baseline. There is no Cosmos chapter: the UI's Cosmos banner still cites
the pilot's balanced accuracy of 0.70, not the shadow study's 0.6, until that is decided.

`record.mjs` holds the takes and is the source of truth for the captions. This file lists them. The times below are
from the final take (Oct 22). They move by a few tenths of a second with every recording.

| # | Chapter | Starts at | Length | What is on screen |
|---|---|---|---|---|
| 1 | title | 0:00.0 | 6.5 s | Title card |
| 2 | live | 0:06.5 | 41.6 s | Chapter card, then the Live tab: basic scene with the wall, rule-based baseline, plan, checks, execution, judge |
| 3 | repair | 0:48.1 | 59.6 s | Chapter card, then the Replay tab: attempt `3:test:1`, two rejections, the repair, execution, judge |
| 4 | refusal | 1:47.7 | 22.5 s | Chapter card, then the Replay tab: attempt `3:test:3`, a correct refusal |
| 5 | results | 2:10.2 | 17.0 s | Results card |
| 6 | closing | 2:27.2 | 6.0 s | Closing card |
| | **Total** | | **2:33.2** | Modules in action (chapters 2 to 4): 123.7 s |

Waits are cut. The page load before each UI chapter, the scene load, the planner run and the replay load keep their
first 0.6 s and last 0.4 s.

## 1 title (6.5 s)

Card: **vplan LLMs plan, physics verifies**. "NVIDIA Nemotron proposes a pick-and-place plan for a robot arm. Kinematic
and collision checks verify it before anything moves, and an independent judge scores the result in physics
simulation." Pipeline: 1 Plan → 2 Verify → 3 Execute → 4 Judge. Footer: "Nebius × NVIDIA Global AI Hackathon ·
Physical AI track · simulation only".

## 2 live (41.6 s)

Card (4.5 s): "Part 1 of 3. A live run in the browser. Plan, verify, execute and judge one task. The planner here is
the rule-based baseline, so no API key is needed."

| Time in video | Caption | Box or click |
|---|---|---|
| 0:11.2 | A simulated Franka Panda arm, two cubes and a bin. The task is typed in plain English. | box: Instruction |
| 0:15.7 | Put a wall between the cubes and the bin. | click: "Basic scene with wall"; box: scene picture |
| 0:20.2 | The planner here is the rule-based baseline: no LLM and no API key. | box: Planner |
| 0:23.7 | One click runs the whole loop: plan, verify, execute, judge. | click: "Plan, verify, execute" |
| 0:26.1 | 1 Plan: a typed plan with the grasp, waypoints over the wall and the drop-off point. | box: Plan (JSON) |
| 0:30.7 | 2 Verify: eight kinematic and collision checks run in a preview simulation. All pass, so the plan is approved. | box: Round 1 card |
| 0:37.4 | 3 Execute: only the approved plan runs, in PyBullet physics: over the wall and into the bin. | box: execution |
| 0:42.6 | 4 Judge: an independent judge scores the result: cube in the bin, no unsafe contact, all limits kept. | box: Judge |

## 3 repair (59.6 s)

Card (5 s): "Part 2 of 3. Nemotron repairs its own plan. A recorded NVIDIA Nemotron attempt from the held-out
evaluation, replayed without keys. The checks reject two plans; the third passes."

| Time in video | Caption | Box or click |
|---|---|---|
| 0:53.3 | A recorded attempt by NVIDIA Nemotron 3.5 Lightning on a held-out task. Replay needs no API key, GPU or network. | box: run and seed-set folders |
| 0:57.8 | Pick the attempt and replay it. | clicks: Attempt list, "Task 1 (test set) · Executed · success", Replay |
| 1:02.0 | Task: pick the object next to the red cylinder and place it in the bin. Nemotron needed three plans. | box: outcome card |
| 1:07.6 | Round 1: Nemotron's first plan sets the drop-off point 8 cm too high. The checks reject it before anything moves. | box: Round 1 card |
| 1:13.7 | Round 2: told what failed, it fixes the height. Now the carried cylinder passes 0.95 cm from the wall; the minimum is 1 cm. | box: Round 2 card |
| 1:21.3 | The checks draw the parts involved in magenta: the wall, and the cylinder in the gripper. | click: Round 2 close-up; box: gallery |
| 1:27.6 | Round 3: Nemotron adds high waypoints (left) to carry the cylinder over the wall. All eight checks pass. | boxes: Plan (JSON), Round 3 card |
| 1:34.6 | 3 Execute: the approved plan runs in PyBullet physics, played back from the recording. | box: execution |
| 1:40.9 | 4 Judge: the independent judge confirms it: the cylinder rests in the bin and nothing unsafe happened. | box: Judge |

## 4 refusal (22.5 s)

Card (4 s): "Part 3 of 3. Knowing when to refuse. Also recorded: a task the arm cannot do, from the same held-out
evaluation."

| Time in video | Caption | Box or click |
|---|---|---|
| 1:51.9 | Another recorded Nemotron attempt from the held-out set. | clicks: Attempt list, "Task 3 (test set) · Refused · success", Replay |
| 1:55.3 | The yellow cylinder is out of the arm's reach. The planner is not told this. | box: outcome card (ground truth) |
| 1:59.8 | Nemotron refuses instead of planning, and says why: the cylinder is about 1.2 m away, beyond the 0.95 m reach. | box: Round 1 card |
| 2:05.8 | The judge accepts the refusal: refusing is correct only when the task is impossible. (top of the screen) | box: Judge |

## 5 results (17 s)

The card builds up in steps. Its numbers come from `docs/results/metrics.json`.
- **Results on 300 held-out tasks.** NVIDIA Nemotron 3.5 Lightning on Nebius Token Factory. Share of solvable tasks
  done correctly:
- **24.7%** first plan, verified (95% CI 20.1–29.8%) → **65.0%** with up to 3 repairs from the checks' findings (95%
  CI 59.4–70.2%)
- **+29.3 pts** over retrying the same number of times without feedback (95% CI 23–36)
- **0 unsafe** verified executions. Without the checks: 169 of 346 broke a safety rule.
- **100%** rule-based baseline with the same checks. By design, every solvable task is one it can solve.
- Footer: "Held-out set: 300 solvable and 100 impossible generated tasks. The planner prompt was frozen before it was
  generated."

## 6 closing (6 s)

Card: **vplan LLMs plan, physics verifies**. "Planner: NVIDIA Nemotron, served by Nebius Token Factory. Checks:
ManipulaPy kinematics and PyBullet collision checks, before anything moves. Judge: independent scoring in PyBullet
physics." Footer: "Open source under AGPL-3.0 · Every recorded attempt replays without keys".
