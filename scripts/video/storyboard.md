# vplan demo video: storyboard

The demo video for the Nebius x NVIDIA Global AI Hackathon. It must stay under 3 minutes, with at least 1 minute of
the modules in action. It follows the storyboard in `docs/spec.md` §3. Live Nemotron calls cost money, so the Nemotron
chapters replay attempts recorded in the held-out evaluation (run `c3-test`, Nemotron 3.5 Lightning on Nebius Token
Factory). The live chapters use the rule-based baseline. The optional Cosmos chapter runs NVIDIA Cosmos3-Edge locally
and presents it as an advisory, learned check, never as verification.

`record.mjs` holds the takes and is the source of truth for the captions. This file lists them. The times below are
from the final take (Oct 23, D83). They move by a few tenths of a second with every recording.

| # | Chapter | Starts at | Length | What is on screen |
|---|---|---|---|---|
| 1 | title | 0:00.0 | 4.5 s | Title card |
| 2 | architecture | 0:04.5 | 27.5 s | The system-overview diagram, one part at a time |
| 3 | live | 0:32.1 | 36.6 s | Chapter card, then the Live tab: basic scene with the wall, rule-based baseline, plan, checks, execution, judge |
| 4 | repair | 1:08.7 | 48.0 s | Chapter card, then the Replay tab: attempt `3:test:1`, two rejections, the repair, execution, judge |
| 5 | refusal | 1:56.7 | 20.0 s | Chapter card, then the Replay tab: attempt `3:test:3`, a correct refusal |
| 6 | cosmos (optional) | 2:16.6 | 18.8 s | Chapter card, then the Live tab with the Cosmos gate on: basic scene, rule-based baseline, one round, Cosmos's verdicts and key frames |
| 7 | results | 2:35.4 | 17.0 s | Results card: Lightning (the headline) and Ultra |
| 8 | closing | 2:52.4 | 4.5 s | Closing card |
| | **Total** | | **2:56.9** | Modules in action (chapters 3 to 6): 123.4 s. Without the cosmos chapter: 2:38.1 |

Waits are cut. The page load before each UI chapter, the scene load, the planner run and the replay load keep their
first 0.6 s and last 0.4 s.

## 1 title (4.5 s)

Card: **vplan LLMs plan, physics verifies**. "NVIDIA Nemotron proposes a pick-and-place plan for a robot arm. Kinematic
and collision checks verify it before anything moves, and an independent judge scores the result in physics
simulation." Pipeline: 1 Plan → 2 Verify → 3 Execute → 4 Judge. Footer: "Nebius × NVIDIA Global AI Hackathon ·
Physical AI track · simulation only".

## 2 architecture (27.5 s)

Card: kicker "How it works", heading **The model proposes, physics decides**, and the SVG of
`docs/diagrams/system-overview.html` without its legend strip. Each caption boxes the parts it names.

| Time in video | Caption | Box |
|---|---|---|
| 0:05.4 | A harness generates solvable and provably impossible tasks, runs them and builds the report. | the evaluation harness band |
| 0:08.9 | NVIDIA Nemotron on Nebius Token Factory answers each task with a typed plan, or a refusal. | Nemotron planner |
| 0:13.0 | The plan becomes a joint trajectory. Eight kinematic and collision checks test it in a preview simulation. | Motion layer and Verifier |
| 0:18.0 | A rejected plan goes back with findings: what failed, by how much, and what to change. Up to 3 repairs. | Verifier; Nemotron planner |
| 0:23.0 | Only an approved plan runs in PyBullet physics. An independent judge, sharing no code with the checks, scores it. | Executor and Judge |
| 0:28.1 | Optional: NVIDIA Cosmos3-Edge looks at rendered frames after the checks pass. It can only block. | Cosmos visual gate |

## 3 live (36.6 s)

Card (3 s): "Part 1 of 3. A live run in the browser. Plan, verify, execute and judge one task. The planner here is
the rule-based baseline, so no API key is needed."

| Time in video | Caption | Box or click |
|---|---|---|
| 0:35.2 | A simulated Franka Panda arm, two cubes and a bin. The task is typed in plain English. | box: Instruction |
| 0:39.2 | Put a wall between the cubes and the bin. The planner is the rule-based baseline: no LLM, no API key. | click: "Basic scene with wall"; boxes: scene picture, then Planner |
| 0:45.7 | One click runs the whole loop: plan, verify, execute, judge. | click: "Plan, verify, execute" |
| 0:48.1 | 1 Plan: a typed plan with the grasp, waypoints over the wall and the drop-off point. | box: Plan (JSON) |
| 0:52.2 | 2 Verify: eight kinematic and collision checks run in a preview simulation. All pass, so the plan is approved. | box: Round 1 card |
| 0:58.0 | 3 Execute: only the approved plan runs, in PyBullet physics: over the wall and into the bin. | box: execution |
| 1:03.1 | 4 Judge: an independent judge scores the result: cube in the bin, no unsafe contact, all limits kept. | box: Judge |

## 4 repair (48.0 s)

Card (3.5 s): "Part 2 of 3. Nemotron repairs its own plan. A recorded NVIDIA Nemotron attempt from the held-out
evaluation, replayed without keys. The checks reject two plans; the third passes."

| Time in video | Caption | Box or click |
|---|---|---|
| 1:12.3 | A recorded attempt by NVIDIA Nemotron 3.5 Lightning on a held-out task. Replay needs no API key, GPU or network. | box: run and seed-set folders |
| 1:16.4 | Pick the attempt and replay it. | clicks: Attempt list, "Task 1 (test set) · Executed · success", Replay |
| 1:20.5 | Task: pick the object next to the red cylinder and place it in the bin. Nemotron needed three plans. | box: outcome card |
| 1:25.7 | Round 1: Nemotron's first plan sets the drop-off point 8 cm too high. The checks reject it before anything moves. | box: Round 1 card |
| 1:31.2 | Round 2: told what failed, it fixes the height. Now the carried cylinder passes 0.95 cm from the wall; the minimum is 1 cm. | box: Round 2 card |
| 1:37.5 | Round 3: Nemotron adds high waypoints (left) to carry the cylinder over the wall. All eight checks pass. | boxes: Plan (JSON), Round 3 card |
| 1:44.0 | 3 Execute: the approved plan runs in PyBullet physics, played back from the recording. | box: execution |
| 1:50.1 | 4 Judge: the independent judge confirms it: the cylinder rests in the bin and nothing unsafe happened. | box: Judge |

The D78 cut also showed the magenta close-up of the failed parts after round 2; it is left out to fit the architecture
chapter under 3 minutes.

## 5 refusal (20.0 s)

Card (3 s): "Part 3 of 3. Knowing when to refuse. Also recorded: a task the arm cannot do, from the same held-out
evaluation."

| Time in video | Caption | Box or click |
|---|---|---|
| 1:59.8 | Another recorded Nemotron attempt from the held-out set. | clicks: Attempt list, "Task 3 (test set) · Refused · success", Replay |
| 2:03.1 | The yellow cylinder is out of the arm's reach. The planner is not told this. | box: outcome card (ground truth) |
| 2:07.2 | Nemotron refuses instead of planning, and says why: the cylinder is about 1.2 m away, beyond the 0.95 m reach. | box: Round 1 card |
| 2:12.7 | The judge accepts the refusal: refusing is correct only when the task is impossible. (top of the screen) | box: Judge |

## 6 cosmos (18.8 s, optional)

Recorded alone (`node record.mjs --cosmos`), with the UI started with `--cosmos-url` and a local Cosmos3-Edge server.
The Repair rounds slider is set to 0 in the cut page load, so the chapter shows one plan. Cosmos's verdicts vary from
run to run (D73: false rejections and replies the strict parser cannot read). The take reads the verdict from the page
and uses the caption written for it, so the third and fourth captions depend on the run. The final take got **Fail**
at the grasp frame: Cosmos said the cube was already held, a false rejection of a plan that passed every check.

Card (2.5 s): "Optional. A learned visual check. NVIDIA Cosmos3-Edge, served locally, looks at rendered key frames
after the checks pass. It is advisory, never verification."

| Time in video | Caption | Box or click |
|---|---|---|
| 2:19.3 | Now with the Cosmos visual check on: the same rule-based baseline, on the basic scene without the wall. (top) | click: "Plan, verify, execute" |
| 2:22.6 | It is advisory, not verification: in our shadow study its balanced accuracy was 0.61 on real attempts and 0.52 on injected errors. (top) | box: Cosmos banner |
| 2:27.5 | The eight checks pass, but Cosmos rejects the plan at the grasp frame, so nothing moves. | box: Round 1 card with the verdict table |
| 2:31.5 | These are the frames it saw. The plan passed every check, yet it was stopped: this is why Cosmos stays advisory. (top) | box: Cosmos key frames |

The third caption on other outcomes:
- **Unsure:** "The eight checks pass, but Cosmos gives no usable answer, even with a third view, so nothing moves." The
  fourth caption is the same as for Fail.
- **Pass:** "The eight checks pass, and Cosmos agrees at every key frame, so the plan runs." The fourth caption is then
  "These are the frames it saw. Even when it agrees, it only advises: the checks decide."

## 7 results (17 s)

The card builds up in steps: the heading, the Lightning row at 0.8 s, the Ultra row at 5.4 s, the merged-findings row at
10 s and the footer at 13.6 s. Its numbers come from `docs/results/metrics.json` (Lightning, D70),
`docs/results/ultra/metrics.json` (Ultra, D82) and `docs/results/a1/` with the paired comparisons in D85.
- **Results on 300 held-out tasks.** "Share of solvable tasks done correctly. NVIDIA Nemotron on Nebius Token Factory."
- **Nemotron 3.5 Lightning**, "the headline: prompt and code frozen before the held-out set": **24.7% → 65.0%** first
  plan verified → up to 3 repairs from the findings (95% CI 20.1–29.8% → 59.4–70.2%); **+29.3 pts** over retrying as
  often without feedback (95% CI 23–36); **0 of 215** verified runs unsafe, without the checks 169 of 346.
- **Nemotron 3 Ultra**, "run later, on code changed after the freeze": **75.7% → 92.7%** (95% CI 70.5–80.2% →
  89.1–95.1%); **−2.0 pts** over retrying, "the stronger model gains nothing from feedback" (95% CI −5.7 to +1.7);
  **0 of 298** verified runs unsafe, without the checks 42 of 304.
- **Richer findings**, "merged after the evaluation: each collision says what hit what, and how far": **76.0%**
  Lightning with repair, up from 65.0% on the same tasks (paired +11.0 pts, 95% CI +4.7 to +17.3); **96.7%** Ultra with
  repair, was 92.7% ("+4.0 pts, mostly from first plans, not repairs"); **0 of 559** verified runs unsafe, both models.
- Footer: "Held-out set: 300 solvable and 100 impossible generated tasks. Later runs are reported beside the headline,
  not instead of it."

## 8 closing (4.5 s)

Card: **vplan LLMs plan, physics verifies**. "Planner: NVIDIA Nemotron 3.5 Lightning and 3 Ultra, served by Nebius
Token Factory. Checks: ManipulaPy kinematics and PyBullet collision checks, before anything moves. Judge: independent
scoring in PyBullet physics." Footer: "Open source under AGPL-3.0 · Every recorded attempt replays without keys".
