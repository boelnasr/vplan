# vplan: LLMs plan, physics verifies

## Inspiration

A language model can turn "put the red cube in the bin" into a robot plan, but nothing in the model checks that the plan is reachable, collision-free or clear of singularities. I wanted the model to propose and physics to decide, with every rejection coming back as evidence the model can act on: what failed, by how much and which plan field to change. It is for robotics developers who review LLM-proposed plans and need to see why one is unsafe and whether a repair fixed it.

## What it does

A user gives a plain-language instruction for a simulated Franka Panda arm. Nemotron answers with a typed plan or a refusal. A motion layer turns the plan into a joint trajectory, and a Verifier runs eight kinematic and collision checks on it in a preview simulation. Each failure becomes a finding with the measured value, the threshold and the plan field to change, such as a carried object 0.95 cm from a wall, against a 1 cm minimum. Findings go back to Nemotron for up to three repairs. Only a plan that passes gets an approval token, bound to its exact trajectory, scene, target and configuration, and executes in PyBullet. An independent Judge that shares no code with the Verifier scores the run. A Gradio UI shows every step live and replays any recorded attempt without keys.

## How I built it

Python 3.10, PyBullet and ManipulaPy 1.4.2, my open-source robotics library, updated for this project during the submission period. A task generator builds feasible tasks with a witness plan and provably impossible ones: target out of reach, bin out of reach, enclosed target, absent object. A supervised, resumable runner evaluates hash-frozen seed sets, and a self-contained HTML report gives Wilson and paired-bootstrap intervals.

## How I use NVIDIA models

- **Nemotron 3.5 Lightning** (`nvidia/Nemotron-3_5-Lightning`) on Nebius Token Factory plans in every evaluation configuration. Every Nemotron call forces one tool, `submit_plan`, whose parameters are the plan schema. The final evaluation made 2,510 calls for about $1.61 at catalog prices, and a later rerun of the repair configuration 976 more for $0.71.
- **Nemotron 3 Ultra** (`nvidia/Nemotron-3-Ultra-550b-a55b`) on Token Factory is the planner for live demos in the UI. In my smoke test it returned 20 valid tool calls of 20 (median 8.6 s, slowest 28.5 s). I also ran it on the same held-out tasks as a second evaluation: 1,417 calls for about $14.70, and 493 more for a rerun of the repair configuration ($5.65). The video makes no live Nemotron calls; its Nemotron chapters replay recorded Lightning attempts from the evaluation.
- **Cosmos3-Edge**, served locally with vLLM on a 16 GB RTX 5070 Ti, is an opt-in, advisory learned visual check. After the Verifier passes a plan, it looks at rendered pre-grasp, grasp and pre-release frames and can only add rejections. A shadow study over all 215 executed repair attempts (1,337 questions) showed it is not a usable gate: balanced accuracy 0.61 on real attempts and 0.52 on injected errors, 27% false rejections of correct pre-release frames, and 1 of 159 wrong-object pre-grasps caught. It never gates the evaluation. The video's Cosmos chapter shows it stopping a fully verified plan with a false rejection at the grasp frame.

## Challenges I ran into

- **Missing waypoints.** In obstacle scenes, 212 of 218 unverified Lightning first plans had no transport waypoints, so the carried object moved straight into the obstacles.
- **A conservative Verifier.** Verified first attempts succeed less often (24.7%) than unverified ones (43.3%), because the Verifier also blocks plans the Judge would pass, such as clearances just under 1 cm.
- **Provider latency.** On the same development tasks, answered calls took 8.7 s median on Oct 19 and 29.2 s on Oct 21 with similar token counts, against a 60 s deadline.

## Accomplishments that I'm proud of

On 300 held-out, reference-solvable tasks, verifier-feedback repair raised Lightning's success from 24.7% (95% CI 20.1–29.8%) for verified first attempts to 65.0% (59.4–70.2%). At equal budget, feedback beat resampling by 29.3 points (23–36), which my 20-task Ultra pilot could not show (17 against 17). No verified execution broke a safety predicate (0 of 215 with repair, 0 of 123 with resampling), while 169 of 346 unverified executions did.

I then ran Nemotron 3 Ultra on the same 300 tasks. It reached 92.7% with repair (89.1–95.1%), up from 75.7% for verified first attempts, and again no verified execution was unsafe, while 42 of 304 unverified ones collided. These runs came after the freeze and after I had seen the Lightning results, so I report them beside the headline, not instead of it.

After the evaluation I made collision findings richer: each one now names which part came too close to what, in which phase, and by how much. Rerun on the same 300 tasks, Lightning's repair rose from 65.0% to 76.0% (paired +11.0 points, CI +4.7 to +17.3), from better repairs: its first plans did slightly worse than before. Ultra's rose from 92.7% to 96.7%, but mostly from better first plans, which the change cannot cause. No verified execution was unsafe (0 of 256 and 0 of 303). These runs also came after the freeze, so they sit beside the headline as well.

## What I learned

- The repair gain over first attempts (+40.3 points, 34.7–46.0) mixes feedback with extra attempts; only the equal-budget control separates the two.
- Feedback matters most for the weaker model. Ultra plans transport waypoints on its own (184 of 222 first plans in obstacle scenes, against 6 of 218 for Lightning), and for Ultra, retrying at equal budget did as well as repair (94.7% against 92.7%; −2.0 points, CI −5.7 to +1.7). The verifier is what keeps both safe.
- Numbers in a prompt seem to anchor the model: with stated lift and clearance values, repair waypoints sat lower and development-set repair dropped from 50 to 42 of 75 (−10.7 points, CI −26.7 to +4.0, crossing zero).

## What's next

- Finish testing a revised prompt that states the motion facts without numbers, on top of the richer findings.
- Refusals: Lightning never refused an enclosed target (0 of 25 in every configuration); Ultra refused 17–24 of 25.
- Cosmos: 49 of its 53 false pre-release rejections said the object was inside the bin, not over it. A reworded question needs fresh frames, not the held-out ones.
