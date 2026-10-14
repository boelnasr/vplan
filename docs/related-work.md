# Related work

A survey of the prior work closest to vplan, as of 2026-10-14. It is a targeted search (about 25 queries), not a systematic review.

## Summary

**No single prior work combines all of vplan's pieces, but every piece has close prior art:**
- **Closest systems:** PRoC3S (CoRL 2024) and LLM³ (IROS 2024). Both check LLM manipulation plans in PyBullet before execution and feed structured failures back to the model.
- **Hong et al. (ECCV 2026 Workshop):** schema-validated typed tool calls in front of a deterministic MoveIt verify-then-act gate.
- **RoboCritics (HRI 2026):** motion-level critics that return structured feedback.
- **Approval binding:** concurrent work (Heydari Shahna et al., arXiv, 9 Oct 2026) binds an approval to the exact generated motion and rechecks authority and measured joint drift at dispatch.

**Two more pieces already exist:**
- **Refusal tests on impossible tasks** with machine-checkable ground truth: RoboAbstention 2026, plan-failure-bench 2026, and PlanBench's unsolvable instances.
- **"VLM success judges are weak":** in FailBench 2026, the best of 13 detectors reaches only 0.77 mean balanced accuracy.

**The defensible gap is methodological.** No manipulation-planning paper found does all three of these:
- compares verifier-feedback repair with independent resampling at an equal budget, from a shared first attempt;
- scores the result with a judge that shares no code with the verifier;
- puts provably impossible tasks into the same evaluation.

The robotics papers compare feedback against no-feedback or single-shot arms (PRoC3S, AutoTAMP), or leave the no-feedback arm underspecified (LLM³).

**The pilot's tie (17/20 repair vs 17/20 resample) is what prior work predicts.**
- Stechly et al. found that "merely re-prompting with a sound verifier maintains most of the benefits".
- Valmeekam et al. found that binary versus detailed feedback "showed minimal impact".
- So a gain from first attempt to repair must not be presented as an effect of the feedback.

## How it was checked

- **IDs:** every arXiv ID (about 75) was resolved through the arXiv API (title, authors, date, comments, journal-ref).
- **Venues:** these come from arXiv comments or journal-ref, PMLR, Crossref or Semantic Scholar. Venues from Semantic Scholar alone are marked [S2].
- **Numbers:** every number is quoted from an abstract or a targeted full-text check. Full text was checked for:
  - LLM³, PRoC3S, AutoTAMP, LLM-Modulo, Huang et al. and Snell et al.;
  - Cosmos-Reason1, PlanBench-o1, RoboCritics, PerceptTwin, Hong et al. and Khatana et al.
- **Relations to vplan:** most rest on the abstract. Judge independence is marked n.s. wherever the full text was not checked.

**Legend**
- **Verifier:** D = deterministic, L = learned.
- **What it checks:** Sym = symbolic, Geo = geometric or kinematic, Phys = physics simulation, Vis = visual.
- **When:** Pre = before execution, Step = during execution, Post = after execution.
- **"Resample / refusal / indep. judge":**
  - whether there is an equal-budget independent-resampling control;
  - whether refusal on impossible tasks is tested;
  - whether success is scored independently of the feedback source.
- **Values:** Y, N, n.s. (not stated or not checked) or n/a.

## 1. Works by theme

### Theme 1: LLM planning with external verification or feedback

| # | Work | What it does | Verifier · when | Feedback to LLM | Resample / refusal / indep. judge | Relation to vplan |
|---|---|---|---|---|---|---|
| 1 | Kambhampati et al., "LLMs Can't Plan, But Can Help Planning in LLM-Modulo Frameworks", ICML 2024 [2402.01817](https://arxiv.org/abs/2402.01817) | Position paper. The LLM generates candidates and a bank of hard (sound) and soft critics checks them; a meta-controller back-prompts. Says critics "can be simulators" and that VLMs can critique robot behaviour style | Any; hard critics D · Pre | Processed critiques | n/a | vplan is an LLM-Modulo instance: physics as the hard critic, Cosmos as a soft critic. Cite it as the source of the architecture |
| 2 | VAL/PDDL loops: Valmeekam et al., NeurIPS 2023 [2305.15771](https://arxiv.org/abs/2305.15771); Guan et al., NeurIPS 2023 [2305.14909](https://arxiv.org/abs/2305.14909); LLM+P, Liu et al., arXiv 2023 [2304.11477](https://arxiv.org/abs/2304.11477); ISR-LLM, Zhou et al., arXiv 2023 [2308.13724](https://arxiv.org/abs/2308.13724) | The LLM writes a plan or PDDL, and a sound planner or validator checks it. Errors are back-prompted (Valmeekam, ISR-LLM) or used to fix the domain model (Guan). GPT-4 alone solves about 12% (Valmeekam) | D Sym · Pre | Validator error reports | N / N / the validator is the success criterion | The same external sound-verifier idea, but purely symbolic |
| 3 | SayCan, Ahn et al., CoRL 2022 [2204.01691](https://arxiv.org/abs/2204.01691) | Weights LLM skill proposals by learned affordance (value) scores | L · Step (filter before each skill) | None (scores only) | N / N / n.s. | Learned per-skill feasibility, not a deterministic check of the whole trajectory |
| 4 | Inner Monologue, Huang et al., CoRL 2022 [2207.05608](https://arxiv.org/abs/2207.05608) | Closed-loop language feedback from success detection, scene description and a human | L + human · Step/Post | Natural-language observations | N / N / n.s. | Detects failures after acting; vplan checks before anything moves |
| 5 | Code as Policies, Liang et al., ICRA 2023 [2209.07753](https://arxiv.org/abs/2209.07753); ProgPrompt, Singh et al., ICRA 2023 [2209.11302](https://arxiv.org/abs/2209.11302); VoxPoser, Huang et al., CoRL 2023 [2307.05973](https://arxiv.org/abs/2307.05973); RoboTool, Xu et al., arXiv 2023 [2310.13065](https://arxiv.org/abs/2310.13065) | The LLM writes policy code, situated programs, 3D value maps or tool-use code | No external verifier (RoboTool's modules are all LLMs) | n/a | N / N / n.s. | The unverified-generator family that vplan's verifier sits in front of |
| 6 | Text2Motion, Lin et al., Autonomous Robots 2023 [2303.12153](https://arxiv.org/abs/2303.12153) | LLM search guided by skill Q-function feasibility, plus geometric feasibility planning over skill sequences. 82% vs 13% for prior language planners | L Geo · Pre | Feasibility scores inside the search | N / N / n.s. | A learned check with no textual findings and no repair loop |
| 7 | AutoTAMP, Chen et al., ICRA 2024 [2306.06531](https://arxiv.org/abs/2306.06531); CLAIRify, Skreta et al., arXiv 2023 [2303.14100](https://arxiv.org/abs/2303.14100) | Language to STL, with a rule-based syntax checker and an LLM semantic checker that trigger re-prompting (up to 5 syntax and 3 semantic rounds). CLAIRify re-prompts with verifier errors for a chemistry DSL | D Sym (+ L semantic) · Pre | Specific syntax errors; semantic mismatch | Only a "No Corrections" arm, not equal-budget / N / n.s. | Checks the specification, not the robot's motion |
| 8 | LLM³, Wang et al., IROS 2024 [2403.11552](https://arxiv.org/abs/2403.11552) | The LLM proposes actions and continuous parameters. BiRRT with IK and collision checks in a PyBullet box-packing task; qualitative demo on a Franka FR3 | D Geo · Pre (sim) | Templates: "in collision with object", "no feasible IK solution", "collision-free and reachable" | Partial: a no-feedback variant under the same 20-call cap, but what it sees is unspecified. Motion-planning calls: random sampling 663.1, LLM 70.2, LLM + feedback 53.2 / N / n.s. | Closest ablation precedent; already shows that feedback saves calls |
| 9 | PRoC3S, Curtis et al., CoRL 2024 (PMLR 270) [2406.05572](https://arxiv.org/abs/2406.05572) | The LLM writes a parameterized program, which becomes a continuous constraint problem solved by sampling. PyBullet checks IK reach, collisions, and grasp and placement stability. Re-prompts the LLM if no solution exists; runs open-loop on a Franka Panda | D Phys/Geo · Pre | The two most common violations, with skill and plan step ("Collision detected between object o12, gripper") | N: the no-feedback arm is a single program with no retry / N / n.s. | Closest technical prior. vplan adds measured value, threshold and plan field in each finding, an approval binding, a code-independent judge, impossible tasks and an equal-budget resample arm |

### Theme 2: Repair vs resampling, and the limits of self-correction

None of these are robotics papers; the robotics analogues are rows 8 and 9.

| # | Work | What it finds | Verifier · feedback | Resample / refusal / indep. judge | Relation to vplan |
|---|---|---|---|---|---|
| 10 | Olausson et al., "Is Self-Repair a Silver Bullet for Code Generation?", ICLR 2024 [2306.09896](https://arxiv.org/abs/2306.09896) | Once repair cost is counted, "performance gains are often modest, vary a lot between subsets of the data, and are sometimes not present at all". Feedback quality is the bottleneck: stronger-model or human feedback gives much larger gains | D unit tests · natural-language explanations | Y / N / tests | The same question in code; the method to copy |
| 11 | Stechly et al., ICLR [S2] [2402.08115](https://arxiv.org/abs/2402.08115); Stechly et al., arXiv 2023 [2310.12397](https://arxiv.org/abs/2310.12397); Valmeekam et al., arXiv 2023 [2310.08118](https://arxiv.org/abs/2310.08118) | GPT-4 on Game of 24, graph colouring and STRIPS planning. Self-critique causes "performance collapse"; a sound external verifier helps; "merely re-prompting with a sound verifier maintains most of the benefits". The content of the critique is "largely irrelevant", since gains come from the right answer appearing among the samples. Binary vs detailed feedback "showed minimal impact" | L self vs D sound · binary vs detailed | Effectively Y (binary re-prompting ablations) / N / external verifier | Predicts the pilot tie; the most important prior to engage with |
| 12 | Huang et al., "LLMs Cannot Self-Correct Reasoning Yet", ICLR 2024 [2310.01798](https://arxiv.org/abs/2310.01798) | Self-correction without external feedback fails, and multi-agent debate is no better than self-consistency at an equal number of responses. Says it is "crucial to compare it to baselines with equivalent inference costs" | L self | Y / N / n/a | Justifies the equal-budget arm |
| 13 | Snell et al., arXiv 2024 [2408.03314](https://arxiv.org/abs/2408.03314); Brown et al., "Large Language Monkeys", arXiv 2024 [2407.21787](https://arxiv.org/abs/2407.21787) | At an equal budget, "easier questions benefit from purely sequential" revisions, while harder ones need a mix of sequential and parallel; their reviser is a PaLM 2-S* model fine-tuned to revise. Coverage grows log-linearly with the number of samples; with automatic verification, SWE-bench Lite goes from 15.9% to 56% at 250 samples | Verifier/PRM; revisions vs samples | Y / N / n/a | Resampling against a deterministic verifier is a strong baseline. The answer may depend on difficulty, so report the easy and hard cohorts separately |
| 14 | Self-Refine, Madaan et al., NeurIPS 2023 [2303.17651](https://arxiv.org/abs/2303.17651); Reflexion, Shinn et al., NeurIPS 2023 [2303.11366](https://arxiv.org/abs/2303.11366); Kamoi et al., TACL 2024 [2406.01297](https://arxiv.org/abs/2406.01297) | The optimistic side: self-feedback refinement, and verbal reflection kept in memory. Kamoi's survey finds self-correction "works well in tasks that can use reliable external feedback", flags unfair evaluations and gives a checklist | Self or external | Self-Refine and Reflexion compare against single-pass generation or a base agent | vplan meets Kamoi's conditions: a reliable external verifier and a fair baseline |
| 15 | Tyen et al., ACL Findings 2024 [2311.08516](https://arxiv.org/abs/2311.08516) | LLMs cannot find their own reasoning errors, but "can correct them given the error location" | Location-only feedback | n/a | Supports findings that name the plan field without giving the fix |

### Theme 3: Infeasible tasks, refusal and asking for help

The 2026 benchmarks are in Theme 7.

| # | Work | What it does | Verifier · when | Refusal relevance | Relation to vplan |
|---|---|---|---|---|---|
| 16 | KnowNo, Ren et al., CoRL 2023 [2307.01928](https://arxiv.org/abs/2307.01928) | Conformal prediction lets the planner ask for help with statistical task-completion guarantees | L calibrated · Pre | Ambiguity, not impossibility | vplan leaves ambiguity out of scope; the classic citation for "knowing when it doesn't know" |
| 17 | CLARA, Park et al., IEEE RA-L [S2] [2306.10376](https://arxiv.org/abs/2306.10376) | Classifies commands as clear, ambiguous or infeasible, then asks clarifying questions | L · Pre | Has an infeasible class | Learned classification; vplan uses provable impossibility and a deterministic verifier |
| 18 | Valmeekam et al., "LLMs Still Can't Plan; Can LRMs?", arXiv 2024 [2409.13373](https://arxiv.org/abs/2409.13373) | On PlanBench, o1-preview identifies 27% (Blocksworld) and 16% (Randomized Mystery Blocksworld) of unsolvable instances, and wrongly calls 0% and 11.5% of solvable ones impossible | D Sym ground truth | Measures both missed refusals and false refusals | Precedent for the refusal confusion matrix |

### Theme 4: Safety guardrails and formal verification

| # | Work | What it does | Verifier · when | Relation to vplan |
|---|---|---|---|---|
| 19 | RoboPAIR, Robey et al., arXiv 2024 [2410.13691](https://arxiv.org/abs/2410.13691); RoboGuard, Ravichandran et al., IEEE RA-L [S2] [2503.07885](https://arxiv.org/abs/2503.07885) | RoboPAIR jailbreaks LLM-controlled robots, "often achieving 100% attack success rates". RoboGuard uses a protected LLM to turn safety rules into temporal-logic specs, then control synthesis resolves conflicts; unsafe plans executed fall from over 92% to below 3% | L spec + D LTL synthesis · Pre | Semantic and adversarial safety. vplan checks geometric and physical safety and does not address jailbreaks |
| 20 | Safety Chip, Yang et al., ICRA 2024 [2309.09919](https://arxiv.org/abs/2309.09919); SELP, Wu et al., ICRA 2025 [2409.19471](https://arxiv.org/abs/2409.19471); VeriPlan, Lee et al., CHI 2025 [2502.17898](https://arxiv.org/abs/2502.17898); VerifyLLM, Grigorev et al., IROS 2025 [2507.05118](https://arxiv.org/abs/2507.05118) | LTL constraint modules with violation explanations and pruning of unsafe actions; LTL-constrained decoding with equivalence voting; model checking of end-user plans; LLM pre-execution plan analysis via LTL | D Sym, formal (VerifyLLM is L) · Pre | Formal temporal-logic constraints over symbolic actions, with no kinematics. vplan's checks are sample-based and give no formal guarantee |
| 21 | SafePlan, Obi et al., arXiv 2025 [2503.06892](https://arxiv.org/abs/2503.06892); SafeGate, Obi et al., arXiv 2026 [2604.05427](https://arxiv.org/abs/2604.05427) | Chain-of-thought checks of the prompt, invariants and pre- and postconditions (90.5% fewer harmful prompts accepted). A "deterministic decision gate to authorize or reject execution", with Task Safety Contracts checked by Z3 | L + D · Pre | A command-level gate (is the request safe?); vplan's gate is motion-level (is this trajectory feasible and safe?) |
| 22 | Heydari Shahna et al., "From LLM Plans to Authorized Motion: Execution-Bound Authorization for Excavators", arXiv 9 Oct 2026 [2610.14245](https://arxiv.org/abs/2610.14245); CapLease, Xu et al., arXiv 2026 [2608.01710](https://arxiv.org/abs/2608.01710); Necula, "Proof-carrying code", POPL 1997 [doi:10.1145/263699.263712](https://doi.org/10.1145/263699.263712) | A one-use lease binds the action, the generated motion, the mission contract and a tolerance. Dispatch atomically rechecks authority and measured joint drift, with zero unauthorized dispatches across a 672-case suite. CapLease binds a user's confirmation durably to one canonical action | D · between approval and dispatch | vplan's token (a hash of trajectory, scene, target and config, plus a live start-state check) is a simpler, concurrent version: cite it, do not claim it |

### Theme 5: VLMs and world models as success or failure detectors

| # | Work | What it does | Verifier · when | Relation to vplan |
|---|---|---|---|---|
| 23 | SuccessVQA, Du et al., CoLLAs 2023 [2303.07280](https://arxiv.org/abs/2303.07280); REFLECT, Liu et al., CoRL 2023 [2306.15724](https://arxiv.org/abs/2306.15724); AHA, Duan et al., ICLR 2025 [S2] [2410.00371](https://arxiv.org/abs/2410.00371) | Success detection as VQA; LLM failure explanation over multisensory summaries (the RoboFail dataset); a VLM fine-tuned on procedurally perturbed sim failures (+10.3% over GPT-4o in-context) | L Vis · Post | Learned and after the fact. vplan uses Cosmos only as a soft check on rendered key frames |
| 24 | Cosmos-Reason1, NVIDIA, arXiv 2025 [2503.15558](https://arxiv.org/abs/2503.15558) | A physical-AI reasoning VLM. On RoboFail (affordance and task-completion checks): 7B (SFT) 60.0, 56B 66.2, vs 67.0 for Gemini 2.0 Flash. The authors say RoboFail "remains consistently challenging" | L Vis | Same family as vplan's gate; the D34 balanced accuracy of about 0.70 fits this range |

### Theme 6: Physics simulation as a check before acting

The main entries are rows 8–9 (LLM³, PRoC3S) and rows 25–27. Also:
- **Mind's Eye** (Liu et al., arXiv 2022, [2210.05359](https://arxiv.org/abs/2210.05359)): MuJoCo outcomes are fed into LLM physics question answering (+27.9% zero-shot). Simulation is a reasoning aid, not a gate.
- **SayPlan** (Rana et al., CoRL 2023, [2307.06135](https://arxiv.org/abs/2307.06135)): iterative replanning from scene-graph simulator feedback; symbolic only.
- **PRISM** (Hsu et al., arXiv 2026, [2601.05356](https://arxiv.org/abs/2601.05356)): an Omniverse digital twin checks LLM-generated lab-robot protocols before execution.
- **SIMMER** (Lu et al., COLM 2026, [2606.14574](https://arxiv.org/abs/2606.14574)): a symbolic state-machine executor finds latent failures; at most 17% of plans are error-free.

### Theme 7: Closest 2025–2026 work

| # | Work | What it does | Verifier · when · feedback | Resample / refusal / indep. judge | Relation to vplan |
|---|---|---|---|---|---|
| 25 | Hong et al., "Bridging Semantics and Physics with Constrained LLMs for Safe and Trustworthy Robotic Manipulation", ECCV 2026 Workshop [2608.29379](https://arxiv.org/abs/2608.29379) | A typed contract: MCP schema-validated tool calls grounded deterministically in MoveIt Task Constructor, with "verify-then-act" kinematic and collision checks. On a UFactory 850: up to 80% on pouring and 90% on grasp-and-place | D Geo · Pre · structured failures ("PlanningFailed", "TargetUnreachable") back to the agent | N / halts on ambiguity, infeasibility not evaluated / judge not specified | Closest architecture to vplan's typed plan into a deterministic gate; no token and no resample arm |
| 26 | RoboCritics, Kim et al., HRI 2026 [2603.06842](https://arxiv.org/abs/2603.06842) | Five motion-level critics (space use, collision, joint speed, end-effector pose, pinch point) run on URDF-derived traces before deployment to a UR3e; user study with n = 18 | D Geo · Pre · flags, hints and a candidate revision ("one-click fix") | N / N / N: the same critics score quality | Closest analogue to structured findings, but it suggests fixes and grades itself, which vplan avoids on purpose |
| 27 | PerceptTwin, Gauthier et al., ICRA 2026 [2606.04226](https://arxiv.org/abs/2606.04226) | Builds interactive sims from the robot's perception. Precondition errors plus an LLM judge over state diffs, up to 5 feedback rounds; plan success rises about 39% on average | D Sym preconditions + L judge · Pre | N / partial (flags unsafe plans) / N: the judge is the feedback LLM (answers checked by hand) | Simulate-before-act with feedback, but a learned judge and no physics checks |
| 28 | RoboAbstention ("The Yes-Man Syndrome"), Yeke et al., arXiv 2026 [2605.20544](https://arxiv.org/abs/2605.20544) | An abstention taxonomy (ambiguous, physically infeasible, false-premise) built with "deterministic constraint derivation", 6,069 instructions. The best VLM abstains on 39.0%, Gemini Robotics ER 1.6 on 16.5%; prompting raises this to 93.6% | D ground truth | n/a / Y / D | Verifiable infeasibility already exists, but grounded in images rather than kinematics |
| 29 | plan-failure-bench, Kazmi, Zenodo preprint 2026 [doi:10.5281/zenodo.21756817](https://doi.org/10.5281/zenodo.21756817) | 60 instructions in 2 symbolic worlds with six trap types, including unreachable goals with proofs. A deterministic checker scores them, "no human or LLM judging anywhere"; valid tasks serve as false-positive controls | D Sym | n/a / Y / D | Closest analogue to the impossible-task generator plus a deterministic judge, but symbolic rather than geometric |
| 30 | FailBench, Navasardyan et al., arXiv 2026 [2609.03611](https://arxiv.org/abs/2609.03611) | 2,197 attempts from 14 sources. The best of 13 VLM detectors reaches 0.77 mean balanced accuracy, under 0.60 on contact-rich assembly. Judges lean towards "success", and fine-tuned detectors do worse than general VLMs | L Vis · Post | n/a | The Cosmos result (D34) replicates this; cite it rather than claiming the finding |
| 31 | Repair or Resample?, Luan et al., arXiv 2026 [2608.25920](https://arxiv.org/abs/2608.25920); VRR-Stop, Wu et al., arXiv 2026 [2607.17641](https://arxiv.org/abs/2607.17641) | In multi-agent systems, plain reruns reproduce 67.97% of failures and repair only 6.90%, while targeted symptom-driven repair fixes 20.15%. With noisy verifiers, "reported acceptance keeps rising while true validity falls" | n/a | Frames repair vs resample / n/a / argues for independent judging | Evidence from outside robotics on both the repair-vs-resample question and why an independent judge matters |

### Also relevant

- **Refusal, uncertainty and safety benchmarks:**
  - **CAPE** (ICRA 2024, [2211.09935](https://arxiv.org/abs/2211.09935)): corrective actions from precondition errors instead of retrying.
  - **Introspective Planning** (NeurIPS 2024, [2402.06529](https://arxiv.org/abs/2402.06529)): aligns planner uncertainty with task ambiguity.
  - **LBAP** (arXiv 2024, [2403.13198](https://arxiv.org/abs/2403.13198)): Bayesian calibration of planner confidence.
  - **SafeAgentBench** (arXiv 2024, [2412.13178](https://arxiv.org/abs/2412.13178)): the most safety-conscious baseline rejects only 10% of detailed hazardous tasks.
  - **Agent Planning Benchmark** (arXiv 2026, [2606.04874](https://arxiv.org/abs/2606.04874)): an unsolvable-task split; calibrated refusal is weak across 12 multimodal LLMs.
  - **Refusals That Bend** (arXiv 2026, [2609.38971](https://arxiv.org/abs/2609.38971)): adding ordinary objects to the scene flips 20.2% of refused tasks to compliance.
- **Repair loops:**
  - **REx** (NeurIPS 2024, [2405.17503](https://arxiv.org/abs/2405.17503)): refinement as explore vs exploit; more problems solved with fewer calls.
  - **Kiecker et al.** (arXiv 2026, NIER, [2607.05197](https://arxiv.org/abs/2607.05197)): most repair gains come in the first 3–4 iterations, which supports k = 3.
- **Verifiers, judges and gates:**
  - **Guardian** (arXiv 2025, [2512.01946](https://arxiv.org/abs/2512.01946)): a VLM failure verifier, with Cosmos-Reason among its baselines.
  - **VISOR** (arXiv 2026, [2605.10408](https://arxiv.org/abs/2605.10408)): a VLM's own uncertainty does not predict whether its verdict is right.
  - **Khatana et al.** (IROS 2026 workshop, [2610.12761](https://arxiv.org/abs/2610.12761)): a geometric verifier returns structured conflict feedback (agents, location, time, minimum separation); no resample control.
  - **Agentic Harnesses** (arXiv 2026, [2608.09857](https://arxiv.org/abs/2608.09857)): an ensemble of LLM judges gates plans before they reach the robot; learned, not deterministic.
  - **LAD-VF** (ICRA 2026, [2509.18384](https://arxiv.org/abs/2509.18384)): prompt optimisation from formal-verification feedback.
  - **RoboMonkey** (arXiv 2025, [2506.17811](https://arxiv.org/abs/2506.17811)): samples VLA actions and picks one with a VLM verifier; the resampling side of test-time scaling.
- **Survey:** Kim et al. (arXiv 2026, [2606.05660](https://arxiv.org/abs/2606.05660)), on safety in long-horizon manipulation.
- **NVIDIA Cosmos 3** ([blog, 31 May 2026](https://blogs.nvidia.com/blog/cosmos-3-physical-ai-open-world-foundation-model/)): the model family behind the demo gate.

## 2. Positioning

### Claim by claim

| vplan element | Closest prior | Verdict |
|---|---|---|
| LLM proposes, an external sound checker decides | LLM-Modulo (simulators named as critics); VAL loops | Established: cite, don't claim |
| Typed plan through a forced tool call | Hong et al. 2026 (MCP schema-validated calls) | Established |
| Checking LLM manipulation plans in physics or kinematics before execution | PRoC3S, LLM³, Hong et al., RoboCritics | Established |
| Structured findings returned to the LLM | PRoC3S, LLM³, Khatana et al., RoboCritics | Established. vplan's form (measured value, threshold and plan field, with no suggested fix) is an unablated design detail |
| Wide check set (singularity, velocity limits, carried-object collision, release height) | PRoC3S (IK, collision, grasp, placement); RoboCritics (joint speed, collision, end-effector pose) | Wider than usual; an engineering point, not a research claim |
| Approval token bound to the exact trajectory, with a live start-state check | Heydari Shahna et al. 2026 (concurrent and richer); CapLease; proof-carrying code | Established or concurrent |
| Judge that shares no code with the verifier | Ground-truth scoring is standard. RoboCritics and PerceptTwin grade themselves; VRR-Stop shows the risk | Good practice; a modest differentiator |
| Paired, equal-budget repair vs independent resampling | Olausson, Huang, Snell, Stechly (outside robotics); LLM³ partially; PRoC3S and AutoTAMP have no such arm | Not found in manipulation planning: **the defensible novelty** |
| "Repair needs fewer calls" | LLM³, REx | Not a new observation; only the paired, equal-budget measurement is new |
| Provably impossible tasks with refusal and false-refusal metrics | RoboAbstention, plan-failure-bench, PlanBench, CLARA | Established. vplan's twist is reach proofs from kinematics in a continuous 3D scene |
| Cosmos at about 0.70 balanced accuracy, so evaluation uses physics only | FailBench; Cosmos-Reason1 on RoboFail | A corroborating data point, not a new finding |

### Claims vplan must not make

1. "First to verify LLM robot plans in (physics) simulation before execution." Prior: PRoC3S, LLM³, Hong et al., RoboCritics, PerceptTwin.
2. "First or novel structured verifier feedback to an LLM planner." Prior: PRoC3S, LLM³, AutoTAMP, CLAIRify, SayPlan, the VAL loops, RoboCritics.
3. "A new architecture." It is LLM-Modulo.
4. "A novel approval or execution binding." Prior: Heydari Shahna et al. 2026, CapLease.
5. "The first impossible-task or refusal benchmark for robot planners." Prior: RoboAbstention, plan-failure-bench, PlanBench, CLARA, SafeAgentBench.
6. "Verifier feedback raised success from 45% to 85%", or any first-attempt-to-repair gain presented as an effect of feedback. Equal-budget resampling reached the same 17/20 in the R7 pilot (D46), and Stechly et al. attribute such gains to the extra samples. The current headline template in [`spec.md`](spec.md) §1 is worded this way (see "Open" below).
7. "Feedback beats resampling." The pilot is a tie at n = 20.
8. "First to show that feedback saves calls." Prior: LLM³, REx.
9. "VLM judges are unreliable" as our own finding. FailBench and Cosmos-Reason1 already show it.
10. "Formally verified" or "guaranteed safe." The checks are sample-based on a kinematic preview with attachment-assisted grasping, unlike RoboGuard, SELP and SafeGate.

### Gap statement

> Prior systems already check LLM-generated manipulation plans in simulation before execution and feed failures back to the model (PRoC3S; LLM³; Hong et al., 2026), and concurrent work binds approvals to exact motions (Heydari Shahna et al., 2026). What we did not find is a controlled test of whether such feedback is worth more than drawing more independent plans at the same call budget, scored by a judge that shares no code with the verifier, with provably impossible tasks in the same evaluation so refusals and false refusals are counted alongside success. vplan provides that test for an open NVIDIA model in simulation.

### Framing advice

- **Lead with repair vs equal-budget resampling.** Give success with intervals and calls-to-success, and keep first-attempt-only as a reference arm.
- **If W3 ties again on success,** the honest claim becomes "same success with fewer calls and no repeated plans", and only if the intervals separate. The evaluation harness itself is also a contribution. Report the easy and hard cohorts separately (Snell et al.).
- **Naming:**
  - call the token an "execution-bound approval (cf. Heydari Shahna et al., 2026)";
  - present the Cosmos result as a replication of FailBench on a new model.
- **Open:** the spec §1 headline template credits the first-attempt-to-repair gain to feedback (claim 6). Changing the binding spec needs the owner's decision and a new entry in [`decisions.md`](decisions.md).

## 3. Must-cite

1. Kambhampati et al., ICML 2024: LLM-Modulo, the origin of the architecture.
2. Curtis et al., CoRL 2024: PRoC3S, the closest technical prior.
3. Wang et al., IROS 2024: LLM³, motion-failure feedback, call savings, the nearest ablation.
4. Hong et al., ECCV 2026 Workshop: typed calls plus deterministic verify-then-act; concurrent.
5. Olausson et al., ICLR 2024: a budget-aware comparison of repair and sampling.
6. Stechly et al., ICLR [S2], with Valmeekam et al. 2023: with a sound verifier, feedback content barely matters.
7. Huang et al., ICLR 2024: the need for equal-cost baselines.
8. Yeke et al., arXiv 2026, RoboAbstention: verifiable infeasibility and abstention.
9. Heydari Shahna et al., arXiv 2026: execution-bound authorization, concurrent with the approval token.
10. Navasardyan et al., arXiv 2026, FailBench: the limits of VLM success judges.

Next tier:
- **Classics reviewers expect:** KnowNo, Text2Motion, SayCan, Inner Monologue.
- **Others:** RoboGuard, RoboCritics, plan-failure-bench, Snell et al.
- **NVIDIA models section:** Cosmos-Reason1 goes there anyway.

## 4. Contested or uncertain

- **Does richer feedback help?** In code, yes: a stronger feedback source gives substantially larger gains (Olausson et al.). In planning, content barely matters (Stechly et al., Valmeekam et al.). This bears on vplan's no-fix findings, which have not been ablated.
- **LLM³'s no-feedback arm** shares the 20-call cap, but whether it sees earlier attempts or only a binary signal is not stated. Whether it counts as resampling is unclear.
- **PRoC3S:** the feedback-iteration limit renders as "55" in the HTML, which may be an artifact.
- **Cosmos-Reason1:** the text says RL did not improve RoboFail, but the extracted Table 9 shows 7B going from 60.0 to 62.0.
- **Snell et al.:** the sequential-revision advantage comes from a model fine-tuned to revise and may not carry over to an off-the-shelf Nemotron.
- **FailBench** is a v1 preprint, and secondary summaries disagreed about what 0.77 means. This survey uses the abstract's wording: "best model reaches 0.77 mean balanced accuracy".

## 5. Not verified

- **Venues:**
  - RoboPAIR, SafePlan, Snell et al., Brown et al., RoboMonkey, Mind's Eye and CLAIRify's journal version are listed as arXiv.
  - RoboGuard, CLARA, Stechly et al. and AHA have venues from Semantic Scholar only.
- **Excluded:** Kim, Electronics Letters 2026, "Physical Simulation-Based Correction of LLM-Generated Tool-Using Primitives". Found in one search, but the page did not open.
- **"Not found":** these claims rest on targeted searches up to 2026-10-14.
- **Unchecked numbers:** the results of plan-failure-bench and the Agent Planning Benchmark.
