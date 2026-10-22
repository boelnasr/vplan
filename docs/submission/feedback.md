# Feedback on the Nebius and NVIDIA tools

We used Nebius Token Factory to serve NVIDIA Nemotron 3.5 Lightning (the evaluation planner) and Nemotron 3 Ultra (the live demo planner), and we served NVIDIA Cosmos3-Edge locally with vLLM as an advisory visual check. Every number below comes from our decision log ([`docs/decisions.md`](../decisions.md), cited by D-number) or the committed results ([`docs/results/`](../results/)).

## Nebius Token Factory

### What worked well

- **Drop-in OpenAI compatibility.** The base URL `https://api.tokenfactory.nebius.com/v1/` takes a Bearer key. Our `openai`-SDK planner moved from OpenRouter, which we used for development, to Token Factory by changing the base URL, key and model ID (D7, D13).
- **Forced tool calls work as documented.** `tool_choice` accepts `auto`, `required` or a named function (D13). In our smoke test a forced named call worked for both models: Lightning returned 19 valid calls of 20 (the other was a 21 s connect timeout on our Windows client, before any reply) and Ultra 20 of 20. No fallback was needed, and every reply carried the `kind` field our schema requires (D13 addendum).
- **Reliable at batch scale.** The final evaluation made 2,510 calls. 35 of them (1.4%) failed: 25 at our own 60 s deadline, the rest as connection errors or connect timeouts. None was a rate-limit error (D70). Nemotron 3 Ultra's runs on the same tasks made 1,417 calls, of which 8 (0.6%) failed (D82).
- **Inexpensive.** All 2,510 calls cost about $1.61 at catalog prices (D70): Lightning lists $0.06 / $0.24 per 1M input / output tokens, Ultra $1 / $3 (D13). The smoke test, the final evaluation and the week-4 development runs together come to about $5 at catalog prices; Ultra on the held-out set added about $14.70 (D82). That is about $20 of the $25 hackathon credit (D57).
- **Clear default rate limits.** The rate-limits page gives 60 requests and 400k tokens per minute by default. We sized three parallel runs from it, each paced to at most 15 requests per minute, and never hit a limit (D69, D70).
- **Interactive latency on a normal day.** Median 8.2 s for Lightning and 8.6 s for Ultra in the smoke test, with Ultra's slowest call at 28.5 s (D13), 7.5–8.7 s per call across the final evaluation configurations, and 6.8–7.3 s for Ultra on the held-out set (`metrics.json`, D82).
- **Removals are announced.** The deprecation notice lists removed models, for example Cosmos3-Super-Reasoner on 2026-08-31. Removed models are not rerouted (D7), so we pinned model IDs and re-checked them before first use (D62).

### What could be better

- **Model IDs differ from the names used elsewhere.** Token Factory's `nvidia/Nemotron-3_5-Lightning` and OpenRouter's `nvidia/nemotron-3.5-lightning` differ in case and in `_` for `.` (D7). We never typed an ID by hand; we copied them from the account's `GET /v1/models` (D13). A consistent scheme, or accepting the dotted form as an alias, would avoid one class of errors.
- **Capability tags changed during the event.** On Sep 30 the public catalog tagged Lightning "function calling" but not Nano, Super or Ultra (D13). On Oct 4 Ultra's entry had no tags (D30). By Oct 18 it listed `function_calling` for Ultra (D13 addendum), and Ultra's forced tool calls worked when we tested them. Because we could not plan from the tags, the demo model's core requirement stayed an open risk until a paid smoke test. Tags that match what each endpoint accepts would remove that step.
- **No JSON mode for Nemotron.** `response_format` with `json_schema` works only for models tagged "JSON mode", and no Nemotron was (D13). Forced tool calls were enough for us, but a server-enforced schema would be a useful second path for structured output.
- **Latency varies from day to day.** With the same model, prompt and `dev` tasks, answered calls took 8.7 s median on Oct 19 and 29.2 s on Oct 21, with similar completion tokens (2,066 against 1,973). Attempts that ended at our 60 s deadline rose from 4 to 17, and the Oct 21 run's first session stopped after three consecutive deadline failures (D76). Another run that day had 10 tasks hit the deadline (D77). Under a fixed per-call deadline (our spec's 60 s), a slow day becomes task failures and biases comparisons across days. Only our own records, similar tokens at three times the latency, showed that the provider changed, not our code. A per-model latency or load signal to check before starting a batch would help.
- **Abandoned calls report no usage.** The SDK's timeout bounds each read, not the whole call, so our planner stops waiting after 60 s of wall-clock time and abandons the request, which the server may still finish (D63). Such calls return no usage to us: none of the 35 failed calls in the final runs has a token count (`metrics.json` counts 11, 13 and 11 calls with missing usage in configs 1, 3 and 7). Our costs are therefore reported tokens at catalog prices, and we could not verify whether abandoned requests are billed (D63). Documenting how cancelled or abandoned requests are billed would let us state batch costs exactly.
- **Serverless Jobs went unused.** The hackathon credit covers tokens, not Jobs compute, so the batch evaluation the rules suggest running on Serverless Jobs ran locally and in our container instead (D57). A small compute credit would help Physical AI teams whose main load is simulation.

## NVIDIA Nemotron 3.5 Lightning and Nemotron 3 Ultra

### What worked well

- **Schema discipline.** Once `kind` was required in the tool schema (D67), every Lightning and Ultra reply in the smoke test carried it (D13). Neither model sent the plan as a JSON-encoded string, which Nemotron Super did in 10 of 10 calls (D64).
- **Lightning uses structured feedback.** On 300 held-out tasks, verifier findings raised Lightning's success from 24.7% (95% CI 20.1–29.8%) to 65.0% (59.4–70.2%), and beat equal-budget resampling by 29.3 points (23–36) (D70).
- **Few false refusals.** Lightning wrongly refused 1 feasible task in 900 attempts across three configurations (D70).
- **Low cost per call:** about $0.0006 for Lightning (D69).

### What could be better

- **Obstacles rarely become waypoints.** The task lists obstacle boxes, yet 212 of Lightning's 218 unverified first plans for feasible obstacle tasks had no transport waypoints (D70). Our prompt did not say that motion between key poses is straight (D70). When a revised prompt stated it with numbers, repair waypoints sat lower (lowest waypoint, median 0.150 m against 0.200 m) and repair success went from 50 to 42 of 75, an interval that crosses zero (D75).
- **Enclosed targets are never refused.** No configuration refused a target sealed by four walls and a lid (0 of 25, D70). Lightning refused 40–57 of the 100 impossible tasks overall.
- **Lightning relies on the feedback loop.** On the same 20 hard development tasks (free OpenRouter endpoints, a pilot, not a result), first attempts succeeded 0 of 20 for Lightning against 9 of 20 for Ultra, and 9 against 17 with repair (D60).

## NVIDIA Cosmos3-Edge with vLLM

### What worked well

- **Runs on a 16 GB consumer Blackwell card.** vLLM 0.31.0 with torch 2.13.0+cu130 on an RTX 5070 Ti (sm_120), in BF16 (D34). The shadow study sent 1,695 requests in about 52 minutes with 0 endpoint errors, at 1.30 s median and 2.72 s p90 per request with thinking on (D73).
- **Thinking helps.** In our 30-frame pilot, balanced accuracy was 0.50 with thinking off and 0.70 with it on (lenient parser), at 0.21 s against 1.18 s median latency (D34).
- **It catches wrong objects at pre-release.** In the 19 executed rounds that grasped the wrong object, it rejected 16 of 19 pre-release frames (84.2%, [`shadow/summary.md`](../results/shadow/summary.md)).

### What could be better

- **Serving needed several non-obvious settings.** We ran `--max-model-len 8192 --max-num-seqs 1 --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":3,"video":0}'`, the image-size limits as `--mm-processor-kwargs '{"do_resize":true,"min_pixels":4096,"max_pixels":409600}'`, and `VLLM_USE_FLASHINFER_SAMPLER=0`, because the FlashInfer sampler JIT-compiles at startup and needs `nvcc` (D34, D73). Our first record of the settings left out `do_resize` and `min_pixels`; only the serve log had them (D73). It used 15.2 of 16.3 GB (D73), which leaves little headroom, and Cosmos3-Nano (16B) does not fit 16 GB in BF16 (D34). A tested 16 GB serving example for Cosmos3-Edge, with these image settings, would save the next team those steps.
- **The reply format is not followed reliably.** We ask for two lines, `VERDICT:` and `REASON:`. 161 of the 168 final `uncertain` verdicts in the shadow study were replies our strict parser rejected: a VERDICT line without a REASON, or a bare `pass` or `fail` (D73). Read leniently, balanced accuracy would have been 0.71 and 0.58 instead of 0.61 and 0.52 (D73). In the demo video's Cosmos chapter, the replies could not be read even after a third view, so a plan that passed all eight geometric checks did not move.
- **"Inside" versus "over" the bin.** At pre-release, 49 of its 53 false rejections said the object was inside the bin, not over it (D73). The frame shows the end of the lowering motion, with the release pose below the bin rim (D47), so our question's "over" is partly to blame. We did not reword it on the held-out frames, to avoid tuning against them.
- **Biased toward pass at pre-grasp and grasp.** It caught 1 of 159 wrong-object pre-grasps and 25 of 196 misaligned grasps (D73).

## What we could not verify

- Whether Token Factory bills requests we abandoned at the deadline, and our costs against the console bill: we report tokens at catalog prices.
- Our account's effective rate limits beyond the documented defaults.
- Whether Token Factory passes `enable_thinking` or `chat_template_kwargs` through to Nemotron.
- Cosmos3-Edge on Nebius: we served it locally only.
