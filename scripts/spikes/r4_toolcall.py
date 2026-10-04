"""R4: forced submit_plan tool calls with an object-root {plan: ...} schema; every output is validated.

Throwaway spike (plan Task 9). Usage: PYTHONPATH='src;.' python scripts/spikes/r4_toolcall.py [forced|required|content] [n]
Even calls ask for the red cube (expect a plan); odd calls ask for an absent purple cube (expect a refusal).
"""
import json
import sys
import time

from openai import OpenAI
from pydantic import ValidationError

from vplan.records import PickPlacePlan, Refusal, SubmitPlanArgs

MODE = sys.argv[1] if len(sys.argv) > 1 else "forced"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20

env = {}
for line in open(".env", encoding="utf-8"):
    line = line.strip()
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
client = OpenAI(base_url=env["LLM_BASE_URL"], api_key=env["LLM_API_KEY"], max_retries=0, timeout=180)
tool = {"type": "function", "function": {"name": "submit_plan", "description": "Submit the plan or a refusal.",
                                          "parameters": SubmitPlanArgs.model_json_schema()}}
tool_choice = {"forced": {"type": "function", "function": {"name": "submit_plan"}}, "required": "required"}.get(MODE)
facts = {"objects": [{"id": "cube_red", "color": "red", "shape": "cube", "size": [0.04] * 3, "position": [0.5, -0.15, 0.02]},
                     {"id": "cube_green", "color": "green", "shape": "cube", "size": [0.04] * 3, "position": [0.55, 0.0, 0.02]}],
         "bin": {"floor_center": [0.45, 0.28, 0.0], "inner_size_xy": [0.16, 0.16], "wall_height": 0.08},
         "robot": {"base": [0, 0, 0], "reach_m": 0.855, "gripper_max_opening_m": 0.08}}
system = ("You plan pick-and-place for a Franka Panda. World frame, z up, meters. "
          + ("Call submit_plan exactly once with {\"plan\": ...}." if MODE != "content" else
             "Reply with only a JSON object matching this schema, no prose: " + json.dumps(SubmitPlanArgs.model_json_schema())))
counts = {"tool_valid": 0, "tool_invalid": 0, "content_valid": 0, "content_invalid": 0, "no_output": 0, "rejected": 0}
kinds = {"plan_for_red": 0, "refusal_for_purple": 0, "wrong_kind": 0}
latencies, prompt_toks, completion_toks = [], [], []
print(f"model={env['LLM_MODEL']} mode={MODE} n={N}")
for i in range(N):
    feasible = i % 2 == 0
    instr = "Put the red cube in the bin." if feasible else "Put the purple cube in the bin."
    kwargs = {"tools": [tool], "tool_choice": tool_choice} if MODE != "content" else {}
    t0 = time.perf_counter()
    try:
        r = client.chat.completions.create(
            model=env["LLM_MODEL"], temperature=0.2,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": json.dumps({"scene": facts, "instruction": instr})}], **kwargs)
    except Exception as e:  # provider rejected the request (tool_choice, schema, rate limit): record it
        counts["rejected"] += 1
        print(f"[{i}] request rejected:", type(e).__name__, str(e)[:300])
        continue
    dt = time.perf_counter() - t0
    latencies.append(dt)
    u = r.usage
    if u is not None:
        prompt_toks.append(u.prompt_tokens)
        completion_toks.append(u.completion_tokens)
    msg = r.choices[0].message
    raw, key = (msg.tool_calls[0].function.arguments, "tool") if msg.tool_calls else (msg.content, "content")
    tok = f"tokens in/out={u.prompt_tokens}/{u.completion_tokens}" if u is not None else "tokens=n/a"
    if not raw:
        counts["no_output"] += 1
        print(f"[{i}] no output ({dt:.1f}s, {tok}) finish={r.choices[0].finish_reason}")
        continue
    if key == "content":  # tolerate a markdown fence around content JSON
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        plan = SubmitPlanArgs.model_validate_json(raw).plan
        counts[f"{key}_valid"] += 1
        if feasible and isinstance(plan, PickPlacePlan) and plan.target_id == "cube_red":
            kinds["plan_for_red"] += 1
        elif not feasible and isinstance(plan, Refusal):
            kinds["refusal_for_purple"] += 1
        else:
            kinds["wrong_kind"] += 1
        print(f"[{i}] {key} valid {type(plan).__name__} ({dt:.1f}s, {tok})")
    except ValidationError as e:
        counts[f"{key}_invalid"] += 1
        print(f"[{i}] {key} invalid ({dt:.1f}s, {tok}):", e.errors()[:2], "| raw:", raw[:300])


def median(xs):
    return sorted(xs)[len(xs) // 2] if xs else None


print(counts)
print(kinds)
print("median latency s:", median(latencies), "| median tokens in/out:", median(prompt_toks), median(completion_toks),
      "| total tokens in/out:", sum(prompt_toks), sum(completion_toks))
