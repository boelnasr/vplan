"""Cosmos visual gate client (spec §11): stage questions, one OpenAI-compatible request per stage, strict parsing.
Imports neither pybullet nor ManipulaPy, so it runs from a venv that only reaches the local vLLM server."""
from __future__ import annotations

import base64
import re
import time
from typing import Literal

import openai

from vplan.records import TaskSpec, VisualVerdict
from vplan.tasks.generator import resolve_instruction

Stage = Literal["pregrasp", "grasp", "prerelease"]
Verdict = Literal["pass", "fail", "uncertain"]
DEFAULT_MODEL = "nvidia/Cosmos3-Edge"
QUESTIONS: dict[str, str] = {
    "pregrasp": "Is the gripper approaching the {obj}?",
    "grasp": "Does the {obj} appear between the fingers, with a plausible closing direction?",
    "prerelease": "Is the {obj} positioned over the bin?",
}
REPLY_FORMAT = ("Answer pass if yes, fail if no, uncertain if the images do not show it clearly. "
                "Reply with exactly two lines:\nVERDICT: pass|fail|uncertain\nREASON: <one line>")
_VERDICT_LINE = re.compile(r"^VERDICT:(.*)$", re.MULTILINE)
_REASON_LINE = re.compile(r"^REASON:(.*)$", re.MULTILINE)


class CosmosUnavailable(RuntimeError):
    """Transport or HTTP failure: the demo shows a banner and executes on the geometric checks alone (spec §11)."""


def stage_question(stage: Stage, task: TaskSpec) -> str:
    """The object is named only by its TaskSpec color and shape; ids would leak nothing visual and oracle data
    must never reach the gate."""
    obj = next(o for o in task.scene.objects if o.id == resolve_instruction(task.instruction, task.scene))
    return ("The images show a simulated Franka Panda robot arm, a table with objects and a blue bin, "
            "from different camera viewpoints. " + QUESTIONS[stage].format(obj=f"{obj.color} {obj.shape}")
            + "\n" + REPLY_FORMAT)


def parse_verdict(raw: str) -> tuple[Verdict, str]:
    """Strict: exactly one `VERDICT:` line with pass/fail/uncertain after any reasoning block, else uncertain."""
    if "</think>" in raw:
        raw = raw.rsplit("</think>", 1)[1]
    elif "<think>" in raw:
        raw = ""  # truncated reasoning: any verdict-looking text is still thought, not the answer
    values = [v.strip().lower() for v in _VERDICT_LINE.findall(raw)]
    if len(values) != 1 or values[0] not in ("pass", "fail", "uncertain"):
        return "uncertain", "unparseable reply"
    reason = _REASON_LINE.search(raw)
    return values[0], reason.group(1).strip() if reason else ""


def ask_cosmos(images: dict[str, bytes], question: str, base_url: str, *, model: str = DEFAULT_MODEL,
               enable_thinking: bool = False, client: openai.OpenAI | None = None,
               timeout: float = 30.0) -> VisualVerdict:
    """images: view name -> PNG bytes, sent in dict order before the question. timeout in s."""
    if client is None:
        # retries would hide outages and inflate latency; the local vLLM server ignores the key
        client = openai.OpenAI(base_url=base_url, api_key="EMPTY", max_retries=0, timeout=timeout)
    content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(b).decode()}}
               for b in images.values()] + [{"type": "text", "text": question}]
    t0 = time.perf_counter()
    try:
        r = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": content}], temperature=0,
            max_tokens=1024 if enable_thinking else 128, timeout=timeout,
            extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}})
    except openai.APIError as e:
        raise CosmosUnavailable(f"{type(e).__name__}: {e}") from e
    latency = time.perf_counter() - t0
    raw = (r.choices[0].message.content if r.choices else None) or ""
    verdict, reason = parse_verdict(raw)
    return VisualVerdict(verdict=verdict, reason=reason, latency_s=latency, raw=raw)


def score(truths: list[bool], verdicts: list[str]) -> dict[str, float | int]:
    """Pilot scoring: `pass` <=> predicate true. A final `uncertain` counts as wrong in both classes; its rate is
    reported separately as abstention. Raises ValueError unless both classes are present."""
    pos = [v for t, v in zip(truths, verdicts, strict=True) if t]
    neg = [v for t, v in zip(truths, verdicts, strict=True) if not t]
    if not pos or not neg:
        raise ValueError("balanced accuracy needs at least one predicate-true and one predicate-false sample")
    tpr = sum(v == "pass" for v in pos) / len(pos)
    tnr = sum(v == "fail" for v in neg) / len(neg)
    return {"n": len(truths), "tpr": tpr, "tnr": tnr, "balanced_accuracy": (tpr + tnr) / 2,
            "abstention_rate": sum(v == "uncertain" for v in verdicts) / len(verdicts)}
