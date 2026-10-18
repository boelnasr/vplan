"""HTML for the W3.5 UI panels, built from ui.views. Every string from a record is escaped: plan rationales, refusal
explanations, Cosmos reasons and error messages come from a model or a provider and must never become markup."""
from __future__ import annotations

import base64
from collections.abc import Sequence
from html import escape

from vplan.ui.views import AttemptView, Banner, JudgeView, RoundView, latency_text

# Passed to Blocks.launch(css=...). Badge colours keep white text readable in Gradio's light and dark themes.
CSS = """
.vp-badge{display:inline-block;padding:1px 8px;border-radius:9px;color:#fff;font-size:0.85em;font-weight:600}
.vp-pass{background:#1a7f37}.vp-fail{background:#cf222e}.vp-partial{background:#9a6700}
.vp-unknown{background:#6e7781}.vp-off{background:#8c959f}.vp-info{background:#0969da}.vp-warn{background:#9a6700}
.vp-round{border:1px solid var(--border-color-primary,#d0d7de);border-radius:8px;padding:8px 12px;margin:8px 0}
.vp-round-head{margin-bottom:4px}.vp-note{opacity:0.75;font-size:0.9em;margin-top:4px}
.vp-checks{border-collapse:collapse;width:100%;font-size:0.92em}
.vp-checks td,.vp-checks th{border-bottom:1px solid var(--border-color-primary,#d0d7de);padding:3px 6px;text-align:left;
vertical-align:top}
.vp-banner{padding:10px 14px;border-radius:8px;font-weight:600;margin:4px 0}
.vp-banner-warning{background:#fff1c2;color:#4d2d00;border:2px solid #d4a72c}
.vp-banner-info{background:#ddf4ff;color:#0a3069;border:1px solid #54aeff}
.vp-verdict{font-size:1.4em;font-weight:700}
.vp-pre{white-space:pre-wrap;font-size:0.85em;margin:4px 0}
.vp-exec img{max-width:100%;border-radius:6px}
"""

_OUTCOME_CLASS = {"executed": "pass", "blocked": "fail", "refused": "info", "planning": "unknown",
                  "verifying": "unknown", "pending": "unknown"}


def badge(text: str, kind: str) -> str:
    return f'<span class="vp-badge vp-{escape(kind)}">{escape(text)}</span>'


def banner_html(banner: Banner | None) -> str:
    if banner is None:
        return ""
    return f'<div class="vp-banner vp-banner-{escape(banner.level)}">{escape(banner.text)}</div>'


def _outcome_badge(r: RoundView) -> str:
    kind = _OUTCOME_CLASS.get(r.outcome, "warn")
    if r.outcome == "executed" and r.judge is not None and not r.judge.success:
        kind = "fail"
    return badge(r.outcome, kind)


def round_html(r: RoundView) -> str:
    head = [f"<b>Round {r.index}</b>", _outcome_badge(r)]
    if r.planner:
        head.append(escape(r.planner))
    if latency_text(r.latency_s):
        head.append(escape(latency_text(r.latency_s)))
    parts = [f'<div class="vp-round"><div class="vp-round-head">{" &middot; ".join(head)} &mdash; '
             f"{escape(r.summary)}</div>"]
    if r.problems:
        parts.append("<ul>" + "".join(f"<li>{escape(p)}</li>" for p in r.problems) + "</ul>")
    if r.plan_json is not None:
        label = "Refusal" if r.plan_kind == "refusal" else "Plan"
        parts.append(f'<details><summary>{label} (JSON)</summary><pre class="vp-pre">{escape(r.plan_json)}</pre>'
                     "</details>")
    if r.checks:
        rows = "".join(f"<tr><td>{escape(c.code)} {escape(c.title)}</td><td>{badge(c.status, c.status)}</td>"
                       f"<td>{escape(c.evidence)}</td></tr>" for c in r.checks)
        parts.append(f'<table class="vp-checks"><tr><th>Check</th><th>Status</th><th>Evidence</th></tr>{rows}'
                     "</table>")
    if r.checks_note:
        parts.append(f'<div class="vp-note">Verifier: {escape(r.checks_note)}</div>')
    if r.visual_status is not None:
        kind = {"pass": "pass", "fail": "fail"}.get(r.visual_status, "warn")
        parts.append(f'<div class="vp-round-head">Cosmos (learned visual check, not verification): '
                     f"{badge(r.visual_status, kind)}"
                     + (f" {escape(r.visual_note)}" if r.visual_note else "") + "</div>")
        if r.visual:
            rows = "".join(f"<tr><td>{escape(v.stage)}</td><td>{escape(', '.join(v.views))}</td>"
                           f"<td>{badge(v.verdict, {'pass': 'pass', 'fail': 'fail'}.get(v.verdict, 'warn'))}</td>"
                           f"<td>{escape(v.reason)}</td><td>{escape(latency_text(v.latency_s))}</td></tr>"
                           for v in r.visual)
            parts.append('<table class="vp-checks"><tr><th>Stage</th><th>Views</th><th>Verdict</th><th>Reason</th>'
                         f"<th>Latency</th></tr>{rows}</table>")
    parts.append("</div>")
    return "".join(parts)


def timeline_html(rounds: Sequence[RoundView]) -> str:
    if not rounds:
        return '<div class="vp-note">No rounds yet.</div>'
    return "".join(round_html(r) for r in rounds)


def judge_html(judge: JudgeView | None, note: str = "") -> str:
    if judge is None:
        return f'<div class="vp-note">{escape(note or "No Judge verdict: nothing was executed or refused.")}</div>'
    rows = "".join(f"<tr><td>{escape(p.id)}</td><td>{badge('pass' if p.passed else 'fail', 'pass' if p.passed else 'fail')}"
                   f"</td><td>{escape(p.meaning)}</td></tr>" for p in judge.predicates)
    out = [f'<div class="vp-verdict">Judge: {badge(judge.verdict, "pass" if judge.success else "fail")}</div>',
           f'<table class="vp-checks"><tr><th>Predicate</th><th>Result</th><th>Meaning</th></tr>{rows}</table>']
    if judge.violations:
        out.append(f'<div class="vp-note">Violations: {escape(", ".join(judge.violations))}</div>')
    if judge.details:
        out.append("<details><summary>Judge details</summary>" + "".join(
            f'<div class="vp-pre"><b>{escape(k)}</b>: {escape(v)}</div>' for k, v in judge.details) + "</details>")
    if note:
        out.append(f'<div class="vp-note">{escape(note)}</div>')
    return "".join(out)


def status_html(lines: Sequence[tuple[str, str]], headline: str = "", kind: str = "info") -> str:
    """A headline badge and label: value lines (task, planner, outcome, ground truth)."""
    head = f'<div class="vp-verdict">{badge(headline, kind)}</div>' if headline else ""
    body = "".join(f"<div><b>{escape(k)}</b>: {escape(v)}</div>" for k, v in lines)
    return head + body


def gif_html(gif: bytes | None, caption: str) -> str:
    """The execution as an animated GIF in a data URI, so Gradio serves no files from disk for it."""
    if not gif:
        return f'<div class="vp-note">{escape(caption)}</div>'
    data = base64.b64encode(gif).decode("ascii")
    return (f'<div class="vp-exec"><img alt="execution" src="data:image/gif;base64,{data}"/>'
            f'<div class="vp-note">{escape(caption)}</div></div>')


def attempt_status(view: AttemptView) -> tuple[str, str]:
    """(headline, badge kind) for an attempt."""
    if view.success:
        return view.headline, "pass"
    return view.headline, "fail" if view.outcome in ("executed", "refused", "budget_exhausted") else "warn"
