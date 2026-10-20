"""HTML for the W3.5 UI panels, built from ui.views. Every string from a record is escaped: plan rationales, refusal
explanations, Cosmos reasons and error messages come from a model or a provider and must never become markup.

Statuses are never colour alone (W4.1): every badge pairs its colour with an icon and a word, so the page reads the
same to colour-blind viewers and in a screen reader (the icon is aria-hidden)."""
from __future__ import annotations

import base64
from collections.abc import Sequence
from html import escape

from vplan.ui.views import (CHECK_HELP, OUTCOME_WORDS, AttemptView, Banner, JudgeView, RoundView, latency_text,
                            predicate_name, stage_name)

# Passed to Blocks.launch(css=...) as CSS (below). Status colours are GitHub's Primer light palette (white on the
# solid colours and the dark text on the tints are all >= 4.5:1); .dark overrides keep them readable in Gradio's dark
# theme. Neutrals use Gradio's theme variables, so the panels follow the theme.
_VARS = """
:root{--vp-pass:#1a7f37;--vp-pass-bg:#dafbe1;--vp-pass-fg:#116329;--vp-fail:#cf222e;--vp-fail-bg:#ffebe9;
--vp-fail-fg:#a40e26;--vp-warn:#9a6700;--vp-warn-bg:#fff8c5;--vp-warn-fg:#6c4400;--vp-info:#0969da;--vp-info-bg:#ddf4ff;
--vp-info-fg:#0a3069;--vp-muted:#6e7781;--vp-muted-bg:#eef1f4;--vp-muted-fg:#424a53}
.dark{--vp-pass-bg:rgba(46,160,67,.18);--vp-pass-fg:#56d364;--vp-fail-bg:rgba(248,81,73,.18);--vp-fail-fg:#ff7b72;
--vp-warn-bg:rgba(187,128,9,.2);--vp-warn-fg:#e3b341;--vp-info-bg:rgba(56,139,253,.18);--vp-info-fg:#79c0ff;
--vp-muted-bg:rgba(110,118,129,.2);--vp-muted-fg:#c9d1d9}
.gradio-container{width:100% !important;max-width:1600px !important;margin:0 auto !important}
.vp-align-end{align-items:flex-end !important}
.vp-align-end > button{margin-bottom:var(--block-padding, 10px)}
.gradio-container textarea:disabled{background:var(--background-fill-secondary);color:var(--body-text-color-subdued);
cursor:not-allowed}
"""
# Rules for the HTML this module writes. gr.HTML puts it inside .prose, whose ".gradio-container-6-30-0 .prose *"
# colour and table-border rules would win over single-class selectors, so _scoped() prefixes every selector here.
_RULES = """
.vp-header{padding:4px 0 2px}
.vp-header h1{font-size:2rem;font-weight:700;margin:0;line-height:1.2;letter-spacing:-0.01em}
.vp-header h1 .vp-brand{color:var(--color-accent)}
.vp-header p{margin:6px 0 0;font-size:1.05rem;color:var(--body-text-color-subdued);max-width:80ch;line-height:1.5}
.vp-flow{display:flex;flex-wrap:wrap;gap:8px;align-items:center;list-style:none;padding:0;margin:12px 0 0}
.vp-flow li{display:inline-flex;align-items:center;gap:6px;font-size:.95rem;font-weight:600;line-height:1.5;margin:0;
padding:0}
.vp-flow li+li::before{content:"\\2192";color:var(--body-text-color-subdued);font-weight:400;margin-right:4px}
.vp-n{display:inline-flex;align-items:center;justify-content:center;width:1.6em;height:1.6em;border-radius:50%;
background:var(--color-accent);font-size:.78em;font-weight:700;line-height:1;flex:none}
.vp-n, .vp-n *{color:#fff}
.vp-step{display:flex;align-items:center;gap:10px;margin:20px 0 4px;padding-bottom:8px;
border-bottom:1px solid var(--border-color-primary)}
.vp-step h2{font-size:1.35rem;font-weight:700;margin:0;line-height:1.3}
.vp-step-sub{color:var(--body-text-color-subdued);font-size:.98rem}
.vp-badge{display:inline-flex;align-items:center;gap:4px;padding:1px 10px;border-radius:999px;font-size:.88em;
font-weight:600;white-space:nowrap;border:1px solid transparent;line-height:1.5}
.vp-badge.vp-pass, .vp-badge.vp-pass *{background:var(--vp-pass-bg);color:var(--vp-pass-fg);border-color:var(--vp-pass)}
.vp-badge.vp-fail, .vp-badge.vp-fail *{background:var(--vp-fail-bg);color:var(--vp-fail-fg);border-color:var(--vp-fail)}
.vp-badge.vp-partial, .vp-badge.vp-partial *, .vp-badge.vp-warn, .vp-badge.vp-warn *{background:var(--vp-warn-bg);
color:var(--vp-warn-fg);border-color:var(--vp-warn)}
.vp-badge.vp-info, .vp-badge.vp-info *{background:var(--vp-info-bg);color:var(--vp-info-fg);border-color:var(--vp-info)}
.vp-badge.vp-unknown, .vp-badge.vp-unknown *, .vp-badge.vp-off, .vp-badge.vp-off *{background:var(--vp-muted-bg);
color:var(--vp-muted-fg);border-color:var(--vp-muted)}
.vp-badge *{border:0}
.vp-verdict{display:inline-flex;align-items:center;gap:8px;padding:5px 14px;border-radius:8px;font-size:1.15rem;
font-weight:700;line-height:1.35;white-space:nowrap}
.vp-verdict, .vp-verdict *{color:#fff}
.vp-verdict.vp-pass{background:var(--vp-pass)}
.vp-verdict.vp-fail{background:var(--vp-fail)}
.vp-verdict.vp-warn, .vp-verdict.vp-partial{background:var(--vp-warn)}
.vp-verdict.vp-info{background:var(--vp-info)}
.vp-verdict.vp-unknown, .vp-verdict.vp-off{background:var(--vp-muted)}
.vp-outcome{display:flex;flex-wrap:wrap;align-items:center;gap:12px}
.vp-outcome-text{font-size:1.15rem;font-weight:600}
.vp-card{border:1px solid var(--border-color-primary);border-radius:10px;padding:14px 16px;
background:var(--block-background-fill)}
.vp-kv{display:grid;grid-template-columns:fit-content(15em) 1fr;gap:5px 18px;margin:12px 0 0;font-size:1rem}
.vp-kv dt{font-weight:600;color:var(--body-text-color-subdued);margin:0}
.vp-kv dd{margin:0}
.vp-round{border:1px solid var(--border-color-primary);border-radius:10px;padding:10px 14px;margin:0 0 10px;
background:var(--block-background-fill)}
.vp-round-head{display:flex;flex-wrap:wrap;align-items:center;gap:8px}
.vp-round-title{font-weight:700;font-size:1.05rem}
.vp-meta{color:var(--body-text-color-subdued);font-size:.9em}
.vp-round-summary{margin:4px 0 6px}
.vp-note{color:var(--body-text-color-subdued);font-size:.92em;margin-top:6px}
.vp-empty{color:var(--body-text-color-subdued);padding:14px 2px;font-style:italic}
table.vp-checks{border-collapse:collapse;width:100%;font-size:.95em;margin-top:6px;border:0}
.vp-checks tr{border:0}
.vp-checks td, .vp-checks th{border:0;border-bottom:1px solid var(--border-color-primary);padding:5px 10px;
text-align:left;vertical-align:top}
.vp-checks th{font-size:.78em;text-transform:uppercase;letter-spacing:.05em;color:var(--body-text-color-subdued);
font-weight:600}
.vp-checks td:first-child{white-space:nowrap;font-weight:500}
.vp-checks td:nth-child(2), .vp-checks th:nth-child(2){width:1%;white-space:nowrap}
.vp-checks tr.vp-row-fail td{background:var(--vp-fail-bg)}
.vp-checks [title]{cursor:help;text-decoration:underline dotted var(--border-color-primary);text-underline-offset:3px}
.vp-chips{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin-top:8px}
.vp-chips-label{font-size:.82em;font-weight:600;text-transform:uppercase;letter-spacing:.05em;
color:var(--body-text-color-subdued);margin-right:2px}
.vp-chips [title]{cursor:help}
.vp-visual{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin-top:10px;font-weight:600}
.vp-banner{display:flex;align-items:flex-start;gap:10px;padding:10px 14px;border-radius:8px;font-weight:600;
margin:4px 0;border:1px solid}
.vp-banner-icon{display:inline-flex;align-items:center;justify-content:center;flex:none;width:1.4em;height:1.4em;
border-radius:50%;font-size:.85em;font-weight:700;margin-top:1px}
.vp-banner-warning, .vp-banner-warning *{background:var(--vp-warn-bg);color:var(--vp-warn-fg);border-color:var(--vp-warn)}
.vp-banner-info, .vp-banner-info *{background:var(--vp-info-bg);color:var(--vp-info-fg);border-color:var(--vp-info)}
.vp-banner-warning .vp-banner-icon{background:var(--vp-warn);color:#fff}
.vp-banner-info .vp-banner-icon{background:var(--vp-info);color:#fff}
.vp-pre{white-space:pre-wrap;font-size:.85em;margin:4px 0;font-family:var(--font-mono)}
.vp-exec{margin:0}
.vp-exec img{width:100%;max-width:640px;border-radius:8px;border:1px solid var(--border-color-primary);display:block}
details summary{cursor:pointer;color:var(--body-text-color-subdued);font-size:.92em;margin-top:6px}
.vp-problems{margin:4px 0 4px 18px}
.vp-problems li{color:var(--vp-fail-fg)}
"""
SCOPE = ".gradio-container .prose "


def _scoped(rules: str) -> str:
    """Every selector in rules prefixed with SCOPE (one rule per line group; selectors split on ', ')."""
    out = []
    for block in rules.strip().split("}"):
        if "{" not in block:
            continue
        selectors, body = block.split("{", 1)
        scoped = ",".join(SCOPE + sel.strip() for sel in selectors.replace("\n", " ").split(", "))
        out.append(f"{scoped}{{{body.strip().replace(chr(10), '')}}}")  # one rule per line
    return "\n".join(out)


CSS = _VARS + _scoped(_RULES)

# (icon, word) per check status; the icon is decoration, the word carries the meaning
STATUS = {"pass": ("✓", "Pass"), "fail": ("✕", "Fail"), "partial": ("◐", "Partly checked"),
          "unknown": ("?", "Not reported"), "off": ("–", "Not run")}
VISUAL = {"pass": ("✓", "Pass", "pass"), "fail": ("✕", "Fail", "fail"), "uncertain": ("?", "Unsure", "warn"),
          "unavailable": ("!", "Unavailable", "warn"), "skipped": ("–", "Skipped", "off")}
# Round outcomes: green executed, amber rejected (stopped before anything moved), blue refused, grey in progress
_OUTCOME_CLASS = {"executed": "pass", "blocked": "warn", "refused": "info", "planning": "unknown",
                  "verifying": "unknown", "pending": "unknown"}
_OUTCOME_ICON = {"blocked": "■", "refused": "⊘", "planning": "…", "verifying": "…", "pending": "…"}
_ICON = {"pass": "✓", "fail": "✕", "info": "⊘", "unknown": "…", "warn": "!", "partial": "◐", "off": "–"}


def badge(text: str, kind: str, icon: str | None = None) -> str:
    icon = _ICON.get(kind, "") if icon is None else icon
    mark = f'<span aria-hidden="true">{escape(icon)}</span> ' if icon else ""
    return f'<span class="vp-badge vp-{escape(kind)}">{mark}{escape(text)}</span>'


def verdict(text: str, kind: str, icon: str | None = None) -> str:
    """A large, solid status chip (attempt outcome, Judge verdict)."""
    icon = _ICON.get(kind, "") if icon is None else icon
    mark = f'<span aria-hidden="true">{escape(icon)}</span>' if icon else ""
    return f'<span class="vp-verdict vp-{escape(kind)}">{mark}<span>{escape(text)}</span></span>'


def status_badge(status: str) -> str:
    icon, word = STATUS.get(status, ("", status))
    return badge(word, status, icon)


def banner_html(banner: Banner | None) -> str:
    if banner is None:
        return ""
    icon = "!" if banner.level == "warning" else "i"
    return (f'<div class="vp-banner vp-banner-{escape(banner.level)}" role="status">'
            f'<span class="vp-banner-icon" aria-hidden="true">{icon}</span><span>{escape(banner.text)}</span></div>')


def header_html(brand: str, tagline: str, subtitle: str, flow: Sequence[str]) -> str:
    steps = "".join(f'<li><span class="vp-n">{i}</span>{escape(s)}</li>' for i, s in enumerate(flow, 1))
    return (f'<header class="vp-header"><h1><span class="vp-brand">{escape(brand)}</span> '
            f'<span class="vp-tagline">{escape(tagline)}</span></h1><p>{escape(subtitle)}</p>'
            f'<ol class="vp-flow" aria-label="Pipeline">{steps}</ol></header>')


def step_html(number: int | None, title: str, subtitle: str = "") -> str:
    """A section heading in the story's order; the number matches the header's pipeline."""
    n = f'<span class="vp-n" aria-hidden="true">{number}</span>' if number is not None else ""
    sub = f'<span class="vp-step-sub">{escape(subtitle)}</span>' if subtitle else ""
    return f'<div class="vp-step">{n}<h2>{escape(title)}</h2>{sub}</div>'


def empty_html(text: str) -> str:
    return f'<div class="vp-empty">{escape(text)}</div>'


def _outcome_kind(r: RoundView) -> str:
    kind = _OUTCOME_CLASS.get(r.outcome, "warn")
    if r.outcome == "executed" and r.judge is not None and not r.judge.success:
        kind = "fail"
    return kind


def round_html(r: RoundView) -> str:
    kind = _outcome_kind(r)
    title = r.title or f"Round {r.index + 1}"
    icon = _ICON[kind] if r.outcome == "executed" else _OUTCOME_ICON.get(r.outcome, "!")
    head = [f'<span class="vp-round-title">{escape(title)}</span>',
            badge(OUTCOME_WORDS.get(r.outcome, r.outcome.replace("_", " ")), kind, icon)]
    meta = [m for m in (r.planner, latency_text(r.latency_s)) if m]
    if meta:
        head.append(f'<span class="vp-meta">{escape(" · ".join(meta))}</span>')
    parts = [f'<div class="vp-round vp-k-{kind}"><div class="vp-round-head">{"".join(head)}</div>'
             f'<div class="vp-round-summary">{escape(r.summary)}</div>']
    if r.problems:
        parts.append('<ul class="vp-problems">' + "".join(f"<li>{escape(p)}</li>" for p in r.problems) + "</ul>")
    # Checks that failed, were only partly checked or went unreported get a row with their evidence; passed and
    # not-run checks are compact chips, so a rejected round shows its failure first
    issues = [c for c in r.checks if c.status not in ("pass", "off")]
    if issues:
        rows = "".join(
            f'<tr class="vp-row-{escape(c.status)}"><td><span title="{escape(_check_tip(c.code, c.check))}">'
            f"{escape(c.title)}</span></td><td>{status_badge(c.status)}</td><td>{escape(c.evidence)}</td></tr>"
            for c in issues)
        parts.append(f'<table class="vp-checks"><tr><th>Check</th><th>Result</th><th>Details</th></tr>{rows}</table>')
    for status, label in (("pass", "Passed"), ("off", "Not run")):
        group = [c for c in r.checks if c.status == status]
        if group:
            icon = STATUS[status][0]
            chips = "".join(f'<span title="{escape(_check_tip(c.code, c.check))}">{badge(c.title, status, icon)}</span>'
                            for c in group)
            parts.append(f'<div class="vp-chips"><span class="vp-chips-label">{label}</span>{chips}</div>')
    if r.checks_note:
        parts.append(f'<div class="vp-note">{escape(r.checks_note[:1].upper() + r.checks_note[1:])}.</div>')
    if r.visual_status is not None:
        icon, word, vkind = VISUAL.get(r.visual_status, ("!", r.visual_status, "warn"))
        parts.append('<div class="vp-visual">Cosmos visual check <span class="vp-meta">(learned, not verification)'
                     f"</span>{badge(word, vkind, icon)}"
                     + (f' <span class="vp-meta">{escape(r.visual_note)}</span>' if r.visual_note else "") + "</div>")
        if r.visual:
            rows = "".join(
                f"<tr><td>{escape(stage_name(v.stage))}</td><td>{escape(', '.join(stage_name(x) for x in v.views))}</td>"
                f"<td>{badge(*_visual_badge(v.verdict))}</td><td>{escape(v.reason)}</td>"
                f"<td>{escape(latency_text(v.latency_s))}</td></tr>" for v in r.visual)
            parts.append('<table class="vp-checks"><tr><th>Moment</th><th>Views</th><th>Verdict</th><th>Reason</th>'
                         f"<th>Time</th></tr>{rows}</table>")
    if r.plan_json is not None:
        label = "Refusal" if r.plan_kind == "refusal" else "Plan"
        parts.append(f'<details><summary>{label} of this round (JSON)</summary><pre class="vp-pre">'
                     f"{escape(r.plan_json)}</pre></details>")
    parts.append("</div>")
    return "".join(parts)


def _check_tip(code: str, check: str) -> str:
    return f"Verifier check {code}: {CHECK_HELP.get(check, check)}"


def _visual_badge(verdict_: str) -> tuple[str, str, str]:
    icon, word, kind = VISUAL.get(verdict_, ("!", verdict_, "warn"))
    return word, kind, icon


def timeline_html(rounds: Sequence[RoundView]) -> str:
    if not rounds:
        return empty_html("No rounds yet.")
    return "".join(round_html(r) for r in rounds)


def judge_html(judge: JudgeView | None, note: str = "") -> str:
    if judge is None:
        return empty_html(note or "No judge verdict: nothing was executed or refused.")
    rows = "".join(
        f'<tr class="vp-row-{"pass" if p.passed else "fail"}"><td><span title="{escape(f"{p.id}: {p.meaning}")}">'
        f"{escape(p.name or p.id)}</span></td><td>{status_badge('pass' if p.passed else 'fail')}</td></tr>"
        for p in judge.predicates)
    out = [f'<div>{verdict(judge.verdict, "pass" if judge.success else "fail")}</div>',
           f'<table class="vp-checks"><tr><th>Criterion</th><th>Result</th></tr>{rows}</table>']
    failed = [p.name or p.id for p in judge.predicates if not p.passed]
    if failed:
        out.append(f'<div class="vp-note">Failed: {escape(", ".join(failed))}</div>')
    if judge.violations:
        out.append(f'<div class="vp-note">Unsafe contacts: {escape("; ".join(judge.violations))}</div>')
    if judge.details:
        out.append('<details><summary>Measurements</summary>' + "".join(
            f'<div class="vp-pre"><b>{escape(predicate_name(k).replace("_", " "))}</b>: {escape(v)}</div>'
            for k, v in judge.details) + "</details>")
    if note:
        out.append(f'<div class="vp-note">{escape(note)}</div>')
    return "".join(out)


def status_html(lines: Sequence[tuple[str, str]], headline: str = "", kind: str = "info",
                notes: Sequence[str] = (), icon: str | None = None) -> str:
    """A card: the headline ("Word: sentence" becomes a solid chip with the word and the sentence beside it), then
    label/value lines (task, planner, ground truth) and muted notes."""
    word, _, sentence = headline.partition(": ")
    head = (f'<div class="vp-outcome">{verdict(word, kind, icon)}'
            + (f'<span class="vp-outcome-text">{escape(sentence)}</span>' if sentence else "") + "</div>"
            if headline else "")
    body = "".join(f"<dt>{escape(k)}</dt><dd>{escape(v)}</dd>" for k, v in lines)
    extra = "".join(f'<div class="vp-note">{escape(n)}</div>' for n in notes)
    return f'<div class="vp-card">{head}<dl class="vp-kv">{body}</dl>{extra}</div>'


def gif_html(gif: bytes | None, caption: str) -> str:
    """The execution as an animated GIF in a data URI, so Gradio serves no files from disk for it."""
    if not gif:
        return empty_html(caption)
    data = base64.b64encode(gif).decode("ascii")
    return (f'<figure class="vp-exec"><img alt="The approved plan executing in the simulator" '
            f'src="data:image/gif;base64,{data}"/><figcaption class="vp-note">{escape(caption)}</figcaption></figure>')


def attempt_status(view: AttemptView) -> tuple[str, str, str]:
    """(headline, colour kind, icon) for an attempt: green success, red judge failure or false refusal, amber blocked
    (no plan passed the checks, so nothing moved) and other stops."""
    if view.success:
        return view.headline, "pass", "✓"
    if view.outcome in ("executed", "refused"):
        return view.headline, "fail", "✕"
    return view.headline, "warn", "■" if view.outcome == "budget_exhausted" else "!"
