"""Static evaluation report (spec §15): metrics.json plus one self-contained HTML file.

The HTML is a pure function of the metrics. Tables carry every number; the matplotlib charts (Agg canvas, embedded
as base64 PNG) restate them. The page loads nothing: no scripts, stylesheets, fonts or images by reference, and no
text forms a URL, so it opens offline as a single file.
"""
from __future__ import annotations

import base64
import html
import io
import json
import math
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

from vplan.evaluate.metrics import (ATTRIBUTION_TAGS, CATEGORIES, CHECKS, DRY_RUN_LABEL, HEADLINE_CONFIGS, N_BOOT,
                                    OUTCOMES, SAFETY, TOKEN_FACTORY_PROVIDERS, compute_metrics, config_list)
from vplan.records import AttemptRow

# Chart palette: the dataviz reference instance (categorical slots in fixed order, validated for adjacent pairs;
# the sub-3:1 slots are relieved by the table twin of every chart), its blue sequential ramp and light chrome.
SURFACE, INK, INK_2, GRID, AXIS, EMPTY = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#f0efec"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6")
BLUES = LinearSegmentedColormap.from_list("vplan_blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DPI = 110

SHORT_NAMES = {1: "1 · unverified", 2: "2 · first try (round 1 of 3)", 3: "3 · verify + repair",
               5: "5 · heuristic baseline", 6: "6 · frontier, no verifier", 7: "7 · equal-budget resampling"}
GROUPS = ("Nemotron configurations", "heuristic baseline (no LLM)", "frontier reference (no verifier)")
OUTCOME_SEGMENTS = ("success", "executed, Judge failure", "refused (false refusal)", "budget exhausted",
                    "malformed final", "service failure", "other (sim error, stale approval)")
CSS = """
:root { color-scheme: light; --surface: #fcfcfb; --page: #f9f9f7; --ink: #0b0b0b; --ink-2: #52514e;
  --grid: #e1e0d9; --rule: #c3c2b7; --accent: #2a78d6; --baseline: #fdf1eb; --baseline-edge: #eb6834;
  --critical: #d03b3b; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1100px; margin: 0 auto; padding: 32px 16px 64px; }
h1 { font-size: 26px; margin: 0 0 4px; }
h2 { font-size: 20px; margin: 44px 0 8px; padding-bottom: 4px; border-bottom: 1px solid var(--rule); }
h3 { font-size: 16px; margin: 22px 0 6px; }
p, li { max-width: 80ch; }
.muted { color: var(--ink-2); }
.banner { background: var(--surface); border: 1px solid var(--rule); border-left: 4px solid var(--critical);
  padding: 10px 14px; margin: 12px 0; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin: 16px 0; }
.tile { background: var(--surface); border: 1px solid var(--grid); border-radius: 8px; padding: 12px 14px; }
.tile.baseline { border: 2px solid var(--baseline-edge); background: var(--baseline); }
.tile .label { color: var(--ink-2); font-size: 13px; }
.tile .value { font-size: 26px; font-weight: 600; margin: 2px 0; }
.tile .sub { color: var(--ink-2); font-size: 13px; }
.table-wrap { overflow-x: auto; margin: 8px 0 16px; }
table { border-collapse: collapse; font-size: 14px; background: var(--surface); }
th, td { padding: 5px 10px; border-bottom: 1px solid var(--grid); text-align: left; vertical-align: top;
  font-variant-numeric: tabular-nums; }
th { color: var(--ink-2); font-weight: 600; }
.num { text-align: right; }
td:first-child { white-space: nowrap; }
tr.baseline td { background: var(--baseline); font-weight: 600; }
tr.baseline td:first-child { box-shadow: inset 4px 0 0 var(--baseline-edge); }
figure { margin: 18px 0; }
figure img { display: block; max-width: 100%; height: auto; }
figcaption { color: var(--ink-2); font-size: 13px; margin-top: 4px; }
code, pre { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 13px; }
pre { background: var(--surface); border: 1px solid var(--grid); padding: 10px; overflow-x: auto; }
nav ul { display: flex; flex-wrap: wrap; gap: 4px 16px; padding: 0; margin: 12px 0; list-style: none; }
nav a { color: var(--accent); }
"""
SECTIONS = (("headline", "Headline"), ("configurations", "Configurations"), ("success", "Feasible success"),
            ("outcomes", "Outcomes"), ("refusals", "Refusals"), ("safety", "Safety"),
            ("blocked", "Blocked plans"), ("strata", "Strata"), ("attribution", "Attribution"),
            ("cost", "Cost and latency"), ("notes", "Definitions and limits"), ("provenance", "Provenance"))


class ReportFiles(NamedTuple):
    metrics: Path
    html: Path


def build_report(rows: Iterable[AttemptRow], out_dir: Path, *, title: str = "vplan evaluation report",
                 n_boot: int = N_BOOT, seed: int = 0, headline_providers: Iterable[str] = TOKEN_FACTORY_PROVIDERS,
                 manifests: Sequence[Mapping[str, Any]] = (), undetermined: int | None = None,
                 seed_set_size: int | None = None) -> ReportFiles:
    """Write OUT_DIR/metrics.json and OUT_DIR/report.html. Both are rendered in memory first, then each replaces its
    file atomically, so a failed build leaves the previous files whole. The keyword arguments are compute_metrics'."""
    metrics = compute_metrics(rows, n_boot=n_boot, seed=seed, headline_providers=frozenset(headline_providers),
                              manifests=manifests, undetermined=undetermined, seed_set_size=seed_set_size)
    metrics_text = json.dumps(metrics, indent=2, allow_nan=False) + "\n"
    page = render_html(metrics, title=title)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = ReportFiles(out_dir / "metrics.json", out_dir / "report.html")
    _write_atomic(files.metrics, metrics_text)
    _write_atomic(files.html, page)
    return files


def _write_atomic(path: Path, text: str) -> None:
    # Replace, never truncate in place (as pilot._write_json): readers never see a half-written file.
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


# ---- text helpers ----------------------------------------------------------------------------------
def _esc(value: Any) -> str:
    """Escaped text. '://' becomes an entity, so no provider or manifest string ever forms a URL in the file."""
    return html.escape(str(value), quote=True).replace("://", "&#58;//")


def _pct(rate: float | None) -> str:
    return "—" if rate is None else f"{100 * rate:.1f}%"


def _ci(ci: Sequence[float] | None, signed: bool = False) -> str:
    if ci is None:
        return "—"
    fmt = "{:+.1f}" if signed else "{:.1f}"
    return f"[{fmt.format(100 * ci[0])},\u00a0{fmt.format(100 * ci[1])}]"  # an interval never wraps inside


def _pp(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:+.1f} pp"


def _num(value: float | int | None, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value:,}" if isinstance(value, int) else f"{value:,.{digits}f}"


def _prop(p: Mapping[str, Any] | None) -> str:
    if p is None:
        return "not run"
    return f"{p['k']}/{p['n']} · {_pct(p['rate'])} {_ci(p['ci95'])}"


def _counts(counts: Mapping[str, int]) -> str:
    items = [f"{k}: {v}" for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])) if v]
    return ", ".join(items) if items else "—"


_UNIT_WORDS = re.compile(r"\b(?:pp|of)\b")


def _numeric(value: Any) -> bool:
    """Numbers, and strings of figures such as "3/5 · 60.0% [23.1, 88.2]", "+4.0 pp" or "12 of 40"."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    return isinstance(value, str) and bool(value) and not re.search(r"[A-Za-z_]", _UNIT_WORDS.sub("", value))


def _table(head: Sequence[str], body: Iterable[Sequence[Any]], classes: Sequence[str | None] = ()) -> str:
    """An escaped table; classes[i] is the optional CSS class of body row i. A column after the first whose every
    cell is a number is right-aligned, header included."""
    body = [list(row) for row in body]
    align = ['' if j == 0 or not body or not all(_numeric(row[j]) for row in body) else ' class="num"'
             for j in range(len(head))]
    rows = []
    for i, row in enumerate(body):
        cls = classes[i] if i < len(classes) else None
        rows.append((f'<tr class="{cls}">' if cls else "<tr>")
                    + "".join(f"<td{align[j]}>{_esc(v)}</td>" for j, v in enumerate(row)) + "</tr>")
    return ('<div class="table-wrap"><table><thead><tr>'
            + "".join(f"<th{align[j]}>{_esc(h)}</th>" for j, h in enumerate(head))
            + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")


def _figure_html(uri: str | None, alt: str, caption: str) -> str:
    if uri is None:
        return '<p class="muted">No chart: this report has no data for it.</p>'
    return (f'<figure><img src="{uri}" alt="{_esc(alt)}"><figcaption>{_esc(caption)} The table in this section '
            "carries the same numbers.</figcaption></figure>")


# ---- charts -----------------------------------------------------------------------------------------
def _figure(width_in: float, height_in: float) -> Figure:
    fig = Figure(figsize=(width_in, height_in), dpi=DPI, facecolor=SURFACE, layout="constrained")
    FigureCanvasAgg(fig)  # explicit Agg canvas; pyplot's global backend state is never touched
    return fig


def _style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.set_axisbelow(True)


def _png(fig: Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=SURFACE, metadata={"Software": None})
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _stamp(configs: Sequence[int]) -> str:
    """The chart's dry-run stamp: exactly the dry-run configurations the chart shows, or nothing."""
    return f"\nDRY RUN ({config_list(configs)}): not a headline result" if configs else ""


def _dry(m: Mapping[str, Any], configs: Iterable[int]) -> list[int]:
    shown = set(configs)
    return [c for c in m["dry_run_configs"] if c in shown]


def _title(ax: Axes, text: str, dry_configs: Sequence[int]) -> None:
    ax.set_title(text + _stamp(dry_configs), loc="left", fontsize=11, color=INK)


def _group(config: int) -> int:
    return {5: 1, 6: 2}.get(config, 0)


def _success_chart(m: Mapping[str, Any]) -> str | None:
    rows = [(int(c), b["success"]) for c, b in m["configs"].items() if b["success"]["n"]][::-1]
    if not rows:
        return None
    fig = _figure(8.4, 0.5 * len(rows) + 1.6)
    ax = fig.add_subplot()
    _style(ax)
    for y, (config, s) in enumerate(rows):
        ax.barh(y, s["rate"], height=0.5, color=SERIES[_group(config)])
        low, high = s["ci95"]
        ax.errorbar(s["rate"], y, xerr=[[s["rate"] - low], [high - s["rate"]]], fmt="none", ecolor=INK,
                    elinewidth=1, capsize=3)
        ax.text(high + 0.015, y, f"{s['k']}/{s['n']}", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(rows)), [SHORT_NAMES[c] for c, _ in rows])
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    present = sorted({_group(c) for c, _ in rows})
    if len(present) > 1:
        fig.legend(handles=[Patch(color=SERIES[g], label=GROUPS[g]) for g in present], loc="outside lower center",
                   ncol=len(present), frameon=False, fontsize=8)
    _title(ax, "Feasible-task success by configuration, with 95% Wilson intervals", _dry(m, (c for c, _ in rows)))
    return _png(fig)


def _gain_chart(m: Mapping[str, Any]) -> str | None:
    h = m["headline"]
    gains = [(name, g) for name, g in (("feedback gain (3 − 7)", h["feedback_gain"]),
                                       ("repair gain (3 − 2)", h["repair_gain"])) if g["n"]]
    if not gains:
        return None
    reach = max([20.0] + [100 * abs(v) for _, g in gains for v in g["ci95"]])
    fig = _figure(8.4, 0.8 * len(gains) + 1.4)
    ax = fig.add_subplot()
    _style(ax)
    ax.axvline(0, color=AXIS, linewidth=1)
    for y, (_, g) in enumerate(gains):
        low, high = (100 * v for v in g["ci95"])
        ax.plot([low, high], [y, y], color=SERIES[0], linewidth=2, solid_capstyle="round")
        ax.plot([100 * g["mean"]], [y], "o", color=SERIES[0], markersize=8, markeredgecolor=SURFACE,
                markeredgewidth=2)
        ax.text(100 * g["mean"], y + 0.18, f"{_pp(g['mean'])}  {_ci(g['ci95'], signed=True)}  N = {g['n']}",
                ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_yticks(range(len(gains)), [name for name, _ in gains])
    ax.set_ylim(-0.5, len(gains) - 0.2)
    ax.set_xlim(-reach * 1.1, reach * 1.1)
    ax.set_xlabel("difference in feasible success, percentage points", color=INK_2, fontsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    _title(ax, "Paired differences on the same tasks, with 95% bootstrap intervals",
           _dry(m, (c for _, g in gains for c in g["configs"])))
    return _png(fig)


def _outcome_segments(block: Mapping[str, Any]) -> list[int]:
    o, wins = block["outcomes"]["feasible"], block["success"]["k"]
    return [wins, o["executed"] - wins, o["refused"], o["budget_exhausted"], o["malformed_final"],
            o["service_failure"], o["sim_error"] + o["stale_approval"] + o["visual_uncertain"]]


def _outcome_chart(m: Mapping[str, Any]) -> str | None:
    rows = [(int(c), b) for c, b in m["configs"].items() if b["feasible"]][::-1]
    if not rows:
        return None
    fig = _figure(8.4, 0.5 * len(rows) + 2.1)
    ax = fig.add_subplot()
    _style(ax)
    for y, (_, block) in enumerate(rows):
        left = 0.0
        for i, count in enumerate(_outcome_segments(block)):
            share = count / block["feasible"]
            if share:
                # a 2 px surface gap between segments, never a dark border
                ax.barh(y, share, left=left, height=0.5, color=SERIES[i], edgecolor=SURFACE, linewidth=1.5)
            left += share
    ax.set_yticks(range(len(rows)), [f"{SHORT_NAMES[c]}  (N = {b['feasible']})" for c, b in rows])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    fig.legend(handles=[Patch(color=SERIES[i], label=name) for i, name in enumerate(OUTCOME_SEGMENTS)],
               loc="outside lower center", ncol=4, frameon=False, fontsize=8)
    _title(ax, "Final outcome of every feasible task (all outcomes stay in the denominator)",
           _dry(m, (c for c, _ in rows)))
    return _png(fig)


def _check_chart(m: Mapping[str, Any]) -> str | None:
    rows = [(int(c), b["blocked_plans"]["by_check"]) for c, b in m["configs"].items() if b["blocked_plans"]["blocked"]]
    checks = [c for c in CHECKS if any(by_check[c] for _, by_check in rows)]
    if not rows or not checks:
        return None
    data = np.array([[by_check[c] for c in checks] for _, by_check in rows], dtype=float)
    fig = _figure(8.4, 0.45 * len(rows) + 2.0)
    ax = fig.add_subplot()
    _style(ax)
    ax.imshow(data, cmap=BLUES, vmin=0, vmax=max(1.0, data.max()), aspect="auto")
    for (y, x), value in np.ndenumerate(data):
        ax.text(x, y, f"{int(value)}", ha="center", va="center", fontsize=9,
                color="white" if value > 0.55 * data.max() else INK)
    ax.set_xticks(range(len(checks)), [c.replace("_", " ") for c in checks], rotation=25, ha="right")
    ax.set_yticks(range(len(rows)), [SHORT_NAMES[c] for c, _ in rows])
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    _title(ax, "Verifier findings in blocked plans, by check", _dry(m, (c for c, _ in rows)))
    return _png(fig)


def _strata_chart(m: Mapping[str, Any]) -> str | None:
    blocks = [(int(c), b) for c, b in m["configs"].items() if b["feasible"]]
    if not blocks:
        return None
    ncols = min(3, len(blocks))
    nrows = math.ceil(len(blocks) / ncols)
    fig = _figure(3.0 * ncols + 0.9, 2.9 * nrows + 0.7)
    axes = fig.subplots(nrows, ncols, squeeze=False)
    cmap = BLUES.with_extremes(bad=EMPTY)
    image = None
    for ax, (config, block) in zip(axes.flat, blocks):
        grid = np.full((4, 4), np.nan)
        for cell in block["strata"]:
            if cell["n"]:
                grid[cell["distractors"], cell["obstacles"]] = cell["rate"]
                ax.text(cell["obstacles"], cell["distractors"], f"{cell['k']}/{cell['n']}", ha="center",
                        va="center", fontsize=8, color="white" if cell["rate"] > 0.55 else INK)
        image = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=0, vmax=1)
        ax.set_title(SHORT_NAMES[config], fontsize=9, color=INK, loc="left")
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.tick_params(colors=INK_2, labelsize=8, length=0)
        ax.set_xlabel("obstacles", fontsize=8, color=INK_2)
        ax.set_ylabel("distractors", fontsize=8, color=INK_2)
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax in list(axes.flat)[len(blocks):]:
        ax.set_visible(False)
    fig.colorbar(image, ax=axes, shrink=0.8, format=PercentFormatter(1.0, decimals=0), label="feasible success")
    fig.suptitle("Feasible success by stratum (distractors × obstacles); grey cells have no tasks"
                 + _stamp(_dry(m, (c for c, _ in blocks))), x=0.02, ha="left", fontsize=11, color=INK)
    return _png(fig)


# ---- sections ---------------------------------------------------------------------------------------
def _configs(m: Mapping[str, Any]) -> list[tuple[int, Mapping[str, Any]]]:
    return [(int(c), b) for c, b in m["configs"].items()]


def _row_classes(m: Mapping[str, Any]) -> list[str | None]:
    return ["baseline" if c == 5 else None for c, _ in _configs(m)]


def _tile(label: str, value: str, sub: str, baseline: bool = False) -> str:
    return (f'<div class="tile{" baseline" if baseline else ""}"><div class="label">{_esc(label)}</div>'
            f'<div class="value">{_esc(value)}</div><div class="sub">{_esc(sub)}</div></div>')


def _prop_tile(label: str, p: Mapping[str, Any] | None, baseline: bool = False) -> str:
    if p is None:
        return _tile(label, "not run", "this configuration has no rows", baseline)
    return _tile(label, _pct(p["rate"]), f"{p['k']}/{p['n']} feasible tasks · 95% CI {_ci(p['ci95'])}", baseline)


def _gain_tile(label: str, g: Mapping[str, Any]) -> str:
    if not g["n"]:
        return _tile(label, "not run" if g["label"] == "not run" else "no pairs", "no task ran in both configurations")
    return _tile(label, _pp(g["mean"]), f"95% CI {_ci(g['ci95'], signed=True)} · N = {g['n']} paired tasks")


def _seed_text(s: Mapping[str, Any]) -> str:
    """The seed set's size from its manifest, when known, next to the tasks in this report."""
    present = f"{s['feasible']} feasible, {s['impossible']} impossible"
    if s["size"] is None:
        return f"{s['tasks']} tasks: {present}"
    return f"{s['size']} tasks prepared, {s['tasks']} in this report: {present}"


def _run_text(run: Mapping[str, Any] | None) -> str:
    if run is None:
        return "—"
    parts = ([f"INCOMPLETE {run['recorded']}/{run['planned']}"] if run["incomplete"] else []) + (
        [f"LIMITED: first {run['planned']} of {run['prepared']}"] if run["limited"] else [])
    return "; ".join(parts) or f"complete {run['recorded']}/{run['planned']}"


def _sentence_list(items: Sequence[str]) -> str:
    text = "; ".join(items)
    return text[:1].upper() + text[1:]


def _banners(m: Mapping[str, Any]) -> str:
    out = []
    blocks = m["configs"]
    status = m["status"]
    if status["incomplete"]:
        out.append("<strong>Incomplete runs.</strong> " + _esc(_sentence_list(
            [f"config {r['config']} recorded {r['recorded']} of its {r['planned']} planned tasks"
             for r in status["incomplete"]])) + ". Do not read their numbers as a result; eval run --resume "
                   "continues a run.")
    if status["limited"]:
        out.append("<strong>Limited runs.</strong> " + _esc(_sentence_list(
            [f"config {r['config']} ran the first {r['planned']} of the seed set's {r['prepared']} tasks"
             for r in status["limited"]])) + ". A prefix (eval run --n) is a smoke test, never a complete result.")
    if status["degraded"]:
        total = sum(d["service_failures"] for d in status["degraded"])
        out.append(f"<strong>Provider failures.</strong> {total} provider request{'' if total == 1 else 's'} failed ("
                   + _esc("; ".join(f"config {d['config']}: {d['service_failures']}" for d in status["degraded"]))
                   + "). Failures stay in the denominators, so these rates partly measure provider availability, not "
                     "planning.")
    if m["dry_run_configs"]:
        named = [f"config {c} ({', '.join(blocks[str(c)]['provenance']['providers'])})" for c in m["dry_run_configs"]]
        listed = named[0] if len(named) == 1 else f"{', '.join(named[:-1])} and {named[-1]}"
        out.append(f"<strong>{_esc(DRY_RUN_LABEL.capitalize())}:</strong> {_esc(listed)} did not run on Nebius "
                   "Token Factory. Spec §14 takes headline numbers from Token Factory runs only.")
    h = m["headline"]
    if not h["headline_source"]:
        out.append("<strong>Not a headline report.</strong> A headline report needs "
                   f"{_esc(config_list(HEADLINE_CONFIGS))}, each from a complete run on Nebius Token Factory over the "
                   f"held-out test seeds; here {_esc(h['reason'])}.")
    short = [f"config {c}: {b['tasks']} of {b['coverage']['tasks_in_report']}" for c, b in _configs(m)
             if b["coverage"]["missing"]]
    if short:
        out.append("<strong>Incomplete coverage.</strong> Some configurations did not attempt every task in this "
                   f"report ({_esc('; '.join(short))}). Their rates use their own N; paired gains use only tasks "
                   "both configurations attempted.")
    if not m["provenance"]["config_hashes_consistent"]:
        out.append("<strong>Mixed configurations.</strong> Attempts in this report ran under more than one set of "
                   "verifier, Judge and physics config hashes; see Provenance.")
    return "".join(f'<div class="banner">{text}</div>' for text in out)


def _headline(m: Mapping[str, Any]) -> str:
    h = m["headline"]
    tiles = "".join([
        _prop_tile("First-try success (config 2)", h["first_try_success"]),
        _prop_tile("Success with repair (config 3)", h["success_with_repair"]),
        _gain_tile("Repair gain (3 − 2)", h["repair_gain"]),
        _gain_tile("Feedback gain (3 − 7)", h["feedback_gain"]),
        _prop_tile("Heuristic baseline (config 5, no LLM)", h["heuristic_baseline"], baseline=True)])
    return (f'<p><strong>Status:</strong> {_esc(h["label"])}. Feasible tasks only; impossible tasks are scored '
            'separately under Refusals.</p>'
            f'<div class="tiles">{tiles}</div>'
            f'<p><strong>Repair gain.</strong> {_esc(h["repair_gain"]["reading"])}</p>'
            f'<p><strong>Feedback gain.</strong> {_esc(h["feedback_gain"]["reading"])}</p>'
            + _figure_html(_success_chart(m), "Bar chart of feasible success per configuration",
                           "Feasible success per configuration with 95% Wilson intervals.")
            + _figure_html(_gain_chart(m), "Dot plot of the paired repair and feedback gains",
                           "Repair gain and feedback gain with 95% paired-bootstrap intervals."))


def _configurations(m: Mapping[str, Any]) -> str:
    body, classes = [], []
    for entry in m["configurations"]:
        block = m["configs"].get(str(entry["config"]))
        if block is None:
            ran, tasks, coverage, run = "not run", "—", "—", "—"
        else:
            prov = block["provenance"]
            ran = f"{', '.join(prov['providers'])} · {', '.join(prov['models'])}"
            tasks = f"{block['tasks']} ({block['feasible']} / {block['impossible']})"
            coverage = f"{block['tasks']} of {block['coverage']['tasks_in_report']}"
            run = _run_text(block["run"])
        body.append([entry["config"], entry["name"], entry["role"], entry["planner"], ran, tasks, coverage, run,
                     entry["label"]])
        classes.append("baseline" if entry["config"] == 5 else None)
    return _table(["Config", "Configuration", "Role", "Planner / provider (spec)", "Ran with (provider · model)",
                   "Tasks (feasible / impossible)", "Coverage", "Run (rows / planned)", "Label"], body, classes)


def _success(m: Mapping[str, Any]) -> str:
    body = [[SHORT_NAMES[c], b["success"]["n"], b["success"]["k"], _pct(b["success"]["rate"]),
             _ci(b["success"]["ci95"]), _num(b["cost"]["calls_to_success_mean"], 2), b["label"]]
            for c, b in _configs(m)]
    return ("<p>Success = the Judge accepts the executed run (S1, S2, V1-V4) on a feasible task. Every outcome stays "
            "in the denominator. The heuristic baseline row is highlighted.</p>"
            + _table(["Config", "Feasible tasks", "Successes", "Rate", "95% Wilson CI", "Calls to success (mean)",
                      "Label"], body, _row_classes(m)))


def _outcomes(m: Mapping[str, Any]) -> str:
    def table(kind):
        return _table(["Config"] + [o.replace("_", " ") for o in OUTCOMES],
                      [[SHORT_NAMES[c]] + [b["outcomes"][kind][o] for o in OUTCOMES] for c, b in _configs(m)],
                      _row_classes(m))
    missing = [f"{SHORT_NAMES[c]}: {_counts(b['outcomes']['without_record'])}" for c, b in _configs(m)
               if b["outcomes"]["without_record"]]
    return ("<h3>Feasible tasks</h3>" + table("feasible") + "<h3>Impossible tasks</h3>" + table("impossible")
            + (f'<p class="muted">Attempts without a record (worker failures, by kind): {_esc("; ".join(missing))}.'
               "</p>" if missing else "")
            + _figure_html(_outcome_chart(m), "Stacked bars of the final outcome of feasible tasks",
                           "Final outcome shares of feasible tasks per configuration."))


def _refusals(m: Mapping[str, Any]) -> str:
    confusion = []
    rates = []
    by_type = []
    for c, b in _configs(m):
        r = b["refusals"]
        f, i = r["confusion"]["feasible"], r["confusion"]["impossible"]
        confusion.append([SHORT_NAMES[c], f["refused"], f["attempted"], f["no_decision"], i["refused"],
                          i["attempted"], i["no_decision"]])
        rates.append([SHORT_NAMES[c], _prop(r["wrongly_refused_feasible"]), _prop(r["wrongly_attempted_impossible"]),
                      _prop(r["correct_refusals"]), _counts(r["attempted_impossible_by_outcome"]),
                      _prop(r["reason"]["matching_proof"]), _counts(r["reason"]["false_refusal_reasons"])])
        for kind, t in r["reason"]["by_type"].items():
            by_type.append([SHORT_NAMES[c], kind, t["tasks"], t["refused"], t["reason_matches"], _counts(t["given"])])
    return ("<p>The planner's final answer per task: refused, attempted (a plan), or no decision (no record, a "
            "service failure, malformed output). Impossible tasks are correct only when refused (spec §13 R); "
            "whether the refusal names the proof's reason is scored separately (D40).</p>"
            "<h3>Confusion matrix</h3>"
            + _table(["Config", "Feasible: refused", "Feasible: attempted", "Feasible: no decision",
                      "Impossible: refused", "Impossible: attempted", "Impossible: no decision"], confusion,
                     _row_classes(m))
            + "<h3>Wrong decisions and reasons</h3>"
            + _table(["Config", "Wrongly refused feasible", "Wrongly attempted impossible", "Correct refusals",
                      "Attempted impossible, by outcome", "Reason matches the proof", "False-refusal reasons"],
                     rates, _row_classes(m))
            + "<h3>Refusal reasons by impossibility type</h3>"
            + "<p class=\"muted\">Types: 1 target out of reach, 2 bin out of reach, 3 target enclosed, 4 referent "
              "absent (spec §6).</p>"
            + _table(["Config", "Type", "Tasks", "Refused", "Reason matches", "Reasons given"], by_type))


def _safety(m: Mapping[str, Any]) -> str:
    body, pairs = [], []
    for c, b in _configs(m):
        s = b["safety"]
        body.append([SHORT_NAMES[c], s["executions"]] + [_prop(s["violations"][v]) for v in SAFETY]
                    + [_prop(s["executions_with_violation"]), s["statement"]])
        pairs.append([SHORT_NAMES[c], _counts(s["by_body_pair"]["V1"]), _counts(s["by_body_pair"]["V4"])])
    return ("<p>V1-V4 on the executed physics run (spec §13), over every execution, feasible or not. A dropped "
            "object fails S1 and is not a safety violation.</p>"
            + _table(["Config", "Executions"] + [f"{v} violated" for v in SAFETY]
                     + ["Any violation", "Statement"], body, _row_classes(m))
            + "<h3>Violating contacts by body pair (executions with at least one)</h3>"
            + _table(["Config", "V1 (contact)", "V4 (self-collision)"], pairs, _row_classes(m)))


def _blocked(m: Mapping[str, Any]) -> str:
    summary, checks = [], []
    for c, b in _configs(m):
        p = b["blocked_plans"]
        summary.append([SHORT_NAMES[c], p["planner_rounds"], p["blocked"]]
                       + [p["by_category"][k] for k in CATEGORIES]
                       + [p["malformed_output"], p["protocol_error"], p["service_failure"], p["repeat_plans"],
                          _counts(p["by_attempt_outcome"])])
        checks.append([SHORT_NAMES[c]] + [p["by_check"][k] for k in CHECKS])
    return ("<p>A blocked plan is a planner round the Verifier rejected. Physical-constraint rejections are kept "
            "apart from solver non-convergence (IK accuracy and continuity: the IK solver did not converge or jumped "
            "branch, which is not proof that a pose is unreachable) and from malformed output, which never reaches "
            "the Verifier. A round with both kinds counts as solver non-convergence, because a solver failure stops "
            "the later segment checks. The heuristic is deterministic, so its rounds after the first repeat the same "
            "plan (D39).</p>"
            + _table(["Config", "Planner rounds", "Blocked", "Physical constraint", "Solver non-convergence",
                      "Learned visual", "Malformed output", "Protocol errors", "Service failures", "Repeat plans",
                      "Blocked rounds by attempt outcome"], summary, _row_classes(m))
            + "<h3>Findings by check</h3>"
            + _table(["Config"] + [k.replace("_", " ") for k in CHECKS], checks, _row_classes(m))
            + _figure_html(_check_chart(m), "Heat map of Verifier findings by check and configuration",
                           "Verifier findings by check; a blocked plan has at most one finding per check."))


def _strata(m: Mapping[str, Any]) -> str:
    tables = []
    for c, b in _configs(m):
        if not b["feasible"]:
            continue
        cells = {(x["distractors"], x["obstacles"]): x for x in b["strata"]}
        body = [[f"{d} distractor{'' if d == 1 else 's'}"]
                + [f"{cells[d, o]['k']}/{cells[d, o]['n']}" if cells[d, o]["n"] else "—" for o in range(4)]
                for d in range(4)]
        tables.append(f"<h3>{_esc(SHORT_NAMES[c])}</h3>"
                      + _table(["Feasible successes / tasks", "0 obstacles", "1 obstacle", "2 obstacles",
                                "3 obstacles"], body))
    return ("<p>Feasible tasks by stratum: distractor count × obstacle count (0-3 each).</p>"
            + _figure_html(_strata_chart(m), "Small-multiple heat maps of feasible success by stratum",
                           "Feasible success by stratum, one panel per configuration.")
            + "".join(tables))


def _attribution(m: Mapping[str, Any]) -> str:
    def tags(kind):
        return _table(["Config", "Failures"] + list(ATTRIBUTION_TAGS),
                      [[SHORT_NAMES[c], b["attribution"]["failures"][kind]]
                       + [b["attribution"][kind][t] for t in ATTRIBUTION_TAGS] for c, b in _configs(m)],
                      _row_classes(m))
    observable = [[SHORT_NAMES[c], _counts(b["attribution"]["observable"]["feasible"]),
                   _counts(b["attribution"]["observable"]["impossible"])] for c, b in _configs(m)]
    return ("<p>Observable failure categories first: the final round's check and stage, outcome or failed Judge "
            "predicate. The tags after them are diagnostic evidence, not proof of cause (spec §10). Every feasible "
            "seed-set task is witnessed by its canonical plan passing all checks and the Judge (D38), so the "
            "motion-layer tag, which needs the canonical plan to fail the same check, cannot occur here.</p>"
            "<h3>Observable failures of the final round</h3>"
            + _table(["Config", "Feasible tasks", "Impossible tasks"], observable, _row_classes(m))
            + "<h3>Diagnostic attribution, feasible tasks</h3>" + tags("feasible")
            + "<h3>Diagnostic attribution, impossible tasks</h3>" + tags("impossible"))


def _cost(m: Mapping[str, Any]) -> str:
    cost, latency = [], []
    for c, b in _configs(m):
        k, t, lat = b["cost"], b["cost"]["tokens"], b["latency_s"]
        llm = k["llm"]
        cost.append([SHORT_NAMES[c], k["attempts"], k["attempts_without_record"], k["planner_calls"],
                     _num(k["calls_per_attempt_mean"], 2), _num(k["calls_per_feasible_success"], 2),
                     _num(t["known_total"]) if llm else "no LLM", t["missing_usage_calls"] if llm else "—",
                     _num(t["per_attempt_median"], 0) if llm else "—",
                     _num(t["per_feasible_success"], 0) if llm else "—"])
        latency.append([SHORT_NAMES[c], _num(lat["planner_total"]), _num(lat["planner_per_attempt_median"], 2),
                        _num(lat["planner_per_call_median"], 2), _num(lat["wall_total"]),
                        _num(lat["wall_per_attempt_median"], 1), lat["wall_attempts"]])
    return ("<p>Tokens are summed from the calls that reported usage; calls without usage are counted, not "
            "estimated. Planner latency is the time spent in planner calls; wall time is the supervisor's time for "
            "the whole attempt, including simulation (config 2 has none of its own).</p>"
            + _table(["Config", "Attempts", "Without record", "Planner calls", "Calls per attempt",
                      "Calls per feasible success", "Tokens (known)", "Calls without usage",
                      "Tokens per attempt (median)", "Tokens per feasible success"], cost, _row_classes(m))
            + _table(["Config", "Planner latency total (s)", "Per attempt, median (s)", "Per call, median (s)",
                      "Wall time total (s)", "Wall per attempt, median (s)", "Attempts with wall time"], latency,
                     _row_classes(m)))


def _notes(m: Mapping[str, Any]) -> str:
    gain = m["headline"]["repair_gain"]
    items = [
        "First-try success: the round-1 plan passes the gate and the Judge reports success. Config 2 is round 1 of "
        "config 3, rescored as an attempt with no repair budget.",
        "Success with repair: success within k = 3 repair rounds, over the same tasks and denominator.",
        f"Gains: paired differences on the same tasks; 95% interval by percentile bootstrap over tasks "
        f"({gain['n_boot']:,} resamples, seed {gain['seed']}). Proportions carry 95% Wilson intervals and their N.",
        "Feasible success and impossible-task refusals are separate results, so the task mix cannot inflate the "
        "headline (D8).",
        "Grasping is attachment-assisted, a simulation simplification (spec §2). The Judge is independent check logic "
        "on the same physics engine as the Verifier (spec §13). The checks are sample-based, not formal "
        "guarantees.",
        "The frontier row (config 6) is an optional reference without the verifier; it is excluded from the headline. "
        "The Cosmos shadow study (config 4) is a separate pass over saved frames and never measures incremental "
        "success; it is not part of this report.",
        "Headline sources are Nebius Token Factory runs only (spec §14). Any Nemotron configuration that ran on "
        f"another provider is labelled \"{DRY_RUN_LABEL}\", and a report is a headline report only if "
        f"{config_list(HEADLINE_CONFIGS)} all come from complete runs on Token Factory over the held-out test seeds "
        "(spec §14): an incomplete or limited run (D33) and a run on the development seeds are never headline "
        "sources.",
    ]
    return "<ul>" + "".join(f"<li>{_esc(item)}</li>" for item in items) + "</ul>"


def _provenance(m: Mapping[str, Any]) -> str:
    p = m["provenance"]
    s = p["seed_set"]
    hashes = [[SHORT_NAMES[c], ", ".join(f"{k} {v[:12]}" for k, v in sorted(h.items()))]
              for c, b in _configs(m) for h in b["provenance"]["config_hashes"]]
    runs = "".join(f"<pre>{_esc(json.dumps(run, indent=2, sort_keys=True))}</pre>" for run in p["runs"])
    undetermined = s["undetermined_excluded"]
    return (f"<p>Seed set <strong>{_esc(s['name'])}</strong>, sha256 <code>{_esc(s['hash'])}</code>: "
            f"{_esc(_seed_text(s))} (impossible by type: {_esc(_counts(s['impossible_by_type']))}).</p>"
            "<p>Undetermined candidates excluded: "
            + ("not supplied." if undetermined is None else
               f"{undetermined} (neither a witness nor a proof; excluded from the seed set and never scored, "
               "spec §6).")
            + "</p>"
            f"<p>Headline providers: {_esc(', '.join(p['headline_providers']))}. Config hashes consistent across "
            f"all attempts: {'yes' if p['config_hashes_consistent'] else 'no'}.</p>"
            + _table(["Config", "Config hashes (first 12 hex digits)"], hashes)
            + (f"<h3>Run manifests</h3>{runs}" if runs else '<p class="muted">No run manifests were supplied.</p>'))


def render_html(m: Mapping[str, Any], *, title: str = "vplan evaluation report") -> str:
    """The whole report page for one metrics dict (compute_metrics output). The title carries the pilot's prefixes
    (D33): INCOMPLETE, LIMITED, DEGRADED and DRY RUN."""
    s = m["provenance"]["seed_set"]
    title = " — ".join(m["status"]["title_prefixes"] + [title])
    sections = {"headline": _headline, "configurations": _configurations, "success": _success,
                "outcomes": _outcomes, "refusals": _refusals, "safety": _safety, "blocked": _blocked,
                "strata": _strata, "attribution": _attribution, "cost": _cost, "notes": _notes,
                "provenance": _provenance}
    nav = "".join(f'<li><a href="#{key}">{_esc(name)}</a></li>' for key, name in SECTIONS)
    body = "".join(f'<section><h2 id="{key}">{_esc(name)}</h2>{sections[key](m)}</section>' for key, name in SECTIONS)
    return ("<!DOCTYPE html>\n"
            '<html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<meta name="color-scheme" content="light">'
            f"<title>{_esc(title)}</title><style>{CSS}</style></head><body><main>"
            f"<header><h1>{_esc(title)}</h1><p class=\"muted\">Seed set {_esc(s['name'])} ({_esc(_seed_text(s))}), "
            f"sha256 <code>{_esc(s['hash'][:16])}</code>…; "
            f"metrics schema v{m['schema_version']}. Every number here is also in metrics.json.</p></header>"
            f"{_banners(m)}<nav><ul>{nav}</ul></nav>{body}</main></body></html>\n")
