"""W3.2 report: metrics.json plus one self-contained static HTML file, both replaced atomically."""
import base64
import json
import os
import re
from pathlib import Path

import pytest

from tests import eval_rows as ev
from vplan.evaluate import report as report_module
from vplan.evaluate.metrics import DRY_RUN_LABEL, compute_metrics
from vplan.evaluate.report import build_report

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(scope="module")
def rows():
    return ev.scenario_rows()


@pytest.fixture(scope="module")
def built(rows, tmp_path_factory):
    out = tmp_path_factory.mktemp("report")
    return out, build_report(rows, out)


@pytest.fixture(scope="module")
def html(built):
    return built[1].html.read_text(encoding="utf-8")


def test_build_report_writes_exactly_metrics_json_and_one_html_file(built, rows):
    out, files = built
    assert sorted(p.name for p in out.iterdir()) == ["metrics.json", "report.html"]
    assert files.metrics == out / "metrics.json" and files.html == out / "report.html"
    assert json.loads(files.metrics.read_text(encoding="utf-8")) == compute_metrics(rows)


def test_html_is_self_contained(html):
    lowered = html.lower()
    assert "http://" not in lowered and "https://" not in lowered
    for needle in ("<script", "<link", "<iframe", "<object", "<embed", "@import", "url("):
        assert needle not in lowered
    sources = re.findall(r'\bsrc="([^"]*)"', html)
    assert len(sources) >= 5
    for src in sources:
        assert src.startswith("data:image/png;base64,")
        assert base64.b64decode(src.split(",", 1)[1], validate=True).startswith(PNG_SIGNATURE)
    assert all(href.startswith("#") for href in re.findall(r'\bhref="([^"]*)"', html))
    assert html.startswith("<!DOCTYPE html>") and html.rstrip().endswith("</html>")


def test_html_labels_dry_runs_the_baseline_and_the_reference_rows(html):
    assert DRY_RUN_LABEL in html
    assert 'class="baseline"' in html  # the heuristic row is highlighted
    assert "excluded from the headline" in html  # config 6
    assert "shadow" in html  # config 4 is listed with its scope
    assert "does not by itself credit the feedback" in html
    assert "Zero observed violations in 3 executions." in html


def test_html_shows_the_headline_numbers(html):
    assert "1/5" in html and "2/5" in html  # first try (config 2) and with repair (config 3)
    assert "+20.0 pp" in html  # repair gain over the five feasible tasks


def test_html_counts_undetermined_candidates_only_when_supplied(html, rows, tmp_path):
    assert "Undetermined candidates excluded: not supplied." in html
    page = build_report(rows, tmp_path, undetermined=4).html.read_text(encoding="utf-8")
    assert "Undetermined candidates excluded: 4" in page


def test_html_never_carries_a_url_even_from_a_provider_name(tmp_path):
    source = ("https://llm.example.invalid/v1", "model-x")
    rows = [ev.scripted_row(3, 0, ev.FEASIBLE, [ev.call(ev.GOOD)], [ev.judged()], source=source)]
    page = build_report(rows, tmp_path).html.read_text(encoding="utf-8")
    assert "https://" not in page and "http://" not in page and "llm.example.invalid/v1" in page


def test_files_are_replaced_atomically(tmp_path, rows, monkeypatch):
    replaced = []
    real_replace = os.replace

    def spy(src, dst):
        replaced.append((Path(src), Path(dst)))
        real_replace(src, dst)
    monkeypatch.setattr(report_module.os, "replace", spy)
    build_report(rows, tmp_path)
    assert sorted(dst.name for _, dst in replaced) == ["metrics.json", "report.html"]
    assert all(src.parent == dst.parent and src.name != dst.name for src, dst in replaced)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["metrics.json", "report.html"]


def test_a_failed_replace_keeps_the_previous_files_and_leaves_no_temp_file(tmp_path, rows, monkeypatch):
    (tmp_path / "report.html").write_text("previous report", encoding="utf-8")
    (tmp_path / "metrics.json").write_text("{}", encoding="utf-8")

    def fail(src, dst):
        raise OSError("disk full")
    monkeypatch.setattr(report_module.os, "replace", fail)
    with pytest.raises(OSError, match="disk full"):
        build_report(rows, tmp_path)
    assert (tmp_path / "report.html").read_text(encoding="utf-8") == "previous report"
    assert (tmp_path / "metrics.json").read_text(encoding="utf-8") == "{}"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["metrics.json", "report.html"]


def test_report_renders_one_configuration_and_a_larger_cohort(tmp_path):
    only_heuristic = build_report([r for r in ev.scenario_rows() if r.config == 5], tmp_path / "one")
    assert "not run" in only_heuristic.html.read_text(encoding="utf-8")
    cohort = build_report(ev.random_rows(n_feasible=32, n_impossible=8), tmp_path / "cohort")
    page = cohort.html.read_text(encoding="utf-8")
    assert "https://" not in page and page.count("data:image/png;base64,") >= 5
