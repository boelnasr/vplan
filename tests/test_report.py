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


def test_html_names_every_dry_run_config_in_the_banner(html):
    assert ("Dry run, not a headline result:</strong> config 1 (openrouter), config 2 (openrouter), config 3 "
            "(openrouter) and config 7 (openrouter) did not run on Nebius Token Factory.") in html
    assert "<strong>Not a headline report.</strong>" in html
    assert "Status:</strong> not a headline result: configs 2, 3 and 7 are dry runs" in html


def test_a_token_factory_headline_with_a_dry_config_1_says_exactly_that(tmp_path, monkeypatch):
    stamps = []
    real = report_module._stamp
    monkeypatch.setattr(report_module, "_stamp", lambda configs: stamps.append(list(configs)) or real(configs))
    rows = ev.scenario_rows(source_override={3: ev.TOKEN_FACTORY, 7: ev.TOKEN_FACTORY})
    page = build_report(rows, tmp_path).html.read_text(encoding="utf-8")
    assert "Dry run, not a headline result:</strong> config 1 (openrouter) did not run" in page
    assert "Not a headline report" not in page
    assert "Status:</strong> Nebius Token Factory run (headline source)" in page
    # Every chart stamps the dry-run configs it shows, and only those: config 1 is in the success, outcome and strata
    # charts; the gain chart (configs 2, 3 and 7) carries no stamp.
    assert len(stamps) == 5 and all(s in ([], [1]) for s in stamps) and stamps.count([1]) == 3
    assert real([1]) == "\nDRY RUN (config 1): not a headline result" and real([]) == ""
    assert real([2, 3, 7]) == "\nDRY RUN (configs 2, 3 and 7): not a headline result"


def test_a_missing_config_7_is_named_in_the_not_headline_banner(tmp_path):
    rows = [r for r in ev.scenario_rows(source_override={3: ev.TOKEN_FACTORY}) if r.config != 7]
    page = build_report(rows, tmp_path).html.read_text(encoding="utf-8")
    assert "Dry run, not a headline result:</strong> config 1 (openrouter) did not run" in page
    assert ("<strong>Not a headline report.</strong> A headline report needs configs 2, 3 and 7, each from a complete "
            "run on Nebius Token Factory over the held-out test seeds; here config 7 not run.") in page


def test_a_token_factory_report_on_dev_seeds_says_why_it_is_not_a_headline(tmp_path):
    rows = [r.model_copy(update={"seed_set": "dev"})
            for r in ev.scenario_rows(source_override={3: ev.TOKEN_FACTORY, 7: ev.TOKEN_FACTORY})]
    page = build_report(rows, tmp_path).html.read_text(encoding="utf-8")
    reason = "development seeds, not the held-out test set"
    assert ("<strong>Not a headline report.</strong> A headline report needs configs 2, 3 and 7, each from a complete "
            f"run on Nebius Token Factory over the held-out test seeds; here {reason}.") in page
    assert f"Status:</strong> not a headline result: {reason}." in page
    assert "(headline source)" not in page and f"Nebius Token Factory run, not a headline result: {reason}" in page


def test_an_incomplete_or_limited_token_factory_run_raises_the_not_headline_banner(tmp_path):
    rows = ev.scenario_rows(source_override={3: ev.TOKEN_FACTORY, 7: ev.TOKEN_FACTORY})
    files = build_report(rows, tmp_path, manifests=[ev.run_manifest(3, 7, 400), ev.run_manifest(7, 400, 400)],
                         seed_set_size=400)
    page = files.html.read_text(encoding="utf-8")
    reason = "config 3 covers only the first 7 of 400 tasks; config 7 run is incomplete (7/400 tasks)"
    assert ("<strong>Not a headline report.</strong> A headline report needs configs 2, 3 and 7, each from a complete "
            f"run on Nebius Token Factory over the held-out test seeds; here {reason}.") in page
    assert f"Status:</strong> not a headline result: {reason}." in page
    assert "(headline source)" not in page  # no row or tile calls these runs a headline source
    assert "Nebius Token Factory run, not a headline result: limited run" in page
    assert "Nebius Token Factory run, not a headline result: incomplete run" in page
    metrics = json.loads(files.metrics.read_text(encoding="utf-8"))
    assert metrics["status"]["complete"] is False and metrics["headline"]["status"] == "incomplete"


def test_html_makes_no_fixed_claim_that_feedback_beats_resampling(html):
    from tests.test_metrics import CLAIMS
    lowered = html.lower()
    assert "separates the effect of feedback from retrying at the same budget" in html
    assert not [claim for claim in CLAIMS if claim in lowered]


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


def test_html_and_metrics_carry_no_local_path_from_a_run_manifest(tmp_path, rows):
    files = build_report(rows, tmp_path, manifests=[ev.local_manifest()])
    for path in (files.html, files.metrics):
        text = path.read_text(encoding="utf-8")
        assert not [needle for needle in ev.LOCAL_PATHS if needle in text], path.name
        assert "e" * 40 in text and "1.4.2" in text  # the ManipulaPy checkout's SHA and version stay


def test_incomplete_and_limited_runs_title_the_page_and_raise_banners(tmp_path, rows):
    from tests.test_metrics import COVERAGE
    page = build_report(rows, tmp_path, manifests=COVERAGE, seed_set_size=9).html.read_text(encoding="utf-8")
    title = ("INCOMPLETE: config 6 2/7 tasks; config 7 7/9 tasks — LIMITED: config 3 first 7 of 9 tasks; config 6 "
             "first 7 of 9 tasks — DEGRADED: 1 service failure — DRY RUN: configs 1, 2, 3 and 7 — vplan evaluation "
             "report")
    assert f"<title>{title}</title>" in page and f"<h1>{title}</h1>" in page
    assert ("<strong>Incomplete runs.</strong> Config 6 recorded 2 of its 7 planned tasks; config 7 recorded 7 of "
            "its 9 planned tasks.") in page
    assert ("<strong>Limited runs.</strong> Config 3 ran the first 7 of the seed set&#x27;s 9 tasks; config 6 ran "
            "the first 7 of the seed set&#x27;s 9 tasks.") in page
    assert "9 tasks prepared, 7 in this report" in page
    assert "INCOMPLETE 2/7" in page and "LIMITED: first 7 of 9" in page  # the configurations table


def test_a_complete_report_has_no_incomplete_or_limited_label(tmp_path, rows):
    full = [r for r in rows if r.config in (3, 7)]
    page = build_report(full, tmp_path, manifests=[ev.run_manifest(3, 7, 7), ev.run_manifest(7, 7, 7)],
                        seed_set_size=7).html.read_text(encoding="utf-8")
    assert "INCOMPLETE" not in page and "LIMITED" not in page and "Incomplete runs" not in page
    assert "<title>DEGRADED: 1 service failure — DRY RUN: configs 2, 3 and 7 — vplan evaluation report</title>" in page
    assert ("<strong>Provider failures.</strong> 1 provider request failed (config 7: 1). Failures stay in the "
            "denominators, so these rates partly measure provider availability, not planning.") in page
    assert "complete 7/7" in page


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
