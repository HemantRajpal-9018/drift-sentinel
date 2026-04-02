"""Tests for HTML report generation."""

import numpy as np
import pytest
from pathlib import Path

from drift_sentinel.reports.html_report import HTMLReportGenerator
from drift_sentinel.detectors.base import DriftResult


def _make_result(name="PSI", is_drift=False, score=0.1, feature="feat_0"):
    return DriftResult(
        detector_name=name,
        is_drift=is_drift,
        score=score,
        threshold=0.2,
        p_value=0.05,
        feature_name=feature,
    )


class TestHTMLReportGenerator:
    def test_generate_returns_html(self):
        gen = HTMLReportGenerator(title="Test Report")
        results = [_make_result(), _make_result("KS", True, 0.5, "feat_1")]
        html = gen.generate(results)
        assert "<html" in html
        assert "Test Report" in html

    def test_contains_results_table(self):
        gen = HTMLReportGenerator()
        results = [_make_result()]
        html = gen.generate(results)
        assert "PSI" in html

    def test_drift_status_displayed(self):
        gen = HTMLReportGenerator()
        results = [_make_result(is_drift=True, score=0.5)]
        html = gen.generate(results)
        assert "DRIFT" in html

    def test_ok_status_displayed(self):
        gen = HTMLReportGenerator()
        results = [_make_result(is_drift=False)]
        html = gen.generate(results)
        assert "OK" in html

    def test_chart_embedded(self):
        gen = HTMLReportGenerator()
        results = [_make_result()]
        html = gen.generate(results)
        assert "data:image/png;base64" in html

    def test_distribution_charts(self):
        gen = HTMLReportGenerator()
        results = [_make_result(feature="age")]
        rng = np.random.default_rng(42)
        ref = {"age": rng.normal(0, 1, 100)}
        cur = {"age": rng.normal(0, 1, 100)}
        html = gen.generate(results, reference=ref, current=cur)
        assert "Distribution" in html

    def test_write_to_file(self, tmp_path):
        gen = HTMLReportGenerator()
        results = [_make_result()]
        out = tmp_path / "report.html"
        gen.generate(results, output_path=str(out))
        assert out.exists()
        assert "<html" in out.read_text()

    def test_multiple_results(self):
        gen = HTMLReportGenerator()
        results = [
            _make_result("PSI", False, 0.05, "feat_0"),
            _make_result("KS", True, 0.8, "feat_0"),
            _make_result("PSI", True, 0.4, "feat_1"),
        ]
        html = gen.generate(results)
        assert html.count("YES") >= 2
