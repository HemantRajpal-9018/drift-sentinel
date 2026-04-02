"""Integration tests for the full drift-sentinel pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from drift_sentinel import DriftMonitor, PSI, KSTest, JensenShannonDivergence, ChiSquaredTest, MMD
from drift_sentinel.detectors.concept import ADWIN, PageHinkley, DDM
from drift_sentinel.detectors.base import DriftResult
from drift_sentinel.alerts.base import BaseAlert
from drift_sentinel.alerts.slack import SlackAlert
from drift_sentinel.alerts.email import EmailAlert
from drift_sentinel.alerts.pagerduty import PagerDutyAlert
from drift_sentinel.config import load_config, build_detectors, build_alerts
from drift_sentinel.reports.html_report import HTMLReportGenerator


class MockAlert(BaseAlert):
    def __init__(self):
        self.sent = []

    def send(self, results, message=None):
        self.sent.append(results)
        return True


class TestEndToEndPipeline:
    """Full pipeline: detect -> alert -> report."""

    def test_full_pipeline_no_drift(self, rng):
        ref = rng.normal(0, 1, 1000)
        cur = rng.normal(0, 1, 1000)
        alert = MockAlert()

        monitor = DriftMonitor(
            detectors=[PSI(), KSTest(), JensenShannonDivergence()],
            alerts=[alert],
        )
        results = monitor.check(ref, cur)

        assert not monitor.has_drift
        assert len(alert.sent) == 0
        assert len(results) == 3

        summary = monitor.summary()
        assert summary["status"] == "ok"

        gen = HTMLReportGenerator(title="Test Report")
        html = gen.generate(results)
        assert "OK" in html
        assert "<html" in html

    def test_full_pipeline_with_drift(self, rng):
        ref = rng.normal(0, 1, 1000)
        cur = rng.normal(5, 2, 1000)
        alert = MockAlert()

        monitor = DriftMonitor(
            detectors=[PSI(), KSTest(), JensenShannonDivergence()],
            alerts=[alert],
        )
        results = monitor.check(ref, cur)

        assert monitor.has_drift
        assert len(alert.sent) == 1
        drifted = [r for r in results if r.is_drift]
        assert len(drifted) >= 2

        summary = monitor.summary()
        assert summary["status"] == "drift"
        assert summary["drifted"] >= 2

    def test_multifeature_pipeline(self, rng):
        ref = {
            "age": rng.normal(30, 5, 500),
            "income": rng.normal(50000, 10000, 500),
            "score": rng.normal(700, 50, 500),
        }
        cur = {
            "age": rng.normal(30, 5, 500),
            "income": rng.normal(80000, 10000, 500),  # shifted
            "score": rng.normal(700, 50, 500),
        }

        monitor = DriftMonitor(detectors=[PSI(), KSTest()])
        results = monitor.check(ref, cur)

        # 2 detectors x 3 features = 6 results
        assert len(results) == 6

        # income should drift
        income_results = [r for r in results if r.feature_name == "income"]
        assert any(r.is_drift for r in income_results)

    def test_config_driven_pipeline(self, tmp_path, rng):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("""
detectors:
  - type: psi
    threshold: 0.2
  - type: ks
    threshold: 0.05
  - type: jensen_shannon
    threshold: 0.1
feature_names:
  - feat_0
  - feat_1
""")
        config = load_config(str(cfg_path))
        detectors = build_detectors(config)
        assert len(detectors) == 3

        ref = rng.normal(0, 1, (500, 2))
        cur = rng.normal(0, 1, (500, 2))

        monitor = DriftMonitor(detectors=detectors)
        results = monitor.check(ref, cur, feature_names=config["feature_names"])
        assert len(results) == 6  # 3 detectors x 2 features

    def test_streaming_then_batch(self, rng):
        monitor = DriftMonitor(detectors=[PSI(), ADWIN()])

        # Streaming updates
        for v in [0.5] * 20:
            monitor.update_streaming({"ADWIN": v})

        # Batch check
        ref = rng.normal(0, 1, 500)
        cur = rng.normal(0, 1, 500)
        results = monitor.check(ref, cur)
        assert len(results) == 1  # Only PSI runs on batch

    def test_multiple_checks_history(self, rng):
        monitor = DriftMonitor(detectors=[PSI()])
        for _ in range(5):
            ref = rng.normal(0, 1, 500)
            cur = rng.normal(0, 1, 500)
            monitor.check(ref, cur)

        assert len(monitor.history) == 5
        assert all("timestamp" in h for h in monitor.history)

    def test_report_with_distributions(self, rng, tmp_path):
        ref_dict = {"age": rng.normal(30, 5, 500), "income": rng.normal(50000, 10000, 500)}
        cur_dict = {"age": rng.normal(30, 5, 500), "income": rng.normal(50000, 10000, 500)}

        monitor = DriftMonitor(detectors=[PSI(), KSTest()])
        results = monitor.check(ref_dict, cur_dict)

        gen = HTMLReportGenerator(title="Distribution Report")
        out_path = tmp_path / "report.html"
        html = gen.generate(results, reference=ref_dict, current=cur_dict, output_path=str(out_path))

        assert out_path.exists()
        content = out_path.read_text()
        assert "Distribution" in content
        assert "data:image/png;base64" in content


class TestDetectorEdgeCases:
    """Edge case tests for all detectors."""

    def test_psi_identical_arrays(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 100)
        result = PSI().detect(arr, arr.copy())
        assert not result.is_drift
        assert result.score < 0.01

    def test_psi_small_sample(self):
        ref = np.array([1.0, 2.0, 3.0])
        cur = np.array([4.0, 5.0, 6.0])
        result = PSI(n_bins=2).detect(ref, cur)
        assert isinstance(result, DriftResult)

    def test_ks_identical_data(self):
        arr = np.arange(100, dtype=float)
        result = KSTest().detect(arr, arr.copy())
        assert not result.is_drift
        assert result.p_value == 1.0

    def test_js_uniform_distributions(self, rng):
        ref = rng.uniform(0, 1, 1000)
        cur = rng.uniform(0, 1, 1000)
        result = JensenShannonDivergence().detect(ref, cur)
        assert not result.is_drift

    def test_chi_squared_single_category(self):
        ref = np.array(["a"] * 100)
        cur = np.array(["a"] * 100)
        result = ChiSquaredTest().detect(ref, cur)
        assert not result.is_drift

    def test_chi_squared_new_category_in_current(self):
        ref = np.array(["a", "b"] * 50)
        cur = np.array(["a"] * 30 + ["b"] * 30 + ["c"] * 40)
        result = ChiSquaredTest().detect(ref, cur)
        assert isinstance(result, DriftResult)
        assert result.is_drift  # new category should trigger drift

    def test_mmd_single_sample(self):
        ref = np.array([[1.0, 2.0]])
        cur = np.array([[3.0, 4.0]])
        # Should not crash even with single samples
        result = MMD(n_permutations=10).detect(ref, cur, seed=42)
        assert isinstance(result, DriftResult)

    def test_mmd_high_dimensional(self, rng):
        ref = rng.normal(0, 1, (50, 10))
        cur = rng.normal(0, 1, (50, 10))
        result = MMD(n_permutations=20).detect(ref, cur, seed=42)
        assert isinstance(result, DriftResult)

    def test_adwin_single_value(self):
        adwin = ADWIN()
        result = adwin.update(1.0)
        assert not result.is_drift

    def test_page_hinkley_constant_input(self):
        ph = PageHinkley(threshold=50, min_instances=10)
        results = [ph.update(1.0) for _ in range(100)]
        assert not any(r.is_drift for r in results)

    def test_ddm_all_correct(self):
        ddm = DDM(min_instances=10)
        results = [ddm.update(0) for _ in range(200)]
        assert not any(r.is_drift for r in results)

    def test_ddm_all_errors(self):
        ddm = DDM(min_instances=10, drift_level=2.0)
        # Start with some correct, then all errors
        for _ in range(50):
            ddm.update(0)
        drifted = False
        for _ in range(200):
            r = ddm.update(1)
            if r.is_drift:
                drifted = True
                break
        assert drifted


class TestAlertEdgeCases:
    """Edge case tests for alert integrations."""

    def test_slack_custom_message(self):
        with patch("drift_sentinel.alerts.slack.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            mock_post.return_value.raise_for_status = MagicMock()
            alert = SlackAlert(webhook_url="https://hooks.slack.com/test")
            result = DriftResult("Test", True, 0.5, 0.2)
            assert alert.send([result], message="Custom alert!") is True
            call_data = json.loads(mock_post.call_args[1]["data"])
            assert call_data["text"] == "Custom alert!"

    def test_email_with_tls_and_auth(self):
        with patch("drift_sentinel.alerts.email.smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_server)
            mock_smtp.return_value.__exit__ = MagicMock(return_value=False)

            alert = EmailAlert(
                smtp_host="smtp.test.com",
                smtp_port=587,
                username="user",
                password="pass",
                from_addr="test@example.com",
                to_addrs=["recipient@example.com"],
                use_tls=True,
            )
            result = DriftResult("Test", True, 0.5, 0.2)
            assert alert.send([result]) is True
            mock_server.starttls.assert_called_once()
            mock_server.login.assert_called_once_with("user", "pass")

    def test_pagerduty_severity_levels(self):
        for severity in ["info", "warning", "error", "critical"]:
            with patch("drift_sentinel.alerts.pagerduty.requests.post") as mock_post:
                mock_post.return_value = MagicMock(status_code=200)
                mock_post.return_value.raise_for_status = MagicMock()
                alert = PagerDutyAlert(routing_key="key", severity=severity)
                result = DriftResult("Test", True, 0.5, 0.2)
                alert.send([result])
                call_data = json.loads(mock_post.call_args[1]["data"])
                assert call_data["payload"]["severity"] == severity

    def test_alert_format_message_multiple_drifted(self):
        alert = MockAlert()
        results = [
            DriftResult("PSI", True, 0.5, 0.2, feature_name="age"),
            DriftResult("KS", True, 0.8, 0.05, p_value=0.001, feature_name="income"),
            DriftResult("JS", False, 0.01, 0.1),
        ]
        msg = alert.format_message(results)
        assert "age" in msg
        assert "income" in msg
        assert "p-value" in msg


class TestMonitorEdgeCases:
    """Edge case tests for DriftMonitor."""

    def test_monitor_no_detectors(self, rng):
        monitor = DriftMonitor(detectors=[])
        ref = rng.normal(0, 1, 100)
        cur = rng.normal(0, 1, 100)
        results = monitor.check(ref, cur)
        assert results == []
        assert not monitor.has_drift

    def test_monitor_alert_failure_doesnt_crash(self, rng):
        class FailingAlert(BaseAlert):
            def send(self, results, message=None):
                raise RuntimeError("Alert send failed")

        monitor = DriftMonitor(
            detectors=[PSI()],
            alerts=[FailingAlert()],
        )
        ref = rng.normal(0, 1, 1000)
        cur = rng.normal(5, 1, 1000)
        # Should not raise despite alert failure
        results = monitor.check(ref, cur)
        assert monitor.has_drift

    def test_monitor_disable_alerts(self, rng):
        alert = MockAlert()
        monitor = DriftMonitor(
            detectors=[PSI()],
            alerts=[alert],
            alert_on_drift=False,
        )
        ref = rng.normal(0, 1, 1000)
        cur = rng.normal(5, 1, 1000)
        monitor.check(ref, cur)
        assert monitor.has_drift
        assert len(alert.sent) == 0

    def test_monitor_mixed_detectors(self, rng):
        monitor = DriftMonitor(
            detectors=[PSI(), KSTest(), ADWIN(), DDM()],
        )
        ref = rng.normal(0, 1, 500)
        cur = rng.normal(0, 1, 500)
        # Batch check only runs batch detectors
        results = monitor.check(ref, cur)
        names = [r.detector_name for r in results]
        assert "PSI" in names
        assert "KS Test" in names
        assert "ADWIN" not in names
        assert "DDM" not in names

    def test_monitor_2d_array_auto_feature_names(self, rng):
        monitor = DriftMonitor(detectors=[PSI()])
        ref = rng.normal(0, 1, (500, 4))
        cur = rng.normal(0, 1, (500, 4))
        results = monitor.check(ref, cur)
        assert len(results) == 4
        feature_names = [r.feature_name for r in results]
        assert "feature_0" in feature_names
        assert "feature_3" in feature_names

    def test_monitor_summary_keys(self, rng):
        monitor = DriftMonitor(detectors=[PSI(), KSTest()])
        ref = rng.normal(0, 1, 500)
        cur = rng.normal(0, 1, 500)
        monitor.check(ref, cur)

        summary = monitor.summary()
        assert "status" in summary
        assert "total_detectors" in summary
        assert "drifted" in summary
        assert "detectors" in summary
        assert "total_checks" in summary

    def test_monitor_history_entries_structure(self, rng):
        monitor = DriftMonitor(detectors=[PSI()])
        ref = rng.normal(0, 1, 500)
        cur = rng.normal(0, 1, 500)
        monitor.check(ref, cur)

        assert len(monitor.history) == 1
        entry = monitor.history[0]
        assert "timestamp" in entry
        assert "has_drift" in entry
        assert "n_detectors" in entry
        assert "n_drifted" in entry
        assert "results" in entry


class TestConfigEdgeCases:
    """Edge case tests for config loading."""

    def test_config_all_detector_types(self, tmp_path):
        cfg_path = tmp_path / "all_detectors.yaml"
        cfg_path.write_text("""
detectors:
  - type: psi
  - type: ks
  - type: ks_test
  - type: jensen_shannon
  - type: js
  - type: chi_squared
  - type: mmd
  - type: adwin
  - type: page_hinkley
  - type: ddm
""")
        config = load_config(str(cfg_path))
        detectors = build_detectors(config)
        assert len(detectors) == 10

    def test_config_all_alert_types(self, tmp_path):
        cfg_path = tmp_path / "all_alerts.yaml"
        cfg_path.write_text("""
alerts:
  - type: slack
    webhook_url: https://hooks.slack.com/test
  - type: email
    smtp_host: smtp.test.com
    to_addrs:
      - test@example.com
  - type: pagerduty
    routing_key: test-key
""")
        config = load_config(str(cfg_path))
        alerts = build_alerts(config)
        assert len(alerts) == 3

    def test_config_detector_with_params(self, tmp_path):
        cfg_path = tmp_path / "params.yaml"
        cfg_path.write_text("""
detectors:
  - type: psi
    threshold: 0.5
    n_bins: 20
  - type: mmd
    threshold: 0.01
    n_permutations: 200
""")
        config = load_config(str(cfg_path))
        detectors = build_detectors(config)
        assert detectors[0].threshold == 0.5
        assert detectors[0].n_bins == 20
        assert detectors[1].threshold == 0.01
        assert detectors[1].n_permutations == 200


class TestDriftResultSerialization:
    """Tests for DriftResult serialization."""

    def test_to_dict_all_fields(self):
        result = DriftResult(
            detector_name="PSI",
            is_drift=True,
            score=0.5,
            threshold=0.2,
            p_value=0.01,
            feature_name="age",
            details={"n_bins": 10},
        )
        d = result.to_dict()
        assert d["detector_name"] == "PSI"
        assert d["is_drift"] is True
        assert d["score"] == 0.5
        assert d["threshold"] == 0.2
        assert d["p_value"] == 0.01
        assert d["feature_name"] == "age"
        assert d["details"]["n_bins"] == 10
        assert "timestamp" in d

    def test_to_dict_json_serializable(self):
        result = DriftResult("Test", True, 0.5, 0.2, p_value=0.01)
        d = result.to_dict()
        # Should be JSON serializable
        json_str = json.dumps(d, default=str)
        parsed = json.loads(json_str)
        assert parsed["detector_name"] == "Test"

    def test_result_timestamp_is_utc(self):
        result = DriftResult("Test", False, 0.0, 0.2)
        assert result.timestamp.tzinfo is not None


class TestFeatureImportanceEdgeCases:
    """Edge cases for feature importance shift detection."""

    def test_empty_importance_dicts(self):
        from drift_sentinel.detectors.feature_importance import FeatureImportanceShift
        fis = FeatureImportanceShift(threshold=0.1)
        result = fis.detect_from_importance({}, {})
        assert not result.is_drift

    def test_single_feature_importance(self):
        from drift_sentinel.detectors.feature_importance import FeatureImportanceShift
        fis = FeatureImportanceShift(threshold=0.1, method="absolute")
        ref = {"feat_a": 1.0}
        cur = {"feat_a": 0.5}
        result = fis.detect_from_importance(ref, cur)
        assert result.is_drift
        assert "feat_a" in result.drifted_features

    def test_many_features(self):
        from drift_sentinel.detectors.feature_importance import FeatureImportanceShift
        fis = FeatureImportanceShift(threshold=0.1)
        ref = {f"feat_{i}": 1.0 / 20 for i in range(20)}
        cur = {f"feat_{i}": 1.0 / 20 for i in range(20)}
        result = fis.detect_from_importance(ref, cur)
        assert not result.is_drift

    def test_importance_shift_result_to_dict(self):
        from drift_sentinel.detectors.feature_importance import FeatureImportanceShift
        fis = FeatureImportanceShift()
        ref = {"a": 0.3, "b": 0.7}
        cur = {"a": 0.7, "b": 0.3}
        result = fis.detect_from_importance(ref, cur)
        d = result.to_dict()
        assert "is_drift" in d
        assert "overall_score" in d
        assert "feature_scores" in d
        assert "drifted_features" in d
        assert "timestamp" in d


class TestCLIEdgeCases:
    """Edge case CLI tests."""

    def test_cli_help(self):
        from click.testing import CliRunner
        from drift_sentinel.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Drift Sentinel" in result.output

    def test_cli_check_help(self):
        from click.testing import CliRunner
        from drift_sentinel.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["check", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output

    def test_cli_monitor_help(self):
        from click.testing import CliRunner
        from drift_sentinel.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["monitor", "--help"])
        assert result.exit_code == 0

    def test_cli_report_help(self):
        from click.testing import CliRunner
        from drift_sentinel.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["report", "--help"])
        assert result.exit_code == 0

    def test_cli_csv_data(self, tmp_path, config_yaml, rng):
        import pandas as pd
        from click.testing import CliRunner
        from drift_sentinel.cli import cli

        ref_df = pd.DataFrame(rng.normal(0, 1, (200, 2)), columns=["a", "b"])
        cur_df = pd.DataFrame(rng.normal(0, 1, (200, 2)), columns=["a", "b"])
        ref_path = tmp_path / "ref.csv"
        cur_path = tmp_path / "cur.csv"
        ref_df.to_csv(ref_path, index=False)
        cur_df.to_csv(cur_path, index=False)

        runner = CliRunner()
        result = runner.invoke(cli, [
            "check", "-c", str(config_yaml),
            "-r", str(ref_path), "-C", str(cur_path),
        ])
        assert result.exit_code == 0

    def test_cli_npz_data(self, tmp_path, config_yaml, rng):
        from click.testing import CliRunner
        from drift_sentinel.cli import cli

        ref = rng.normal(0, 1, (200, 2))
        cur = rng.normal(0, 1, (200, 2))
        ref_path = tmp_path / "ref.npz"
        cur_path = tmp_path / "cur.npz"
        np.savez(ref_path, data=ref)
        np.savez(cur_path, data=cur)

        runner = CliRunner()
        result = runner.invoke(cli, [
            "check", "-c", str(config_yaml),
            "-r", str(ref_path), "-C", str(cur_path),
        ])
        assert result.exit_code == 0


class TestPackageImports:
    """Verify all public imports work."""

    def test_top_level_imports(self):
        from drift_sentinel import (
            PSI, KSTest, JensenShannonDivergence,
            ChiSquaredTest, MMD, ADWIN, PageHinkley,
            DDM, DriftMonitor,
        )
        assert PSI is not None
        assert DriftMonitor is not None

    def test_version(self):
        from drift_sentinel import __version__
        assert __version__ == "0.1.0"

    def test_alert_imports(self):
        from drift_sentinel.alerts import SlackAlert, EmailAlert, PagerDutyAlert, BaseAlert
        assert SlackAlert is not None
        assert BaseAlert is not None

    def test_report_imports(self):
        from drift_sentinel.reports import HTMLReportGenerator
        assert HTMLReportGenerator is not None

    def test_detector_imports(self):
        from drift_sentinel.detectors import (
            BaseDetector, PSI, KSTest, JensenShannonDivergence,
            ChiSquaredTest, MMD, ADWIN, PageHinkley, DDM,
        )
        assert BaseDetector is not None

    def test_feature_importance_import(self):
        from drift_sentinel.detectors.feature_importance import (
            FeatureImportanceShift,
            ImportanceShiftResult,
        )
        assert FeatureImportanceShift is not None
