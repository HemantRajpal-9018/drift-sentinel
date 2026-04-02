"""Tests for DriftMonitor."""

import numpy as np
import pytest

from drift_sentinel.detectors.statistical import PSI, KSTest, JensenShannonDivergence
from drift_sentinel.detectors.concept import ADWIN, DDM
from drift_sentinel.detectors.base import DriftResult
from drift_sentinel.alerts.base import BaseAlert
from drift_sentinel.monitor import DriftMonitor


class MockAlert(BaseAlert):
    def __init__(self):
        self.sent = []

    def send(self, results, message=None):
        self.sent.append(results)
        return True


class TestDriftMonitor:
    def test_check_no_drift(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 1000)
        cur = rng.normal(0, 1, 1000)
        monitor = DriftMonitor(detectors=[PSI(), KSTest()])
        results = monitor.check(ref, cur)
        assert isinstance(results, list)
        assert all(isinstance(r, DriftResult) for r in results)
        assert not monitor.has_drift

    def test_check_with_drift(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 1000)
        cur = rng.normal(5, 1, 1000)
        monitor = DriftMonitor(detectors=[PSI(), KSTest()])
        results = monitor.check(ref, cur)
        assert monitor.has_drift

    def test_multiple_features_2d(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, (500, 3))
        cur = rng.normal(0, 1, (500, 3))
        monitor = DriftMonitor(detectors=[PSI()])
        results = monitor.check(ref, cur, feature_names=["a", "b", "c"])
        assert len(results) == 3

    def test_dict_input(self):
        rng = np.random.default_rng(42)
        ref = {"age": rng.normal(30, 5, 500), "income": rng.normal(50000, 10000, 500)}
        cur = {"age": rng.normal(30, 5, 500), "income": rng.normal(50000, 10000, 500)}
        monitor = DriftMonitor(detectors=[KSTest()])
        results = monitor.check(ref, cur)
        assert len(results) == 2

    def test_history_recorded(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 500)
        cur = rng.normal(0, 1, 500)
        monitor = DriftMonitor(detectors=[PSI()])
        monitor.check(ref, cur)
        monitor.check(ref, cur)
        assert len(monitor.history) == 2

    def test_alert_triggered_on_drift(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 1000)
        cur = rng.normal(5, 1, 1000)
        alert = MockAlert()
        monitor = DriftMonitor(detectors=[PSI()], alerts=[alert])
        monitor.check(ref, cur)
        assert len(alert.sent) == 1

    def test_alert_not_triggered_no_drift(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 1000)
        cur = rng.normal(0, 1, 1000)
        alert = MockAlert()
        monitor = DriftMonitor(detectors=[PSI()], alerts=[alert])
        monitor.check(ref, cur)
        assert len(alert.sent) == 0

    def test_add_detector(self):
        monitor = DriftMonitor()
        monitor.add_detector(PSI())
        assert len(monitor.detectors) == 1

    def test_add_alert(self):
        monitor = DriftMonitor()
        monitor.add_alert(MockAlert())
        assert len(monitor.alerts) == 1

    def test_summary_no_checks(self):
        monitor = DriftMonitor()
        summary = monitor.summary()
        assert summary["status"] == "no_checks_run"

    def test_summary_after_check(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 500)
        cur = rng.normal(0, 1, 500)
        monitor = DriftMonitor(detectors=[PSI()])
        monitor.check(ref, cur)
        summary = monitor.summary()
        assert "total_detectors" in summary
        assert "drifted" in summary

    def test_streaming_update(self):
        monitor = DriftMonitor(detectors=[ADWIN()])
        result = monitor.update_streaming({"ADWIN": 1.0})
        assert len(result) == 1

    def test_last_results(self):
        rng = np.random.default_rng(42)
        monitor = DriftMonitor(detectors=[PSI()])
        monitor.check(rng.normal(0, 1, 500), rng.normal(0, 1, 500))
        assert len(monitor.last_results) > 0
