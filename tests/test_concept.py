"""Tests for concept drift detectors."""

import pytest

from drift_sentinel.detectors.concept import ADWIN, DDM, PageHinkley
from drift_sentinel.detectors.base import DriftResult


class TestADWIN:
    def test_no_drift_stationary(self):
        adwin = ADWIN(delta=0.002)
        results = []
        for v in [0.5] * 50:
            results.append(adwin.update(v))
        assert not any(r.is_drift for r in results)

    def test_drift_mean_shift(self):
        adwin = ADWIN(delta=0.01)
        for v in [0.0] * 100:
            adwin.update(v)
        drifted = False
        for v in [1.0] * 100:
            result = adwin.update(v)
            if result.is_drift:
                drifted = True
                break
        assert drifted

    def test_reset(self):
        adwin = ADWIN()
        for v in [1.0] * 20:
            adwin.update(v)
        adwin.reset()
        assert adwin.window_size == 0

    def test_window_grows(self):
        adwin = ADWIN()
        for v in [0.5] * 10:
            adwin.update(v)
        assert adwin.window_size > 0

    def test_result_type(self):
        adwin = ADWIN()
        result = adwin.update(1.0)
        assert isinstance(result, DriftResult)

    def test_name(self):
        assert ADWIN().name == "ADWIN"

    def test_mean_property(self):
        adwin = ADWIN()
        assert adwin.mean == 0.0
        adwin.update(4.0)
        adwin.update(6.0)
        assert abs(adwin.mean - 5.0) < 1e-9


class TestPageHinkley:
    def test_no_drift_stationary(self):
        ph = PageHinkley(threshold=50, min_instances=10)
        results = []
        for v in [1.0] * 50:
            results.append(ph.update(v))
        assert not any(r.is_drift for r in results)

    def test_drift_mean_shift(self):
        ph = PageHinkley(threshold=10, min_instances=10, delta=0.005)
        for v in [0.0] * 50:
            ph.update(v)
        drifted = False
        for v in [5.0] * 100:
            result = ph.update(v)
            if result.is_drift:
                drifted = True
                break
        assert drifted

    def test_reset(self):
        ph = PageHinkley()
        for v in range(20):
            ph.update(float(v))
        ph.reset()
        assert ph._n == 0

    def test_min_instances(self):
        ph = PageHinkley(threshold=0.001, min_instances=100)
        for v in range(50):
            result = ph.update(float(v * 100))
        assert not result.is_drift  # not enough instances yet

    def test_result_details(self):
        ph = PageHinkley()
        result = ph.update(1.0)
        assert "n_samples" in result.details
        assert "mean" in result.details

    def test_name(self):
        assert PageHinkley().name == "Page-Hinkley"


class TestDDM:
    def test_no_drift_low_error(self):
        ddm = DDM(min_instances=10)
        results = []
        for _ in range(100):
            results.append(ddm.update(0))
        assert not any(r.is_drift for r in results)

    def test_drift_increasing_error(self):
        ddm = DDM(min_instances=10, drift_level=2.5)
        # Good period: low error
        for _ in range(50):
            ddm.update(0)
        # Bad period: high error — needs many samples because DDM uses overall avg
        drifted = False
        for _ in range(500):
            result = ddm.update(1)
            if result.is_drift:
                drifted = True
                break
        assert drifted

    def test_warning_before_drift(self):
        ddm = DDM(min_instances=10, warning_level=1.0, drift_level=100.0)
        for _ in range(50):
            ddm.update(0)
        warned = False
        for _ in range(500):
            result = ddm.update(1)
            if result.details.get("warning"):
                warned = True
                break
        assert warned

    def test_rejects_non_binary(self):
        ddm = DDM()
        with pytest.raises(ValueError, match="binary"):
            ddm.update(0.5)

    def test_accepts_float_binary(self):
        ddm = DDM()
        ddm.update(0.0)
        ddm.update(1.0)

    def test_reset(self):
        ddm = DDM()
        for _ in range(20):
            ddm.update(0)
        ddm.reset()
        assert ddm._n == 0

    def test_name(self):
        assert DDM().name == "DDM"

    def test_result_details(self):
        ddm = DDM()
        result = ddm.update(0)
        assert "error_rate" in result.details
        assert "n_samples" in result.details
