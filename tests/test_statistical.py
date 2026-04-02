"""Tests for statistical drift detectors."""

import numpy as np
import pytest

from drift_sentinel.detectors.statistical import (
    PSI,
    ChiSquaredTest,
    JensenShannonDivergence,
    KSTest,
    MMD,
)
from drift_sentinel.detectors.base import DriftResult


class TestPSI:
    def test_no_drift_same_distribution(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 1000)
        cur = rng.normal(0, 1, 1000)
        result = PSI().detect(ref, cur)
        assert isinstance(result, DriftResult)
        assert not result.is_drift
        assert result.score < 0.2

    def test_drift_different_distribution(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 1000)
        cur = rng.normal(3, 1, 1000)
        result = PSI().detect(ref, cur)
        assert result.is_drift
        assert result.score > 0.2

    def test_custom_threshold(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 1000)
        cur = rng.normal(0.5, 1, 1000)
        result_strict = PSI(threshold=0.01).detect(ref, cur)
        result_lenient = PSI(threshold=1.0).detect(ref, cur)
        assert result_strict.is_drift or not result_lenient.is_drift

    def test_custom_bins(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 500)
        cur = rng.normal(0, 1, 500)
        result = PSI(n_bins=20).detect(ref, cur)
        assert result.details["n_bins"] == 20

    def test_result_has_correct_name(self):
        result = PSI().detect(np.array([1, 2, 3]), np.array([1, 2, 3]))
        assert result.detector_name == "PSI"

    def test_score_is_non_negative(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 500)
        cur = rng.normal(0, 1, 500)
        result = PSI().detect(ref, cur)
        assert result.score >= 0

    def test_details_contain_bin_psi(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 500)
        cur = rng.normal(0, 1, 500)
        result = PSI(n_bins=5).detect(ref, cur)
        assert "bin_psi" in result.details
        assert len(result.details["bin_psi"]) == 5


class TestKSTest:
    def test_no_drift_same_distribution(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 1000)
        cur = rng.normal(0, 1, 1000)
        result = KSTest().detect(ref, cur)
        assert not result.is_drift
        assert result.p_value > 0.05

    def test_drift_different_distribution(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 1000)
        cur = rng.normal(2, 1, 1000)
        result = KSTest().detect(ref, cur)
        assert result.is_drift
        assert result.p_value < 0.05

    def test_p_value_present(self):
        result = KSTest().detect(np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4]))
        assert result.p_value is not None

    def test_custom_threshold(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 500)
        cur = rng.normal(0, 1, 500)
        result = KSTest(threshold=0.99).detect(ref, cur)
        assert result.is_drift  # almost everything drifts at alpha=0.99

    def test_score_between_0_and_1(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 500)
        cur = rng.normal(0, 1, 500)
        result = KSTest().detect(ref, cur)
        assert 0 <= result.score <= 1


class TestJensenShannonDivergence:
    def test_no_drift(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 1000)
        cur = rng.normal(0, 1, 1000)
        result = JensenShannonDivergence().detect(ref, cur)
        assert not result.is_drift

    def test_drift(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 1000)
        cur = rng.normal(5, 1, 1000)
        result = JensenShannonDivergence().detect(ref, cur)
        assert result.is_drift

    def test_score_bounded(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 500)
        cur = rng.normal(0, 1, 500)
        result = JensenShannonDivergence().detect(ref, cur)
        assert 0 <= result.score <= 1.0

    def test_name(self):
        assert JensenShannonDivergence().name == "Jensen-Shannon Divergence"


class TestChiSquaredTest:
    def test_no_drift_same_categories(self):
        rng = np.random.default_rng(42)
        ref = rng.choice(["a", "b", "c"], size=500, p=[0.5, 0.3, 0.2])
        cur = rng.choice(["a", "b", "c"], size=500, p=[0.5, 0.3, 0.2])
        result = ChiSquaredTest().detect(ref, cur)
        assert not result.is_drift

    def test_drift_different_categories(self):
        rng = np.random.default_rng(42)
        ref = rng.choice(["a", "b", "c"], size=500, p=[0.5, 0.3, 0.2])
        cur = rng.choice(["a", "b", "c"], size=500, p=[0.1, 0.1, 0.8])
        result = ChiSquaredTest().detect(ref, cur)
        assert result.is_drift

    def test_p_value_present(self):
        ref = np.array(["x", "y", "x", "y", "x"])
        cur = np.array(["x", "y", "x", "y", "x"])
        result = ChiSquaredTest().detect(ref, cur)
        assert result.p_value is not None

    def test_integer_categories(self):
        rng = np.random.default_rng(42)
        ref = rng.choice([1, 2, 3], size=300)
        cur = rng.choice([1, 2, 3], size=300)
        result = ChiSquaredTest().detect(ref, cur)
        assert isinstance(result, DriftResult)

    def test_details_contain_categories(self):
        ref = np.array(["a", "b", "c", "a", "b"])
        cur = np.array(["a", "b", "c", "a", "c"])
        result = ChiSquaredTest().detect(ref, cur)
        assert "categories" in result.details


class TestMMD:
    def test_no_drift_same_distribution(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, (100, 2))
        cur = rng.normal(0, 1, (100, 2))
        result = MMD(n_permutations=50).detect(ref, cur, seed=42)
        assert not result.is_drift

    def test_drift_different_distribution(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, (100, 2))
        cur = rng.normal(3, 1, (100, 2))
        result = MMD(n_permutations=50).detect(ref, cur, seed=42)
        assert result.is_drift

    def test_1d_input(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 100)
        cur = rng.normal(5, 1, 100)
        result = MMD(n_permutations=50).detect(ref, cur, seed=42)
        assert result.is_drift

    def test_custom_bandwidth(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, (50, 2))
        cur = rng.normal(0, 1, (50, 2))
        result = MMD(kernel_bandwidth=1.0, n_permutations=30).detect(ref, cur, seed=42)
        assert result.details["kernel_bandwidth"] == 1.0

    def test_p_value_present(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, (50, 2))
        cur = rng.normal(0, 1, (50, 2))
        result = MMD(n_permutations=30).detect(ref, cur, seed=42)
        assert result.p_value is not None
        assert 0 <= result.p_value <= 1
