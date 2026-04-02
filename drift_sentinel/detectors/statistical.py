"""Statistical drift detection tests."""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon

from drift_sentinel.detectors.base import BaseDetector, DriftResult


class PSI(BaseDetector):
    """Population Stability Index for detecting distribution shifts."""

    def __init__(self, threshold: float = 0.2, n_bins: int = 10, eps: float = 1e-6):
        super().__init__(threshold=threshold)
        self.n_bins = n_bins
        self.eps = eps

    @property
    def name(self) -> str:
        return "PSI"

    def detect(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        **kwargs,
    ) -> DriftResult:
        reference = np.asarray(reference, dtype=float).ravel()
        current = np.asarray(current, dtype=float).ravel()

        breakpoints = np.histogram_bin_edges(reference, bins=self.n_bins)
        ref_counts = np.histogram(reference, bins=breakpoints)[0] + self.eps
        cur_counts = np.histogram(current, bins=breakpoints)[0] + self.eps

        ref_pct = ref_counts / ref_counts.sum()
        cur_pct = cur_counts / cur_counts.sum()

        psi_value = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

        return DriftResult(
            detector_name=self.name,
            is_drift=psi_value > self.threshold,
            score=psi_value,
            threshold=self.threshold,
            details={
                "n_bins": self.n_bins,
                "ref_size": len(reference),
                "cur_size": len(current),
                "bin_psi": ((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)).tolist(),
            },
        )


class KSTest(BaseDetector):
    """Kolmogorov-Smirnov test for comparing two distributions."""

    def __init__(self, threshold: float = 0.05):
        super().__init__(threshold=threshold)

    @property
    def name(self) -> str:
        return "KS Test"

    def detect(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        **kwargs,
    ) -> DriftResult:
        reference = np.asarray(reference, dtype=float).ravel()
        current = np.asarray(current, dtype=float).ravel()

        statistic, p_value = stats.ks_2samp(reference, current)

        return DriftResult(
            detector_name=self.name,
            is_drift=p_value < self.threshold,
            score=float(statistic),
            threshold=self.threshold,
            p_value=float(p_value),
            details={
                "ref_size": len(reference),
                "cur_size": len(current),
            },
        )


class JensenShannonDivergence(BaseDetector):
    """Jensen-Shannon divergence for measuring distribution similarity."""

    def __init__(self, threshold: float = 0.1, n_bins: int = 10, eps: float = 1e-6):
        super().__init__(threshold=threshold)
        self.n_bins = n_bins
        self.eps = eps

    @property
    def name(self) -> str:
        return "Jensen-Shannon Divergence"

    def detect(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        **kwargs,
    ) -> DriftResult:
        reference = np.asarray(reference, dtype=float).ravel()
        current = np.asarray(current, dtype=float).ravel()

        breakpoints = np.histogram_bin_edges(reference, bins=self.n_bins)
        ref_counts = np.histogram(reference, bins=breakpoints)[0] + self.eps
        cur_counts = np.histogram(current, bins=breakpoints)[0] + self.eps

        ref_pct = ref_counts / ref_counts.sum()
        cur_pct = cur_counts / cur_counts.sum()

        js_value = float(jensenshannon(ref_pct, cur_pct) ** 2)

        return DriftResult(
            detector_name=self.name,
            is_drift=js_value > self.threshold,
            score=js_value,
            threshold=self.threshold,
            details={
                "n_bins": self.n_bins,
                "ref_size": len(reference),
                "cur_size": len(current),
            },
        )


class ChiSquaredTest(BaseDetector):
    """Chi-squared test for categorical feature drift detection."""

    def __init__(self, threshold: float = 0.05):
        super().__init__(threshold=threshold)

    @property
    def name(self) -> str:
        return "Chi-Squared Test"

    def detect(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        **kwargs,
    ) -> DriftResult:
        reference = np.asarray(reference).ravel()
        current = np.asarray(current).ravel()

        all_categories = np.union1d(np.unique(reference), np.unique(current))

        ref_counts = np.array(
            [np.sum(reference == cat) for cat in all_categories], dtype=float
        )
        cur_counts = np.array(
            [np.sum(current == cat) for cat in all_categories], dtype=float
        )

        # Scale expected counts to match observed total
        expected = ref_counts * (cur_counts.sum() / ref_counts.sum())
        # Avoid zero expected
        mask = expected > 0
        if mask.sum() == 0:
            return DriftResult(
                detector_name=self.name,
                is_drift=False,
                score=0.0,
                threshold=self.threshold,
                p_value=1.0,
            )

        statistic, p_value = stats.chisquare(cur_counts[mask], f_exp=expected[mask])

        return DriftResult(
            detector_name=self.name,
            is_drift=p_value < self.threshold,
            score=float(statistic),
            threshold=self.threshold,
            p_value=float(p_value),
            details={
                "categories": all_categories.tolist(),
                "ref_counts": ref_counts.tolist(),
                "cur_counts": cur_counts.tolist(),
            },
        )


class MMD(BaseDetector):
    """Maximum Mean Discrepancy for multivariate distribution comparison.

    Uses an RBF (Gaussian) kernel.
    """

    def __init__(
        self,
        threshold: float = 0.05,
        kernel_bandwidth: float | None = None,
        n_permutations: int = 100,
    ):
        super().__init__(threshold=threshold)
        self.kernel_bandwidth = kernel_bandwidth
        self.n_permutations = n_permutations

    @property
    def name(self) -> str:
        return "MMD"

    @staticmethod
    def _rbf_kernel(X: np.ndarray, Y: np.ndarray, bandwidth: float) -> np.ndarray:
        XX = np.sum(X ** 2, axis=1, keepdims=True)
        YY = np.sum(Y ** 2, axis=1, keepdims=True)
        dists = XX + YY.T - 2.0 * X @ Y.T
        return np.exp(-dists / (2.0 * bandwidth ** 2))

    def _compute_mmd2(
        self, X: np.ndarray, Y: np.ndarray, bandwidth: float
    ) -> float:
        n = len(X)
        m = len(Y)

        Kxx = self._rbf_kernel(X, X, bandwidth)
        Kyy = self._rbf_kernel(Y, Y, bandwidth)
        Kxy = self._rbf_kernel(X, Y, bandwidth)

        # Unbiased estimator
        np.fill_diagonal(Kxx, 0)
        np.fill_diagonal(Kyy, 0)

        mmd2 = (
            Kxx.sum() / (n * (n - 1))
            + Kyy.sum() / (m * (m - 1))
            - 2 * Kxy.sum() / (n * m)
        )
        return float(mmd2)

    def detect(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        **kwargs,
    ) -> DriftResult:
        reference = np.atleast_2d(np.asarray(reference, dtype=float))
        current = np.atleast_2d(np.asarray(current, dtype=float))

        if reference.ndim == 1:
            reference = reference.reshape(-1, 1)
        if current.ndim == 1:
            current = current.reshape(-1, 1)

        if self.kernel_bandwidth is None:
            combined = np.vstack([reference, current])
            from scipy.spatial.distance import pdist

            dists = pdist(combined, "sqeuclidean")
            bandwidth = float(np.sqrt(np.median(dists))) if len(dists) > 0 else 1.0
        else:
            bandwidth = self.kernel_bandwidth

        mmd2_observed = self._compute_mmd2(reference, current, bandwidth)

        # Permutation test for p-value
        combined = np.vstack([reference, current])
        n = len(reference)
        rng = np.random.default_rng(kwargs.get("seed", 42))
        count = 0
        for _ in range(self.n_permutations):
            perm = rng.permutation(len(combined))
            perm_X = combined[perm[:n]]
            perm_Y = combined[perm[n:]]
            mmd2_perm = self._compute_mmd2(perm_X, perm_Y, bandwidth)
            if mmd2_perm >= mmd2_observed:
                count += 1
        p_value = (count + 1) / (self.n_permutations + 1)

        return DriftResult(
            detector_name=self.name,
            is_drift=p_value < self.threshold,
            score=mmd2_observed,
            threshold=self.threshold,
            p_value=p_value,
            details={
                "kernel_bandwidth": bandwidth,
                "n_permutations": self.n_permutations,
                "ref_size": len(reference),
                "cur_size": len(current),
            },
        )
