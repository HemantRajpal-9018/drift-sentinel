"""Concept drift detectors for streaming data."""

from __future__ import annotations

import math
from collections import deque

from drift_sentinel.detectors.base import BaseStreamingDetector, DriftResult


class ADWIN(BaseStreamingDetector):
    """ADaptive WINdowing method for change detection.

    Maintains a variable-length window and shrinks it when a statistically
    significant change in the mean is detected between two sub-windows.
    """

    def __init__(self, delta: float = 0.002):
        super().__init__(threshold=delta)
        self.delta = delta
        self._window: list[float] = []
        self._total = 0.0
        self._count = 0
        self._width = 0

    @property
    def name(self) -> str:
        return "ADWIN"

    def reset(self) -> None:
        self._window = []
        self._total = 0.0
        self._count = 0
        self._width = 0
        self._drift_detected = False
        self._warning_detected = False

    @property
    def window_size(self) -> int:
        return len(self._window)

    @property
    def mean(self) -> float:
        if not self._window:
            return 0.0
        return sum(self._window) / len(self._window)

    def update(self, value: float) -> DriftResult:
        self._window.append(value)
        self._drift_detected = False

        if len(self._window) < 5:
            return DriftResult(
                detector_name=self.name,
                is_drift=False,
                score=0.0,
                threshold=self.delta,
                details={"window_size": len(self._window)},
            )

        found_cut = True
        while found_cut and len(self._window) > 2:
            found_cut = False
            n = len(self._window)
            for i in range(1, n):
                sub1 = self._window[:i]
                sub2 = self._window[i:]
                n1, n2 = len(sub1), len(sub2)
                if n1 < 2 or n2 < 2:
                    continue
                mean1 = sum(sub1) / n1
                mean2 = sum(sub2) / n2
                m = 1.0 / (1.0 / n1 + 1.0 / n2)
                epsilon = math.sqrt(
                    (1.0 / (2.0 * m)) * math.log(4.0 * n / self.delta)
                )
                if abs(mean1 - mean2) >= epsilon:
                    self._window = self._window[i:]
                    self._drift_detected = True
                    found_cut = True
                    break

        return DriftResult(
            detector_name=self.name,
            is_drift=self._drift_detected,
            score=abs(
                sum(self._window[: len(self._window) // 2])
                / max(1, len(self._window) // 2)
                - sum(self._window[len(self._window) // 2 :])
                / max(1, len(self._window) - len(self._window) // 2)
            )
            if len(self._window) > 1
            else 0.0,
            threshold=self.delta,
            details={"window_size": len(self._window)},
        )


class PageHinkley(BaseStreamingDetector):
    """Page-Hinkley test for change detection in sequential data.

    Detects changes in the mean of a Gaussian signal.
    """

    def __init__(
        self,
        delta: float = 0.005,
        threshold: float = 50.0,
        alpha: float = 0.9999,
        min_instances: int = 30,
    ):
        super().__init__(threshold=threshold)
        self.delta = delta
        self.alpha = alpha
        self.min_instances = min_instances
        self._n = 0
        self._sum = 0.0
        self._cumulative = 0.0
        self._min_cumulative = float("inf")

    @property
    def name(self) -> str:
        return "Page-Hinkley"

    def reset(self) -> None:
        self._n = 0
        self._sum = 0.0
        self._cumulative = 0.0
        self._min_cumulative = float("inf")
        self._drift_detected = False
        self._warning_detected = False

    def update(self, value: float) -> DriftResult:
        self._n += 1
        self._sum += value
        mean = self._sum / self._n

        self._cumulative = self.alpha * self._cumulative + (value - mean - self.delta)
        self._min_cumulative = min(self._min_cumulative, self._cumulative)

        ph_value = self._cumulative - self._min_cumulative
        self._drift_detected = (
            ph_value > self.threshold and self._n >= self.min_instances
        )

        return DriftResult(
            detector_name=self.name,
            is_drift=self._drift_detected,
            score=float(ph_value),
            threshold=self.threshold,
            details={
                "n_samples": self._n,
                "mean": mean,
                "cumulative": self._cumulative,
            },
        )


class DDM(BaseStreamingDetector):
    """Drift Detection Method (DDM).

    Monitors error rate of a classifier and triggers when it increases
    beyond expected thresholds based on the binomial distribution.

    Uses a sliding window approach: tracks running error rate (p) and
    standard deviation (s = sqrt(p*(1-p)/n)). Compares current p+s against
    historical minimum p_min + s_min.
    """

    def __init__(
        self,
        warning_level: float = 2.0,
        drift_level: float = 3.0,
        min_instances: int = 30,
    ):
        super().__init__(threshold=drift_level)
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.min_instances = min_instances
        self._n = 0
        self._p = 0.0
        self._s = 0.0
        self._ps_min = float("inf")
        self._p_min = float("inf")
        self._s_min = float("inf")

    @property
    def name(self) -> str:
        return "DDM"

    def reset(self) -> None:
        self._n = 0
        self._p = 0.0
        self._s = 0.0
        self._ps_min = float("inf")
        self._p_min = float("inf")
        self._s_min = float("inf")
        self._drift_detected = False
        self._warning_detected = False

    def update(self, value: float) -> DriftResult:
        """Update with a prediction error (0 or 1)."""
        if value not in (0, 1, 0.0, 1.0):
            raise ValueError("DDM expects binary error values (0 or 1)")

        self._n += 1
        self._p += (value - self._p) / self._n
        self._s = math.sqrt(self._p * (1.0 - self._p) / self._n) if self._n > 1 else 0.0

        self._drift_detected = False
        self._warning_detected = False

        current_ps = self._p + self._s

        if self._n >= self.min_instances:
            if current_ps < self._ps_min:
                self._ps_min = current_ps
                self._p_min = self._p
                self._s_min = self._s

            if self._s_min > 0:
                drift_threshold = self._p_min + self.drift_level * self._s_min
                warning_threshold = self._p_min + self.warning_level * self._s_min
            else:
                # When s_min is 0 (e.g. perfect initial accuracy), use
                # a small epsilon so any error increase triggers detection
                eps = 1.0 / (2.0 * self._n)
                drift_threshold = self._p_min + self.drift_level * eps
                warning_threshold = self._p_min + self.warning_level * eps

            if current_ps >= drift_threshold:
                self._drift_detected = True
            elif current_ps >= warning_threshold:
                self._warning_detected = True

        return DriftResult(
            detector_name=self.name,
            is_drift=self._drift_detected,
            score=float(current_ps),
            threshold=float(self._p_min + self.drift_level * self._s_min)
            if self._s_min != float("inf")
            else 0.0,
            details={
                "n_samples": self._n,
                "error_rate": self._p,
                "std_dev": self._s,
                "min_error_rate": self._p_min if self._p_min != float("inf") else None,
                "warning": self._warning_detected,
            },
        )
