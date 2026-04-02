"""High-level DriftMonitor that combines multiple detectors."""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import numpy as np

from drift_sentinel.alerts.base import BaseAlert
from drift_sentinel.detectors.base import BaseDetector, BaseStreamingDetector, DriftResult

logger = logging.getLogger(__name__)


class DriftMonitor:
    """Orchestrates multiple drift detectors, tracks history, and triggers alerts.

    Example::

        from drift_sentinel import DriftMonitor, PSI, KSTest

        monitor = DriftMonitor(detectors=[PSI(), KSTest()])
        results = monitor.check(reference_data, current_data, feature_names=["age", "income"])
        if monitor.has_drift:
            print("Drift detected!")
    """

    def __init__(
        self,
        detectors: list[BaseDetector | BaseStreamingDetector] | None = None,
        alerts: list[BaseAlert] | None = None,
        alert_on_drift: bool = True,
    ):
        self.detectors = detectors or []
        self.alerts = alerts or []
        self.alert_on_drift = alert_on_drift
        self._history: list[dict[str, Any]] = []
        self._last_results: list[DriftResult] = []

    @property
    def has_drift(self) -> bool:
        return any(r.is_drift for r in self._last_results)

    @property
    def last_results(self) -> list[DriftResult]:
        return list(self._last_results)

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._history)

    def add_detector(self, detector: BaseDetector | BaseStreamingDetector) -> None:
        self.detectors.append(detector)

    def add_alert(self, alert: BaseAlert) -> None:
        self.alerts.append(alert)

    def check(
        self,
        reference: np.ndarray | dict[str, np.ndarray],
        current: np.ndarray | dict[str, np.ndarray],
        feature_names: list[str] | None = None,
    ) -> list[DriftResult]:
        """Run all batch detectors on reference vs current data.

        Args:
            reference: Reference data — 1D/2D array or dict mapping feature names to arrays.
            current: Current data — same format as reference.
            feature_names: Feature names when passing 2D arrays.

        Returns:
            List of DriftResult from each detector × feature combination.
        """
        results: list[DriftResult] = []

        # Normalize to dict of {feature_name: (ref_array, cur_array)}
        pairs = self._normalize_inputs(reference, current, feature_names)

        for detector in self.detectors:
            if isinstance(detector, BaseStreamingDetector):
                continue
            for feat_name, (ref_arr, cur_arr) in pairs.items():
                try:
                    result = detector.detect(ref_arr, cur_arr)
                    result.feature_name = feat_name
                    results.append(result)
                except Exception as e:
                    logger.warning(
                        "Detector %s failed on feature %s: %s",
                        detector.name, feat_name, e,
                    )

        self._last_results = results
        self._record_history(results)

        if self.alert_on_drift and any(r.is_drift for r in results):
            self._send_alerts(results)

        return results

    def update_streaming(self, values: dict[str, float]) -> list[DriftResult]:
        """Feed new values to streaming detectors.

        Args:
            values: Mapping of detector name (or index) to current value.
        """
        results: list[DriftResult] = []
        for detector in self.detectors:
            if isinstance(detector, BaseStreamingDetector):
                key = detector.name
                if key in values:
                    result = detector.update(values[key])
                    results.append(result)

        self._last_results = results
        self._record_history(results)

        if self.alert_on_drift and any(r.is_drift for r in results):
            self._send_alerts(results)

        return results

    def _normalize_inputs(
        self,
        reference: np.ndarray | dict[str, np.ndarray],
        current: np.ndarray | dict[str, np.ndarray],
        feature_names: list[str] | None,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        if isinstance(reference, dict) and isinstance(current, dict):
            keys = sorted(set(reference.keys()) & set(current.keys()))
            return {k: (np.asarray(reference[k]), np.asarray(current[k])) for k in keys}

        ref = np.asarray(reference)
        cur = np.asarray(current)

        if ref.ndim == 1:
            name = feature_names[0] if feature_names else "feature_0"
            return {name: (ref, cur)}

        n_features = ref.shape[1]
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        return {
            feature_names[i]: (ref[:, i], cur[:, i]) for i in range(n_features)
        }

    def _record_history(self, results: list[DriftResult]) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "has_drift": any(r.is_drift for r in results),
            "n_detectors": len(results),
            "n_drifted": sum(1 for r in results if r.is_drift),
            "results": [r.to_dict() for r in results],
        }
        self._history.append(entry)

    def _send_alerts(self, results: list[DriftResult]) -> None:
        for alert in self.alerts:
            try:
                alert.send(results)
            except Exception as e:
                logger.error("Alert %s failed: %s", type(alert).__name__, e)

    def summary(self) -> dict[str, Any]:
        """Return a summary of the latest check."""
        if not self._last_results:
            return {"status": "no_checks_run"}

        drifted = [r for r in self._last_results if r.is_drift]
        return {
            "status": "drift" if drifted else "ok",
            "total_detectors": len(self._last_results),
            "drifted": len(drifted),
            "detectors": {
                r.detector_name + (f"_{r.feature_name}" if r.feature_name else ""): {
                    "is_drift": r.is_drift,
                    "score": r.score,
                    "threshold": r.threshold,
                    "p_value": r.p_value,
                }
                for r in self._last_results
            },
            "total_checks": len(self._history),
        }
