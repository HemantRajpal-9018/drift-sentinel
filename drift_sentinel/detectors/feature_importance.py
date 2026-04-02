"""Feature importance shift detection using SHAP values."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from scipy import stats


@dataclass
class ImportanceShiftResult:
    """Result of a feature importance comparison."""

    is_drift: bool
    overall_score: float
    threshold: float
    feature_scores: dict[str, float]
    reference_importance: dict[str, float]
    current_importance: dict[str, float]
    drifted_features: list[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_drift": self.is_drift,
            "overall_score": self.overall_score,
            "threshold": self.threshold,
            "feature_scores": self.feature_scores,
            "drifted_features": self.drifted_features,
            "timestamp": self.timestamp.isoformat(),
        }


class FeatureImportanceShift:
    """Detect shifts in feature importance using SHAP values.

    Compares SHAP-based feature importance between reference and current
    datasets to identify which features have changed in relative contribution.
    """

    def __init__(
        self,
        threshold: float = 0.1,
        method: str = "correlation",
    ):
        self.threshold = threshold
        self.method = method

    def compute_importance(
        self,
        model: Any,
        data: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> dict[str, float]:
        """Compute SHAP-based feature importance."""
        try:
            import shap
        except ImportError:
            raise ImportError(
                "SHAP is required for feature importance shift detection. "
                "Install it with: pip install drift-sentinel[shap]"
            )

        data_array = np.asarray(data)
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(data_array.shape[1])]

        explainer = shap.Explainer(model, data_array)
        shap_values = explainer(data_array)

        importance = np.abs(shap_values.values).mean(axis=0)
        if importance.ndim > 1:
            importance = importance.mean(axis=-1)

        total = importance.sum()
        if total > 0:
            importance = importance / total

        return dict(zip(feature_names, importance.tolist()))

    def detect_from_importance(
        self,
        reference_importance: dict[str, float],
        current_importance: dict[str, float],
    ) -> ImportanceShiftResult:
        """Detect drift from pre-computed importance scores."""
        all_features = sorted(
            set(reference_importance.keys()) | set(current_importance.keys())
        )
        ref_vals = np.array([reference_importance.get(f, 0.0) for f in all_features])
        cur_vals = np.array([current_importance.get(f, 0.0) for f in all_features])

        feature_scores = {}
        drifted_features = []
        for i, feat in enumerate(all_features):
            diff = abs(ref_vals[i] - cur_vals[i])
            feature_scores[feat] = float(diff)
            if diff > self.threshold:
                drifted_features.append(feat)

        if self.method == "correlation" and len(ref_vals) > 1:
            correlation, _ = stats.spearmanr(ref_vals, cur_vals)
            overall_score = 1.0 - max(0.0, float(correlation))
        else:
            overall_score = float(np.mean(np.abs(ref_vals - cur_vals)))

        return ImportanceShiftResult(
            is_drift=len(drifted_features) > 0,
            overall_score=overall_score,
            threshold=self.threshold,
            feature_scores=feature_scores,
            reference_importance=reference_importance,
            current_importance=current_importance,
            drifted_features=drifted_features,
        )

    def detect(
        self,
        model: Any,
        reference_data: np.ndarray,
        current_data: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> ImportanceShiftResult:
        """Full detection: compute SHAP importance and compare."""
        ref_importance = self.compute_importance(model, reference_data, feature_names)
        cur_importance = self.compute_importance(model, current_data, feature_names)
        return self.detect_from_importance(ref_importance, cur_importance)
