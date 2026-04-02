"""Base alert interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from drift_sentinel.detectors.base import DriftResult


class BaseAlert(ABC):
    """Abstract base class for alert integrations."""

    @abstractmethod
    def send(self, results: list[DriftResult], message: str | None = None) -> bool:
        """Send an alert. Returns True if successful."""
        ...

    def format_message(self, results: list[DriftResult]) -> str:
        """Format drift results into a human-readable message."""
        drifted = [r for r in results if r.is_drift]
        if not drifted:
            return "No drift detected."

        lines = [f"⚠️ Drift detected in {len(drifted)} detector(s):\n"]
        for r in drifted:
            feat = f" ({r.feature_name})" if r.feature_name else ""
            lines.append(
                f"  • {r.detector_name}{feat}: "
                f"score={r.score:.4f} (threshold={r.threshold:.4f})"
            )
            if r.p_value is not None:
                lines.append(f"    p-value={r.p_value:.6f}")
        return "\n".join(lines)
