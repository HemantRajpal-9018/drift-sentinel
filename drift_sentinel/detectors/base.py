"""Base detector interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class DriftResult:
    """Result of a drift detection test."""

    detector_name: str
    is_drift: bool
    score: float
    threshold: float
    p_value: float | None = None
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    feature_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector_name": self.detector_name,
            "is_drift": self.is_drift,
            "score": self.score,
            "threshold": self.threshold,
            "p_value": self.p_value,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "feature_name": self.feature_name,
        }


class BaseDetector(ABC):
    """Abstract base class for all drift detectors."""

    def __init__(self, threshold: float | None = None):
        self.threshold = threshold

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def detect(self, reference: Any, current: Any, **kwargs: Any) -> DriftResult:
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(threshold={self.threshold})"


class BaseStreamingDetector(ABC):
    """Abstract base class for streaming (concept) drift detectors."""

    def __init__(self, threshold: float | None = None):
        self.threshold = threshold
        self._drift_detected = False
        self._warning_detected = False

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def update(self, value: float) -> DriftResult:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...

    @property
    def drift_detected(self) -> bool:
        return self._drift_detected

    @property
    def warning_detected(self) -> bool:
        return self._warning_detected

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(threshold={self.threshold})"
