"""Configuration loading from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from drift_sentinel.detectors.statistical import (
    PSI,
    ChiSquaredTest,
    JensenShannonDivergence,
    KSTest,
    MMD,
)
from drift_sentinel.detectors.concept import ADWIN, DDM, PageHinkley
from drift_sentinel.detectors.base import BaseDetector, BaseStreamingDetector
from drift_sentinel.alerts.base import BaseAlert
from drift_sentinel.alerts.slack import SlackAlert
from drift_sentinel.alerts.email import EmailAlert
from drift_sentinel.alerts.pagerduty import PagerDutyAlert

DETECTOR_MAP: dict[str, type] = {
    "psi": PSI,
    "ks": KSTest,
    "ks_test": KSTest,
    "jensen_shannon": JensenShannonDivergence,
    "js": JensenShannonDivergence,
    "chi_squared": ChiSquaredTest,
    "mmd": MMD,
    "adwin": ADWIN,
    "page_hinkley": PageHinkley,
    "ddm": DDM,
}

ALERT_MAP: dict[str, type] = {
    "slack": SlackAlert,
    "email": EmailAlert,
    "pagerduty": PagerDutyAlert,
}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def build_detectors(config: dict[str, Any]) -> list[BaseDetector | BaseStreamingDetector]:
    """Build detector instances from config dict."""
    detectors = []
    for det_config in config.get("detectors", []):
        det_type = det_config.get("type", "").lower()
        params = {k: v for k, v in det_config.items() if k != "type"}
        cls = DETECTOR_MAP.get(det_type)
        if cls is None:
            raise ValueError(f"Unknown detector type: {det_type}")
        detectors.append(cls(**params))
    return detectors


def build_alerts(config: dict[str, Any]) -> list[BaseAlert]:
    """Build alert instances from config dict."""
    alerts = []
    for alert_config in config.get("alerts", []):
        alert_type = alert_config.get("type", "").lower()
        params = {k: v for k, v in alert_config.items() if k != "type"}
        cls = ALERT_MAP.get(alert_type)
        if cls is None:
            raise ValueError(f"Unknown alert type: {alert_type}")
        alerts.append(cls(**params))
    return alerts
