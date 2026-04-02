"""PagerDuty alerting integration."""

from __future__ import annotations

import json
import logging

import requests

from drift_sentinel.alerts.base import BaseAlert
from drift_sentinel.detectors.base import DriftResult

logger = logging.getLogger(__name__)


class PagerDutyAlert(BaseAlert):
    """Send drift alerts to PagerDuty via Events API v2."""

    EVENTS_URL = "https://events.pagerduty.com/v2/enqueue"

    def __init__(
        self,
        routing_key: str,
        severity: str = "warning",
        source: str = "drift-sentinel",
    ):
        self.routing_key = routing_key
        self.severity = severity
        self.source = source

    def send(self, results: list[DriftResult], message: str | None = None) -> bool:
        drifted = [r for r in results if r.is_drift]
        summary = message or self.format_message(results)

        payload = {
            "routing_key": self.routing_key,
            "event_action": "trigger",
            "payload": {
                "summary": summary[:1024],
                "severity": self.severity,
                "source": self.source,
                "custom_details": {
                    "total_detectors": len(results),
                    "drifted_detectors": len(drifted),
                    "details": [r.to_dict() for r in drifted],
                },
            },
        }

        try:
            resp = requests.post(
                self.EVENTS_URL,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            resp.raise_for_status()
            return True
        except Exception as e:
            logger.error("PagerDuty alert failed: %s", e)
            return False
