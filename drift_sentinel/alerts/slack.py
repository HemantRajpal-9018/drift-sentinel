"""Slack webhook alerting."""

from __future__ import annotations

import json
import logging

import requests

from drift_sentinel.alerts.base import BaseAlert
from drift_sentinel.detectors.base import DriftResult

logger = logging.getLogger(__name__)


class SlackAlert(BaseAlert):
    """Send drift alerts to Slack via incoming webhook."""

    def __init__(self, webhook_url: str, channel: str | None = None, username: str = "Drift Sentinel"):
        self.webhook_url = webhook_url
        self.channel = channel
        self.username = username

    def send(self, results: list[DriftResult], message: str | None = None) -> bool:
        text = message or self.format_message(results)
        payload: dict = {
            "text": text,
            "username": self.username,
        }
        if self.channel:
            payload["channel"] = self.channel

        try:
            resp = requests.post(
                self.webhook_url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            resp.raise_for_status()
            return True
        except Exception as e:
            logger.error("Slack alert failed: %s", e)
            return False
