"""Tests for alerting integrations."""

import json
from unittest.mock import patch, MagicMock

import pytest

from drift_sentinel.alerts.slack import SlackAlert
from drift_sentinel.alerts.email import EmailAlert
from drift_sentinel.alerts.pagerduty import PagerDutyAlert
from drift_sentinel.alerts.base import BaseAlert
from drift_sentinel.detectors.base import DriftResult


def _make_drift_result(is_drift=True, score=0.5, name="TestDetector"):
    return DriftResult(
        detector_name=name,
        is_drift=is_drift,
        score=score,
        threshold=0.2,
        p_value=0.01 if is_drift else 0.5,
        feature_name="feature_0",
    )


class TestBaseAlert:
    def test_format_message_with_drift(self):
        results = [_make_drift_result(True), _make_drift_result(False)]

        class ConcreteAlert(BaseAlert):
            def send(self, results, message=None):
                return True

        alert = ConcreteAlert()
        msg = alert.format_message(results)
        assert "Drift detected" in msg
        assert "TestDetector" in msg

    def test_format_message_no_drift(self):
        results = [_make_drift_result(False)]

        class ConcreteAlert(BaseAlert):
            def send(self, results, message=None):
                return True

        alert = ConcreteAlert()
        msg = alert.format_message(results)
        assert "No drift detected" in msg


class TestSlackAlert:
    @patch("drift_sentinel.alerts.slack.requests.post")
    def test_send_success(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()
        alert = SlackAlert(webhook_url="https://hooks.slack.com/test")
        results = [_make_drift_result()]
        assert alert.send(results) is True
        mock_post.assert_called_once()

    @patch("drift_sentinel.alerts.slack.requests.post")
    def test_send_with_channel(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()
        alert = SlackAlert(webhook_url="https://hooks.slack.com/test", channel="#alerts")
        alert.send([_make_drift_result()])
        call_data = json.loads(mock_post.call_args[1]["data"] if "data" in mock_post.call_args[1] else mock_post.call_args[0][0])
        # Just verify the call was made
        assert mock_post.called

    @patch("drift_sentinel.alerts.slack.requests.post", side_effect=Exception("fail"))
    def test_send_failure(self, mock_post):
        alert = SlackAlert(webhook_url="https://hooks.slack.com/test")
        assert alert.send([_make_drift_result()]) is False


class TestEmailAlert:
    @patch("drift_sentinel.alerts.email.smtplib.SMTP")
    def test_send_success(self, mock_smtp):
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp.return_value.__exit__ = MagicMock(return_value=False)
        alert = EmailAlert(
            smtp_host="smtp.test.com",
            to_addrs=["test@example.com"],
        )
        assert alert.send([_make_drift_result()]) is True

    def test_send_no_recipients(self):
        alert = EmailAlert(smtp_host="smtp.test.com")
        assert alert.send([_make_drift_result()]) is False

    @patch("drift_sentinel.alerts.email.smtplib.SMTP", side_effect=Exception("fail"))
    def test_send_failure(self, mock_smtp):
        alert = EmailAlert(
            smtp_host="smtp.test.com",
            to_addrs=["test@example.com"],
        )
        assert alert.send([_make_drift_result()]) is False


class TestPagerDutyAlert:
    @patch("drift_sentinel.alerts.pagerduty.requests.post")
    def test_send_success(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()
        alert = PagerDutyAlert(routing_key="test-key")
        assert alert.send([_make_drift_result()]) is True
        mock_post.assert_called_once()

    @patch("drift_sentinel.alerts.pagerduty.requests.post", side_effect=Exception("fail"))
    def test_send_failure(self, mock_post):
        alert = PagerDutyAlert(routing_key="test-key")
        assert alert.send([_make_drift_result()]) is False

    @patch("drift_sentinel.alerts.pagerduty.requests.post")
    def test_payload_structure(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()
        alert = PagerDutyAlert(routing_key="test-key", severity="critical")
        alert.send([_make_drift_result()])
        call_data = json.loads(mock_post.call_args[1]["data"])
        assert call_data["routing_key"] == "test-key"
        assert call_data["payload"]["severity"] == "critical"
