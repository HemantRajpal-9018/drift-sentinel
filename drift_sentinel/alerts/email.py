"""Email alerting via SMTP."""

from __future__ import annotations

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from drift_sentinel.alerts.base import BaseAlert
from drift_sentinel.detectors.base import DriftResult

logger = logging.getLogger(__name__)


class EmailAlert(BaseAlert):
    """Send drift alerts via SMTP email."""

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int = 587,
        username: str | None = None,
        password: str | None = None,
        from_addr: str = "drift-sentinel@localhost",
        to_addrs: list[str] | None = None,
        use_tls: bool = True,
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_addr = from_addr
        self.to_addrs = to_addrs or []
        self.use_tls = use_tls

    def send(self, results: list[DriftResult], message: str | None = None) -> bool:
        if not self.to_addrs:
            logger.error("No recipients configured for email alert")
            return False

        text = message or self.format_message(results)
        drifted = [r for r in results if r.is_drift]

        msg = MIMEMultipart()
        msg["From"] = self.from_addr
        msg["To"] = ", ".join(self.to_addrs)
        msg["Subject"] = f"[Drift Sentinel] Drift detected ({len(drifted)} detector(s))"
        msg.attach(MIMEText(text, "plain"))

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                if self.username and self.password:
                    server.login(self.username, self.password)
                server.sendmail(self.from_addr, self.to_addrs, msg.as_string())
            return True
        except Exception as e:
            logger.error("Email alert failed: %s", e)
            return False
