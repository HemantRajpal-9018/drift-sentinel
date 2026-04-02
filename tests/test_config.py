"""Tests for config loading."""

import pytest
from pathlib import Path

from drift_sentinel.config import load_config, build_detectors, build_alerts
from drift_sentinel.detectors.statistical import PSI, KSTest
from drift_sentinel.detectors.concept import ADWIN


class TestConfig:
    def test_load_config(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("""
detectors:
  - type: psi
    threshold: 0.25
  - type: ks
    threshold: 0.01
alerts:
  - type: slack
    webhook_url: https://hooks.slack.com/test
""")
        config = load_config(str(cfg_file))
        assert "detectors" in config
        assert len(config["detectors"]) == 2

    def test_build_detectors(self):
        config = {
            "detectors": [
                {"type": "psi", "threshold": 0.3},
                {"type": "ks"},
                {"type": "adwin", "delta": 0.01},
            ]
        }
        detectors = build_detectors(config)
        assert len(detectors) == 3
        assert isinstance(detectors[0], PSI)
        assert detectors[0].threshold == 0.3
        assert isinstance(detectors[1], KSTest)
        assert isinstance(detectors[2], ADWIN)

    def test_build_unknown_detector(self):
        config = {"detectors": [{"type": "unknown_detector"}]}
        with pytest.raises(ValueError, match="Unknown detector"):
            build_detectors(config)

    def test_build_alerts(self):
        config = {
            "alerts": [
                {"type": "slack", "webhook_url": "https://hooks.slack.com/test"},
            ]
        }
        alerts = build_alerts(config)
        assert len(alerts) == 1

    def test_build_unknown_alert(self):
        config = {"alerts": [{"type": "unknown_alert"}]}
        with pytest.raises(ValueError, match="Unknown alert"):
            build_alerts(config)

    def test_empty_config(self, tmp_path):
        cfg_file = tmp_path / "empty.yaml"
        cfg_file.write_text("")
        config = load_config(str(cfg_file))
        assert config == {}
        detectors = build_detectors(config)
        assert detectors == []
