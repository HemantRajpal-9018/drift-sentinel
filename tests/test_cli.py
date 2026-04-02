"""Tests for CLI."""

import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from drift_sentinel.cli import cli


@pytest.fixture
def sample_data(tmp_path):
    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1, (200, 2))
    cur = rng.normal(0, 1, (200, 2))
    ref_path = tmp_path / "ref.npy"
    cur_path = tmp_path / "cur.npy"
    np.save(ref_path, ref)
    np.save(cur_path, cur)
    return ref_path, cur_path


@pytest.fixture
def drifted_data(tmp_path):
    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1, (200, 2))
    cur = rng.normal(5, 1, (200, 2))
    ref_path = tmp_path / "ref.npy"
    cur_path = tmp_path / "cur.npy"
    np.save(ref_path, ref)
    np.save(cur_path, cur)
    return ref_path, cur_path


@pytest.fixture
def config_file(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("""
detectors:
  - type: psi
    threshold: 0.2
  - type: ks
    threshold: 0.05
""")
    return cfg


class TestCLI:
    def test_version(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "drift-sentinel" in result.output

    def test_check_no_drift(self, sample_data, config_file):
        runner = CliRunner()
        ref_path, cur_path = sample_data
        result = runner.invoke(cli, [
            "check", "-c", str(config_file),
            "-r", str(ref_path), "-C", str(cur_path),
        ])
        assert result.exit_code == 0
        assert "OK" in result.output

    def test_check_with_drift(self, drifted_data, config_file):
        runner = CliRunner()
        ref_path, cur_path = drifted_data
        result = runner.invoke(cli, [
            "check", "-c", str(config_file),
            "-r", str(ref_path), "-C", str(cur_path),
        ])
        assert result.exit_code == 1
        assert "DRIFT" in result.output

    def test_monitor_output_json(self, sample_data, config_file, tmp_path):
        runner = CliRunner()
        ref_path, cur_path = sample_data
        out = tmp_path / "results.json"
        result = runner.invoke(cli, [
            "monitor", "-c", str(config_file),
            "-r", str(ref_path), "-C", str(cur_path),
            "-o", str(out),
        ])
        assert out.exists()
        data = json.loads(out.read_text())
        assert "status" in data

    def test_report_generates_html(self, sample_data, config_file, tmp_path):
        runner = CliRunner()
        ref_path, cur_path = sample_data
        out = tmp_path / "report.html"
        result = runner.invoke(cli, [
            "report", "-c", str(config_file),
            "-r", str(ref_path), "-C", str(cur_path),
            "-o", str(out),
        ])
        assert result.exit_code == 0
        assert out.exists()
        assert "<html" in out.read_text()
