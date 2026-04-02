"""Tests for FastAPI dashboard."""

import pytest
import numpy as np

try:
    from fastapi.testclient import TestClient
    from drift_sentinel.dashboard.app import app, _monitor
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")


@pytest.fixture
def client():
    return TestClient(app)


class TestDashboard:
    def test_homepage(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Drift Sentinel" in resp.text

    def test_api_status(self, client):
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "total_checks" in data

    def test_api_history(self, client):
        resp = client.get("/api/history")
        assert resp.status_code == 200
        assert "history" in resp.json()

    def test_api_summary(self, client):
        resp = client.get("/api/summary")
        assert resp.status_code == 200

    def test_api_check(self, client):
        rng = np.random.default_rng(42)
        payload = {
            "reference": rng.normal(0, 1, (100, 2)).tolist(),
            "current": rng.normal(0, 1, (100, 2)).tolist(),
        }
        resp = client.post("/api/check", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "has_drift" in data
        assert "results" in data

    def test_api_check_empty(self, client):
        resp = client.post("/api/check", json={"reference": [], "current": []})
        assert resp.status_code == 400

    def test_api_report_no_data(self, client):
        # Report may 404 if no checks run, or return if previous tests populated state
        resp = client.get("/api/report")
        assert resp.status_code in (200, 404)
