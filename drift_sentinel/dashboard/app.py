"""FastAPI dashboard for drift monitoring."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    from fastapi import FastAPI, HTTPException, UploadFile, File
    from fastapi.responses import HTMLResponse, JSONResponse
except ImportError:
    raise ImportError(
        "FastAPI is required for the dashboard. "
        "Install with: pip install drift-sentinel[api]"
    )

from drift_sentinel import __version__
from drift_sentinel.detectors.base import DriftResult
from drift_sentinel.monitor import DriftMonitor
from drift_sentinel.detectors.statistical import PSI, KSTest, JensenShannonDivergence
from drift_sentinel.reports.html_report import HTMLReportGenerator

app = FastAPI(
    title="Drift Sentinel Dashboard",
    version=__version__,
    description="ML model drift monitoring dashboard",
)

# In-memory state
_monitor = DriftMonitor(detectors=[PSI(), KSTest(), JensenShannonDivergence()])
_check_history: list[dict[str, Any]] = []

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Drift Sentinel Dashboard</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               background: #0f172a; color: #e2e8f0; }
        .header { background: #1e293b; padding: 1.5rem 2rem; border-bottom: 1px solid #334155; }
        .header h1 { font-size: 1.5rem; color: #f1f5f9; }
        .header .version { color: #64748b; font-size: 0.85rem; }
        .container { max-width: 1200px; margin: 0 auto; padding: 2rem; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem; margin-bottom: 2rem; }
        .card { background: #1e293b; border-radius: 12px; padding: 1.5rem; border: 1px solid #334155; }
        .card h3 { color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; }
        .card .value { font-size: 2.5rem; font-weight: 700; margin-top: 0.5rem; }
        .ok { color: #22c55e; }
        .drift { color: #ef4444; }
        .table { width: 100%; border-collapse: collapse; background: #1e293b;
                 border-radius: 12px; overflow: hidden; border: 1px solid #334155; }
        .table th, .table td { padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid #334155; }
        .table th { color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; }
        .badge { padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.8rem; font-weight: 600; }
        .badge-ok { background: #052e16; color: #22c55e; }
        .badge-drift { background: #450a0a; color: #ef4444; }
        #status { margin-bottom: 1rem; }
        button { background: #3b82f6; color: white; border: none; padding: 0.75rem 1.5rem;
                 border-radius: 8px; cursor: pointer; font-size: 0.9rem; margin-right: 0.5rem; }
        button:hover { background: #2563eb; }
        .section-title { font-size: 1.2rem; color: #f1f5f9; margin: 2rem 0 1rem; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Drift Sentinel</h1>
        <span class="version">v""" + __version__ + """</span>
    </div>
    <div class="container">
        <div class="grid">
            <div class="card">
                <h3>Status</h3>
                <div class="value" id="status">—</div>
            </div>
            <div class="card">
                <h3>Total Checks</h3>
                <div class="value" id="total-checks">0</div>
            </div>
            <div class="card">
                <h3>Drift Events</h3>
                <div class="value" id="drift-events">0</div>
            </div>
            <div class="card">
                <h3>Last Check</h3>
                <div class="value" id="last-check" style="font-size:1.2rem;">Never</div>
            </div>
        </div>

        <h2 class="section-title">Recent History</h2>
        <table class="table" id="history-table">
            <thead><tr>
                <th>Time</th><th>Status</th><th>Detectors</th><th>Drifted</th>
            </tr></thead>
            <tbody id="history-body"></tbody>
        </table>
    </div>
    <script>
        async function refresh() {
            const resp = await fetch('/api/status');
            const data = await resp.json();
            document.getElementById('status').textContent = data.status.toUpperCase();
            document.getElementById('status').className = 'value ' + (data.has_drift ? 'drift' : 'ok');
            document.getElementById('total-checks').textContent = data.total_checks;
            document.getElementById('drift-events').textContent = data.drift_events;
            document.getElementById('last-check').textContent = data.last_check || 'Never';

            const body = document.getElementById('history-body');
            body.innerHTML = '';
            (data.history || []).slice(-20).reverse().forEach(h => {
                const row = document.createElement('tr');
                row.innerHTML = `<td>${h.timestamp}</td>
                    <td><span class="badge ${h.has_drift ? 'badge-drift' : 'badge-ok'}">${h.has_drift ? 'DRIFT' : 'OK'}</span></td>
                    <td>${h.n_detectors}</td><td>${h.n_drifted}</td>`;
                body.appendChild(row);
            });
        }
        refresh();
        setInterval(refresh, 5000);
    </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML


@app.get("/api/status")
async def api_status():
    history = _monitor.history
    drift_events = sum(1 for h in history if h.get("has_drift"))
    return {
        "status": "drift" if _monitor.has_drift else "ok",
        "has_drift": _monitor.has_drift,
        "total_checks": len(history),
        "drift_events": drift_events,
        "last_check": history[-1]["timestamp"] if history else None,
        "history": history[-50:],
    }


@app.get("/api/history")
async def api_history(limit: int = 50):
    return {"history": _monitor.history[-limit:]}


@app.get("/api/summary")
async def api_summary():
    return _monitor.summary()


@app.post("/api/check")
async def api_check(payload: dict[str, Any]):
    """Run a drift check. Expects {reference: [...], current: [...]}."""
    ref = np.array(payload.get("reference", []))
    cur = np.array(payload.get("current", []))
    if ref.size == 0 or cur.size == 0:
        raise HTTPException(400, "reference and current arrays are required")

    feature_names = payload.get("feature_names")
    results = _monitor.check(ref, cur, feature_names=feature_names)

    return JSONResponse(content=_to_json_safe({
        "has_drift": _monitor.has_drift,
        "results": [r.to_dict() for r in results],
        "summary": _monitor.summary(),
    }))


def _to_json_safe(obj: Any) -> Any:
    """Recursively convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


@app.get("/api/report", response_class=HTMLResponse)
async def api_report():
    if not _monitor.last_results:
        raise HTTPException(404, "No checks have been run yet")
    gen = HTMLReportGenerator(title="Dashboard Report")
    return gen.generate(_monitor.last_results)


def create_app(detectors=None, alerts=None) -> FastAPI:
    """Create a configured app instance."""
    global _monitor
    if detectors or alerts:
        _monitor = DriftMonitor(detectors=detectors or [], alerts=alerts or [])
    return app
