"""HTML drift report generation with embedded charts."""

from __future__ import annotations

import base64
import io
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from jinja2 import Template

from drift_sentinel.detectors.base import DriftResult

REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Drift Sentinel Report — {{ title }}</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               background: #f5f7fa; color: #333; line-height: 1.6; }
        .container { max-width: 1100px; margin: 0 auto; padding: 2rem; }
        h1 { font-size: 1.8rem; margin-bottom: 0.5rem; color: #1a1a2e; }
        .subtitle { color: #666; margin-bottom: 2rem; }
        .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                        gap: 1rem; margin-bottom: 2rem; }
        .card { background: white; border-radius: 8px; padding: 1.5rem;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .card h3 { font-size: 0.85rem; text-transform: uppercase; color: #888; margin-bottom: 0.5rem; }
        .card .value { font-size: 2rem; font-weight: 700; }
        .status-ok { color: #10b981; }
        .status-drift { color: #ef4444; }
        .status-warning { color: #f59e0b; }
        table { width: 100%; border-collapse: collapse; background: white;
                border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        th, td { padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid #eee; }
        th { background: #f8f9fa; font-weight: 600; font-size: 0.85rem; text-transform: uppercase; color: #555; }
        .drift-yes { background: #fef2f2; color: #dc2626; font-weight: 600; }
        .drift-no { color: #10b981; }
        .chart-container { margin: 2rem 0; background: white; border-radius: 8px;
                          padding: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .chart-container img { max-width: 100%; height: auto; }
        .section-title { font-size: 1.3rem; margin: 2rem 0 1rem; color: #1a1a2e; }
        .footer { text-align: center; color: #999; font-size: 0.8rem; margin-top: 3rem; }
    </style>
</head>
<body>
<div class="container">
    <h1>{{ title }}</h1>
    <p class="subtitle">Generated on {{ generated_at }}</p>

    <div class="summary-grid">
        <div class="card">
            <h3>Status</h3>
            <div class="value {{ 'status-drift' if has_drift else 'status-ok' }}">
                {{ "DRIFT" if has_drift else "OK" }}
            </div>
        </div>
        <div class="card">
            <h3>Detectors Run</h3>
            <div class="value">{{ total_detectors }}</div>
        </div>
        <div class="card">
            <h3>Drifted</h3>
            <div class="value {{ 'status-drift' if n_drifted > 0 else 'status-ok' }}">{{ n_drifted }}</div>
        </div>
        <div class="card">
            <h3>Features Checked</h3>
            <div class="value">{{ n_features }}</div>
        </div>
    </div>

    <h2 class="section-title">Detection Results</h2>
    <table>
        <thead>
            <tr>
                <th>Detector</th>
                <th>Feature</th>
                <th>Drift?</th>
                <th>Score</th>
                <th>Threshold</th>
                <th>p-value</th>
            </tr>
        </thead>
        <tbody>
        {% for r in results %}
            <tr class="{{ 'drift-yes' if r.is_drift else '' }}">
                <td>{{ r.detector_name }}</td>
                <td>{{ r.feature_name or '—' }}</td>
                <td class="{{ 'drift-yes' if r.is_drift else 'drift-no' }}">{{ "YES" if r.is_drift else "No" }}</td>
                <td>{{ "%.4f"|format(r.score) }}</td>
                <td>{{ "%.4f"|format(r.threshold) }}</td>
                <td>{{ "%.6f"|format(r.p_value) if r.p_value is not none else '—' }}</td>
            </tr>
        {% endfor %}
        </tbody>
    </table>

    {% for chart in charts %}
    <div class="chart-container">
        <h3>{{ chart.title }}</h3>
        <img src="data:image/png;base64,{{ chart.image }}" alt="{{ chart.title }}">
    </div>
    {% endfor %}

    <p class="footer">Drift Sentinel v{{ version }} — Lightweight ML Model Monitoring</p>
</div>
</body>
</html>"""


class HTMLReportGenerator:
    """Generate HTML drift reports with embedded charts."""

    def __init__(self, title: str = "Drift Report"):
        self.title = title

    @staticmethod
    def _fig_to_base64(fig: plt.Figure) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def _make_score_chart(self, results: list[DriftResult]) -> dict[str, str]:
        fig, ax = plt.subplots(figsize=(10, max(3, len(results) * 0.4)))
        names = [
            f"{r.detector_name}\n({r.feature_name})" if r.feature_name else r.detector_name
            for r in results
        ]
        scores = [r.score for r in results]
        thresholds = [r.threshold for r in results]
        colors = ["#ef4444" if r.is_drift else "#10b981" for r in results]

        y_pos = range(len(names))
        ax.barh(y_pos, scores, color=colors, alpha=0.8, height=0.6)
        for i, (threshold, name) in enumerate(zip(thresholds, names)):
            ax.plot([threshold, threshold], [i - 0.35, i + 0.35], "k--", linewidth=1.5, alpha=0.6)

        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("Score")
        ax.set_title("Drift Scores vs Thresholds")
        ax.invert_yaxis()
        fig.tight_layout()

        return {"title": "Drift Scores Overview", "image": self._fig_to_base64(fig)}

    def _make_distribution_charts(
        self,
        reference: dict[str, np.ndarray],
        current: dict[str, np.ndarray],
    ) -> list[dict[str, str]]:
        charts = []
        for feat_name in sorted(reference.keys()):
            if feat_name not in current:
                continue
            ref = np.asarray(reference[feat_name])
            cur = np.asarray(current[feat_name])

            fig, ax = plt.subplots(figsize=(8, 3.5))
            bins = np.histogram_bin_edges(np.concatenate([ref, cur]), bins=30)
            ax.hist(ref, bins=bins, alpha=0.5, label="Reference", color="#6366f1", density=True)
            ax.hist(cur, bins=bins, alpha=0.5, label="Current", color="#f97316", density=True)
            ax.set_title(f"Distribution: {feat_name}", fontsize=11)
            ax.legend(fontsize=9)
            ax.set_ylabel("Density")
            fig.tight_layout()

            charts.append({
                "title": f"Distribution Comparison — {feat_name}",
                "image": self._fig_to_base64(fig),
            })
        return charts

    def generate(
        self,
        results: list[DriftResult],
        reference: dict[str, np.ndarray] | None = None,
        current: dict[str, np.ndarray] | None = None,
        output_path: str | Path | None = None,
    ) -> str:
        """Generate an HTML report.

        Returns the HTML string and optionally writes to a file.
        """
        from drift_sentinel import __version__

        charts = [self._make_score_chart(results)]
        if reference and current:
            charts.extend(self._make_distribution_charts(reference, current))

        features = {r.feature_name for r in results if r.feature_name}

        template = Template(REPORT_TEMPLATE)
        html = template.render(
            title=self.title,
            generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            has_drift=any(r.is_drift for r in results),
            total_detectors=len(results),
            n_drifted=sum(1 for r in results if r.is_drift),
            n_features=len(features) or 1,
            results=results,
            charts=charts,
            version=__version__,
        )

        if output_path:
            Path(output_path).write_text(html, encoding="utf-8")

        return html
