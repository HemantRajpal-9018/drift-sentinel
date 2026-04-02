"""CLI for drift-sentinel."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np
import yaml

from drift_sentinel import __version__


@click.group()
@click.version_option(__version__, prog_name="drift-sentinel")
def cli():
    """Drift Sentinel — Lightweight ML model monitoring."""


@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML")
@click.option("--reference", "-r", required=True, type=click.Path(exists=True), help="Reference data (CSV/NPY)")
@click.option("--current", "-C", required=True, type=click.Path(exists=True), help="Current data (CSV/NPY)")
@click.option("--output", "-o", type=click.Path(), default=None, help="Output JSON path")
def monitor(config, reference, current, output):
    """Run drift detection with the specified config."""
    from drift_sentinel.config import load_config, build_detectors, build_alerts
    from drift_sentinel.monitor import DriftMonitor

    cfg = load_config(config)
    detectors = build_detectors(cfg)
    alerts = build_alerts(cfg)

    ref_data = _load_data(reference)
    cur_data = _load_data(current)

    mon = DriftMonitor(detectors=detectors, alerts=alerts)
    results = mon.check(ref_data, cur_data)

    summary = mon.summary()
    click.echo(json.dumps(summary, indent=2, default=str))

    if output:
        Path(output).write_text(json.dumps(summary, indent=2, default=str))
        click.echo(f"Results written to {output}")

    sys.exit(1 if mon.has_drift else 0)


@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML")
@click.option("--reference", "-r", required=True, type=click.Path(exists=True), help="Reference data (CSV/NPY)")
@click.option("--current", "-C", required=True, type=click.Path(exists=True), help="Current data (CSV/NPY)")
@click.option("--output", "-o", type=click.Path(), default="drift_report.html", help="Output HTML path")
@click.option("--title", "-t", default="Drift Report", help="Report title")
def report(config, reference, current, output, title):
    """Generate an HTML drift report."""
    from drift_sentinel.config import load_config, build_detectors
    from drift_sentinel.monitor import DriftMonitor
    from drift_sentinel.reports.html_report import HTMLReportGenerator

    cfg = load_config(config)
    detectors = build_detectors(cfg)

    ref_data = _load_data(reference)
    cur_data = _load_data(current)

    mon = DriftMonitor(detectors=detectors)
    feature_names = cfg.get("feature_names")
    results = mon.check(ref_data, cur_data, feature_names=feature_names)

    # Build feature dicts for distribution charts
    ref_array = np.asarray(ref_data)
    cur_array = np.asarray(cur_data)
    if feature_names is None:
        n = ref_array.shape[1] if ref_array.ndim > 1 else 1
        feature_names = [f"feature_{i}" for i in range(n)]

    if ref_array.ndim == 1:
        ref_dict = {feature_names[0]: ref_array}
        cur_dict = {feature_names[0]: cur_array}
    else:
        ref_dict = {feature_names[i]: ref_array[:, i] for i in range(ref_array.shape[1])}
        cur_dict = {feature_names[i]: cur_array[:, i] for i in range(cur_array.shape[1])}

    gen = HTMLReportGenerator(title=title)
    gen.generate(results, reference=ref_dict, current=cur_dict, output_path=output)
    click.echo(f"Report generated: {output}")


@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML")
@click.option("--reference", "-r", required=True, type=click.Path(exists=True), help="Reference data (CSV/NPY)")
@click.option("--current", "-C", required=True, type=click.Path(exists=True), help="Current data (CSV/NPY)")
def check(config, reference, current):
    """Quick drift check — exits 0 if no drift, 1 if drift detected."""
    from drift_sentinel.config import load_config, build_detectors
    from drift_sentinel.monitor import DriftMonitor

    cfg = load_config(config)
    detectors = build_detectors(cfg)

    ref_data = _load_data(reference)
    cur_data = _load_data(current)

    mon = DriftMonitor(detectors=detectors)
    results = mon.check(ref_data, cur_data)

    drifted = [r for r in results if r.is_drift]
    if drifted:
        click.echo(f"DRIFT DETECTED — {len(drifted)}/{len(results)} detector(s) triggered", err=True)
        for r in drifted:
            feat = f" [{r.feature_name}]" if r.feature_name else ""
            click.echo(f"  {r.detector_name}{feat}: score={r.score:.4f}", err=True)
        sys.exit(1)
    else:
        click.echo(f"OK — {len(results)} detector(s) passed")
        sys.exit(0)


def _load_data(path: str) -> np.ndarray:
    """Load data from CSV or NPY file."""
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p)
    elif p.suffix == ".npz":
        data = np.load(p)
        return data[list(data.keys())[0]]
    else:
        import pandas as pd
        df = pd.read_csv(p)
        return df.values


if __name__ == "__main__":
    cli()
