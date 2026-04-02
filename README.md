# Drift Sentinel

[![PyPI version](https://img.shields.io/pypi/v/drift-sentinel.svg)](https://pypi.org/project/drift-sentinel/)
[![Python](https://img.shields.io/pypi/pyversions/drift-sentinel.svg)](https://pypi.org/project/drift-sentinel/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-50%2B-green.svg)]()

Lightweight ML model monitoring SDK for detecting data drift, concept drift, and feature importance shifts. Pip-installable, production-ready, with built-in alerting and dashboards.

## Installation

```bash
pip install drift-sentinel
```

With all extras (SHAP, FastAPI dashboard, Plotly):

```bash
pip install drift-sentinel[all]
```

## Quick Start

```python
from drift_sentinel import DriftMonitor, PSI, KSTest

# Create a monitor with detectors
monitor = DriftMonitor(detectors=[PSI(), KSTest()])

# Check for drift
results = monitor.check(reference_data, current_data)

if monitor.has_drift:
    print("Drift detected!")
    print(monitor.summary())
```

## 3-Line Usage

```python
from drift_sentinel import DriftMonitor, PSI
monitor = DriftMonitor(detectors=[PSI()])
results = monitor.check(training_data, production_data)
```

## Features

### Statistical Drift Tests

| Detector | Best For | Default Threshold |
|----------|----------|-------------------|
| `PSI` | Overall distribution shift | 0.2 |
| `KSTest` | Continuous feature drift | 0.05 (p-value) |
| `JensenShannonDivergence` | Distribution similarity | 0.1 |
| `ChiSquaredTest` | Categorical feature drift | 0.05 (p-value) |
| `MMD` | Multivariate distribution comparison | 0.05 (p-value) |

```python
from drift_sentinel import PSI, KSTest, JensenShannonDivergence, ChiSquaredTest, MMD

# Each detector has a .detect() method
psi = PSI(threshold=0.2)
result = psi.detect(reference_array, current_array)
print(result.is_drift, result.score, result.p_value)
```

### Concept Drift Detectors (Streaming)

For real-time monitoring of model performance:

```python
from drift_sentinel.detectors.concept import ADWIN, PageHinkley, DDM

# ADWIN for adaptive windowing
adwin = ADWIN(delta=0.002)
for value in streaming_data:
    result = adwin.update(value)
    if result.is_drift:
        print("Distribution change detected!")

# DDM for classification error monitoring
ddm = DDM(warning_level=2.0, drift_level=3.0)
for error in prediction_errors:
    result = ddm.update(error)  # 0 or 1
```

### Feature Importance Shift

Track which features are changing in predictive importance:

```python
from drift_sentinel.detectors.feature_importance import FeatureImportanceShift

fis = FeatureImportanceShift(threshold=0.1)
result = fis.detect(model, reference_data, current_data,
                    feature_names=["age", "income", "score"])
print(result.drifted_features)
```

### Multi-Feature Monitoring

```python
import numpy as np
from drift_sentinel import DriftMonitor, PSI, KSTest

monitor = DriftMonitor(detectors=[PSI(), KSTest()])

# Dict-based input
results = monitor.check(
    reference={"age": ref_ages, "income": ref_incomes},
    current={"age": cur_ages, "income": cur_incomes},
)

# Or 2D array with feature names
results = monitor.check(ref_2d, cur_2d, feature_names=["age", "income"])
```

### Alerting

```python
from drift_sentinel import DriftMonitor, PSI
from drift_sentinel.alerts import SlackAlert, EmailAlert, PagerDutyAlert

monitor = DriftMonitor(
    detectors=[PSI()],
    alerts=[
        SlackAlert(webhook_url="https://hooks.slack.com/..."),
        EmailAlert(smtp_host="smtp.gmail.com", to_addrs=["team@co.com"]),
        PagerDutyAlert(routing_key="your-key"),
    ],
)
# Alerts fire automatically when drift is detected
monitor.check(reference, current)
```

### HTML Reports

```python
from drift_sentinel.reports import HTMLReportGenerator

gen = HTMLReportGenerator(title="Weekly Drift Report")
html = gen.generate(
    results,
    reference={"age": ref_ages, "income": ref_incomes},
    current={"age": cur_ages, "income": cur_incomes},
    output_path="drift_report.html",
)
```

### CLI

```bash
# Quick drift check (exit code 1 if drift)
drift-sentinel check -c config.yaml -r reference.npy -C current.npy

# Full monitoring with JSON output
drift-sentinel monitor -c config.yaml -r ref.csv -C cur.csv -o results.json

# Generate HTML report
drift-sentinel report -c config.yaml -r ref.npy -C cur.npy -o report.html
```

### FastAPI Dashboard

```bash
# Start the dashboard
pip install drift-sentinel[api]
uvicorn drift_sentinel.dashboard.app:app --port 8000

# Or with Docker
docker-compose up
```

API endpoints:
- `GET /` — Dashboard UI
- `GET /api/status` — Current drift status
- `POST /api/check` — Run drift check
- `GET /api/history` — Check history
- `GET /api/report` — Generate HTML report

## Configuration (YAML)

```yaml
detectors:
  - type: psi
    threshold: 0.2
  - type: ks
    threshold: 0.05
  - type: jensen_shannon
    threshold: 0.1

feature_names:
  - age
  - income
  - credit_score

alerts:
  - type: slack
    webhook_url: https://hooks.slack.com/services/...
  - type: pagerduty
    routing_key: your-key
    severity: warning
```

## API Reference

### Detectors

All detectors implement `.detect(reference, current) -> DriftResult`:

- **`DriftResult`**: `detector_name`, `is_drift`, `score`, `threshold`, `p_value`, `feature_name`, `details`

### Streaming Detectors

Implement `.update(value) -> DriftResult` and `.reset()`:

- **`ADWIN(delta=0.002)`** — Adaptive windowing
- **`PageHinkley(threshold=50, delta=0.005)`** — Sequential change detection
- **`DDM(warning_level=2.0, drift_level=3.0)`** — Error rate monitoring

### DriftMonitor

- `monitor.check(reference, current)` — Run all detectors
- `monitor.has_drift` — Boolean drift status
- `monitor.summary()` — Dict summary
- `monitor.history` — All past results

## Comparison

| Feature | Drift Sentinel | Evidently | WhyLabs |
|---------|---------------|-----------|---------|
| Pip install | Yes | Yes | Yes (client) |
| Statistical tests | PSI, KS, JS, Chi2, MMD | PSI, KS, JS, Chi2, others | KS, Chi2 |
| Concept drift | ADWIN, PH, DDM | No | No |
| SHAP importance | Yes | No | Yes |
| Streaming support | Yes | No | Yes |
| Built-in alerting | Slack, Email, PagerDuty | No (via Grafana) | Built-in |
| Self-hosted dashboard | FastAPI | Streamlit | Cloud only |
| Lines to get started | 3 | 5+ | 10+ |
| Lightweight | ~5 deps | 50+ deps | Cloud SDK |

## Development

```bash
git clone https://github.com/drift-sentinel/drift-sentinel
cd drift-sentinel
make dev    # Install with dev deps
make test   # Run tests
make lint   # Lint
```

## License

MIT
