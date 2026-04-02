"""Shared test fixtures."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from drift_sentinel.detectors.base import DriftResult


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def normal_ref(rng):
    return rng.normal(0, 1, 1000)


@pytest.fixture
def normal_cur(rng):
    return rng.normal(0, 1, 1000)


@pytest.fixture
def shifted_cur(rng):
    return rng.normal(5, 1, 1000)


@pytest.fixture
def multivariate_ref(rng):
    return rng.normal(0, 1, (200, 3))


@pytest.fixture
def multivariate_cur(rng):
    return rng.normal(0, 1, (200, 3))


@pytest.fixture
def categorical_ref(rng):
    return rng.choice(["a", "b", "c"], size=500, p=[0.5, 0.3, 0.2])


@pytest.fixture
def categorical_cur(rng):
    return rng.choice(["a", "b", "c"], size=500, p=[0.5, 0.3, 0.2])


@pytest.fixture
def sample_drift_result():
    return DriftResult(
        detector_name="TestDetector",
        is_drift=True,
        score=0.5,
        threshold=0.2,
        p_value=0.01,
        feature_name="feature_0",
    )


@pytest.fixture
def sample_ok_result():
    return DriftResult(
        detector_name="TestDetector",
        is_drift=False,
        score=0.05,
        threshold=0.2,
        p_value=0.5,
        feature_name="feature_0",
    )


@pytest.fixture
def config_yaml(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("""
detectors:
  - type: psi
    threshold: 0.2
  - type: ks
    threshold: 0.05
feature_names:
  - feat_0
  - feat_1
""")
    return cfg


@pytest.fixture
def sample_npy_data(tmp_path, rng):
    ref = rng.normal(0, 1, (200, 2))
    cur = rng.normal(0, 1, (200, 2))
    ref_path = tmp_path / "ref.npy"
    cur_path = tmp_path / "cur.npy"
    np.save(ref_path, ref)
    np.save(cur_path, cur)
    return ref_path, cur_path


@pytest.fixture
def drifted_npy_data(tmp_path, rng):
    ref = rng.normal(0, 1, (200, 2))
    cur = rng.normal(5, 1, (200, 2))
    ref_path = tmp_path / "ref.npy"
    cur_path = tmp_path / "cur.npy"
    np.save(ref_path, ref)
    np.save(cur_path, cur)
    return ref_path, cur_path
