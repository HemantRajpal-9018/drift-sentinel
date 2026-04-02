"""Tests for base classes."""

import pytest

from drift_sentinel.detectors.base import BaseDetector, BaseStreamingDetector, DriftResult


class TestDriftResult:
    def test_to_dict(self):
        result = DriftResult(
            detector_name="Test",
            is_drift=True,
            score=0.5,
            threshold=0.2,
            p_value=0.01,
            feature_name="age",
        )
        d = result.to_dict()
        assert d["detector_name"] == "Test"
        assert d["is_drift"] is True
        assert d["score"] == 0.5
        assert d["feature_name"] == "age"
        assert "timestamp" in d

    def test_default_values(self):
        result = DriftResult(
            detector_name="Test",
            is_drift=False,
            score=0.0,
            threshold=0.2,
        )
        assert result.p_value is None
        assert result.details == {}
        assert result.feature_name is None


class TestBaseDetector:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseDetector()

    def test_repr(self):
        class ConcreteDetector(BaseDetector):
            @property
            def name(self):
                return "Concrete"

            def detect(self, reference, current, **kwargs):
                return DriftResult("Concrete", False, 0.0, 0.2)

        d = ConcreteDetector(threshold=0.5)
        assert "ConcreteDetector" in repr(d)
        assert "0.5" in repr(d)


class TestBaseStreamingDetector:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseStreamingDetector()

    def test_initial_state(self):
        class ConcreteStreaming(BaseStreamingDetector):
            @property
            def name(self):
                return "Concrete"

            def update(self, value):
                return DriftResult("Concrete", False, 0.0, 0.2)

            def reset(self):
                pass

        d = ConcreteStreaming()
        assert not d.drift_detected
        assert not d.warning_detected
