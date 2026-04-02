"""Tests for feature importance shift detection."""

import pytest

from drift_sentinel.detectors.feature_importance import (
    FeatureImportanceShift,
    ImportanceShiftResult,
)


class TestFeatureImportanceShift:
    def test_no_drift_same_importance(self):
        fis = FeatureImportanceShift(threshold=0.1)
        ref = {"feat_a": 0.4, "feat_b": 0.3, "feat_c": 0.3}
        cur = {"feat_a": 0.4, "feat_b": 0.3, "feat_c": 0.3}
        result = fis.detect_from_importance(ref, cur)
        assert not result.is_drift
        assert len(result.drifted_features) == 0

    def test_drift_shifted_importance(self):
        fis = FeatureImportanceShift(threshold=0.1)
        ref = {"feat_a": 0.5, "feat_b": 0.3, "feat_c": 0.2}
        cur = {"feat_a": 0.1, "feat_b": 0.6, "feat_c": 0.3}
        result = fis.detect_from_importance(ref, cur)
        assert result.is_drift
        assert "feat_a" in result.drifted_features

    def test_result_type(self):
        fis = FeatureImportanceShift()
        ref = {"feat_a": 0.5, "feat_b": 0.5}
        cur = {"feat_a": 0.5, "feat_b": 0.5}
        result = fis.detect_from_importance(ref, cur)
        assert isinstance(result, ImportanceShiftResult)

    def test_to_dict(self):
        fis = FeatureImportanceShift()
        ref = {"a": 0.5, "b": 0.5}
        cur = {"a": 0.3, "b": 0.7}
        result = fis.detect_from_importance(ref, cur)
        d = result.to_dict()
        assert "is_drift" in d
        assert "feature_scores" in d

    def test_correlation_method(self):
        fis = FeatureImportanceShift(threshold=0.1, method="correlation")
        ref = {"a": 0.5, "b": 0.3, "c": 0.2}
        cur = {"a": 0.5, "b": 0.3, "c": 0.2}
        result = fis.detect_from_importance(ref, cur)
        assert result.overall_score < 0.01

    def test_missing_feature_in_current(self):
        fis = FeatureImportanceShift(threshold=0.1)
        ref = {"a": 0.5, "b": 0.3, "c": 0.2}
        cur = {"a": 0.5, "b": 0.5}
        result = fis.detect_from_importance(ref, cur)
        assert "c" in result.feature_scores

    def test_shap_import_error(self):
        fis = FeatureImportanceShift()
        # compute_importance should raise ImportError if shap is not installed
        # We test the error path rather than requiring shap in test deps
        try:
            import shap
            pytest.skip("SHAP is installed")
        except ImportError:
            import numpy as np
            with pytest.raises(ImportError, match="SHAP"):
                fis.compute_importance(None, np.array([[1, 2], [3, 4]]))
