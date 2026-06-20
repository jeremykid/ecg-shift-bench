"""Tests for ECGShiftBench classification metrics."""

import numpy as np
import pytest

from ecgshiftbench.metrics.classification import (
    compute_metrics,
    macro_auroc,
    macro_f1,
    per_label_auroc,
    per_label_f1,
    sensitivity_at_specificity,
)


@pytest.fixture
def binary_data():
    """Simple 2-class, 100-sample perfect prediction scenario."""
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=(100, 3)).astype(np.float32)
    # Perfect scores: positive class gets 0.9, negative gets 0.1
    y_score = np.where(y_true == 1, 0.9, 0.1).astype(np.float32)
    return y_true, y_score


@pytest.fixture
def random_data():
    """Random predictions (chance-level)."""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=(200, 5)).astype(np.float32)
    y_score = rng.uniform(0, 1, size=(200, 5)).astype(np.float32)
    return y_true, y_score


class TestPerLabelAUROC:
    def test_perfect_predictions(self, binary_data):
        y_true, y_score = binary_data
        scores = per_label_auroc(y_true, y_score)
        assert scores.shape == (3,)
        np.testing.assert_allclose(scores, 1.0, atol=1e-6)

    def test_shape(self, random_data):
        y_true, y_score = random_data
        scores = per_label_auroc(y_true, y_score)
        assert scores.shape == (5,)

    def test_nan_for_single_class(self):
        y_true = np.zeros((100, 2), dtype=np.float32)  # no positives in col 0
        y_true[:, 1] = np.tile([0, 1], 50)
        y_score = np.random.default_rng(0).uniform(0, 1, (100, 2)).astype(np.float32)
        scores = per_label_auroc(y_true, y_score)
        assert np.isnan(scores[0])
        assert not np.isnan(scores[1])

    def test_chance_level(self, random_data):
        y_true, y_score = random_data
        scores = per_label_auroc(y_true, y_score)
        valid = scores[~np.isnan(scores)]
        # Chance-level AUROC should be near 0.5
        assert np.all((valid > 0.2) & (valid < 0.8))


class TestMacroAUROC:
    def test_perfect(self, binary_data):
        y_true, y_score = binary_data
        score = macro_auroc(y_true, y_score)
        assert abs(score - 1.0) < 1e-6

    def test_returns_nan_when_all_nan(self):
        y_true = np.zeros((10, 2), dtype=np.float32)
        y_score = np.random.default_rng(0).uniform(0, 1, (10, 2)).astype(np.float32)
        assert np.isnan(macro_auroc(y_true, y_score))


class TestMacroF1:
    def test_perfect(self, binary_data):
        y_true, y_score = binary_data
        score = macro_f1(y_true, y_score, threshold=0.5)
        assert abs(score - 1.0) < 1e-6

    def test_all_wrong(self, binary_data):
        y_true, y_score = binary_data
        # Invert scores
        inverted = 1.0 - y_score
        score = macro_f1(y_true, inverted, threshold=0.5)
        assert score < 0.2


class TestPerLabelF1:
    def test_perfect(self, binary_data):
        y_true, y_score = binary_data
        y_pred = (y_score >= 0.5).astype(int)
        scores = per_label_f1(y_true, y_pred)
        np.testing.assert_allclose(scores, 1.0, atol=1e-6)

    def test_shape(self, random_data):
        y_true, y_score = random_data
        y_pred = (y_score >= 0.5).astype(int)
        scores = per_label_f1(y_true, y_pred)
        assert scores.shape == (5,)

    def test_zero_division(self):
        y_true = np.zeros((10, 2), dtype=np.int64)
        y_pred = np.zeros((10, 2), dtype=np.int64)
        scores = per_label_f1(y_true, y_pred)
        assert np.all(scores == 0.0)


class TestSensitivityAtSpecificity:
    def test_1d_perfect(self):
        y_true = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32)
        y_score = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9], dtype=np.float32)
        sens = sensitivity_at_specificity(y_true, y_score, target_specificity=0.9)
        assert sens >= 0.9

    def test_2d_returns_mean(self, binary_data):
        y_true, y_score = binary_data
        sens = sensitivity_at_specificity(y_true, y_score, target_specificity=0.9)
        assert 0.0 <= sens <= 1.0

    def test_2d_with_label_idx(self, binary_data):
        y_true, y_score = binary_data
        sens = sensitivity_at_specificity(y_true, y_score, target_specificity=0.9, label_idx=0)
        assert 0.0 <= sens <= 1.0


class TestComputeMetrics:
    def test_returns_expected_keys(self, binary_data):
        y_true, y_score = binary_data
        metrics = compute_metrics(y_true, y_score)
        assert "macro_auroc" in metrics
        assert "macro_f1" in metrics
        assert "sensitivity_at_90sp" in metrics

    def test_perfect_predictions(self, binary_data):
        y_true, y_score = binary_data
        metrics = compute_metrics(y_true, y_score)
        assert abs(metrics["macro_auroc"] - 1.0) < 1e-6
        assert abs(metrics["macro_f1"] - 1.0) < 1e-6

    def test_values_in_range(self, random_data):
        y_true, y_score = random_data
        metrics = compute_metrics(y_true, y_score)
        for v in metrics.values():
            assert 0.0 <= v <= 1.0 or np.isnan(v)
