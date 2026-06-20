"""Tests for multi-label and domain-gap metrics."""

import numpy as np
import pytest

from ecg_shift_bench.evaluation.domain_gap import domain_gap
from ecg_shift_bench.evaluation.metrics import multilabel_metrics
from ecg_shift_bench.labels.canonical import CANONICAL_LABELS


def test_metrics_are_valid_on_toy_data() -> None:
    y_true = np.array(
        [
            [1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1],
            [1, 1, 0, 0, 1, 1],
            [0, 0, 1, 1, 0, 0],
        ]
    )
    y_score = y_true * 0.8 + (1 - y_true) * 0.2
    metrics = multilabel_metrics(y_true, y_score)
    for name in ("macro_auroc", "micro_auroc", "macro_auprc", "micro_auprc"):
        assert 0.0 <= metrics[name] <= 1.0
    assert set(metrics["per_label_auroc"]) == set(CANONICAL_LABELS)
    assert set(metrics["per_label_auprc"]) == set(CANONICAL_LABELS)
    assert metrics["macro_auroc"] == pytest.approx(1.0)


def test_undefined_per_label_metric_is_nan() -> None:
    y_true = np.zeros((3, 6), dtype=int)
    y_true[:, 0] = [0, 1, 0]
    y_score = np.full((3, 6), 0.1)
    metrics = multilabel_metrics(y_true, y_score)
    assert np.isnan(metrics["per_label_auroc"]["RBBB"])
    assert np.isfinite(metrics["macro_auroc"])


def test_domain_gap_definition() -> None:
    assert domain_gap(0.88, 0.71) == pytest.approx(0.17)
