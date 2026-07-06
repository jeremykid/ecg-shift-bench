"""Tests for multi-label and domain-gap metrics."""

import numpy as np
import pytest

from ecg_shift_bench.evaluation.domain_gap import domain_gap
from ecg_shift_bench.evaluation.metrics import (
    binary_class_report,
    multilabel_metrics,
    optimal_multilabel_thresholds,
    thresholded_multilabel_metrics,
)
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


def test_thresholded_multilabel_metrics_use_validation_cutoffs() -> None:
    y_true = np.array(
        [
            [1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1],
            [1, 1, 0, 0, 1, 1],
            [0, 0, 1, 1, 0, 0],
        ]
    )
    y_score = y_true * 0.9 + (1 - y_true) * 0.1
    selected_thresholds = optimal_multilabel_thresholds(y_true, y_score)
    assert set(selected_thresholds) == set(CANONICAL_LABELS)
    thresholds = {label: 0.5 for label in CANONICAL_LABELS}
    metrics = thresholded_multilabel_metrics(y_true, y_score, thresholds=thresholds)
    for name in (
        "macro_accuracy",
        "micro_accuracy",
        "macro_f1",
        "micro_f1",
        "macro_sensitivity",
        "micro_sensitivity",
        "macro_specificity",
        "micro_specificity",
    ):
        assert 0.0 <= metrics[name] <= 1.0
    assert set(metrics["per_label_f1"]) == set(CANONICAL_LABELS)
    assert set(metrics["per_label_accuracy"]) == set(CANONICAL_LABELS)
    assert set(metrics["per_label_sensitivity"]) == set(CANONICAL_LABELS)
    assert set(metrics["per_label_specificity"]) == set(CANONICAL_LABELS)
    assert set(metrics["thresholds"]) == set(CANONICAL_LABELS)


def test_thresholded_multilabel_metrics_use_strict_greater_than() -> None:
    y_true = np.array([[1], [0], [0]])
    y_score = np.array([[0.5], [0.5], [0.1]])
    metrics = thresholded_multilabel_metrics(
        y_true,
        y_score,
        label_names=["L"],
        thresholds=[0.5],
    )
    assert metrics["per_label_accuracy"]["L"] == pytest.approx(2 / 3)
    assert metrics["per_label_recall"]["L"] == pytest.approx(0.0)


def test_binary_class_report_keeps_auprc_and_aprec_distinct() -> None:
    y_true = np.array([1, 1, 0, 0])
    y_score = np.array([0.9, 0.4, 0.3, 0.2])
    report = binary_class_report(y_true, y_score, threshold=0.5)
    assert set(report) == {
        "accuracy",
        "auroc",
        "auprc",
        "f1_score",
        "prec",
        "rec",
        "sensitivity",
        "spec",
        "aprec",
        "br_score",
        "tn",
        "fp",
        "fn",
        "tp",
    }
    assert report["auprc"] != report["aprec"]
    assert 0.0 <= report["auroc"] <= 1.0
    assert 0.0 <= report["aprec"] <= 1.0


def test_domain_gap_definition() -> None:
    assert domain_gap(0.88, 0.71) == pytest.approx(0.17)
