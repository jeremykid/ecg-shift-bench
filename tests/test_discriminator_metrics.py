"""Tests for dataset-discriminator metrics."""

from __future__ import annotations

import numpy as np

from ecg_shift_bench.evaluation.discriminator import dataset_classification_metrics


def test_dataset_classification_metrics_return_expected_keys() -> None:
    truth = np.array([0, 1, 2, 1], dtype=np.int64)
    scores = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.05, 0.1, 0.85],
            [0.2, 0.7, 0.1],
        ],
        dtype=np.float32,
    )
    metrics = dataset_classification_metrics(truth, scores, ["PTBXL", "CODE15", "SPH"])
    assert set(metrics) >= {
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "auroc",
        "per_class_auroc",
        "per_class_support",
        "confusion_matrix",
    }
    assert metrics["confusion_matrix"] == [[1, 0, 0], [0, 2, 0], [0, 0, 1]]
    assert metrics["per_class_support"] == {"PTBXL": 1, "CODE15": 2, "SPH": 1}


def test_pairwise_metrics_handle_two_classes() -> None:
    truth = np.array([0, 1, 0, 1], dtype=np.int64)
    scores = np.array([[0.8, 0.2], [0.3, 0.7], [0.75, 0.25], [0.4, 0.6]], dtype=np.float32)
    metrics = dataset_classification_metrics(truth, scores, ["PTBXL", "CODE15"])
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]
    assert metrics["per_class_auroc"]["PTBXL"] == metrics["per_class_auroc"]["CODE15"]
