"""Metrics for the dataset discriminator study."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def dataset_classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_score: np.ndarray,
    label_names: list[str],
) -> dict[str, object]:
    """Compute classification metrics for multiclass or pairwise study runs."""
    truth = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    if scores.ndim != 2:
        raise ValueError("y_score must be a 2D array")
    if truth.ndim != 1:
        raise ValueError("y_true must be a 1D array")
    if truth.shape[0] != scores.shape[0]:
        raise ValueError("y_true and y_score must have the same number of rows")
    if len(label_names) != scores.shape[1]:
        raise ValueError("label_names length must match the number of score columns")
    if not np.isfinite(scores).all():
        raise ValueError("y_score must contain only finite values")

    predictions = scores.argmax(axis=1)
    metrics: dict[str, object] = {
        "accuracy": float(accuracy_score(truth, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, average="macro")),
        "per_class_support": {
            label: int((truth == index).sum()) for index, label in enumerate(label_names)
        },
        "confusion_matrix": confusion_matrix(
            truth, predictions, labels=list(range(len(label_names)))
        ).tolist(),
    }
    metrics["auroc"] = _macro_auroc(truth, scores, label_names)
    metrics["per_class_auroc"] = _per_class_auroc(truth, scores, label_names)
    return metrics


def _macro_auroc(truth: np.ndarray, scores: np.ndarray, label_names: list[str]) -> float:
    per_class = _per_class_auroc(truth, scores, label_names)
    values = np.asarray(list(per_class.values()), dtype=float)
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def _per_class_auroc(
    truth: np.ndarray,
    scores: np.ndarray,
    label_names: list[str],
) -> dict[str, float]:
    values: dict[str, float] = {}
    for index, label in enumerate(label_names):
        binary_truth = (truth == index).astype(int)
        if np.unique(binary_truth).size < 2:
            values[label] = float("nan")
            continue
        try:
            values[label] = float(roc_auc_score(binary_truth, scores[:, index]))
        except ValueError:
            values[label] = float("nan")
    return values
