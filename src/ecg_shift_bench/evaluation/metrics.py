"""Robust multi-label classification metrics."""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import average_precision_score, roc_auc_score

from ecg_shift_bench.labels.canonical import CANONICAL_LABELS


def _validated_arrays(y_true: ArrayLike, y_score: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true)
    scores = np.asarray(y_score, dtype=float)
    if truth.ndim != 2 or scores.ndim != 2 or truth.shape != scores.shape:
        raise ValueError("y_true and y_score must be same-shaped (samples, labels) arrays")
    if not np.isin(truth, [0, 1]).all():
        raise ValueError("y_true must contain only binary values")
    if not np.isfinite(scores).all():
        raise ValueError("y_score must contain only finite values")
    return truth.astype(int), scores


def _per_label_metric(
    metric,
    truth: np.ndarray,
    scores: np.ndarray,
    label_names: list[str],
) -> dict[str, float]:
    values: dict[str, float] = {}
    for index, label in enumerate(label_names):
        if np.unique(truth[:, index]).size < 2:
            values[label] = float("nan")
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            values[label] = float(metric(truth[:, index], scores[:, index]))
    return values


def _nanmean(values: dict[str, float]) -> float:
    array = np.asarray(list(values.values()), dtype=float)
    return float(np.nanmean(array)) if np.isfinite(array).any() else float("nan")


def multilabel_metrics(
    y_true: ArrayLike,
    y_score: ArrayLike,
    label_names: list[str] | None = None,
) -> dict[str, float | dict[str, float]]:
    """Compute macro/micro AUROC, AUPRC, and per-label values.

    Per-label metrics are NaN when a label has only one observed class. Macro
    metrics average the defined per-label values, which makes small evaluation
    subsets usable while retaining explicit undefined entries.
    """
    truth, scores = _validated_arrays(y_true, y_score)
    labels = list(label_names or CANONICAL_LABELS)
    if len(labels) != truth.shape[1]:
        raise ValueError("label_names length must match the second array dimension")

    per_label_auroc = _per_label_metric(roc_auc_score, truth, scores, labels)
    per_label_auprc = _per_label_metric(average_precision_score, truth, scores, labels)
    flattened_truth = truth.ravel()
    if np.unique(flattened_truth).size < 2:
        micro_auroc = float("nan")
        micro_auprc = float("nan")
    else:
        micro_auroc = float(roc_auc_score(flattened_truth, scores.ravel()))
        micro_auprc = float(average_precision_score(flattened_truth, scores.ravel()))
    return {
        "macro_auroc": _nanmean(per_label_auroc),
        "micro_auroc": micro_auroc,
        "macro_auprc": _nanmean(per_label_auprc),
        "micro_auprc": micro_auprc,
        "per_label_auroc": per_label_auroc,
        "per_label_auprc": per_label_auprc,
    }
