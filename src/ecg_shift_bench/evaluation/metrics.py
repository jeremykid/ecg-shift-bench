"""Robust multi-label classification metrics."""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

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


def _validated_binary_arrays(y_true: ArrayLike, y_score: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true)
    scores = np.asarray(y_score, dtype=float)
    if truth.ndim != 1 or scores.ndim != 1 or truth.shape != scores.shape:
        raise ValueError("binary_class_report expects same-shaped 1D arrays")
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


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _safe_roc_auc(truth: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(truth).size < 2:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(roc_auc_score(truth, scores))


def _safe_average_precision(truth: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(truth).size < 2:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(average_precision_score(truth, scores))


def _safe_binary_pr_auc(truth: np.ndarray, pred: np.ndarray) -> float:
    if np.unique(truth).size < 2:
        return float("nan")
    precision, recall, _ = precision_recall_curve(truth, pred)
    return float(auc(recall, precision))


def _safe_brier_score(truth: np.ndarray, scores: np.ndarray) -> float:
    if truth.size == 0:
        return float("nan")
    return float(brier_score_loss(truth, scores, pos_label=int(np.max(truth))))


def _binary_metrics_from_predictions(truth: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    truth = truth.astype(int, copy=False)
    pred = pred.astype(int, copy=False)
    tp = float(np.logical_and(truth == 1, pred == 1).sum())
    tn = float(np.logical_and(truth == 0, pred == 0).sum())
    fp = float(np.logical_and(truth == 0, pred == 1).sum())
    fn = float(np.logical_and(truth == 1, pred == 0).sum())
    support = float((truth == 1).sum())
    negative_support = float((truth == 0).sum())

    accuracy = _safe_divide(tp + tn, tp + tn + fp + fn)
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "f1_score": f1,
        "support": support,
        "negative_support": negative_support,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _threshold_array(
    thresholds: dict[str, float] | list[float] | tuple[float, ...] | np.ndarray | None,
    labels: list[str],
) -> np.ndarray:
    if thresholds is None:
        return np.full(len(labels), 0.5, dtype=float)
    if isinstance(thresholds, dict):
        return np.asarray([float(thresholds.get(label, 0.5)) for label in labels], dtype=float)
    values = np.asarray(thresholds, dtype=float)
    if values.shape != (len(labels),):
        raise ValueError("thresholds must match the number of labels")
    return values


def optimal_multilabel_thresholds(
    y_true: ArrayLike,
    y_score: ArrayLike,
    label_names: list[str] | None = None,
) -> dict[str, float]:
    """Choose a per-label threshold using Youden's J statistic on ROC curves."""
    truth, scores = _validated_arrays(y_true, y_score)
    labels = list(label_names or CANONICAL_LABELS)
    if len(labels) != truth.shape[1]:
        raise ValueError("label_names length must match the second array dimension")

    thresholds: dict[str, float] = {}
    for index, label in enumerate(labels):
        if np.unique(truth[:, index]).size < 2:
            thresholds[label] = 0.5
            continue
        fpr, tpr, thr = roc_curve(truth[:, index], scores[:, index])
        if thr.size == 0:
            thresholds[label] = 0.5
            continue
        j_scores = tpr - fpr
        max_j = np.nanmax(j_scores)
        best_candidates = np.flatnonzero(j_scores == max_j)
        if best_candidates.size == 0:
            thresholds[label] = 0.5
            continue
        thresholds[label] = float(thr[int(best_candidates[-1])])
    return thresholds


def binary_class_report(
    y_true: ArrayLike,
    y_score: ArrayLike,
    threshold: float,
) -> dict[str, float | int]:
    """Return the per-label report contract used by the external generate_report.py."""
    truth, scores = _validated_binary_arrays(y_true, y_score)

    pred = (scores > float(threshold)).astype(int)
    metrics = _binary_metrics_from_predictions(truth, pred)
    return {
        "accuracy": metrics["accuracy"],
        "auroc": _safe_roc_auc(truth, scores),
        "auprc": _safe_binary_pr_auc(truth, pred),
        "f1_score": metrics["f1_score"],
        "prec": metrics["precision"],
        "rec": metrics["recall"],
        "sensitivity": metrics["sensitivity"],
        "spec": metrics["specificity"],
        "aprec": _safe_average_precision(truth, scores),
        "br_score": _safe_brier_score(truth, scores),
        "tn": int(metrics["tn"]),
        "fp": int(metrics["fp"]),
        "fn": int(metrics["fn"]),
        "tp": int(metrics["tp"]),
    }


def source_script_multilabel_report(
    y_true: ArrayLike,
    y_score: ArrayLike,
    label_names: list[str] | None = None,
    thresholds: dict[str, float] | list[float] | tuple[float, ...] | np.ndarray | None = None,
) -> dict[str, float | dict[str, float | int]]:
    """Compute the source-script report for a multilabel prediction matrix."""
    truth, scores = _validated_arrays(y_true, y_score)
    labels = list(label_names or CANONICAL_LABELS)
    if len(labels) != truth.shape[1]:
        raise ValueError("label_names length must match the second array dimension")

    threshold_array = _threshold_array(thresholds, labels)
    per_label_reports: dict[str, dict[str, float | int]] = {}
    per_label_support: dict[str, int] = {}
    for index, label in enumerate(labels):
        per_label_reports[label] = binary_class_report(truth[:, index], scores[:, index], threshold_array[index])
        per_label_support[label] = int(truth[:, index].sum())

    report_keys = (
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
    )
    summary = {
        f"macro_{key}": _nanmean({label: float(per_label_reports[label][key]) for label in labels})
        for key in report_keys
    }
    summary.update(
        {
            "thresholds": {label: float(threshold_array[index]) for index, label in enumerate(labels)},
            "per_label_reports": per_label_reports,
            "per_label_support": per_label_support,
        }
    )
    return summary


def thresholded_multilabel_metrics(
    y_true: ArrayLike,
    y_score: ArrayLike,
    label_names: list[str] | None = None,
    thresholds: dict[str, float] | list[float] | tuple[float, ...] | np.ndarray | None = None,
) -> dict[str, float | dict[str, float]]:
    """Compute classification-style multilabel metrics using per-label thresholds."""
    truth, scores = _validated_arrays(y_true, y_score)
    labels = list(label_names or CANONICAL_LABELS)
    if len(labels) != truth.shape[1]:
        raise ValueError("label_names length must match the second array dimension")

    threshold_array = _threshold_array(thresholds, labels)
    predictions = (scores > threshold_array).astype(int)

    per_label_accuracy: dict[str, float] = {}
    per_label_precision: dict[str, float] = {}
    per_label_recall: dict[str, float] = {}
    per_label_sensitivity: dict[str, float] = {}
    per_label_specificity: dict[str, float] = {}
    per_label_f1: dict[str, float] = {}
    per_label_support: dict[str, float] = {}
    per_label_negative_support: dict[str, float] = {}
    per_label_thresholds: dict[str, float] = {}

    for index, label in enumerate(labels):
        binary_metrics = _binary_metrics_from_predictions(truth[:, index], predictions[:, index])
        per_label_accuracy[label] = binary_metrics["accuracy"]
        per_label_precision[label] = binary_metrics["precision"]
        per_label_recall[label] = binary_metrics["recall"]
        per_label_sensitivity[label] = binary_metrics["sensitivity"]
        per_label_specificity[label] = binary_metrics["specificity"]
        per_label_f1[label] = binary_metrics["f1_score"]
        per_label_support[label] = binary_metrics["support"]
        per_label_negative_support[label] = binary_metrics["negative_support"]
        per_label_thresholds[label] = float(threshold_array[index])

    flattened_truth = truth.ravel()
    flattened_pred = predictions.ravel()
    micro_metrics = _binary_metrics_from_predictions(flattened_truth, flattened_pred)

    return {
        "thresholds": per_label_thresholds,
        "macro_accuracy": _nanmean(per_label_accuracy),
        "micro_accuracy": micro_metrics["accuracy"],
        "macro_precision": _nanmean(per_label_precision),
        "micro_precision": micro_metrics["precision"],
        "macro_recall": _nanmean(per_label_recall),
        "micro_recall": micro_metrics["recall"],
        "macro_sensitivity": _nanmean(per_label_sensitivity),
        "micro_sensitivity": micro_metrics["sensitivity"],
        "macro_specificity": _nanmean(per_label_specificity),
        "micro_specificity": micro_metrics["specificity"],
        "macro_f1": _nanmean(per_label_f1),
        "micro_f1": micro_metrics["f1_score"],
        "per_label_accuracy": per_label_accuracy,
        "per_label_precision": per_label_precision,
        "per_label_recall": per_label_recall,
        "per_label_sensitivity": per_label_sensitivity,
        "per_label_specificity": per_label_specificity,
        "per_label_f1": per_label_f1,
        "per_label_support": per_label_support,
        "per_label_negative_support": per_label_negative_support,
    }


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
