"""Classification metrics for multi-label ECG evaluation.

All functions accept NumPy arrays with shape ``(N, L)`` where *N* is the
number of samples and *L* is the number of labels.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)


def per_label_auroc(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> np.ndarray:
    """Compute per-label AUROC scores.

    Args:
        y_true: Binary ground-truth labels, shape ``(N, L)``.
        y_score: Predicted probabilities, shape ``(N, L)``.

    Returns:
        Float array of shape ``(L,)`` with per-label AUROC.
        Labels with no positive or no negative examples receive ``NaN``.
    """
    n_labels = y_true.shape[1]
    scores = np.full(n_labels, np.nan)
    for i in range(n_labels):
        if len(np.unique(y_true[:, i])) > 1:
            scores[i] = roc_auc_score(y_true[:, i], y_score[:, i])
    return scores


def macro_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mean AUROC across all labels (ignoring NaN labels).

    Args:
        y_true: Binary ground-truth labels, shape ``(N, L)``.
        y_score: Predicted probabilities, shape ``(N, L)``.

    Returns:
        Macro-averaged AUROC scalar.
    """
    per_label = per_label_auroc(y_true, y_score)
    valid = per_label[~np.isnan(per_label)]
    if len(valid) == 0:
        return float("nan")
    return float(np.mean(valid))


def per_label_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Compute per-label F1 scores from binary predictions.

    Args:
        y_true: Binary ground-truth labels, shape ``(N, L)``.
        y_pred: Binary predicted labels, shape ``(N, L)``.

    Returns:
        Float array of shape ``(L,)`` with per-label F1 scores.
    """
    n_labels = y_true.shape[1]
    scores = np.zeros(n_labels)
    for i in range(n_labels):
        scores[i] = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
    return scores


def macro_f1(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Macro-averaged F1 score.

    Args:
        y_true: Binary ground-truth labels, shape ``(N, L)``.
        y_score: Predicted probabilities, shape ``(N, L)``.
        threshold: Decision threshold for binarising *y_score*.

    Returns:
        Macro-averaged F1 scalar.
    """
    y_pred = (y_score >= threshold).astype(int)
    per_label = per_label_f1(y_true, y_pred)
    return float(np.mean(per_label))


def sensitivity_at_specificity(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_specificity: float = 0.90,
    label_idx: Optional[int] = None,
) -> float:
    """Compute sensitivity at a target specificity threshold.

    Args:
        y_true: Binary ground-truth labels, shape ``(N,)`` or ``(N, L)``.
        y_score: Predicted probabilities, same shape as *y_true*.
        target_specificity: Desired specificity (0–1).
        label_idx: If *y_true* is 2-D, index of the label to use.
            When ``None`` and *y_true* is 2-D, the macro-average is returned.

    Returns:
        Sensitivity scalar.
    """
    from sklearn.metrics import roc_curve

    if y_true.ndim == 2:
        if label_idx is not None:
            y_true = y_true[:, label_idx]
            y_score = y_score[:, label_idx]
        else:
            values = [
                sensitivity_at_specificity(
                    y_true[:, i],
                    y_score[:, i],
                    target_specificity=target_specificity,
                )
                for i in range(y_true.shape[1])
            ]
            valid = [v for v in values if not np.isnan(v)]
            return float(np.mean(valid)) if valid else float("nan")

    fpr, tpr, _ = roc_curve(y_true, y_score)
    specificity = 1.0 - fpr
    # Among all operating points with specificity >= target, return the one
    # with the highest sensitivity (i.e. the least conservative threshold).
    mask = specificity >= target_specificity
    if mask.any():
        return float(tpr[mask].max())
    # If no point meets the target, return the sensitivity at the closest point.
    idx = np.argmin(np.abs(specificity - target_specificity))
    return float(tpr[idx])


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute the full set of ECGShiftBench evaluation metrics.

    Args:
        y_true: Binary ground-truth labels, shape ``(N, L)``.
        y_score: Predicted probabilities, shape ``(N, L)``.
        threshold: Decision threshold for binary F1.

    Returns:
        Dictionary with keys:
        - ``macro_auroc``: Macro-averaged AUROC.
        - ``macro_f1``: Macro-averaged F1 score.
        - ``sensitivity_at_90sp``: Sensitivity at 90 % specificity.
    """
    return {
        "macro_auroc": macro_auroc(y_true, y_score),
        "macro_f1": macro_f1(y_true, y_score, threshold=threshold),
        "sensitivity_at_90sp": sensitivity_at_specificity(
            y_true, y_score, target_specificity=0.90
        ),
    }
