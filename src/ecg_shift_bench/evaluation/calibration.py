"""Calibration metrics for multi-label probabilities."""

import numpy as np
from numpy.typing import ArrayLike


def expected_calibration_error(
    y_true: ArrayLike,
    y_probability: ArrayLike,
    bins: int = 10,
) -> float:
    """Compute pooled equal-width expected calibration error."""
    truth = np.asarray(y_true).ravel()
    probability = np.asarray(y_probability, dtype=float).ravel()
    if truth.shape != probability.shape or bins <= 0:
        raise ValueError("Inputs must have equal shape and bins must be positive")
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("Probabilities must lie in [0, 1]")
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.digitize(probability, edges[1:-1]), bins - 1)
    error = 0.0
    for index in range(bins):
        mask = assignments == index
        if mask.any():
            error += mask.mean() * abs(probability[mask].mean() - truth[mask].mean())
    return float(error)
