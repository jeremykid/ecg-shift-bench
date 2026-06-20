"""Basic, dataset-independent signal quality checks."""

import numpy as np


def signal_quality_flags(signal: np.ndarray, flat_std_threshold: float = 1e-6) -> dict[str, bool]:
    """Flag non-finite values and nearly flat leads in a lead-first array."""
    array = np.asarray(signal)
    if array.ndim != 2:
        raise ValueError("signal must have shape (leads, samples)")
    return {
        "has_non_finite": bool(~np.isfinite(array).all()),
        "has_flat_lead": bool(np.any(np.nanstd(array, axis=-1) < flat_std_threshold)),
        "has_twelve_leads": array.shape[0] == 12,
    }
