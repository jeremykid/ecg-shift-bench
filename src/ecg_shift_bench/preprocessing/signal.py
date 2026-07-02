"""Numerical preprocessing for lead-first ECG arrays."""

from math import gcd

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt


def _as_lead_first(signal: np.ndarray) -> np.ndarray:
    array = np.asarray(signal, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D (leads, samples) array, got shape {array.shape}")
    return array


def resample_signal(signal: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Polyphase-resample a lead-first signal along its time axis."""
    array = _as_lead_first(signal)
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("Sampling rates must be positive")
    if source_rate == target_rate:
        return array.copy()
    divisor = gcd(source_rate, target_rate)
    return resample_poly(array, target_rate // divisor, source_rate // divisor, axis=-1).astype(
        np.float32
    )


def crop_or_pad(
    signal: np.ndarray,
    target_length: int,
    *,
    crop: str = "center",
    pad_value: float = 0.0,
) -> np.ndarray:
    """Center/left-crop or symmetrically pad a lead-first signal."""
    array = _as_lead_first(signal)
    if target_length <= 0:
        raise ValueError("target_length must be positive")
    length = array.shape[-1]
    if length > target_length:
        if crop not in {"center", "left"}:
            raise ValueError("crop must be 'center' or 'left'")
        start = (length - target_length) // 2 if crop == "center" else 0
        return array[:, start : start + target_length].copy()
    if length < target_length:
        missing = target_length - length
        before = missing // 2
        after = missing - before
        return np.pad(
            array,
            ((0, 0), (before, after)),
            mode="constant",
            constant_values=pad_value,
        ).astype(np.float32)
    return array.copy()


def bandpass_filter(
    signal: np.ndarray,
    sampling_rate: int,
    low_hz: float = 0.5,
    high_hz: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    """Apply an optional zero-phase Butterworth bandpass filter.

    This is intentionally opt-in. Benchmark runs must record the chosen cutoff,
    order, and handling of signals too short for zero-phase filtering.
    """
    array = _as_lead_first(signal)
    nyquist = sampling_rate / 2
    if not 0 < low_hz < high_hz < nyquist:
        raise ValueError("Require 0 < low_hz < high_hz < Nyquist frequency")
    sos = butter(order, [low_hz / nyquist, high_hz / nyquist], btype="band", output="sos")
    return sosfiltfilt(sos, array, axis=-1).astype(np.float32)
