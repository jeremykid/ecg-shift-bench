"""ECG signal preprocessing utilities.

All functions operate on NumPy arrays with shape ``(num_leads, signal_length)``
(i.e. channels-first format, consistent with PyTorch convention).
"""

from __future__ import annotations

import numpy as np
import scipy.signal as sig


def bandpass_filter(
    signal: np.ndarray,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    fs: float = 500.0,
    order: int = 4,
) -> np.ndarray:
    """Apply a Butterworth band-pass filter to an ECG signal.

    Args:
        signal: ECG array of shape ``(num_leads, T)`` or ``(T,)``.
        lowcut: Low-cut frequency in Hz.
        highcut: High-cut frequency in Hz.
        fs: Sampling rate in Hz.
        order: Filter order.

    Returns:
        Filtered array with the same shape as *signal*.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = sig.butter(order, [low, high], btype="band")
    if signal.ndim == 1:
        return sig.filtfilt(b, a, signal).astype(np.float32)
    return np.stack(
        [sig.filtfilt(b, a, signal[i]) for i in range(signal.shape[0])],
        axis=0,
    ).astype(np.float32)


def remove_baseline_wander(
    signal: np.ndarray,
    fs: float = 500.0,
    cutoff: float = 0.5,
    order: int = 4,
) -> np.ndarray:
    """Remove baseline wander using a high-pass Butterworth filter.

    Args:
        signal: ECG array of shape ``(num_leads, T)`` or ``(T,)``.
        fs: Sampling rate in Hz.
        cutoff: High-pass cut-off frequency in Hz.
        order: Filter order.

    Returns:
        Baseline-corrected array with the same shape as *signal*.
    """
    nyq = 0.5 * fs
    high = cutoff / nyq
    b, a = sig.butter(order, high, btype="highpass")
    if signal.ndim == 1:
        return sig.filtfilt(b, a, signal).astype(np.float32)
    return np.stack(
        [sig.filtfilt(b, a, signal[i]) for i in range(signal.shape[0])],
        axis=0,
    ).astype(np.float32)


def normalize_signal(
    signal: np.ndarray,
    method: str = "zscore",
    per_lead: bool = True,
) -> np.ndarray:
    """Normalize an ECG signal.

    Args:
        signal: ECG array of shape ``(num_leads, T)`` or ``(T,)``.
        method: One of ``'zscore'`` (zero mean, unit variance) or
            ``'minmax'`` (scale to [0, 1]).
        per_lead: If ``True``, normalise each lead independently.
            If ``False``, use global statistics across all leads.

    Returns:
        Normalised array with the same shape as *signal*.

    Raises:
        ValueError: If *method* is not recognised.
    """
    if method not in ("zscore", "minmax"):
        raise ValueError(f"method must be 'zscore' or 'minmax', got {method!r}")

    signal = signal.astype(np.float32)

    if signal.ndim == 1 or not per_lead:
        if method == "zscore":
            std = signal.std()
            return (signal - signal.mean()) / (std if std > 0 else 1.0)
        else:
            lo, hi = signal.min(), signal.max()
            return (signal - lo) / (hi - lo + 1e-8)

    normalized = np.empty_like(signal)
    for i in range(signal.shape[0]):
        lead = signal[i]
        if method == "zscore":
            std = lead.std()
            normalized[i] = (lead - lead.mean()) / (std if std > 0 else 1.0)
        else:
            lo, hi = lead.min(), lead.max()
            normalized[i] = (lead - lo) / (hi - lo + 1e-8)
    return normalized


def resample_signal(
    signal: np.ndarray,
    orig_fs: float,
    target_fs: float,
) -> np.ndarray:
    """Resample an ECG signal to a target sampling rate.

    Args:
        signal: ECG array of shape ``(num_leads, T)`` or ``(T,)``.
        orig_fs: Original sampling rate in Hz.
        target_fs: Target sampling rate in Hz.

    Returns:
        Resampled array of shape ``(num_leads, T_new)`` or ``(T_new,)``.
    """
    if orig_fs == target_fs:
        return signal

    ratio = target_fs / orig_fs
    if signal.ndim == 1:
        new_len = int(len(signal) * ratio)
        return sig.resample(signal, new_len).astype(np.float32)

    new_len = int(signal.shape[1] * ratio)
    return np.stack(
        [sig.resample(signal[i], new_len) for i in range(signal.shape[0])],
        axis=0,
    ).astype(np.float32)


def pad_or_truncate(
    signal: np.ndarray,
    target_length: int,
    pad_value: float = 0.0,
) -> np.ndarray:
    """Pad or truncate a signal to *target_length* samples.

    Padding is applied at the end of the signal; truncation removes
    samples from the end.

    Args:
        signal: ECG array of shape ``(num_leads, T)`` or ``(T,)``.
        target_length: Desired signal length in samples.
        pad_value: Value used for padding (default 0.0).

    Returns:
        Array of shape ``(num_leads, target_length)`` or
        ``(target_length,)``.
    """
    if signal.ndim == 1:
        current = len(signal)
        if current >= target_length:
            return signal[:target_length].astype(np.float32)
        padded = np.full(target_length, pad_value, dtype=np.float32)
        padded[:current] = signal
        return padded

    current = signal.shape[1]
    if current >= target_length:
        return signal[:, :target_length].astype(np.float32)
    pad_width = target_length - current
    return np.pad(signal, ((0, 0), (0, pad_width)), constant_values=pad_value).astype(np.float32)
