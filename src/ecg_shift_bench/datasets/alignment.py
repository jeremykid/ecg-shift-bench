"""Shared ECG waveform alignment contract for dataset adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ecg_shift_bench.preprocessing.signal import crop_or_pad, per_lead_zscore, resample_signal

CANONICAL_LEAD_ORDER = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]


@dataclass(frozen=True)
class AlignedECGSample:
    """Aligned ECG record with waveform, labels, and provenance fields."""

    signal: np.ndarray
    labels: dict[str, int]
    record_id: str
    patient_id: str | None
    domain: str
    sampling_rate: int
    source_sampling_rate: int
    source_length: int
    meta: dict[str, Any]


def align_ecg_signal(
    signal: np.ndarray,
    *,
    source_rate: int,
    target_rate: int = 500,
    target_length: int = 5000,
) -> np.ndarray:
    """Return a lead-first float32 waveform with common rate, length, and z-score."""
    array = np.asarray(signal, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D lead-first ECG array, got shape {array.shape}")
    if array.shape[0] != 12:
        raise ValueError(f"Expected 12 ECG leads, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("ECG signal contains non-finite values")

    aligned = resample_signal(array, source_rate, target_rate)
    aligned = crop_or_pad(aligned, target_length)
    aligned = per_lead_zscore(aligned)
    return aligned.astype(np.float32, copy=False)


def lead_indices_for_order(lead_names: list[str], record_id: str) -> list[int]:
    """Map source lead names to the canonical order, case-insensitively."""
    observed = [str(name).upper() for name in lead_names]
    expected = [name.upper() for name in CANONICAL_LEAD_ORDER]
    missing = set(expected).difference(observed)
    if missing:
        raise ValueError(f"Record {record_id!r} is missing leads: {sorted(missing)}")
    return [observed.index(lead) for lead in expected]
