"""Shared ECG waveform alignment contract for dataset adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ecg_shift_bench.preprocessing.signal import crop_or_pad, resample_signal

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

CANONICAL_TARGET_UNIT = "mV"
_UNIT_SCALES_TO_MV = {
    "mV": 1.0,
    "uV": 1e-3,
}
_UNIT_ALIASES = {
    "mv": "mV",
    "millivolt": "mV",
    "millivolts": "mV",
    "uv": "uV",
    "microvolt": "uV",
    "microvolts": "uV",
}


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
    source_unit: str = CANONICAL_TARGET_UNIT,
    target_unit: str = CANONICAL_TARGET_UNIT,
) -> np.ndarray:
    """Return a lead-first float32 waveform with common unit, rate, and length."""
    array = np.asarray(signal, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D lead-first ECG array, got shape {array.shape}")
    if array.shape[0] != 12:
        raise ValueError(f"Expected 12 ECG leads, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("ECG signal contains non-finite values")

    aligned = convert_signal_units(array, source_unit=source_unit, target_unit=target_unit)
    aligned = resample_signal(aligned, source_rate, target_rate)
    aligned = crop_or_pad(aligned, target_length)
    return aligned.astype(np.float32, copy=False)


def canonical_unit_name(unit: str) -> str:
    """Return the canonical ECG unit label used by the shared alignment path."""
    canonical = (
        str(unit)
        .strip()
        .replace("\u03bc", "u")
        .replace("\u00b5", "u")
        .casefold()
    )
    try:
        return _UNIT_ALIASES[canonical]
    except KeyError as error:
        raise ValueError(f"Unsupported ECG unit {unit!r}; expected mV or uV") from error


def convert_signal_units(
    signal: np.ndarray,
    *,
    source_unit: str = CANONICAL_TARGET_UNIT,
    target_unit: str = CANONICAL_TARGET_UNIT,
) -> np.ndarray:
    """Convert a waveform between supported physical ECG units."""
    array = np.asarray(signal, dtype=np.float32)
    source = canonical_unit_name(source_unit)
    target = canonical_unit_name(target_unit)
    if source == target:
        return array.copy()
    source_scale = _UNIT_SCALES_TO_MV[source]
    target_scale = _UNIT_SCALES_TO_MV[target]
    return (array * (source_scale / target_scale)).astype(np.float32, copy=False)


def lead_indices_for_order(lead_names: list[str], record_id: str) -> list[int]:
    """Map source lead names to the canonical order, case-insensitively."""
    observed = [str(name).upper() for name in lead_names]
    expected = [name.upper() for name in CANONICAL_LEAD_ORDER]
    missing = set(expected).difference(observed)
    if missing:
        raise ValueError(f"Record {record_id!r} is missing leads: {sorted(missing)}")
    return [observed.index(lead) for lead in expected]
