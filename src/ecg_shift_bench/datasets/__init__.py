"""Dataset interfaces and registry."""

from ecg_shift_bench.datasets.alignment import (
    AlignedECGSample,
    align_ecg_signal,
    canonical_unit_name,
    convert_signal_units,
)
from ecg_shift_bench.datasets.base import BaseECGDataset, ECGRecord

__all__ = [
    "AlignedECGSample",
    "BaseECGDataset",
    "ECGRecord",
    "align_ecg_signal",
    "canonical_unit_name",
    "convert_signal_units",
]


def __getattr__(name: str):
    """Lazily expose registry helpers that depend on optional dataset backends."""
    if name in {"DATASET_REGISTRY", "create_dataset"}:
        from ecg_shift_bench.datasets.registry import DATASET_REGISTRY, create_dataset

        globals()["DATASET_REGISTRY"] = DATASET_REGISTRY
        globals()["create_dataset"] = create_dataset
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
