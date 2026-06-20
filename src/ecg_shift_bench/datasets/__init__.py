"""Dataset interfaces and registry."""

from ecg_shift_bench.datasets.base import BaseECGDataset, ECGRecord
from ecg_shift_bench.datasets.registry import DATASET_REGISTRY, create_dataset

__all__ = ["BaseECGDataset", "ECGRecord", "DATASET_REGISTRY", "create_dataset"]
