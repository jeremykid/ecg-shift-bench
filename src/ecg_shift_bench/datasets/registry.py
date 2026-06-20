"""Dataset adapter registry."""

from pathlib import Path
from typing import Any

from ecg_shift_bench.datasets.base import BaseECGDataset
from ecg_shift_bench.datasets.chapman import ChapmanDataset
from ecg_shift_bench.datasets.code15 import CODE15Dataset
from ecg_shift_bench.datasets.ptbxl import PTBXLDataset
from ecg_shift_bench.datasets.sph import SPHDataset

DATASET_REGISTRY: dict[str, type[BaseECGDataset]] = {
    "ptbxl": PTBXLDataset,
    "chapman": ChapmanDataset,
    "sph": SPHDataset,
    "code15": CODE15Dataset,
}


def create_dataset(
    name: str,
    root: str | Path,
    config: dict[str, Any] | None = None,
) -> BaseECGDataset:
    """Instantiate a registered dataset by case-insensitive name."""
    key = name.lower().replace("-", "")
    try:
        dataset_type = DATASET_REGISTRY[key]
    except KeyError as error:
        choices = sorted(DATASET_REGISTRY)
        raise KeyError(f"Unknown dataset {name!r}; choose from {choices}") from error
    return dataset_type(root=root, config=config)
