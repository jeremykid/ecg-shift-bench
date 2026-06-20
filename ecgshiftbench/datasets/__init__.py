"""
ECG dataset loaders.

Available datasets:
- PTB-XL: Large 12-lead ECG dataset from Germany
- Chapman-Shaoxing: Chinese 12-lead ECG dataset
- CPSC2018: China Physiological Signal Challenge 2018 dataset
- G12EC: Georgia 12-lead ECG Challenge dataset
"""

from ecgshiftbench.datasets.base import ECGDataset, DatasetInfo
from ecgshiftbench.datasets.ptbxl import PTBXLDataset
from ecgshiftbench.datasets.chapman import ChapmanDataset
from ecgshiftbench.datasets.cpsc2018 import CPSC2018Dataset
from ecgshiftbench.datasets.g12ec import G12ECDataset

DATASET_REGISTRY = {
    "ptbxl": PTBXLDataset,
    "chapman": ChapmanDataset,
    "cpsc2018": CPSC2018Dataset,
    "g12ec": G12ECDataset,
}


def load_dataset(name: str, root: str, **kwargs) -> ECGDataset:
    """Load a registered ECG dataset by name.

    Args:
        name: Dataset identifier (e.g. ``'ptbxl'``, ``'chapman'``).
        root: Path to the root directory containing the dataset files.
        **kwargs: Additional keyword arguments forwarded to the dataset class.

    Returns:
        An :class:`ECGDataset` instance.

    Raises:
        ValueError: If *name* is not a registered dataset.
    """
    name = name.lower()
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Available datasets: {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[name](root=root, **kwargs)


__all__ = [
    "ECGDataset",
    "DatasetInfo",
    "PTBXLDataset",
    "ChapmanDataset",
    "CPSC2018Dataset",
    "G12ECDataset",
    "DATASET_REGISTRY",
    "load_dataset",
]
