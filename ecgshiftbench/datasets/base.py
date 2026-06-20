"""Base classes for ECG datasets."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


@dataclass
class DatasetInfo:
    """Metadata about an ECG dataset.

    Attributes:
        name: Human-readable dataset name.
        num_leads: Number of ECG leads (typically 12).
        sampling_rate: Signal sampling rate in Hz.
        duration_seconds: Recording duration in seconds.
        labels: Ordered list of binary label names used in this dataset.
        num_samples: Total number of recordings, or ``None`` if not yet loaded.
        source_country: Country of data collection.
        year: Year of data collection or publication.
    """

    name: str
    num_leads: int
    sampling_rate: int
    duration_seconds: float
    labels: List[str]
    num_samples: Optional[int] = None
    source_country: str = ""
    year: int = 0


class ECGDataset(ABC, Dataset):
    """Abstract base class for 12-lead ECG datasets.

    All datasets in ECGShiftBench extend this class and provide a
    consistent interface for loading signals and multi-label targets.

    Subclasses must implement :meth:`load_metadata` and :meth:`load_signal`.
    """

    #: Metadata describing this dataset — set by subclasses.
    info: DatasetInfo

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        target_transform=None,
        labels: Optional[Sequence[str]] = None,
    ) -> None:
        """Initialise the dataset.

        Args:
            root: Root directory containing the raw dataset files.
            split: Data split to use — one of ``'train'``, ``'val'``, ``'test'``.
            transform: Optional callable applied to each ECG signal array.
            target_transform: Optional callable applied to each label vector.
            labels: Subset of label names to include.  Defaults to all
                labels defined in :attr:`info`.
        """
        super().__init__()
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Dataset root directory not found: {root}")
        self.root = root
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        self._metadata: Optional[pd.DataFrame] = None
        self._active_labels: Optional[List[str]] = None

        self.load_metadata()

        if labels is not None:
            self.set_labels(labels)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def load_metadata(self) -> None:
        """Populate ``self._metadata`` with a :class:`~pandas.DataFrame`.

        The frame must contain at least:
        - ``record_id``: unique identifier for each recording.
        - One column per label defined in :attr:`info.labels` containing
          binary (0/1) integer values.
        """

    @abstractmethod
    def load_signal(self, index: int) -> np.ndarray:
        """Load the raw ECG signal for sample *index*.

        Returns:
            Float32 array of shape ``(num_leads, signal_length)``.
        """

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def metadata(self) -> pd.DataFrame:
        """Return the metadata DataFrame (loaded lazily)."""
        if self._metadata is None:
            raise RuntimeError("Metadata has not been loaded. Call load_metadata() first.")
        return self._metadata

    @property
    def active_labels(self) -> List[str]:
        """Return the list of label columns currently in use."""
        if self._active_labels is None:
            return list(self.info.labels)
        return self._active_labels

    def set_labels(self, labels: Sequence[str]) -> None:
        """Restrict the returned label vector to *labels*.

        Args:
            labels: Subset of label names from :attr:`info.labels`.

        Raises:
            ValueError: If any label in *labels* is not part of the dataset.
        """
        unknown = set(labels) - set(self.info.labels)
        if unknown:
            raise ValueError(f"Unknown labels for {self.info.name}: {unknown}")
        self._active_labels = list(labels)

    def get_labels(self) -> np.ndarray:
        """Return the full label matrix for all samples.

        Returns:
            Integer array of shape ``(num_samples, num_active_labels)``.
        """
        return self.metadata[self.active_labels].values.astype(np.int64)

    def label_counts(self) -> Dict[str, int]:
        """Return the number of positive examples per label."""
        return {lbl: int(self.metadata[lbl].sum()) for lbl in self.active_labels}

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        signal = self.load_signal(index)
        labels = self.metadata.iloc[index][self.active_labels].values.astype(np.float32)

        if self.transform is not None:
            signal = self.transform(signal)
        if self.target_transform is not None:
            labels = self.target_transform(labels)

        return signal, labels

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"split={self.split!r}, "
            f"n_samples={len(self)}, "
            f"labels={self.active_labels})"
        )
