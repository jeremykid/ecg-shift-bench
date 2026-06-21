"""Dataset interface shared by all ECG sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ECGRecord:
    """Lightweight record descriptor that does not load the signal eagerly."""

    record_id: str
    patient_id: str
    domain: str
    labels: dict[str, int]


class BaseECGDataset(ABC):
    """Abstract interface for a manually provisioned ECG dataset."""

    name: str
    domain: str

    def __init__(self, root: str | Path, config: dict[str, Any] | None = None) -> None:
        self.root = Path(root)
        self.config = config or {}
        self.domain = str(self.config.get("domain", self.domain))
        self._metadata: pd.DataFrame | None = None

    @abstractmethod
    def load_metadata(self) -> pd.DataFrame:
        """Load and validate the dataset's record-level metadata."""

    @abstractmethod
    def load_signal(self, record_id: str) -> np.ndarray:
        """Load one signal as a ``(n_leads, n_samples)`` float array."""

    @abstractmethod
    def get_labels(self, record_id: str) -> dict[str, int]:
        """Return harmonized canonical labels for one record."""

    def iter_records(self) -> Iterator[ECGRecord]:
        """Iterate descriptors without loading signal arrays."""
        metadata = self.load_metadata()
        record_col = self.config.get("record_id_column", "record_id")
        patient_col = self.config.get("patient_id_column", "patient_id")
        for row in metadata[[record_col, patient_col]].itertuples(index=False, name=None):
            record_id, patient_id = map(str, row)
            yield ECGRecord(record_id, patient_id, self.domain, self.get_labels(record_id))
