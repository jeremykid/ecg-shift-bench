"""Dataset interface shared by all ECG sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ecg_shift_bench.datasets.alignment import AlignedECGSample, align_ecg_signal


@dataclass(frozen=True)
class ECGRecord:
    """Lightweight record descriptor that does not load the signal eagerly."""

    record_id: str
    patient_id: str | None
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

    def load_aligned_signal(self, record_id: str) -> np.ndarray:
        """Load one ECG after applying the shared alignment contract."""
        signal = self.load_signal(record_id)
        return align_ecg_signal(
            signal,
            source_rate=int(self.config.get("sampling_rate", 500)),
            target_rate=int(self.config.get("target_sampling_rate", 500)),
            target_length=int(self.config.get("target_length", 5000)),
        )

    def load_aligned_sample(self, record_id: str) -> AlignedECGSample:
        """Load one aligned ECG with labels and provenance for downstream tasks."""
        raw_signal = self.load_signal(record_id)
        aligned_signal = align_ecg_signal(
            raw_signal,
            source_rate=int(self.config.get("sampling_rate", 500)),
            target_rate=int(self.config.get("target_sampling_rate", 500)),
            target_length=int(self.config.get("target_length", 5000)),
        )
        return AlignedECGSample(
            signal=aligned_signal,
            labels=self.get_labels(record_id),
            record_id=str(record_id),
            patient_id=self._patient_id_for_record(record_id),
            domain=self.domain,
            sampling_rate=int(self.config.get("target_sampling_rate", 500)),
            source_sampling_rate=int(self.config.get("sampling_rate", 500)),
            source_length=int(raw_signal.shape[-1]),
            meta={
                "target_length": int(self.config.get("target_length", 5000)),
                "normalization": self.config.get("normalization", "per_lead_zscore"),
                "lead_order": self.config.get("lead_order"),
            },
        )

    def iter_records(self) -> Iterator[ECGRecord]:
        """Iterate descriptors without loading signal arrays."""
        metadata = self.load_metadata()
        record_col = self.config.get("record_id_column", "record_id")
        for record_id in metadata[record_col].astype(str):
            patient_id = self._patient_id_for_record(record_id)
            yield ECGRecord(record_id, patient_id, self.domain, self.get_labels(record_id))

    def _patient_id_for_record(self, record_id: str) -> str | None:
        metadata = self.load_metadata()
        record_col = self.config.get("record_id_column", "record_id")
        patient_col = self.config.get("patient_id_column")
        if not patient_col or str(patient_col) not in metadata.columns:
            return None
        matches = metadata[record_col].astype(str) == str(record_id)
        if not matches.any():
            raise KeyError(f"Unknown {self.name} record: {record_id}")
        value = metadata.loc[matches, str(patient_col)].iloc[0]
        return None if pd.isna(value) else str(value)
