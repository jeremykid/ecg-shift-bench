"""Chapman-Shaoxing/Ningbo metadata and CSV waveform loader.

Expected: ``Diagnostics.xlsx`` and waveform CSV files under ``ECGData/``.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ecg_shift_bench.datasets._tabular import TabularSkeletonDataset, _parse_labels
from ecg_shift_bench.datasets.alignment import CANONICAL_LEAD_ORDER, lead_indices_for_order
from ecg_shift_bench.labels.harmonize import harmonize_labels

DEFAULT_CONFIG: dict[str, Any] = {
    "metadata_file": "Diagnostics.xlsx",
    "records_root": "ECGData",
    "record_id_column": "FileName",
    "label_column": "Rhythm",
    "beat_label_column": "Beat",
    "sampling_rate": 500,
}


class ChapmanDataset(TabularSkeletonDataset):
    """Metadata-aware Chapman-Shaoxing/Ningbo adapter."""

    name = "CHAPMAN"
    domain = "chapman_ningbo_china"
    metadata_file = "Diagnostics.xlsx"

    def __init__(self, root: str | Path, config: dict[str, Any] | None = None) -> None:
        merged_config = {**DEFAULT_CONFIG, **(config or {})}
        super().__init__(root, merged_config)

    def load_metadata(self) -> pd.DataFrame:
        """Load Chapman metadata without requiring unavailable patient IDs."""
        if self._metadata is not None:
            return self._metadata
        path = self.root / self.config.get("metadata_file", self.metadata_file)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {self.name} metadata at {path}. Download the dataset manually "
                "and update the dataset config."
            )
        if path.suffix.lower() in {".xls", ".xlsx"}:
            frame = pd.read_excel(path)
        else:
            frame = pd.read_csv(path)
        record_col = self.config.get("record_id_column", "FileName")
        label_col = self.config.get("label_column", "Rhythm")
        missing = {record_col, label_col}.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        if frame[record_col].isna().any():
            raise ValueError(f"{path} contains missing record identifiers")
        if frame[record_col].astype(str).duplicated().any():
            raise ValueError(f"{path} contains duplicate record identifiers")
        self._metadata = frame
        self._record_rows = None
        return frame

    def load_signal(self, record_id: str) -> np.ndarray:
        """Load one Chapman CSV tracing and return canonical ``(12, samples)``."""
        row = self._metadata_row(record_id)
        record_value = str(row[self.config.get("record_id_column", "FileName")])
        path = self._resolve_signal_path(record_value)
        frame = pd.read_csv(path)
        source_leads = [str(column) for column in frame.columns]
        indices = lead_indices_for_order(source_leads, record_id)
        ordered_columns = [frame.columns[index] for index in indices]
        signal = frame[ordered_columns].to_numpy(dtype=np.float32).T
        if signal.ndim != 2 or signal.shape[0] != len(CANONICAL_LEAD_ORDER):
            raise ValueError(f"Chapman record {record_id!r} has unexpected shape {signal.shape}")
        return signal.copy()

    def get_labels(self, record_id: str) -> dict[str, int]:
        """Return canonical labels from Chapman rhythm and beat annotations."""
        row = self._metadata_row(record_id)
        raw_labels = _parse_labels(row[self.config.get("label_column", "Rhythm")])
        beat_col = self.config.get("beat_label_column", "Beat")
        if beat_col in row.index:
            raw_labels.extend(_parse_beat_labels(row[beat_col]))
        return harmonize_labels(raw_labels, self.name)

    def _resolve_signal_path(self, record_value: str) -> Path:
        filename = record_value if record_value.endswith(".csv") else f"{record_value}.csv"
        records_root = self.root / self.config.get("records_root", "ECGData")
        candidates = [
            records_root / filename,
            records_root / "ECGData" / filename,
        ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(f"Missing Chapman waveform for {record_value!r}: {candidates}")


def _parse_beat_labels(value: Any) -> list[str]:
    """Parse Chapman beat annotations, which are commonly whitespace-separated."""
    labels: list[str] = []
    for label in _parse_labels(value):
        labels.extend(str(label).split())
    return [label for label in labels if label and label.upper() != "NONE"]
