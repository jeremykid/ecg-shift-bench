"""SPH ECG metadata and HDF5 waveform loader.

Expected: a normalized ``metadata.csv`` index and waveform files under
``records/``.
"""

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from ecg_shift_bench.datasets._tabular import TabularSkeletonDataset

DEFAULT_CONFIG: dict[str, Any] = {
    "metadata_file": "metadata.csv",
    "records_root": "records",
    "record_id_column": "ECG_ID",
    "patient_id_column": "Patient_ID",
    "label_column": "AHA_Code",
    "signal_dataset": "ecg",
    "sampling_rate": 500,
}


class SPHDataset(TabularSkeletonDataset):
    """Metadata-aware SPH dataset adapter."""

    name = "SPH"
    domain = "sph_china"
    metadata_file = "metadata.csv"

    def __init__(self, root: str | Path, config: dict[str, Any] | None = None) -> None:
        merged_config = {**DEFAULT_CONFIG, **(config or {})}
        super().__init__(root, merged_config)

    def load_metadata(self) -> pd.DataFrame:
        """Load SPH metadata using the public release column names."""
        if self._metadata is not None:
            return self._metadata
        path = self.root / self.config.get("metadata_file", self.metadata_file)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {self.name} metadata at {path}. Download the dataset manually "
                "and update the dataset config."
            )
        frame = pd.read_csv(path)
        required = {
            self.config.get("record_id_column", "ECG_ID"),
            self.config.get("patient_id_column", "Patient_ID"),
            self.config.get("label_column", "AHA_Code"),
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        record_col = self.config.get("record_id_column", "ECG_ID")
        if frame[record_col].isna().any():
            raise ValueError(f"{path} contains missing record identifiers")
        if frame[record_col].astype(str).duplicated().any():
            raise ValueError(f"{path} contains duplicate record identifiers")
        self._metadata = frame
        self._record_rows = None
        return frame

    def load_signal(self, record_id: str) -> np.ndarray:
        """Load one SPH HDF5 tracing as ``(12, samples)``."""
        row = self._metadata_row(record_id)
        record_value = str(row[self.config.get("record_id_column", "ECG_ID")])
        path = self._resolve_signal_path(record_value)
        signal_key = self.config.get("signal_dataset", "ecg")
        with h5py.File(path, "r") as handle:
            if signal_key not in handle:
                raise KeyError(f"SPH record {record_id!r} is missing dataset {signal_key!r}")
            signal = np.asarray(handle[signal_key], dtype=np.float32)
        if signal.ndim != 2 or signal.shape[0] != 12:
            raise ValueError(f"SPH record {record_id!r} has unexpected shape {signal.shape}")
        return signal.copy()

    def _resolve_signal_path(self, record_value: str) -> Path:
        filename = record_value if record_value.endswith(".h5") else f"{record_value}.h5"
        records_root = self.root / self.config.get("records_root", "records")
        candidates = [
            records_root / filename,
            records_root / "records" / filename,
        ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(f"Missing SPH waveform for {record_value!r}: {candidates}")
