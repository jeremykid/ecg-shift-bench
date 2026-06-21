"""CODE-15% metadata and HDF5 signal loader.

Expected: ``exams.csv`` and ``exams_part0.hdf5`` through
``exams_part17.hdf5``. Stored tracings have shape ``(samples, 12)`` at 400 Hz
in I, II, III, aVR, aVL, aVF, V1--V6 order. Every released HDF5 part includes
one trailing all-zero sentinel with exam ID 0; it is not present in metadata.
"""

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from ecg_shift_bench.datasets._tabular import TabularSkeletonDataset
from ecg_shift_bench.labels.harmonize import harmonize_labels

DEFAULT_LABEL_COLUMNS = ["AF", "RBBB", "LBBB", "1dAVb", "SB", "ST"]
DEFAULT_CONFIG: dict[str, Any] = {
    "metadata_file": "exams.csv",
    "records_root": ".",
    "record_id_column": "exam_id",
    "patient_id_column": "patient_id",
    "trace_file_column": "trace_file",
    "label_columns": DEFAULT_LABEL_COLUMNS,
    "signal_dataset": "tracings",
    "signal_layout": "samples_leads",
    "sampling_rate": 400,
}


class CODE15Dataset(TabularSkeletonDataset):
    """CODE-15% adapter with patient linkage and lazy HDF5 indexing."""

    name = "CODE15"
    domain = "code15_brazil"
    metadata_file = "exams.csv"

    def __init__(self, root: str | Path, config: dict[str, Any] | None = None) -> None:
        merged_config = {**DEFAULT_CONFIG, **(config or {})}
        super().__init__(root, merged_config)
        self._hdf5_indices: dict[Path, dict[int, int]] = {}

    def load_metadata(self) -> pd.DataFrame:
        """Load CODE-15% metadata and validate IDs, labels, and part names."""
        if self._metadata is not None:
            return self._metadata
        path = self.root / self.config.get("metadata_file", self.metadata_file)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing CODE15 metadata at {path}. Download the dataset manually "
                "and update the dataset config."
            )
        frame = pd.read_csv(path)
        record_col = self.config.get("record_id_column", "exam_id")
        patient_col = self.config.get("patient_id_column", "patient_id")
        trace_col = self.config.get("trace_file_column", "trace_file")
        label_columns = self.config.get("label_columns", DEFAULT_LABEL_COLUMNS)
        required = {record_col, patient_col, trace_col, *label_columns}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        if frame[[record_col, patient_col, trace_col]].isna().any().any():
            raise ValueError(f"{path} contains missing record, patient, or trace identifiers")
        if frame[record_col].astype(str).duplicated().any():
            raise ValueError(f"{path} contains duplicate record identifiers")
        invalid_labels = {
            column
            for column in label_columns
            if not frame[column].dropna().isin([True, False, 0, 1]).all()
        }
        if invalid_labels:
            raise ValueError(f"{path} has non-binary labels in: {sorted(invalid_labels)}")
        self._metadata = frame
        self._record_rows = None
        return frame

    def get_labels(self, record_id: str) -> dict[str, int]:
        """Return canonical labels from CODE-15%'s six boolean columns."""
        row = self._metadata_row(record_id)
        label_columns = self.config.get("label_columns", DEFAULT_LABEL_COLUMNS)
        raw_labels = [column for column in label_columns if bool(row[column])]
        return harmonize_labels(raw_labels, self.name)

    def load_signal(self, record_id: str) -> np.ndarray:
        """Load one tracing and transpose it to ``(12, samples)``."""
        row = self._metadata_row(record_id)
        trace_col = self.config.get("trace_file_column", "trace_file")
        records_root = self.root / self.config.get("records_root", ".")
        hdf5_path = records_root / str(row[trace_col])
        if not hdf5_path.exists():
            archive_path = hdf5_path.with_suffix(".zip")
            hint = f" Extract {archive_path.name} first." if archive_path.exists() else ""
            raise FileNotFoundError(f"Missing CODE15 signal part at {hdf5_path}.{hint}")

        record_value = int(row[self.config.get("record_id_column", "exam_id")])
        index = self._hdf5_index(hdf5_path)
        if record_value not in index:
            raise KeyError(f"CODE15 exam {record_id!r} is absent from {hdf5_path.name}")
        signal_key = self.config.get("signal_dataset", "tracings")
        with h5py.File(hdf5_path, "r") as handle:
            signal = np.asarray(handle[signal_key][index[record_value]], dtype=np.float32)

        layout = self.config.get("signal_layout", "samples_leads")
        if layout == "samples_leads":
            signal = signal.T
        elif layout != "leads_samples":
            raise ValueError("signal_layout must be 'samples_leads' or 'leads_samples'")
        if signal.ndim != 2 or signal.shape[0] != 12:
            raise ValueError(f"CODE15 exam {record_id!r} has unexpected shape {signal.shape}")
        return signal.copy()

    def _hdf5_index(self, path: Path) -> dict[int, int]:
        """Build an exam-ID to row mapping once per HDF5 part."""
        if path not in self._hdf5_indices:
            with h5py.File(path, "r") as handle:
                exam_ids = np.asarray(handle["exam_id"][:], dtype=np.int64)
            non_sentinel = np.flatnonzero(exam_ids != 0)
            ids = exam_ids[non_sentinel]
            if np.unique(ids).size != ids.size:
                raise ValueError(f"Duplicate non-sentinel exam IDs in {path}")
            self._hdf5_indices[path] = {
                int(exam_id): int(row) for exam_id, row in zip(ids, non_sentinel, strict=True)
            }
        return self._hdf5_indices[path]
