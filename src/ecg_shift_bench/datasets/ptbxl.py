"""PTB-XL metadata and WFDB signal loader.

Expected: ``ptbxl_database.csv`` plus WFDB records in ``records500/`` (or
``records100/``). The adapter uses the record path stored in the metadata and
returns physical signals in mV with canonical 12-lead ordering.
"""

from pathlib import Path
from typing import Any

import numpy as np

from ecg_shift_bench.datasets._tabular import TabularSkeletonDataset

CANONICAL_LEAD_ORDER = [
    "I",
    "II",
    "III",
    "AVR",
    "AVL",
    "AVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]

DEFAULT_CONFIG: dict[str, Any] = {
    "metadata_file": "ptbxl_database.csv",
    "record_id_column": "ecg_id",
    "patient_id_column": "patient_id",
    "label_column": "scp_codes",
    "record_path_column": "filename_hr",
    "sampling_rate": 500,
}


class PTBXLDataset(TabularSkeletonDataset):
    """PTB-XL adapter for the official versioned PhysioNet layout."""

    name = "PTBXL"
    domain = "ptbxl_germany"
    metadata_file = "ptbxl_database.csv"

    def __init__(self, root: str | Path, config: dict[str, Any] | None = None) -> None:
        merged_config = {**DEFAULT_CONFIG, **(config or {})}
        super().__init__(root, merged_config)

    def load_signal(self, record_id: str) -> np.ndarray:
        """Load one high- or low-resolution WFDB record as ``(12, samples)``."""
        import wfdb

        row = self._metadata_row(record_id)
        path_column = self.config.get("record_path_column", "filename_hr")
        if path_column not in row.index:
            raise ValueError(f"PTB-XL metadata is missing record path column {path_column!r}")
        record_path = self.root / str(row[path_column])
        signal, fields = wfdb.rdsamp(str(record_path))
        signal = np.asarray(signal, dtype=np.float32)
        lead_names = [str(name).upper() for name in fields["sig_name"]]
        missing = set(CANONICAL_LEAD_ORDER).difference(lead_names)
        if missing:
            raise ValueError(f"PTB-XL record {record_id!r} is missing leads: {sorted(missing)}")
        indices = [lead_names.index(lead) for lead in CANONICAL_LEAD_ORDER]
        expected_rate = self.config.get("sampling_rate")
        if expected_rate is not None and int(fields["fs"]) != int(expected_rate):
            raise ValueError(
                f"PTB-XL record {record_id!r} has sampling rate {fields['fs']}, "
                f"expected {expected_rate}"
            )
        return signal[:, indices].T.copy()
