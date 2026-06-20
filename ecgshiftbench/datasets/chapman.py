"""Chapman-Shaoxing 12-lead ECG dataset loader.

The Chapman-Shaoxing dataset contains 10,646 12-lead ECGs collected at
Chapman University Hospital and Shaoxing People's Hospital, China.

Reference:
    Zheng et al. "A 12-lead electrocardiogram database for arrhythmia
    research covering more than 10,000 patients." Scientific Data 7 (2020): 48.
    https://doi.org/10.1038/s41597-020-0386-x
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import scipy.io as sio

from ecgshiftbench.datasets.base import DatasetInfo, ECGDataset

# Rhythm labels present in the Chapman dataset
_RHYTHM_LABELS = [
    "AFIB",
    "GSVT",
    "SB",
    "SR",
]


class ChapmanDataset(ECGDataset):
    """Loader for the Chapman-Shaoxing 12-lead ECG dataset.

    Directory layout expected under *root*::

        <root>/
          AttributesDictionary.csv     (mapping of numerical codes → label names)
          Diagnostics.csv              (per-recording rhythm labels)
          ECGData/                     (raw ECG files, one CSV per recording)

    Args:
        root: Path to the Chapman root directory.
        split: One of ``'train'``, ``'val'``, ``'test'``.
            Records are split 70 / 10 / 20 by the ordered record index.
        transform: Optional signal transform.
        target_transform: Optional label transform.
        labels: Subset of rhythm labels to use.
    """

    info = DatasetInfo(
        name="Chapman-Shaoxing",
        num_leads=12,
        sampling_rate=500,
        duration_seconds=10.0,
        labels=_RHYTHM_LABELS,
        source_country="China",
        year=2020,
    )

    def load_metadata(self) -> None:
        diag_path = os.path.join(self.root, "Diagnostics.csv")
        df = pd.read_csv(diag_path)
        df.columns = df.columns.str.strip()
        df["record_id"] = df["FileName"].astype(str).str.strip()

        for lbl in _RHYTHM_LABELS:
            df[lbl] = (df["Rhythm"].str.strip() == lbl).astype(int)

        n = len(df)
        n_test = int(n * 0.2)
        n_val = int(n * 0.1)
        n_train = n - n_test - n_val

        if self.split == "train":
            df = df.iloc[:n_train]
        elif self.split == "val":
            df = df.iloc[n_train : n_train + n_val]
        else:
            df = df.iloc[n_train + n_val :]

        self._metadata = df.reset_index(drop=True)
        self.info.num_samples = len(df)

    def load_signal(self, index: int) -> np.ndarray:
        row = self._metadata.iloc[index]
        record_id = row["record_id"]
        ecg_dir = os.path.join(self.root, "ECGData")
        path = os.path.join(ecg_dir, f"{record_id}.csv")
        data = pd.read_csv(path, header=None).values.T.astype(np.float32)
        return data  # (12, T)
