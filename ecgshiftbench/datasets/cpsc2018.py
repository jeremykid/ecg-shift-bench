"""CPSC 2018 dataset loader.

The China Physiological Signal Challenge (CPSC) 2018 dataset contains
6,877 12-lead ECG recordings labelled with nine rhythm/morphology classes.

Reference:
    Liu et al. "An open access database for evaluating the algorithms
    of electrocardiogram rhythm and morphology abnormality detection."
    Journal of Medical Imaging and Health Informatics 8 (2018): 1368–1373.
    https://doi.org/10.1166/jmihi.2018.2442
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import scipy.io as sio

from ecgshiftbench.datasets.base import DatasetInfo, ECGDataset

_LABELS = [
    "Normal",
    "AF",
    "I-AVB",
    "LBBB",
    "RBBB",
    "PAC",
    "PVC",
    "STD",
    "STE",
]


class CPSC2018Dataset(ECGDataset):
    """Loader for the CPSC 2018 12-lead ECG dataset.

    Directory layout expected under *root*::

        <root>/
          REFERENCE.csv      (columns: Recording, label_1, label_2, …)
          data/              (MATLAB .mat files, one per recording)

    Args:
        root: Path to the CPSC 2018 root directory.
        split: One of ``'train'``, ``'val'``, ``'test'``.
            Split proportions: 70 / 10 / 20 by ordered record index.
        transform: Optional signal transform.
        target_transform: Optional label transform.
        labels: Subset of the nine class labels to use.
    """

    info = DatasetInfo(
        name="CPSC2018",
        num_leads=12,
        sampling_rate=500,
        duration_seconds=10.0,
        labels=_LABELS,
        source_country="China",
        year=2018,
    )

    def load_metadata(self) -> None:
        ref_path = os.path.join(self.root, "REFERENCE.csv")
        df = pd.read_csv(ref_path)
        df.columns = df.columns.str.strip()
        df["record_id"] = df["Recording"].astype(str).str.strip()

        for lbl in _LABELS:
            if lbl in df.columns:
                df[lbl] = df[lbl].fillna(0).astype(int)
            else:
                df[lbl] = 0

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
        mat_path = os.path.join(self.root, "data", f"{record_id}.mat")
        mat = sio.loadmat(mat_path)
        # CPSC mat files store signals in key 'val'
        signal = mat["val"].astype(np.float32)  # (12, T)
        return signal
