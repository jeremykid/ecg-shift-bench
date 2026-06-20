"""Georgia 12-lead ECG Challenge (G12EC) dataset loader.

The G12EC dataset is the training set from the 2020 PhysioNet/Computing
in Cardiology Challenge, provided by Emory University, Georgia, USA.

Reference:
    Reyna et al. "Will Two Do? Varying Dimensions in Electrocardiography:
    The PhysioNet/Computing in Cardiology Challenge 2021."
    Computing in Cardiology (2021).
    https://doi.org/10.23919/CinC53138.2021.9662687
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import wfdb

from ecgshiftbench.datasets.base import DatasetInfo, ECGDataset

# SNOMED-CT codes and their short names for the most frequent classes
# present in G12EC (2020 Challenge scored labels).
_SCORED_LABELS = [
    "270492004",   # I-AVB
    "164889003",   # AF
    "164890007",   # AFL
    "426627000",   # Brady
    "713427006",   # CRBBB
    "713426002",   # IRBBB
    "445118002",   # LQRSV
    "39732003",    # LAD
    "164909002",   # LBBB
    "251146004",   # LQRSV (alternate)
    "698252002",   # NSIVCB
    "10370003",    # PR
    "164947007",   # PAC
    "164917005",   # RBBB
    "47665007",    # RAD
    "427393003",   # SA
    "426177001",   # SB
    "426783006",   # SR
    "427084000",   # ST
    "63593006",    # SVPB
    "164934002",   # TWC
    "59931005",    # TWI
    "17338001",    # VPB
]

_LABEL_NAMES = [
    "IAVB", "AF", "AFL", "Brady", "CRBBB", "IRBBB", "LQRSV", "LAD",
    "LBBB", "LQRSV2", "NSIVCB", "PR", "PAC", "RBBB", "RAD", "SA",
    "SB", "SR", "ST", "SVPB", "TWC", "TWI", "VPB",
]

_CODE_TO_NAME = dict(zip(_SCORED_LABELS, _LABEL_NAMES))


class G12ECDataset(ECGDataset):
    """Loader for the Georgia 12-lead ECG dataset.

    Directory layout expected under *root*::

        <root>/
          <record>.hea        (WFDB header files)
          <record>.mat        (WFDB data files)

    Args:
        root: Path to the G12EC root directory.
        split: One of ``'train'``, ``'val'``, ``'test'``.
        transform: Optional signal transform.
        target_transform: Optional label transform.
        labels: Subset of label names to use.
    """

    info = DatasetInfo(
        name="G12EC",
        num_leads=12,
        sampling_rate=500,
        duration_seconds=10.0,
        labels=_LABEL_NAMES,
        source_country="USA",
        year=2020,
    )

    def load_metadata(self) -> None:
        records = sorted(
            [
                os.path.splitext(f)[0]
                for f in os.listdir(self.root)
                if f.endswith(".hea")
            ]
        )

        rows = []
        for record_id in records:
            hea_path = os.path.join(self.root, record_id)
            header = wfdb.rdheader(hea_path)
            codes_str = ""
            for comment in header.comments:
                if comment.startswith("Dx:"):
                    codes_str = comment.split(":", 1)[1].strip()
                    break
            present_codes = set(codes_str.split(","))
            row: dict = {"record_id": record_id}
            for code, name in _CODE_TO_NAME.items():
                row[name] = int(code in present_codes)
            rows.append(row)

        df = pd.DataFrame(rows)
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
        record_path = os.path.join(self.root, row["record_id"])
        record = wfdb.rdsamp(record_path)
        signal = record[0].T.astype(np.float32)  # (12, T)
        return signal
