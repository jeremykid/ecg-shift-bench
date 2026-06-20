"""PTB-XL dataset loader.

PTB-XL is a large 12-lead ECG dataset with 21,799 clinical records from
9 years (2007–2017) of daily clinical routine at the Physikalisch-Technische
Bundesanstalt (PTB) in Germany.

Reference:
    Wagner et al. "PTB-XL, a large publicly available electrocardiography
    dataset." Scientific Data 7 (2020): 154.
    https://doi.org/10.1038/s41597-020-0495-6
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import ast
import numpy as np
import pandas as pd
import wfdb

from ecgshiftbench.datasets.base import DatasetInfo, ECGDataset

# Superclasses mapped to a canonical 5-class label set
_SUPERCLASS_LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]


class PTBXLDataset(ECGDataset):
    """Loader for the PTB-XL dataset.

    Directory layout expected under *root*::

        <root>/
          ptbxl_database.csv
          scp_statements.csv
          records100/   (100 Hz versions)
          records500/   (500 Hz versions)

    Args:
        root: Path to the PTB-XL root directory.
        split: One of ``'train'``, ``'val'``, ``'test'``.
            PTB-XL uses pre-defined folds (1–10); fold 10 → test, fold 9 → val,
            folds 1–8 → train.
        sampling_rate: ``100`` or ``500`` Hz.  Defaults to ``500``.
        transform: Optional signal transform.
        target_transform: Optional label transform.
        labels: Subset of the 5 superclass labels to use.
    """

    info = DatasetInfo(
        name="PTB-XL",
        num_leads=12,
        sampling_rate=500,
        duration_seconds=10.0,
        labels=_SUPERCLASS_LABELS,
        source_country="Germany",
        year=2020,
    )

    def __init__(
        self,
        root: str,
        split: str = "train",
        sampling_rate: int = 500,
        transform=None,
        target_transform=None,
        labels: Optional[Sequence[str]] = None,
    ) -> None:
        if sampling_rate not in (100, 500):
            raise ValueError(f"sampling_rate must be 100 or 500, got {sampling_rate}")
        self._sampling_rate = sampling_rate
        self.info = DatasetInfo(
            name="PTB-XL",
            num_leads=12,
            sampling_rate=sampling_rate,
            duration_seconds=10.0,
            labels=_SUPERCLASS_LABELS,
            source_country="Germany",
            year=2020,
        )
        super().__init__(
            root=root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            labels=labels,
        )

    def load_metadata(self) -> None:
        db_path = os.path.join(self.root, "ptbxl_database.csv")
        scp_path = os.path.join(self.root, "scp_statements.csv")

        df = pd.read_csv(db_path, index_col="ecg_id")
        df.scp_codes = df.scp_codes.apply(ast.literal_eval)

        scp_df = pd.read_csv(scp_path, index_col=0)
        superclass_map: dict[str, str] = scp_df[scp_df.diagnostic == 1].diagnostic_class.to_dict()

        def _label_vector(scp_codes: dict) -> dict:
            vec = {lbl: 0 for lbl in _SUPERCLASS_LABELS}
            for code, likelihood in scp_codes.items():
                if likelihood >= 50 and code in superclass_map:
                    vec[superclass_map[code]] = 1
            return vec

        label_df = df.scp_codes.apply(_label_vector).apply(pd.Series)
        df = pd.concat([df, label_df], axis=1)

        # Split by pre-defined strat_fold
        if self.split == "test":
            df = df[df.strat_fold == 10]
        elif self.split == "val":
            df = df[df.strat_fold == 9]
        else:
            df = df[df.strat_fold <= 8]

        df = df.reset_index()
        df["record_id"] = df["ecg_id"].astype(str)
        self._metadata = df
        self.info.num_samples = len(df)

    def load_signal(self, index: int) -> np.ndarray:
        row = self._metadata.iloc[index]
        if self._sampling_rate == 500:
            fname = row.get("filename_hr", row.get("filename_lr", None))
        else:
            fname = row.get("filename_lr", row.get("filename_hr", None))
        path = os.path.join(self.root, fname)
        record = wfdb.rdsamp(path)
        signal = record[0].T.astype(np.float32)  # (12, T)
        return signal
