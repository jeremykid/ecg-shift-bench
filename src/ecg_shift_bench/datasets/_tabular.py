"""Shared helpers for the initial tabular-metadata loader skeletons."""

from __future__ import annotations

import ast
from typing import Any

import numpy as np
import pandas as pd

from ecg_shift_bench.datasets.base import BaseECGDataset
from ecg_shift_bench.labels.harmonize import harmonize_labels


class TabularSkeletonDataset(BaseECGDataset):
    """Common metadata behavior; subclasses define source-specific signal I/O."""

    metadata_file: str

    def load_metadata(self) -> pd.DataFrame:
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
        required = {
            self.config.get("record_id_column", "record_id"),
            self.config.get("patient_id_column", "patient_id"),
            self.config.get("label_column", "labels"),
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        self._metadata = frame
        return frame

    def load_signal(self, record_id: str) -> np.ndarray:
        raise NotImplementedError(
            f"TODO: implement and validate {self.name} signal loading for record {record_id!r}. "
            f"Expected records under {self.root / self.config.get('records_root', 'records')}."
        )

    def get_labels(self, record_id: str) -> dict[str, int]:
        frame = self.load_metadata()
        record_col = self.config.get("record_id_column", "record_id")
        label_col = self.config.get("label_column", "labels")
        rows = frame.loc[frame[record_col].astype(str) == str(record_id), label_col]
        if rows.empty:
            raise KeyError(f"Unknown {self.name} record: {record_id}")
        return harmonize_labels(_parse_labels(rows.iloc[0]), self.name)


def _parse_labels(value: Any) -> list[str]:
    """Parse common list, delimited-string, and PTB-XL dictionary encodings."""
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [str(item).strip() for item in value]
    if pd.isna(value):
        return []
    text = str(value).strip()
    if text.startswith(("[", "{", "(")):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, dict):
                return [str(key) for key, score in parsed.items() if float(score) > 0]
            if isinstance(parsed, (list, tuple, set)):
                return [str(item) for item in parsed]
        except (ValueError, SyntaxError, TypeError):
            pass
    separator = ";" if ";" in text else ","
    return [item.strip() for item in text.split(separator) if item.strip()]
