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

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._record_rows: dict[str, int] | None = None

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
        record_col = self.config.get("record_id_column", "record_id")
        if frame[record_col].isna().any():
            raise ValueError(f"{path} contains missing record identifiers")
        if frame[record_col].astype(str).duplicated().any():
            raise ValueError(f"{path} contains duplicate record identifiers")
        self._metadata = frame
        self._record_rows = None
        return frame

    def load_signal(self, record_id: str) -> np.ndarray:
        raise NotImplementedError(
            f"TODO: implement and validate {self.name} signal loading for record {record_id!r}. "
            f"Expected records under {self.root / self.config.get('records_root', 'records')}."
        )

    def get_labels(self, record_id: str) -> dict[str, int]:
        label_col = self.config.get("label_column", "labels")
        row = self._metadata_row(record_id)
        return harmonize_labels(_parse_labels(row[label_col]), self.name)

    def _metadata_row(self, record_id: str) -> pd.Series:
        """Return one metadata row using a lazily built string-keyed index."""
        frame = self.load_metadata()
        record_col = self.config.get("record_id_column", "record_id")
        if self._record_rows is None:
            self._record_rows = {
                str(value): index for index, value in enumerate(frame[record_col].tolist())
            }
        try:
            row_number = self._record_rows[str(record_id)]
        except KeyError as error:
            raise KeyError(f"Unknown {self.name} record: {record_id}") from error
        return frame.iloc[row_number]


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
                # PTB-XL rhythm/form statements commonly carry score 0 because
                # confidence is not applicable. Dictionary presence is the label.
                return [str(key) for key in parsed]
            if isinstance(parsed, (list, tuple, set)):
                return [str(item) for item in parsed]
        except (ValueError, SyntaxError, TypeError):
            pass
    separator = ";" if ";" in text else ","
    return [item.strip() for item in text.split(separator) if item.strip()]
