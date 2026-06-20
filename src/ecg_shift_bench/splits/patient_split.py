"""Patient-grouped train/validation/test splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def patient_level_split(
    metadata: pd.DataFrame,
    *,
    patient_column: str = "patient_id",
    validation_size: float = 0.1,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Split records by patient so a patient occurs in exactly one subset."""
    if patient_column not in metadata:
        raise KeyError(f"Missing patient column: {patient_column}")
    if metadata[patient_column].isna().any():
        raise ValueError("Patient identifiers must not be missing")
    if validation_size < 0 or test_size < 0 or validation_size + test_size >= 1:
        raise ValueError("validation_size and test_size must be non-negative and sum to < 1")

    patients = metadata[patient_column].drop_duplicates().to_numpy(copy=True)
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    n_patients = len(patients)
    n_test = int(round(n_patients * test_size))
    n_validation = int(round(n_patients * validation_size))
    if test_size > 0 and n_patients > 1:
        n_test = max(1, n_test)
    if validation_size > 0 and n_patients - n_test > 1:
        n_validation = max(1, n_validation)
    if n_test + n_validation >= n_patients and n_patients > 0:
        n_validation = max(0, n_patients - n_test - 1)

    test_patients = set(patients[:n_test])
    validation_patients = set(patients[n_test : n_test + n_validation])
    train_patients = set(patients[n_test + n_validation :])

    def select(patient_ids: set[object]) -> pd.DataFrame:
        selected = metadata.loc[metadata[patient_column].isin(patient_ids)].copy()
        return selected.reset_index(drop=True)

    return {
        "train": select(train_patients),
        "validation": select(validation_patients),
        "test": select(test_patients),
    }


def assert_no_patient_overlap(
    splits: dict[str, pd.DataFrame], patient_column: str = "patient_id"
) -> None:
    """Raise if any patient identifier occurs in more than one split."""
    names = list(splits)
    patient_sets = {name: set(frame[patient_column]) for name, frame in splits.items()}
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = patient_sets[left].intersection(patient_sets[right])
            if overlap:
                raise ValueError(f"Patient leakage between {left} and {right}: {sorted(overlap)!r}")
