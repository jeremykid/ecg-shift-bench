"""Patient-grouped train/validation/test splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


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


def patient_level_stratified_split(
    patient_table: pd.DataFrame,
    *,
    patient_column: str = "patient_id",
    strata_column: str = "stratum",
    validation_size: float = 0.1,
    test_size: float = 0.2,
    seed: int = 42,
    ) -> tuple[dict[str, set[object]], bool]:
    """Split patient IDs with label-aware stratification when possible.

    The input must contain one row per patient. If stratification is not feasible
    because some strata are too small, fall back to the same deterministic random
    split used by :func:`patient_level_split`.
    """
    if patient_column not in patient_table:
        raise KeyError(f"Missing patient column: {patient_column}")
    if strata_column not in patient_table:
        raise KeyError(f"Missing strata column: {strata_column}")
    if patient_table[patient_column].isna().any():
        raise ValueError("Patient identifiers must not be missing")
    if patient_table[strata_column].isna().any():
        raise ValueError("Stratification labels must not be missing")
    if validation_size < 0 or test_size < 0 or validation_size + test_size >= 1:
        raise ValueError("validation_size and test_size must be non-negative and sum to < 1")

    table = patient_table[[patient_column, strata_column]].drop_duplicates(subset=[patient_column])
    if table.empty:
        return {"train": set(), "validation": set(), "test": set()}, True

    try:
        trainval, test = train_test_split(
            table,
            test_size=test_size,
            random_state=seed,
            stratify=table[strata_column] if table[strata_column].nunique() > 1 else None,
        )
        if validation_size == 0:
            train = trainval
            validation = table.iloc[0:0].copy()
        else:
            relative_validation = validation_size / (1.0 - test_size)
            train, validation = train_test_split(
                trainval,
                test_size=relative_validation,
                random_state=seed,
                stratify=trainval[strata_column] if trainval[strata_column].nunique() > 1 else None,
            )
    except ValueError:
        return _random_patient_id_split(
            table,
            patient_column=patient_column,
            validation_size=validation_size,
            test_size=test_size,
            seed=seed,
        ), False

    return {
        "train": set(train[patient_column].tolist()),
        "validation": set(validation[patient_column].tolist()),
        "test": set(test[patient_column].tolist()),
    }, True


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


def _random_patient_id_split(
    patient_table: pd.DataFrame,
    *,
    patient_column: str,
    validation_size: float,
    test_size: float,
    seed: int,
) -> dict[str, set[object]]:
    """Fallback deterministic patient-ID split without stratification."""
    patients = patient_table[patient_column].drop_duplicates().to_numpy(copy=True)
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
    return {
        "train": train_patients,
        "validation": validation_patients,
        "test": test_patients,
    }
