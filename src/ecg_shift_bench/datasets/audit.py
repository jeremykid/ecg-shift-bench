"""Dataset audit and split manifest helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ecg_shift_bench.datasets.alignment import CANONICAL_LEAD_ORDER, canonical_unit_name
from ecg_shift_bench.datasets.base import BaseECGDataset
from ecg_shift_bench.labels.canonical import CANONICAL_LABELS
from ecg_shift_bench.splits.patient_split import (
    assert_no_patient_overlap,
    patient_level_stratified_split,
)

DEFAULT_SEED = 42
DEFAULT_VALIDATION_SIZE = 0.1
DEFAULT_TEST_SIZE = 0.2


@dataclass(frozen=True)
class DatasetAuditResult:
    """Audit outputs for one dataset."""

    dataset: str
    audit: dict[str, Any]
    split_manifest: pd.DataFrame
    exclusions: pd.DataFrame
    reproducibility: dict[str, Any]


def audit_dataset(
    dataset: BaseECGDataset,
    *,
    waveform_check_limit: int | None = 0,
) -> DatasetAuditResult:
    """Audit metadata, labels, optional waveforms, and standardized splits."""
    metadata = dataset.load_metadata()
    record_col = str(dataset.config.get("record_id_column", "record_id"))
    patient_col = dataset.config.get("patient_id_column")
    records_total = len(metadata)
    exclusions = _waveform_exclusions(dataset, metadata, record_col, waveform_check_limit)
    excluded_ids = set(exclusions["record_id"].astype(str).tolist()) if not exclusions.empty else set()
    usable_metadata = metadata.loc[~metadata[record_col].astype(str).isin(excluded_ids)].copy()
    split_manifest, split_policy = build_split_manifest(dataset, usable_metadata)
    label_summary = _label_summary(dataset, usable_metadata, record_col)
    split_label_summary = _split_label_summary(dataset, split_manifest)
    patient_count = (
        int(usable_metadata[str(patient_col)].nunique(dropna=True))
        if patient_col and str(patient_col) in usable_metadata.columns
        else None
    )
    records_excluded = int(exclusions["record_id"].nunique()) if not exclusions.empty else 0
    audit = {
        "dataset": dataset.name,
        "domain": dataset.domain,
        "records_total": int(records_total),
        "records_usable": int(len(usable_metadata)),
        "records_excluded": records_excluded,
        "patients_total": patient_count,
        "available_labels": label_summary["available_labels"],
        "missing_labels": label_summary["missing_labels"],
        "positive_counts": label_summary["positive_counts"],
        "positive_prevalence": label_summary["positive_prevalence"],
        "split_label_summary": split_label_summary,
        "waveform_check": _waveform_check_summary(dataset, waveform_check_limit, records_total),
        "split_policy": split_policy,
    }
    return DatasetAuditResult(
        dataset=_dataset_key(dataset),
        audit=audit,
        split_manifest=split_manifest,
        exclusions=exclusions,
        reproducibility=_reproducibility(dataset),
    )


def build_split_manifest(
    dataset: BaseECGDataset,
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create the standardized train/validation/test manifest for one dataset."""
    record_col = str(dataset.config.get("record_id_column", "record_id"))
    patient_col = dataset.config.get("patient_id_column")
    if dataset.name.upper() == "PTBXL":
        splits = _ptbxl_official_splits(metadata)
        policy = _split_policy("official", "patient", "official_strat_fold")
        policy["stratification"] = "official_folds"
    elif patient_col and str(patient_col) in metadata.columns:
        splits, stratified = _patient_level_splits(dataset, metadata, record_col, str(patient_col))
        policy = _split_policy(
            "generated",
            "patient",
            "patient_level_stratified_signature" if stratified else "patient_level_random",
        )
        policy["stratification"] = "patient_label_signature" if stratified else "none"
        if not stratified:
            policy["split_note"] = (
                "Fallback random patient split used because stratified patient split was not feasible"
            )
    else:
        splits = _record_level_split(metadata, seed=DEFAULT_SEED)
        policy = _split_policy(
            "generated",
            "record",
            "record_level_random_no_patient_id",
            leakage_status="unavailable",
            leakage_reason="patient_id_column_unavailable",
        )
        policy["stratification"] = "record_level_random"

    manifest = _manifest_from_splits(
        dataset,
        splits,
        record_col,
        str(patient_col) if patient_col else None,
    )
    if patient_col and str(patient_col) in metadata.columns:
        assert_no_patient_overlap(splits, str(patient_col))
        policy["leakage_check"] = {"status": "passed", "patient_column": str(patient_col)}
    return manifest, policy


def write_alignment_audit_outputs(
    result: DatasetAuditResult,
    output_dir: str | Path,
) -> dict[str, str]:
    """Write audit artifacts under ``outputs/alignment/<dataset>/``."""
    dataset_dir = Path(output_dir) / result.dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "audit": dataset_dir / "audit.json",
        "split_manifest": dataset_dir / "split_manifest.csv",
        "exclusions": dataset_dir / "exclusions.csv",
        "reproducibility": dataset_dir / "reproducibility.json",
    }
    _write_json(paths["audit"], result.audit)
    result.split_manifest.to_csv(paths["split_manifest"], index=False)
    result.exclusions.to_csv(paths["exclusions"], index=False)
    _write_json(paths["reproducibility"], result.reproducibility)
    return {key: str(path) for key, path in paths.items()}


def _dataset_key(dataset: BaseECGDataset) -> str:
    return dataset.name.lower().replace("-", "").replace("%", "")


def _waveform_exclusions(
    dataset: BaseECGDataset,
    metadata: pd.DataFrame,
    record_col: str,
    limit: int | None,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    if limit == 0:
        return pd.DataFrame(columns=["record_id", "reason"])
    records = metadata[record_col].astype(str)
    if limit is not None and limit > 0:
        records = records.head(limit)
    for record_id in records:
        try:
            _validate_aligned_signal(dataset.load_aligned_signal(record_id), dataset)
        except Exception as error:  # noqa: BLE001 - audit must record all exclusion reasons.
            rows.append({"record_id": record_id, "reason": str(error)})
    return pd.DataFrame(rows, columns=["record_id", "reason"])


def _waveform_check_summary(
    dataset: BaseECGDataset,
    waveform_check_limit: int | None,
    records_total: int,
) -> dict[str, Any]:
    source_unit = canonical_unit_name(str(dataset.config.get("source_unit", "mV")))
    target_unit = canonical_unit_name(str(dataset.config.get("target_unit", "mV")))
    if waveform_check_limit is None:
        checked_records = records_total
        mode = "full"
    else:
        checked_records = min(max(waveform_check_limit, 0), records_total)
        mode = "metadata_only" if waveform_check_limit == 0 else "sampled"
    return {
        "checked_records": checked_records,
        "mode": mode,
        "target_sampling_rate": int(dataset.config.get("target_sampling_rate", 500)),
        "target_length": int(dataset.config.get("target_length", 5000)),
        "lead_order": dataset.config.get("lead_order", CANONICAL_LEAD_ORDER),
        "source_unit": source_unit,
        "target_unit": target_unit,
        "unit_converted": source_unit != target_unit,
    }


def _validate_aligned_signal(signal: np.ndarray, dataset: BaseECGDataset) -> None:
    target_length = int(dataset.config.get("target_length", 5000))
    expected_shape = (12, target_length)
    if signal.shape != expected_shape:
        raise ValueError(f"expected aligned shape {expected_shape}, got {signal.shape}")
    if signal.dtype != np.float32:
        raise ValueError(f"expected float32 aligned signal, got {signal.dtype}")
    if not np.isfinite(signal).all():
        raise ValueError("aligned signal contains non-finite values")


def _split_label_summary(
    dataset: BaseECGDataset,
    split_manifest: pd.DataFrame,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split_name, frame in split_manifest.groupby("split", sort=False):
        positive_counts = {label: 0 for label in CANONICAL_LABELS}
        for record_id in frame["record_id"].astype(str):
            labels = dataset.get_labels(record_id)
            for label in CANONICAL_LABELS:
                positive_counts[label] += int(labels[label])
        total = len(frame)
        summary[str(split_name)] = {
            "records": int(total),
            "patients": int(frame["patient_id"].nunique()) if "patient_id" in frame.columns else None,
            "positive_counts": positive_counts,
            "positive_prevalence": {
                label: (count / total if total else 0.0)
                for label, count in positive_counts.items()
            },
        }
    return summary


def _patient_level_splits(
    dataset: BaseECGDataset,
    metadata: pd.DataFrame,
    record_col: str,
    patient_col: str,
) -> tuple[dict[str, pd.DataFrame], bool]:
    patient_rows = _patient_label_table(dataset, metadata, record_col, patient_col)
    patient_ids, stratified = patient_level_stratified_split(
        patient_rows,
        patient_column=patient_col,
        strata_column="stratum",
        validation_size=DEFAULT_VALIDATION_SIZE,
        test_size=DEFAULT_TEST_SIZE,
        seed=DEFAULT_SEED,
    )

    def select(patient_ids_subset: set[object]) -> pd.DataFrame:
        selected = metadata.loc[metadata[patient_col].isin(patient_ids_subset)].copy()
        return selected.reset_index(drop=True)

    splits = {name: select(ids) for name, ids in patient_ids.items()}
    assert_no_patient_overlap(splits, patient_col)
    return splits, stratified


def _patient_label_table(
    dataset: BaseECGDataset,
    metadata: pd.DataFrame,
    record_col: str,
    patient_col: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for patient_id, frame in metadata.groupby(patient_col, sort=False):
        patient_labels = {label: 0 for label in CANONICAL_LABELS}
        for record_id in frame[record_col].astype(str):
            labels = dataset.get_labels(record_id)
            for label in CANONICAL_LABELS:
                patient_labels[label] = max(patient_labels[label], int(labels[label]))
        rows.append(
            {
                patient_col: patient_id,
                "stratum": _label_signature(patient_labels),
            }
        )
    return pd.DataFrame(rows, columns=[patient_col, "stratum"])


def _label_signature(labels: dict[str, int]) -> str:
    positives = [label for label in CANONICAL_LABELS if int(labels.get(label, 0))]
    return "NORMAL" if not positives else "+".join(positives)


def _label_summary(
    dataset: BaseECGDataset,
    metadata: pd.DataFrame,
    record_col: str,
) -> dict[str, Any]:
    positive_counts = {label: 0 for label in CANONICAL_LABELS}
    for record_id in metadata[record_col].astype(str):
        labels = dataset.get_labels(record_id)
        for label in CANONICAL_LABELS:
            positive_counts[label] += int(labels[label])
    total = len(metadata)
    return {
        "available_labels": list(CANONICAL_LABELS),
        "missing_labels": _missing_label_counts(dataset, metadata),
        "positive_counts": positive_counts,
        "positive_prevalence": {
            label: (count / total if total else 0.0) for label, count in positive_counts.items()
        },
    }


def _missing_label_counts(dataset: BaseECGDataset, metadata: pd.DataFrame) -> dict[str, int]:
    if "label_columns" in dataset.config:
        native_to_canonical = {
            "AF": "AF",
            "RBBB": "RBBB",
            "LBBB": "LBBB",
            "1dAVb": "1dAVB",
            "SB": "SB",
            "ST": "ST",
        }
        counts = {label: 0 for label in CANONICAL_LABELS}
        for native, canonical in native_to_canonical.items():
            if native in metadata:
                counts[canonical] = int(metadata[native].isna().sum())
            else:
                counts[canonical] = len(metadata)
        return counts
    label_col = dataset.config.get("label_column")
    if label_col and str(label_col) in metadata:
        missing = int(metadata[str(label_col)].isna().sum())
    else:
        missing = 0
    return {label: missing for label in CANONICAL_LABELS}


def _ptbxl_official_splits(metadata: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if "strat_fold" not in metadata.columns:
        raise ValueError("PTB-XL metadata is missing official strat_fold")
    folds = metadata["strat_fold"].astype(int)
    return {
        "train": metadata.loc[folds.isin(range(1, 9))].copy().reset_index(drop=True),
        "validation": metadata.loc[folds == 9].copy().reset_index(drop=True),
        "test": metadata.loc[folds == 10].copy().reset_index(drop=True),
    }


def _record_level_split(metadata: pd.DataFrame, *, seed: int) -> dict[str, pd.DataFrame]:
    shuffled = metadata.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    total = len(shuffled)
    n_test = int(round(total * DEFAULT_TEST_SIZE))
    n_validation = int(round(total * DEFAULT_VALIDATION_SIZE))
    if DEFAULT_TEST_SIZE > 0 and total > 1:
        n_test = max(1, n_test)
    if DEFAULT_VALIDATION_SIZE > 0 and total - n_test > 1:
        n_validation = max(1, n_validation)
    if n_test + n_validation >= total and total > 0:
        n_validation = max(0, total - n_test - 1)
    return {
        "train": shuffled.iloc[n_test + n_validation :].copy().reset_index(drop=True),
        "validation": shuffled.iloc[n_test : n_test + n_validation].copy().reset_index(drop=True),
        "test": shuffled.iloc[:n_test].copy().reset_index(drop=True),
    }


def _manifest_from_splits(
    dataset: BaseECGDataset,
    splits: dict[str, pd.DataFrame],
    record_col: str,
    patient_col: str | None,
) -> pd.DataFrame:
    frames = []
    for split, frame in splits.items():
        columns = pd.DataFrame(
            {
                "record_id": frame[record_col].astype(str),
                "split": split,
                "domain": dataset.domain,
            }
        )
        if patient_col and patient_col in frame.columns:
            columns.insert(1, "patient_id", frame[patient_col].astype(str))
        frames.append(columns)
    return pd.concat(frames, ignore_index=True)


def _split_policy(
    split_source: str,
    split_level: str,
    method: str,
    *,
    leakage_status: str = "pending",
    leakage_reason: str | None = None,
) -> dict[str, Any]:
    leakage_check = {"status": leakage_status}
    if leakage_reason:
        leakage_check["reason"] = leakage_reason
    return {
        "split_source": split_source,
        "split_level": split_level,
        "method": method,
        "split_algorithm": method,
        "train_fraction": 0.7 if split_source == "generated" else None,
        "validation_fraction": DEFAULT_VALIDATION_SIZE if split_source == "generated" else None,
        "test_fraction": DEFAULT_TEST_SIZE if split_source == "generated" else None,
        "seed": DEFAULT_SEED if split_source == "generated" else None,
        "stratification": "none",
        "leakage_check": leakage_check,
    }


def _reproducibility(dataset: BaseECGDataset) -> dict[str, Any]:
    source_rate = int(dataset.config.get("sampling_rate", 500))
    target_rate = int(dataset.config.get("target_sampling_rate", 500))
    target_length = int(dataset.config.get("target_length", 5000))
    source_unit = canonical_unit_name(str(dataset.config.get("source_unit", "mV")))
    target_unit = canonical_unit_name(str(dataset.config.get("target_unit", "mV")))
    return {
        "dataset": dataset.name,
        "domain": dataset.domain,
        "source_sampling_rate": source_rate,
        "target_sampling_rate": target_rate,
        "target_length": target_length,
        "lead_order": dataset.config.get("lead_order", CANONICAL_LEAD_ORDER),
        "source_unit": source_unit,
        "target_unit": target_unit,
        "unit_converted": source_unit != target_unit,
        "resampling_method": "polyphase_resample_signal",
        "resampling_note": (
            "Resampling standardizes the temporal grid for a common model input contract; "
            "it does not add new physiological information."
        ),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
