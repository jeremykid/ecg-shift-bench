"""Dataset audit and split manifest helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ecg_shift_bench.datasets.alignment import CANONICAL_LEAD_ORDER
from ecg_shift_bench.datasets.base import BaseECGDataset
from ecg_shift_bench.labels.canonical import CANONICAL_LABELS
from ecg_shift_bench.splits.patient_split import assert_no_patient_overlap, patient_level_split

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
    waveform_check_limit: int = 0,
) -> DatasetAuditResult:
    """Audit metadata, labels, optional waveforms, and standardized splits."""
    metadata = dataset.load_metadata()
    record_col = str(dataset.config.get("record_id_column", "record_id"))
    patient_col = dataset.config.get("patient_id_column")
    records_total = len(metadata)
    exclusions = _waveform_exclusions(dataset, metadata, record_col, waveform_check_limit)
    split_manifest, split_policy = build_split_manifest(dataset, metadata)
    label_summary = _label_summary(dataset, metadata, record_col)
    patient_count = (
        int(metadata[str(patient_col)].nunique(dropna=True))
        if patient_col and str(patient_col) in metadata.columns
        else None
    )
    records_excluded = int(exclusions["record_id"].nunique()) if not exclusions.empty else 0
    audit = {
        "dataset": dataset.name,
        "domain": dataset.domain,
        "records_total": int(records_total),
        "records_usable": int(records_total - records_excluded),
        "records_excluded": records_excluded,
        "patients_total": patient_count,
        "available_labels": label_summary["available_labels"],
        "missing_labels": label_summary["missing_labels"],
        "positive_counts": label_summary["positive_counts"],
        "positive_prevalence": label_summary["positive_prevalence"],
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
    elif patient_col and str(patient_col) in metadata.columns:
        splits = patient_level_split(
            metadata,
            patient_column=str(patient_col),
            validation_size=DEFAULT_VALIDATION_SIZE,
            test_size=DEFAULT_TEST_SIZE,
            seed=DEFAULT_SEED,
        )
        policy = _split_policy("generated", "patient", "patient_level_random")
    else:
        splits = _record_level_split(metadata, seed=DEFAULT_SEED)
        policy = _split_policy(
            "generated",
            "record",
            "record_level_random_no_patient_id",
            leakage_status="unavailable",
            leakage_reason="patient_id_column_unavailable",
        )

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
    limit: int,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    if limit <= 0:
        return pd.DataFrame(columns=["record_id", "reason"])
    for record_id in metadata[record_col].astype(str).head(limit):
        try:
            _validate_aligned_signal(dataset.load_aligned_signal(record_id), dataset)
        except Exception as error:  # noqa: BLE001 - audit must record all exclusion reasons.
            rows.append({"record_id": record_id, "reason": str(error)})
    return pd.DataFrame(rows, columns=["record_id", "reason"])


def _waveform_check_summary(
    dataset: BaseECGDataset,
    waveform_check_limit: int,
    records_total: int,
) -> dict[str, Any]:
    return {
        "checked_records": min(max(waveform_check_limit, 0), records_total),
        "mode": "sampled" if waveform_check_limit > 0 else "metadata_only",
        "target_sampling_rate": int(dataset.config.get("target_sampling_rate", 500)),
        "target_length": int(dataset.config.get("target_length", 5000)),
        "lead_order": dataset.config.get("lead_order", CANONICAL_LEAD_ORDER),
        "normalization": dataset.config.get("normalization", "per_lead_zscore"),
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
    if dataset.config.get("normalization", "per_lead_zscore") == "per_lead_zscore":
        means = signal.mean(axis=-1)
        stds = signal.std(axis=-1)
        if not np.allclose(means, 0.0, atol=1e-4):
            raise ValueError("aligned signal is not centered per lead")
        if not np.all((stds < 1e-6) | np.isclose(stds, 1.0, atol=1e-3)):
            raise ValueError("aligned signal is not z-scored per lead")


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
    return {
        "dataset": dataset.name,
        "domain": dataset.domain,
        "source_sampling_rate": source_rate,
        "target_sampling_rate": target_rate,
        "target_length": target_length,
        "lead_order": dataset.config.get("lead_order", CANONICAL_LEAD_ORDER),
        "normalization": dataset.config.get("normalization", "per_lead_zscore"),
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
