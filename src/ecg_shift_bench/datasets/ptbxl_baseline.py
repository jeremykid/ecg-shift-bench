"""Immutable PTB-XL snapshot validation and training dataset utilities."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ecg_shift_bench.datasets._tabular import _parse_labels
from ecg_shift_bench.datasets.ptbxl import PTBXLDataset
from ecg_shift_bench.labels.canonical import CANONICAL_LABELS
from ecg_shift_bench.labels.harmonize import harmonize_labels

SPLIT_FOLDS = {
    "train": tuple(range(1, 9)),
    "validation": (9,),
    "test": (10,),
}


@dataclass(frozen=True)
class PreparedPTBXLSnapshot:
    """Validated metadata, split tables, and immutable identity details."""

    snapshot_id: str
    splits: dict[str, pd.DataFrame]
    split_manifest: pd.DataFrame
    split_manifest_sha256: str
    summary: dict[str, Any]
    source_hashes: dict[str, str]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_split_manifest_bytes(manifest: pd.DataFrame) -> bytes:
    """Serialize the record-to-patient split map for stable hashing."""
    required = ["ecg_id", "patient_id", "split"]
    if list(manifest.columns) != required:
        raise ValueError(f"Split manifest columns must be exactly {required}")
    lines = ["ecg_id,patient_id,split\n"]
    ordered = manifest.sort_values("ecg_id")
    for row in ordered.itertuples(index=False):
        lines.append(f"{int(row.ecg_id)},{int(row.patient_id)},{row.split}\n")
    return "".join(lines).encode("utf-8")


def _target_vector(raw_labels: Any) -> np.ndarray:
    labels = harmonize_labels(_parse_labels(raw_labels), "PTBXL")
    return np.asarray([labels[label] for label in CANONICAL_LABELS], dtype=np.float32)


def prepare_ptbxl_snapshot(
    root: str | Path,
    dataset_config: dict[str, Any],
    snapshot_manifest: dict[str, Any],
) -> PreparedPTBXLSnapshot:
    """Validate release identity, official folds, targets, counts, and leakage."""
    root_path = Path(root)
    source_hashes: dict[str, str] = {}
    for relative_path, expected in snapshot_manifest["source"]["identity_files"].items():
        path = root_path / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Snapshot identity file is missing: {path}")
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(
                f"Snapshot identity mismatch for {relative_path}: "
                f"expected {expected}, observed {observed}"
            )
        source_hashes[relative_path] = observed

    dataset = PTBXLDataset(root_path, dataset_config)
    metadata = dataset.load_metadata()
    required = {"ecg_id", "patient_id", "strat_fold", "scp_codes", "filename_hr"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"PTB-XL snapshot metadata is missing columns: {sorted(missing)}")
    if metadata[["patient_id", "strat_fold", "filename_hr"]].isna().any().any():
        raise ValueError("PTB-XL snapshot contains missing patient, fold, or waveform paths")
    observed_folds = set(metadata["strat_fold"].astype(int).unique())
    if observed_folds != set(range(1, 11)):
        raise ValueError(f"Expected official strat_fold values 1-10, got {sorted(observed_folds)}")

    expected_total = int(snapshot_manifest["counts"]["records_total"])
    if len(metadata) != expected_total:
        raise ValueError(f"Expected {expected_total} PTB-XL records, found {len(metadata)}")

    splits = {
        name: metadata.loc[metadata["strat_fold"].isin(folds)].copy().reset_index(drop=True)
        for name, folds in SPLIT_FOLDS.items()
    }
    expected_splits = snapshot_manifest["counts"]["splits"]
    for name, frame in splits.items():
        expected_count = int(expected_splits[name]["records"])
        if len(frame) != expected_count:
            raise ValueError(f"Expected {expected_count} {name} records, found {len(frame)}")

    patient_sets = {
        name: set(frame["patient_id"].astype(int).tolist()) for name, frame in splits.items()
    }
    overlaps: dict[str, int] = {}
    split_names = list(SPLIT_FOLDS)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            key = f"{left}__{right}"
            overlaps[key] = len(patient_sets[left].intersection(patient_sets[right]))
    if any(overlaps.values()):
        raise ValueError(f"Patient leakage detected across official folds: {overlaps}")

    supports: dict[str, dict[str, int]] = {}
    all_negative: dict[str, int] = {}
    for name, frame in splits.items():
        targets = np.stack([_target_vector(value) for value in frame["scp_codes"]])
        supports[name] = {
            label: int(targets[:, index].sum())
            for index, label in enumerate(CANONICAL_LABELS)
        }
        all_negative[name] = int((targets.sum(axis=1) == 0).sum())
        if any(value == 0 for value in supports[name].values()):
            raise ValueError(f"Split {name} lacks a positive example for at least one target")
        expected = expected_splits[name]
        expected_patients = int(expected["patients"])
        if len(patient_sets[name]) != expected_patients:
            raise ValueError(
                f"Expected {expected_patients} {name} patients, found {len(patient_sets[name])}"
            )
        expected_support = {
            label: int(value) for label, value in expected["positive_support"].items()
        }
        if supports[name] != expected_support:
            raise ValueError(
                f"Canonical label support mismatch for {name}: "
                f"expected {expected_support}, observed {supports[name]}"
            )
        expected_negative = int(expected["all_six_labels_negative"])
        if all_negative[name] != expected_negative:
            raise ValueError(
                f"Expected {expected_negative} all-negative {name} records, "
                f"found {all_negative[name]}"
            )

    split_manifest = pd.concat(
        [
            frame.assign(split=name)[["ecg_id", "patient_id", "split"]]
            for name, frame in splits.items()
        ],
        ignore_index=True,
    )
    manifest_bytes = canonical_split_manifest_bytes(split_manifest)
    split_hash = hashlib.sha256(manifest_bytes).hexdigest()
    expected_hash = snapshot_manifest["split_policy"]["manifest_sha256"]
    if split_hash != expected_hash:
        raise ValueError(
            f"Split manifest mismatch: expected {expected_hash}, observed {split_hash}"
        )

    summary = {
        "snapshot_id": snapshot_manifest["snapshot_id"],
        "records_total": len(metadata),
        "splits": {
            name: {
                "records": len(frame),
                "patients": len(patient_sets[name]),
                "positive_support": supports[name],
                "all_six_labels_negative": all_negative[name],
            }
            for name, frame in splits.items()
        },
        "patient_overlap": overlaps,
        "split_manifest_sha256": split_hash,
        "source_hashes": source_hashes,
    }
    return PreparedPTBXLSnapshot(
        snapshot_id=str(snapshot_manifest["snapshot_id"]),
        splits=splits,
        split_manifest=split_manifest,
        split_manifest_sha256=split_hash,
        summary=summary,
        source_hashes=source_hashes,
    )


class PTBXLClassificationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Load strict 500 Hz, 10-second waveforms for the legacy PTB-XL baseline."""

    def __init__(
        self,
        root: str | Path,
        dataset_config: dict[str, Any],
        metadata: pd.DataFrame,
        input_length: int = 5000,
    ) -> None:
        self.dataset = PTBXLDataset(root, dataset_config)
        self.dataset._metadata = metadata.reset_index(drop=True)
        self.metadata = self.dataset._metadata
        self.input_length = input_length
        self.record_ids = self.metadata["ecg_id"].astype(str).tolist()
        self.targets = np.stack(
            [_target_vector(value) for value in self.metadata["scp_codes"]]
        )

    def __len__(self) -> int:
        return len(self.record_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        signal = self.dataset.load_signal(self.record_ids[index])
        expected_shape = (12, self.input_length)
        if signal.shape != expected_shape:
            raise ValueError(
                f"PTB-XL record {self.record_ids[index]!r} has shape {signal.shape}, "
                f"expected {expected_shape}; cropping and padding are disabled"
            )
        if not np.isfinite(signal).all():
            raise ValueError(f"PTB-XL record {self.record_ids[index]!r} contains non-finite values")
        return torch.from_numpy(signal.copy()), torch.from_numpy(self.targets[index].copy())
