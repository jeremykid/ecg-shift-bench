"""Cohort helpers for the dataset discriminator study."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from ecg_shift_bench.datasets.audit import build_split_manifest
from ecg_shift_bench.datasets.base import BaseECGDataset
from ecg_shift_bench.labels.canonical import CANONICAL_LABELS
from ecg_shift_bench.labels.dataset_ids import (
    DATASET_ID_ORDER,
    canonical_dataset_name,
    dataset_index,
    selected_dataset_names,
)


@dataclass(frozen=True)
class DiscriminatorRecord:
    """One row in the discriminator cohort manifest."""

    dataset_name: str
    dataset_id: int
    record_id: str
    patient_id: str | None
    domain: str
    split: str
    label_signature: str
    is_normal: bool


def canonical_label_signature(labels: Mapping[str, int]) -> str:
    """Return a stable signature for one canonical six-label record."""
    positives = [label for label in CANONICAL_LABELS if int(labels.get(label, 0))]
    return "NORMAL" if not positives else "+".join(positives)


def build_discriminator_cohort(
    datasets: Mapping[str, BaseECGDataset],
) -> pd.DataFrame:
    """Build the aligned cohort manifest without loading waveform arrays."""
    rows: list[dict[str, Any]] = []
    for canonical_name in DATASET_ID_ORDER:
        if canonical_name not in datasets:
            continue
        dataset = datasets[canonical_name]
        metadata = dataset.load_metadata()
        split_manifest, _ = build_split_manifest(dataset, metadata)
        record_col = str(dataset.config.get("record_id_column", "record_id"))
        patient_col = dataset.config.get("patient_id_column")
        split_lookup = dict(
            zip(split_manifest["record_id"].astype(str), split_manifest["split"], strict=True)
        )
        patient_lookup: dict[str, str | None] | None = None
        if patient_col and str(patient_col) in metadata.columns:
            patient_lookup = {
                str(row[record_col]): (
                    None if pd.isna(row[str(patient_col)]) else str(row[str(patient_col)])
                )
                for _, row in metadata[[record_col, str(patient_col)]].iterrows()
            }

        record_ids = metadata[record_col].astype(str).tolist()
        for record_id in record_ids:
            labels = dataset.get_labels(record_id)
            patient_id = patient_lookup.get(record_id) if patient_lookup is not None else None
            split_value = split_lookup.get(record_id)
            if split_value is None:
                raise KeyError(f"Missing split assignment for {canonical_name} record {record_id}")
            rows.append(
                {
                    "dataset_name": canonical_name,
                    "dataset_id": dataset_index(canonical_name),
                    "domain": dataset.domain,
                    "record_id": record_id,
                    "patient_id": patient_id,
                    "split": str(split_value),
                    "label_signature": canonical_label_signature(labels),
                    "is_normal": int(not any(labels.values())),
                }
            )

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        return manifest
    manifest["dataset_id"] = manifest["dataset_id"].astype(int)
    manifest["is_normal"] = manifest["is_normal"].astype(bool)
    return manifest.sort_values(["dataset_id", "split", "record_id"]).reset_index(drop=True)


def select_cohort_subset(
    manifest: pd.DataFrame,
    *,
    selected_datasets: list[str] | None = None,
    subset: str = "uncontrolled",
    seed: int = 42,
) -> pd.DataFrame:
    """Filter the cohort manifest for one study variant."""
    cohort = manifest.copy().reset_index(drop=True)
    if selected_datasets is not None:
        selected = selected_dataset_names(selected_datasets)
        cohort = cohort.loc[cohort["dataset_name"].isin(selected)].reset_index(drop=True)
    if subset == "uncontrolled":
        return cohort
    if subset == "normal_only":
        return cohort.loc[cohort["is_normal"]].reset_index(drop=True)
    if subset == "label_balanced":
        return _balanced_label_subset(cohort, seed=seed)
    if subset == "random_label":
        return cohort
    raise ValueError(f"Unknown subset mode: {subset}")


def add_target_labels(
    manifest: pd.DataFrame,
    *,
    selected_datasets: list[str],
    random_label: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """Attach target dataset IDs for training and evaluation."""
    subset = manifest.copy().reset_index(drop=True)
    canonical = selected_dataset_names(selected_datasets)
    label_map = {name: index for index, name in enumerate(canonical)}
    subset["target_dataset_id"] = subset["dataset_name"].map(label_map)
    if subset["target_dataset_id"].isna().any():
        missing = sorted(subset.loc[subset["target_dataset_id"].isna(), "dataset_name"].unique())
        raise KeyError(f"Missing dataset IDs for: {missing}")
    subset["target_dataset_id"] = subset["target_dataset_id"].astype(int)
    if random_label:
        subset = _permute_training_targets(subset, seed=seed)
    return subset.reset_index(drop=True)


def _balanced_label_subset(manifest: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    """Sample the same number of records per canonical label signature and dataset."""
    selected_frames: list[pd.DataFrame] = []
    rng = torch.Generator().manual_seed(seed)
    for signature in sorted(manifest["label_signature"].unique()):
        signature_frame = manifest.loc[manifest["label_signature"] == signature]
        counts = signature_frame.groupby("dataset_name").size()
        if counts.empty:
            continue
        target_count = int(counts.min())
        if target_count == 0:
            continue
        for dataset_name in DATASET_ID_ORDER:
            if dataset_name not in counts:
                continue
            dataset_frame = signature_frame.loc[signature_frame["dataset_name"] == dataset_name]
            sampled = dataset_frame.sample(n=target_count, random_state=int(rng.initial_seed()))
            selected_frames.append(sampled)
    if not selected_frames:
        return manifest.iloc[0:0].copy()
    return (
        pd.concat(selected_frames, ignore_index=True)
        .sort_values(["dataset_id", "split", "record_id"])
        .reset_index(drop=True)
    )


def _permute_training_targets(manifest: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    """Permute training labels while leaving validation and test labels unchanged."""
    shuffled = manifest.copy().reset_index(drop=True)
    train_mask = shuffled["split"] == "train"
    train_targets = shuffled.loc[train_mask, "target_dataset_id"].to_numpy(copy=True)
    if len(train_targets) == 0:
        return shuffled
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(train_targets), generator=generator).numpy()
    shuffled.loc[train_mask, "target_dataset_id"] = train_targets[permutation]
    return shuffled


class DiscriminatorECGDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Load aligned ECGs and dataset-ID targets from one cohort manifest."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        datasets: Mapping[str, BaseECGDataset],
    ) -> None:
        self.manifest = manifest.reset_index(drop=True).copy()
        self.datasets = dict(datasets)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.manifest.iloc[index]
        dataset_name = canonical_dataset_name(row["dataset_name"])
        dataset = self.datasets[dataset_name]
        sample = dataset.load_aligned_sample(str(row["record_id"]))
        signal = torch.from_numpy(sample.signal)
        target = torch.tensor(int(row["target_dataset_id"]), dtype=torch.long)
        return signal, target
