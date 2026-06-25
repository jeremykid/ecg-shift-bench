"""Tests for the dataset discriminator study."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from ecg_shift_bench.datasets.discriminator import (
    DiscriminatorECGDataset,
    add_target_labels,
    build_discriminator_cohort,
    canonical_label_signature,
    select_cohort_subset,
)
from ecg_shift_bench.labels.dataset_ids import (
    DATASET_ID_ORDER,
    canonical_dataset_name,
    dataset_index,
    selected_dataset_names,
)


@dataclass
class FakeDataset:
    name: str
    domain: str
    config: dict[str, object]
    metadata: pd.DataFrame
    labels: dict[str, dict[str, int]]

    def load_metadata(self) -> pd.DataFrame:
        return self.metadata.copy()

    def get_labels(self, record_id: str) -> dict[str, int]:
        return dict(self.labels[str(record_id)])

    def load_aligned_sample(self, record_id: str):
        signal = np.full((12, 8), float(dataset_index(self.name)), dtype=np.float32)
        return SimpleNamespace(
            signal=signal, labels=self.get_labels(record_id), record_id=str(record_id)
        )


def _fake_datasets() -> dict[str, FakeDataset]:
    records = pd.DataFrame(
        {
            "record_id": ["a1", "a2", "b1", "b2", "c1", "c2"],
            "patient_id": ["p1", "p2", "p3", "p4", "p5", "p6"],
        }
    )
    labels = {
        "a1": {"AF": 0, "RBBB": 0, "LBBB": 0, "1dAVB": 0, "SB": 0, "ST": 0},
        "a2": {"AF": 1, "RBBB": 0, "LBBB": 0, "1dAVB": 0, "SB": 0, "ST": 0},
        "b1": {"AF": 0, "RBBB": 1, "LBBB": 0, "1dAVB": 0, "SB": 0, "ST": 0},
        "b2": {"AF": 0, "RBBB": 0, "LBBB": 1, "1dAVB": 0, "SB": 0, "ST": 0},
        "c1": {"AF": 0, "RBBB": 0, "LBBB": 0, "1dAVB": 1, "SB": 0, "ST": 0},
        "c2": {"AF": 0, "RBBB": 0, "LBBB": 0, "1dAVB": 0, "SB": 1, "ST": 0},
    }
    datasets: dict[str, FakeDataset] = {}
    for name in DATASET_ID_ORDER:
        datasets[name] = FakeDataset(
            name=name,
            domain=f"{name.lower()}_domain",
            config={
                "record_id_column": "record_id",
                "patient_id_column": "patient_id",
                "sampling_rate": 500,
                "target_sampling_rate": 500,
                "target_length": 5000,
                "normalization": "per_lead_zscore",
            },
            metadata=records.assign(record_id=lambda frame: frame["record_id"]),
            labels=labels,
        )
    return datasets


def _fake_split_manifest(
    dataset: FakeDataset, metadata: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, object]]:
    split = pd.DataFrame(
        {
            "record_id": metadata["record_id"].astype(str),
            "split": ["train", "validation", "test", "train", "validation", "test"],
        }
    )
    return split, {"method": "fake"}


def test_dataset_identity_helpers_are_stable() -> None:
    assert canonical_dataset_name("ptb-xl") == "PTBXL"
    assert dataset_index("CODE15") == 1
    assert selected_dataset_names(["sph", "ptb-xl"]) == ["PTBXL", "SPH"]


def test_canonical_label_signature_prefers_normal_for_negative_examples() -> None:
    assert (
        canonical_label_signature({"AF": 0, "RBBB": 0, "LBBB": 0, "1dAVB": 0, "SB": 0, "ST": 0})
        == "NORMAL"
    )


def test_build_and_subset_discriminator_cohort(monkeypatch: pytest.MonkeyPatch) -> None:
    datasets = _fake_datasets()
    monkeypatch.setattr(
        "ecg_shift_bench.datasets.discriminator.build_split_manifest",
        _fake_split_manifest,
    )

    cohort = build_discriminator_cohort(datasets)
    assert set(cohort["dataset_name"]) == set(DATASET_ID_ORDER)
    assert len(cohort) == 24
    assert set(cohort.columns) >= {
        "dataset_name",
        "dataset_id",
        "record_id",
        "patient_id",
        "domain",
        "split",
        "label_signature",
        "is_normal",
    }

    normal_only = select_cohort_subset(
        cohort, selected_datasets=["PTBXL", "CODE15"], subset="normal_only"
    )
    assert set(normal_only["dataset_name"]) == {"PTBXL", "CODE15"}
    assert normal_only["is_normal"].all()

    balanced = select_cohort_subset(cohort, subset="label_balanced")
    assert len(balanced) == len(cohort)

    pairwise = add_target_labels(
        select_cohort_subset(cohort, selected_datasets=["PTBXL", "SPH"], subset="uncontrolled"),
        selected_datasets=["PTBXL", "SPH"],
    )
    assert set(pairwise["target_dataset_id"]) == {0, 1}
    assert set(pairwise["dataset_name"]) == {"PTBXL", "SPH"}


def test_random_label_only_permutes_training_targets() -> None:
    manifest = pd.DataFrame(
        {
            "dataset_name": ["PTBXL", "CODE15", "SPH", "CHAPMAN"],
            "split": ["train", "train", "validation", "test"],
            "target_dataset_id": [0, 1, 2, 3],
        }
    )
    permuted = add_target_labels(
        manifest.assign(patient_id=["p1", "p2", "p3", "p4"], record_id=["r1", "r2", "r3", "r4"]),
        selected_datasets=DATASET_ID_ORDER,
        random_label=True,
        seed=7,
    )
    assert sorted(permuted.loc[permuted["split"] != "train", "target_dataset_id"].tolist()) == [
        2,
        3,
    ]
    assert sorted(permuted.loc[permuted["split"] == "train", "target_dataset_id"].tolist()) == [
        0,
        1,
    ]


def test_discriminator_ecg_dataset_loads_aligned_sample() -> None:
    manifest = pd.DataFrame(
        {
            "dataset_name": ["PTBXL"],
            "record_id": ["a1"],
            "patient_id": ["p1"],
            "split": ["train"],
            "target_dataset_id": [0],
        }
    )
    dataset = DiscriminatorECGDataset(manifest, {"PTBXL": _fake_datasets()["PTBXL"]})
    signal, target = dataset[0]
    assert signal.shape == (12, 8)
    assert target.dtype == torch.long
    assert int(target.item()) == 0
