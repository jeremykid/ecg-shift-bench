"""Tests for the dataset-discriminator training workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from ecg_shift_bench.training.discriminator import (
    DatasetSpec,
    _resolve_device,
    run_dataset_discriminator,
)


@dataclass
class FakeDataset:
    name: str
    domain: str
    config: dict[str, object]
    metadata: pd.DataFrame

    def load_metadata(self) -> pd.DataFrame:
        return self.metadata.copy()

    def get_labels(self, record_id: str) -> dict[str, int]:
        lookup = {
            "r1": {"AF": 0, "RBBB": 0, "LBBB": 0, "1dAVB": 0, "SB": 0, "ST": 0},
            "r2": {"AF": 1, "RBBB": 0, "LBBB": 0, "1dAVB": 0, "SB": 0, "ST": 0},
            "r3": {"AF": 0, "RBBB": 1, "LBBB": 0, "1dAVB": 0, "SB": 0, "ST": 0},
            "r4": {"AF": 0, "RBBB": 0, "LBBB": 1, "1dAVB": 0, "SB": 0, "ST": 0},
        }
        return lookup[str(record_id)]

    def load_aligned_sample(self, record_id: str):
        signal = np.full((12, 8), float(ord(self.name[0]) % 4), dtype=np.float32)
        return type(
            "AlignedSample",
            (),
            {
                "signal": signal,
                "labels": self.get_labels(record_id),
                "record_id": str(record_id),
                "patient_id": str(record_id).replace("r", "p"),
                "domain": self.domain,
            },
        )()


def _dataset_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "record_id": ["r1", "r2", "r3", "r4"],
            "patient_id": ["p1", "p2", "p3", "p4"],
        }
    )


def _fake_split_manifest(
    dataset: FakeDataset, metadata: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, object]]:
    split = pd.DataFrame(
        {
            "record_id": metadata["record_id"].astype(str),
            "split": ["train", "validation", "test", "train"],
        }
    )
    return split, {"method": "fake"}


def _fake_datasets() -> dict[str, FakeDataset]:
    records = _dataset_frame()
    configs = {
        "PTBXL": {"record_id_column": "record_id", "patient_id_column": "patient_id"},
        "CODE15": {"record_id_column": "record_id", "patient_id_column": "patient_id"},
        "CHAPMAN": {"record_id_column": "record_id"},
        "SPH": {"record_id_column": "record_id", "patient_id_column": "patient_id"},
    }
    return {
        name: FakeDataset(
            name=name, domain=f"{name.lower()}_domain", config=configs[name], metadata=records
        )
        for name in configs
    }


def test_resolve_device_rejects_cpu() -> None:
    with pytest.raises(ValueError, match="CUDA only"):
        _resolve_device("cpu")


def test_run_dataset_discriminator_writes_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    datasets = _fake_datasets()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "ecg_shift_bench.datasets.discriminator.build_split_manifest",
        _fake_split_manifest,
    )
    monkeypatch.setattr(
        "ecg_shift_bench.training.discriminator.create_dataset",
        lambda name, root, config: datasets[name],
    )
    monkeypatch.setattr(
        "ecg_shift_bench.training.discriminator._resolve_device",
        lambda requested: torch.device("cpu"),
    )
    monkeypatch.setattr(
        "ecg_shift_bench.training.discriminator.create_model",
        lambda model_config, *, num_labels: captured.update(
            {"model_config": dict(model_config), "num_labels": num_labels}
        )
        or nn.Sequential(nn.Flatten(), nn.Linear(12 * 8, num_labels)),
    )

    config = {
        "experiment": "dataset_discriminator_xresnet1d_v1",
        "model": {"name": "xresnet1d", "in_channels": 12, "channels": 8, "dropout": 0.0},
        "training": {
            "seed": 42,
            "batch_size": 2,
            "epochs": 1,
            "optimizer": "adamw",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "workers": 0,
        },
    }
    config_path = tmp_path / "dataset_discriminator.yaml"
    config_path.write_text("experiment: dataset_discriminator_xresnet1d_v1\n", encoding="utf-8")
    status = run_dataset_discriminator(
        experiment_config=config,
        experiment_config_path=config_path,
        dataset_specs=[
            DatasetSpec(
                name=name,
                root=tmp_path / name.lower(),
                config={},
                config_path=tmp_path / f"{name}.yaml",
            )
            for name in ("PTBXL", "CODE15", "CHAPMAN", "SPH")
        ],
        output_dir=tmp_path / "outputs",
        requested_device="cuda:0",
        command="python scripts/train_dataset_discriminator.py",
        mode="multiclass",
        subset="uncontrolled",
        preflight_only=True,
    )

    assert status["status"] == "preflight_completed"
    assert captured["model_config"]["name"] == "xresnet1d"
    assert captured["num_labels"] == 4
    assert (tmp_path / "outputs" / "run_status.json").exists()
    assert (tmp_path / "outputs" / "cohort_manifest.csv").exists()
    assert (tmp_path / "outputs" / "split_manifest.csv").exists()
    assert not (tmp_path / "outputs" / "best_checkpoint.pt").exists()


def test_run_dataset_discriminator_random_label_preserves_permuted_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = pd.DataFrame(
        {
            "record_id": ["r1", "r2", "r3"],
            "patient_id": ["p1", "p2", "p3"],
        }
    )

    class RandomLabelDataset(FakeDataset):
        pass

    datasets = {
        name: RandomLabelDataset(
            name=name,
            domain=f"{name.lower()}_domain",
            config={"record_id_column": "record_id", "patient_id_column": "patient_id"},
            metadata=records,
        )
        for name in ("PTBXL", "CODE15", "CHAPMAN", "SPH")
    }

    def split_manifest(dataset: FakeDataset, metadata: pd.DataFrame):
        split = pd.DataFrame(
            {
                "record_id": metadata["record_id"].astype(str),
                "split": ["train", "validation", "test"],
            }
        )
        return split, {"method": "fake"}

    monkeypatch.setattr(
        "ecg_shift_bench.datasets.discriminator.build_split_manifest",
        split_manifest,
    )
    monkeypatch.setattr(
        "ecg_shift_bench.training.discriminator.create_dataset",
        lambda name, root, config: datasets[name],
    )
    monkeypatch.setattr(
        "ecg_shift_bench.training.discriminator._resolve_device",
        lambda requested: torch.device("cpu"),
    )
    monkeypatch.setattr(
        "ecg_shift_bench.training.discriminator.create_model",
        lambda model_config, *, num_labels: nn.Sequential(nn.Flatten(), nn.Linear(12 * 8, num_labels)),
    )

    config = {
        "experiment": "dataset_discriminator_xresnet1d_v1",
        "model": {"name": "xresnet1d", "in_channels": 12, "channels": 8, "dropout": 0.0},
        "training": {
            "seed": 42,
            "batch_size": 2,
            "epochs": 1,
            "optimizer": "adamw",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "workers": 0,
        },
    }
    config_path = tmp_path / "dataset_discriminator.yaml"
    config_path.write_text("experiment: dataset_discriminator_xresnet1d_v1\n", encoding="utf-8")
    run_dataset_discriminator(
        experiment_config=config,
        experiment_config_path=config_path,
        dataset_specs=[
            DatasetSpec(
                name=name,
                root=tmp_path / name.lower(),
                config={},
                config_path=tmp_path / f"{name}.yaml",
            )
            for name in ("PTBXL", "CODE15", "CHAPMAN", "SPH")
        ],
        output_dir=tmp_path / "outputs",
        requested_device="cuda:0",
        command="python scripts/train_dataset_discriminator.py",
        mode="multiclass",
        subset="random_label",
        preflight_only=True,
    )

    cohort = pd.read_csv(tmp_path / "outputs" / "cohort_manifest.csv")
    train_targets = cohort.loc[cohort["split"] == "train", "target_dataset_id"].tolist()
    assert train_targets == [2, 3, 0, 1]
