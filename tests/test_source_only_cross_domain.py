"""Workflow tests for the source-only cross-domain baseline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from ecg_shift_bench.labels.canonical import CANONICAL_LABELS
from ecg_shift_bench.training import source_only_cross_domain as source_only


@dataclass
class FakeDataset:
    name: str
    domain: str
    config: dict[str, object]
    metadata: pd.DataFrame
    signals: dict[str, np.ndarray]
    labels: dict[str, dict[str, int]]
    calls: list[str]

    def load_metadata(self) -> pd.DataFrame:
        self.calls.append(f"load_metadata:{self.name}")
        return self.metadata.copy()

    def get_labels(self, record_id: str) -> dict[str, int]:
        return dict(self.labels[str(record_id)])

    def load_aligned_signal(self, record_id: str) -> np.ndarray:
        self.calls.append(f"load_signal:{self.name}:{record_id}")
        return self.signals[str(record_id)].copy()


def _labels(index: int) -> dict[str, int]:
    vectors = [
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ]
    return dict(zip(CANONICAL_LABELS, vectors[index], strict=True))


def _fake_records(
    prefix: str, count: int, *, length: int
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, dict[str, int]]]:
    record_ids = [f"{prefix}{index}" for index in range(1, count + 1)]
    metadata = pd.DataFrame(
        {
            "record_id": record_ids,
            "patient_id": [f"p{index}" for index in range(1, count + 1)],
        }
    )
    signals = {
        record_id: np.full((12, length), float(index), dtype=np.float32)
        for index, record_id in enumerate(record_ids, start=1)
    }
    labels = {
        record_id: _labels((index - 1) % len(CANONICAL_LABELS))
        for index, record_id in enumerate(record_ids, start=1)
    }
    return metadata, signals, labels


def _fake_split_manifest(
    dataset: FakeDataset,
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if dataset.name == "SOURCE":
        manifest = pd.DataFrame(
            {
                "record_id": ["s1", "s2", "s3", "s4", "s5", "s6"],
                "split": ["train", "train", "train", "validation", "validation", "test"],
            }
        )
        policy = {"method": "generated", "split_source": "generated", "seed": 7}
    else:
        manifest = pd.DataFrame(
            {
                "record_id": ["t1", "t2", "t3", "t4", "t5", "t6"],
                "split": ["train", "train", "validation", "validation", "test", "test"],
            }
        )
        policy = {"method": "official", "split_source": "official"}
    manifest["patient_id"] = metadata["patient_id"].astype(str).tolist()
    return manifest, policy


def _write_yaml(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_aligned_dataset_reads_standard_manifest_record_ids() -> None:
    dataset = FakeDataset(
        name="SOURCE",
        domain="source_domain",
        config={
            "record_id_column": "ecg_id",
            "patient_id_column": "patient_id",
            "sampling_rate": 500,
            "target_sampling_rate": 500,
            "target_length": 8,
            "source_unit": "mV",
            "target_unit": "mV",
            "domain": "source_domain",
        },
        metadata=pd.DataFrame(
            {
                "record_id": ["s1", "s2"],
                "patient_id": ["p1", "p2"],
            }
        ),
        signals={
            "s1": np.full((12, 8), 1.0, dtype=np.float32),
            "s2": np.full((12, 8), 2.0, dtype=np.float32),
        },
        labels={
            "s1": _labels(0),
            "s2": _labels(1),
        },
        calls=[],
    )
    dataset_frame = pd.DataFrame(
        {
            "record_id": ["s1", "s2"],
            "patient_id": ["p1", "p2"],
            "split": ["train", "validation"],
        }
    )

    aligned = source_only.AlignedClassificationDataset(dataset, dataset_frame, 8)

    inputs, targets = aligned[0]
    assert inputs.shape == (12, 8)
    assert targets.shape == (6,)


def test_source_only_run_uses_source_data_before_target_and_writes_report_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    source_metadata, source_signals, source_labels = _fake_records("s", 6, length=64)
    target_metadata, target_signals, target_labels = _fake_records("t", 6, length=64)
    datasets = {
        "SOURCE": FakeDataset(
            name="SOURCE",
            domain="source_domain",
            config={
                "record_id_column": "record_id",
                "patient_id_column": "patient_id",
                "sampling_rate": 500,
                "target_sampling_rate": 500,
                "target_length": 64,
                "source_unit": "mV",
                "target_unit": "mV",
                "domain": "source_domain",
            },
            metadata=source_metadata,
            signals=source_signals,
            labels=source_labels,
            calls=calls,
        ),
        "TARGET": FakeDataset(
            name="TARGET",
            domain="target_domain",
            config={
                "record_id_column": "record_id",
                "patient_id_column": "patient_id",
                "sampling_rate": 500,
                "target_sampling_rate": 500,
                "target_length": 64,
                "source_unit": "mV",
                "target_unit": "mV",
                "domain": "target_domain",
            },
            metadata=target_metadata,
            signals=target_signals,
            labels=target_labels,
            calls=calls,
        ),
    }

    source_dataset_config = tmp_path / "source.yaml"
    target_dataset_config = tmp_path / "target.yaml"
    experiment_config_path = tmp_path / "experiment.yaml"
    _write_yaml(
        source_dataset_config,
        "name: SOURCE\nroot: /placeholder/source\nrecord_id_column: record_id\npatient_id_column: patient_id\nsampling_rate: 500\ntarget_sampling_rate: 500\ntarget_length: 64\nsource_unit: mV\ntarget_unit: mV\ndomain: source_domain\n",
    )
    _write_yaml(
        target_dataset_config,
        "name: TARGET\nroot: /placeholder/target\nrecord_id_column: record_id\npatient_id_column: patient_id\nsampling_rate: 500\ntarget_sampling_rate: 500\ntarget_length: 64\nsource_unit: mV\ntarget_unit: mV\ndomain: target_domain\n",
    )
    _write_yaml(
        experiment_config_path,
        "experiment: source_only_source_to_target\nmethod: source_only\n",
    )

    experiment_config = {
        "experiment": "source_only_source_to_target",
        "method": "source_only",
        "source_datasets": ["SOURCE"],
        "target_datasets": ["TARGET"],
        "dataset_configs": {
            "source": str(source_dataset_config),
            "target": str(target_dataset_config),
        },
        "model": {"name": "resnet1d", "width": 4},
        "data": {
            "input_length": 64,
            "preprocessing_version": "shared_alignment_v1",
            "sampling_rate": 500,
            "target_sampling_rate": 500,
            "source_unit": "mV",
            "target_unit": "mV",
            "normalization": "none",
        },
        "training": {
            "seed": 7,
            "batch_size": 2,
            "workers": 0,
            "epochs": 1,
            "optimizer": "adamw",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "amp": False,
        },
    }

    monkeypatch.setattr(source_only, "_git_state", lambda: ("deadbeef", False))
    monkeypatch.setattr(source_only, "_resolve_device", lambda requested: torch.device("cpu"))
    monkeypatch.setattr(
        source_only,
        "create_dataset",
        lambda name, root, config: (
            calls.append(f"create_dataset:{name}")
            or datasets[name]
        ),
    )
    monkeypatch.setattr(source_only, "build_split_manifest", _fake_split_manifest)
    monkeypatch.setattr(source_only, "preflight_forward_backward", lambda *args, **kwargs: 0.25)

    def fake_train_epoch(
        model,
        batches,
        optimizer,
        criterion,
        device,
        amp_enabled,
        *,
        description,
    ):
        calls.append(f"train_epoch:{description}:{batches.dataset.dataset.name}")
        batch = next(iter(batches))
        calls.append(f"train_batch:{float(batch[0].mean()):.1f}")
        return {"loss": 0.1, "min_batch_loss": 0.1, "max_batch_loss": 0.1, "steps": 1, "samples": 2}

    monkeypatch.setattr(source_only, "train_epoch", fake_train_epoch)

    output_dir = tmp_path / "outputs"
    status = source_only.run_source_only_cross_domain(
        experiment_config=experiment_config,
        experiment_config_path=experiment_config_path,
        source_dataset_spec=source_only.DatasetSpec(
            name="SOURCE",
            root=tmp_path / "source_root",
            config=dict(datasets["SOURCE"].config),
            config_path=source_dataset_config,
        ),
        target_dataset_spec=source_only.DatasetSpec(
            name="TARGET",
            root=tmp_path / "target_root",
            config=dict(datasets["TARGET"].config),
            config_path=target_dataset_config,
        ),
        output_dir=output_dir,
        requested_device="cpu",
        command="python scripts/train.py --config experiment.yaml",
    )

    assert status["status"] == "completed"
    assert calls.index("create_dataset:SOURCE") < calls.index("train_epoch:source train 1/1:SOURCE")
    assert calls.index("train_epoch:source train 1/1:SOURCE") < calls.index("create_dataset:TARGET")
    assert not any(entry.startswith("load_signal:TARGET") for entry in calls[: calls.index("create_dataset:TARGET")])

    summary = pd.read_csv(output_dir / "results_summary.csv").iloc[0].to_dict()
    assert summary["source_dataset"] == "SOURCE"
    assert summary["target_dataset"] == "TARGET"
    assert summary["source_split_version"] == "generated_seed7"
    assert summary["target_split_version"] == "official"
    assert summary["split_version"] == "source:generated_seed7|target:official"
    assert summary["preprocessing_version"] == "shared_alignment_v1"
    assert summary["selection_metric"] == "source_validation_macro_auprc"
    assert summary["source_train_records"] == 3
    assert summary["source_validation_records"] == 2
    assert summary["target_test_records"] == 2
    assert summary["model_architecture"] == "resnet1d"
    assert summary["evaluation_metrics"]

    per_class = pd.read_csv(output_dir / "per_class_summary.csv")
    assert per_class["split"].tolist() == [
        "source_train",
        "source_train",
        "source_train",
        "source_train",
        "source_train",
        "source_train",
        "source_validation",
        "source_validation",
        "source_validation",
        "source_validation",
        "source_validation",
        "source_validation",
        "target_test",
        "target_test",
        "target_test",
        "target_test",
        "target_test",
        "target_test",
    ]
    assert per_class["label"].tolist()[:6] == list(CANONICAL_LABELS)

    status_json = json.loads((output_dir / "run_status.json").read_text(encoding="utf-8"))
    assert status_json["protocol"]["target_labels_available_during_training"] is False
    assert status_json["protocol"]["target_unlabeled_data_available_during_training"] is False
    assert status_json["protocol"]["model_updates_during_testing"] is False
    assert status_json["protocol"]["normalization"] == "none"
