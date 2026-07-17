"""Workflow tests for the generic UDA training path."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from ecg_shift_bench.labels.canonical import CANONICAL_LABELS
from ecg_shift_bench.methods.uda import build_uda_method as real_build_uda_method
from ecg_shift_bench.training import uda


@dataclass
class FakeDataset:
    name: str
    domain: str
    config: dict[str, object]
    metadata: pd.DataFrame
    signals: dict[str, np.ndarray]
    labels: dict[str, dict[str, int]]
    calls: list[str]
    label_calls: int = 0

    def load_metadata(self) -> pd.DataFrame:
        self.calls.append(f"load_metadata:{self.name}")
        return self.metadata.copy()

    def get_labels(self, record_id: str) -> dict[str, int]:
        self.label_calls += 1
        self.calls.append(f"get_labels:{self.name}:{record_id}")
        return dict(self.labels[str(record_id)])

    def load_aligned_signal(self, record_id: str) -> np.ndarray:
        self.calls.append(f"load_signal:{self.name}:{record_id}")
        return self.signals[str(record_id)].copy()


class TinyUdaModel(nn.Module):
    def __init__(self, num_labels: int = 6) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(12 * 8, 4), nn.ReLU())
        self.head = nn.Linear(4, num_labels)

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(inputs))


def _labels(index: int) -> dict[str, int]:
    vectors = [
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [1, 1, 0, 0, 1, 1],
        [0, 0, 1, 1, 0, 0],
        [1, 0, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
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
    splits = {
        "SOURCE": ["train", "train", "validation", "validation", "test", "test"],
        "TARGET": ["train", "train", "validation", "validation", "test", "test"],
    }
    policy = {
        "SOURCE": {"method": "generated", "split_source": "generated", "seed": 7},
        "TARGET": {"method": "official", "split_source": "official"},
    }
    manifest = pd.DataFrame(
        {
            "record_id": metadata["record_id"].astype(str).tolist(),
            "split": splits[dataset.name],
            "patient_id": metadata["patient_id"].astype(str).tolist(),
        }
    )
    return manifest, policy[dataset.name]


def _write_yaml(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_uda_run_uses_unlabeled_target_batches_and_writes_full_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    source_metadata, source_signals, source_labels = _fake_records("s", 6, length=8)
    target_metadata, target_signals, target_labels = _fake_records("t", 6, length=8)
    datasets = {
        "SOURCE": FakeDataset(
            name="SOURCE",
            domain="source_domain",
            config={
                "record_id_column": "record_id",
                "patient_id_column": "patient_id",
                "sampling_rate": 500,
                "target_sampling_rate": 500,
                "target_length": 8,
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
                "target_length": 8,
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
        "name: SOURCE\n"
        "root: /placeholder/source\n"
        "record_id_column: record_id\n"
        "patient_id_column: patient_id\n"
        "sampling_rate: 500\n"
        "target_sampling_rate: 500\n"
        "target_length: 8\n"
        "source_unit: mV\n"
        "target_unit: mV\n"
        "domain: source_domain\n",
    )
    _write_yaml(
        target_dataset_config,
        "name: TARGET\n"
        "root: /placeholder/target\n"
        "record_id_column: record_id\n"
        "patient_id_column: patient_id\n"
        "sampling_rate: 500\n"
        "target_sampling_rate: 500\n"
        "target_length: 8\n"
        "source_unit: mV\n"
        "target_unit: mV\n"
        "domain: target_domain\n",
    )
    _write_yaml(
        experiment_config_path,
        "experiment: uda_source_only_source_to_target\nmethod: source_only\n",
    )

    experiment_config = {
        "experiment": "uda_source_only_source_to_target",
        "method": "source_only",
        "source_datasets": ["SOURCE"],
        "target_datasets": ["TARGET"],
        "dataset_configs": {
            "source": str(source_dataset_config),
            "target": str(target_dataset_config),
        },
        "model": {"name": "resnet1d", "width": 4},
        "data": {
            "input_length": 8,
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
            "epochs": 2,
            "optimizer": "adamw",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "amp": False,
        },
        "evaluation": {"selection_metric": "source_validation_macro_auroc"},
        "method_params": {},
        "protocol": {
            "target_inputs_available_during_training": True,
            "target_labels_available_during_training": False,
        },
    }

    monkeypatch.setattr(uda, "_git_state", lambda: ("deadbeef", False))
    monkeypatch.setattr(uda, "_resolve_device", lambda requested: torch.device("cpu"))
    monkeypatch.setattr(
        uda,
        "create_dataset",
        lambda name, root, config: calls.append(f"create_dataset:{name}") or datasets[name],
    )
    monkeypatch.setattr(uda, "build_split_manifest", _fake_split_manifest)
    monkeypatch.setattr(
        uda, "create_model", lambda model_config, *, num_labels: TinyUdaModel(num_labels)
    )

    source_train = np.array(
        [
            [1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1],
        ],
        dtype=int,
    )
    good_scores = source_train * 0.9 + (1 - source_train) * 0.1
    bad_scores = source_train * 0.1 + (1 - source_train) * 0.9

    def fake_collect_predictions(
        model,
        batches,
        device,
        amp_enabled,
        *,
        description,
    ):
        del model, batches, device, amp_enabled
        if "source train" in description:
            return source_train, good_scores
        if "source validation" in description and "1/2" in description:
            return source_train, bad_scores
        if "source validation" in description:
            return source_train, good_scores
        if "target validation" in description or "target test" in description:
            return source_train, good_scores
        raise AssertionError(f"Unexpected description: {description}")

    monkeypatch.setattr(uda, "_collect_predictions", fake_collect_predictions)

    output_dir = tmp_path / "outputs"
    status = uda.run_uda_cross_domain(
        experiment_config=experiment_config,
        experiment_config_path=experiment_config_path,
        source_dataset_spec=uda.DatasetSpec(
            name="SOURCE",
            root=tmp_path / "source_root",
            config=dict(datasets["SOURCE"].config),
            config_path=source_dataset_config,
        ),
        target_dataset_spec=uda.DatasetSpec(
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
    assert status["best_epoch"] == 2
    assert status["protocol"]["target_inputs_available_during_training"] is True
    assert status["protocol"]["target_labels_available_during_training"] is False
    assert datasets["TARGET"].label_calls == 4

    summary = pd.read_csv(output_dir / "results_summary.csv").iloc[0].to_dict()
    assert summary["source_dataset"] == "SOURCE"
    assert summary["target_dataset"] == "TARGET"
    assert summary["selection_metric"] == "source_validation_macro_auroc"
    assert summary["best_epoch"] == 2
    assert summary["source_train_records"] == 2
    assert summary["source_validation_records"] == 2
    assert summary["target_validation_records"] == 2
    assert summary["target_test_records"] == 2
    assert summary["feature_extraction"] == "model.forward_features"

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
        "target_validation",
        "target_validation",
        "target_validation",
        "target_validation",
        "target_validation",
        "target_validation",
        "target_test",
        "target_test",
        "target_test",
        "target_test",
        "target_test",
        "target_test",
    ]
    assert per_class["label"].tolist()[:6] == list(CANONICAL_LABELS)
    training_log = pd.read_csv(output_dir / "training_log.csv")
    assert training_log["phase"].tolist() == ["epoch", "epoch", "final_evaluation"]
    assert "source_classification_loss" in training_log.columns
    assert "adaptation_loss" in training_log.columns
    assert "target_test_macro_auroc" in training_log.columns

    status_json = json.loads((output_dir / "run_status.json").read_text(encoding="utf-8"))
    assert status_json["protocol"]["target_inputs_available_during_training"] is True
    assert status_json["method_metadata"]["name"] == "source_only"
    assert (output_dir / "target_validation_metrics.json").exists()


def test_coral_uda_run_records_method_params_and_adaptation_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    source_metadata, source_signals, source_labels = _fake_records("s", 6, length=8)
    target_metadata, target_signals, target_labels = _fake_records("t", 6, length=8)
    target_signals["t1"] = np.full((12, 8), 2.0, dtype=np.float32)
    target_signals["t2"] = np.full((12, 8), 4.0, dtype=np.float32)
    datasets = {
        "SOURCE": FakeDataset(
            name="SOURCE",
            domain="source_domain",
            config={
                "record_id_column": "record_id",
                "patient_id_column": "patient_id",
                "sampling_rate": 500,
                "target_sampling_rate": 500,
                "target_length": 8,
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
                "target_length": 8,
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
        "name: SOURCE\n"
        "root: /placeholder/source\n"
        "record_id_column: record_id\n"
        "patient_id_column: patient_id\n"
        "sampling_rate: 500\n"
        "target_sampling_rate: 500\n"
        "target_length: 8\n"
        "source_unit: mV\n"
        "target_unit: mV\n"
        "domain: source_domain\n",
    )
    _write_yaml(
        target_dataset_config,
        "name: TARGET\n"
        "root: /placeholder/target\n"
        "record_id_column: record_id\n"
        "patient_id_column: patient_id\n"
        "sampling_rate: 500\n"
        "target_sampling_rate: 500\n"
        "target_length: 8\n"
        "source_unit: mV\n"
        "target_unit: mV\n"
        "domain: target_domain\n",
    )
    _write_yaml(experiment_config_path, "experiment: uda_coral_source_to_target\nmethod: coral\n")

    experiment_config = {
        "experiment": "uda_coral_source_to_target",
        "method": "coral",
        "source_datasets": ["SOURCE"],
        "target_datasets": ["TARGET"],
        "dataset_configs": {
            "source": str(source_dataset_config),
            "target": str(target_dataset_config),
        },
        "model": {"name": "resnet1d", "width": 4},
        "data": {
            "input_length": 8,
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
        "evaluation": {"selection_metric": "source_validation_macro_auroc"},
        "method_params": {"lambda": 0.5},
        "protocol": {
            "target_inputs_available_during_training": True,
            "target_labels_available_during_training": False,
        },
    }

    monkeypatch.setattr(uda, "_git_state", lambda: ("deadbeef", False))
    monkeypatch.setattr(uda, "_resolve_device", lambda requested: torch.device("cpu"))
    monkeypatch.setattr(
        uda,
        "create_dataset",
        lambda name, root, config: calls.append(f"create_dataset:{name}") or datasets[name],
    )
    monkeypatch.setattr(uda, "build_split_manifest", _fake_split_manifest)
    monkeypatch.setattr(
        uda, "create_model", lambda model_config, *, num_labels: TinyUdaModel(num_labels)
    )

    source_train = np.array(
        [
            [1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1],
        ],
        dtype=int,
    )
    good_scores = source_train * 0.9 + (1 - source_train) * 0.1

    def fake_collect_predictions(
        model,
        batches,
        device,
        amp_enabled,
        *,
        description,
    ):
        del model, batches, device, amp_enabled, description
        return source_train, good_scores

    monkeypatch.setattr(uda, "_collect_predictions", fake_collect_predictions)

    output_dir = tmp_path / "outputs"
    status = uda.run_uda_cross_domain(
        experiment_config=experiment_config,
        experiment_config_path=experiment_config_path,
        source_dataset_spec=uda.DatasetSpec(
            name="SOURCE",
            root=tmp_path / "source_root",
            config=dict(datasets["SOURCE"].config),
            config_path=source_dataset_config,
        ),
        target_dataset_spec=uda.DatasetSpec(
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
    assert datasets["TARGET"].label_calls == 4

    training_log = pd.read_csv(output_dir / "training_log.csv")
    assert training_log.loc[0, "adaptation_loss"] > 0.0
    assert training_log.loc[0, "method_coral_loss"] > 0.0
    assert training_log.loc[0, "method_coral_lambda"] == pytest.approx(0.5)

    status_json = json.loads((output_dir / "run_status.json").read_text(encoding="utf-8"))
    assert status_json["method"] == "coral"
    assert status_json["method_params"] == {"lambda": 0.5}
    assert status_json["method_metadata"]["name"] == "coral"
    assert status_json["feature_extraction"] == "model.forward_features"


def test_ecg_adapt_uda_run_trains_method_parameters_and_records_losses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    source_metadata, source_signals, source_labels = _fake_records("s", 6, length=8)
    target_metadata, target_signals, target_labels = _fake_records("t", 6, length=8)
    datasets = {
        "SOURCE": FakeDataset(
            name="SOURCE",
            domain="source_domain",
            config={
                "record_id_column": "record_id",
                "patient_id_column": "patient_id",
                "sampling_rate": 500,
                "target_sampling_rate": 500,
                "target_length": 8,
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
                "target_length": 8,
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
        "name: SOURCE\n"
        "root: /placeholder/source\n"
        "record_id_column: record_id\n"
        "patient_id_column: patient_id\n"
        "sampling_rate: 500\n"
        "target_sampling_rate: 500\n"
        "target_length: 8\n"
        "source_unit: mV\n"
        "target_unit: mV\n"
        "domain: source_domain\n",
    )
    _write_yaml(
        target_dataset_config,
        "name: TARGET\n"
        "root: /placeholder/target\n"
        "record_id_column: record_id\n"
        "patient_id_column: patient_id\n"
        "sampling_rate: 500\n"
        "target_sampling_rate: 500\n"
        "target_length: 8\n"
        "source_unit: mV\n"
        "target_unit: mV\n"
        "domain: target_domain\n",
    )
    _write_yaml(
        experiment_config_path,
        "experiment: uda_ecg_adapt_source_to_target\nmethod: ecg_adapt\n",
    )

    experiment_config = {
        "experiment": "uda_ecg_adapt_source_to_target",
        "method": "ecg_adapt",
        "source_datasets": ["SOURCE"],
        "target_datasets": ["TARGET"],
        "dataset_configs": {
            "source": str(source_dataset_config),
            "target": str(target_dataset_config),
        },
        "model": {"name": "resnet1d", "width": 4},
        "data": {
            "input_length": 8,
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
        "evaluation": {"selection_metric": "source_validation_macro_auroc"},
        "method_params": {
            "lambda": 0.001,
            "hidden_dims": [3],
            "dropout": 0.0,
            "negative_learning_epochs": 2,
            "pseudo_label_threshold": 0.7,
            "negative_label_threshold": 0.3,
        },
        "protocol": {
            "target_inputs_available_during_training": True,
            "target_labels_available_during_training": False,
        },
    }

    created_method = {}
    recorded_method_epochs: list[int] = []
    evaluation_method_modes: list[bool] = []
    captured_optimizer_parameters: list[torch.nn.Parameter] = []

    def build_method(method_name, *, method_params=None):
        method = real_build_uda_method(method_name, method_params=method_params)
        original_adaptation_loss = method.adaptation_loss

        def recording_adaptation_loss(**kwargs):
            recorded_method_epochs.append(int(kwargs["epoch"]))
            return original_adaptation_loss(**kwargs)

        method.adaptation_loss = recording_adaptation_loss
        created_method["method"] = method
        return method

    def create_capturing_optimizer(parameters, name, learning_rate, weight_decay):
        del name
        captured_optimizer_parameters.extend(list(parameters))
        return torch.optim.Adam(
            captured_optimizer_parameters,
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )

    monkeypatch.setattr(uda, "_git_state", lambda: ("deadbeef", False))
    monkeypatch.setattr(uda, "_resolve_device", lambda requested: torch.device("cpu"))
    monkeypatch.setattr(uda, "build_uda_method", build_method)
    monkeypatch.setattr(uda, "create_optimizer", create_capturing_optimizer)
    monkeypatch.setattr(
        uda,
        "create_dataset",
        lambda name, root, config: calls.append(f"create_dataset:{name}") or datasets[name],
    )
    monkeypatch.setattr(uda, "build_split_manifest", _fake_split_manifest)
    monkeypatch.setattr(
        uda,
        "create_model",
        lambda model_config, *, num_labels: TinyUdaModel(num_labels),
    )

    source_train = np.array(
        [
            [1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1],
        ],
        dtype=int,
    )
    good_scores = source_train * 0.9 + (1 - source_train) * 0.1

    def fake_collect_predictions(
        model,
        batches,
        device,
        amp_enabled,
        *,
        description,
    ):
        del model, batches, device, amp_enabled, description
        evaluation_method_modes.append(created_method["method"].training)
        return source_train, good_scores

    monkeypatch.setattr(uda, "_collect_predictions", fake_collect_predictions)

    output_dir = tmp_path / "outputs"
    status = uda.run_uda_cross_domain(
        experiment_config=experiment_config,
        experiment_config_path=experiment_config_path,
        source_dataset_spec=uda.DatasetSpec(
            name="SOURCE",
            root=tmp_path / "source_root",
            config=dict(datasets["SOURCE"].config),
            config_path=source_dataset_config,
        ),
        target_dataset_spec=uda.DatasetSpec(
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
    assert datasets["TARGET"].label_calls == 4
    method = created_method["method"]
    method_parameter_ids = {id(parameter) for parameter in method.parameters()}
    optimizer_parameter_ids = {id(parameter) for parameter in captured_optimizer_parameters}
    assert method_parameter_ids
    assert method_parameter_ids.issubset(optimizer_parameter_ids)

    training_log = pd.read_csv(output_dir / "training_log.csv")
    assert recorded_method_epochs == [0, 0]
    assert evaluation_method_modes and not any(evaluation_method_modes)
    assert method.training is False
    assert training_log.loc[0, "method_ecg_adapt_stage"] == "positive_plus_negative"
    assert training_log.loc[0, "method_discriminator_loss"] > 0.0
    assert training_log.loc[0, "method_source_positive_domain_loss"] > 0.0
    assert training_log.loc[0, "method_source_negative_domain_loss"] > 0.0
    assert training_log.loc[0, "method_target_negative_domain_loss"] >= 0.0
    assert "method_source_positive_AF_count" in training_log.columns
    assert "method_target_pseudo_positive_AF_rate" in training_log.columns
    assert training_log.loc[0, "method_lambda"] == pytest.approx(0.001)

    checkpoint = torch.load(
        output_dir / "best_checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert checkpoint["method"] == "ecg_adapt_multilabel"
    assert checkpoint["method_params"]["lambda"] == pytest.approx(0.001)
    assert "method_state_dict" in checkpoint
    assert any(key.startswith("discriminator.") for key in checkpoint["method_state_dict"])


def test_method_metric_aggregation_sums_counts_and_averages_other_scalars() -> None:
    accumulator: dict[str, float] = {}
    counts: dict[str, int] = {}

    uda._aggregate_scalar_metrics(
        accumulator,
        counts,
        {"selected_source_positive_count": 2, "source_positive_selection_rate": 0.25},
    )
    uda._aggregate_scalar_metrics(
        accumulator,
        counts,
        {"selected_source_positive_count": 3, "source_positive_selection_rate": 0.75},
    )
    aggregated = uda._finalize_scalar_metrics(accumulator, counts)

    assert aggregated["selected_source_positive_count"] == pytest.approx(5.0)
    assert aggregated["source_positive_selection_rate"] == pytest.approx(0.5)
