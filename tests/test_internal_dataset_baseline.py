from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from ecg_shift_bench.labels.canonical import CANONICAL_LABELS


class FakeDataset:
    def __init__(self, record_ids: list[str], labels: dict[str, dict[str, int]], input_length: int) -> None:
        self.record_ids = record_ids
        self.labels = labels
        self.input_length = input_length
        self.name = "FAKE"
        self.domain = "fake_domain"

    def load_metadata(self) -> pd.DataFrame:
        return pd.DataFrame({"record_id": self.record_ids})

    def load_aligned_signal(self, record_id: str) -> np.ndarray:
        value = float(int(record_id[-1]) if record_id[-1].isdigit() else 1)
        return np.full((12, self.input_length), value, dtype=np.float32)

    def get_labels(self, record_id: str) -> dict[str, int]:
        return self.labels[record_id]


def _write_split_manifest(path: Path) -> None:
    frame = pd.DataFrame(
        {
            "record_id": ["1", "2", "3", "4", "5", "6"],
            "patient_id": ["p1", "p2", "p3", "p4", "p5", "p6"],
            "split": ["train", "train", "validation", "validation", "test", "test"],
            "domain": ["fake_domain"] * 6,
        }
    )
    frame.to_csv(path, index=False)


def test_internal_baseline_run_writes_per_dataset_outputs_and_summary(
    tmp_path: Path, monkeypatch
) -> None:
    from ecg_shift_bench.training import internal_dataset_baseline as baseline

    labels = {}
    for index, record_id in enumerate(["1", "2", "3", "4", "5", "6"], start=1):
        labels[record_id] = {
            label: int((index + offset) % 2 == 0)
            for offset, label in enumerate(CANONICAL_LABELS)
        }

    datasets = {
        "ptbxl": FakeDataset(["1", "2", "3", "4", "5", "6"], labels, input_length=64),
        "code15": FakeDataset(["1", "2", "3", "4", "5", "6"], labels, input_length=64),
        "chapman": FakeDataset(["1", "2", "3", "4", "5", "6"], labels, input_length=64),
        "sph": FakeDataset(["1", "2", "3", "4", "5", "6"], labels, input_length=64),
    }

    def fake_create_dataset(name: str, root, config=None):
        return datasets[name]

    monkeypatch.setattr(baseline, "create_dataset", fake_create_dataset)
    monkeypatch.setattr(baseline, "preflight_forward_backward", lambda *args, **kwargs: 0.123)
    monkeypatch.setattr(
        baseline,
        "train_epoch",
        lambda *args, **kwargs: {
            "loss": 0.5,
            "min_batch_loss": 0.4,
            "max_batch_loss": 0.6,
            "steps": 1,
            "samples": 2,
        },
    )

    def fake_evaluate(model, batches, device, amp_enabled, *, description):
        del model, batches, device, amp_enabled, description
        return {
            "macro_auroc": 0.8,
            "micro_auroc": 0.81,
            "macro_auprc": 0.82,
            "micro_auprc": 0.83,
            "per_label_auroc": {label: 0.7 for label in CANONICAL_LABELS},
            "per_label_auprc": {label: 0.6 for label in CANONICAL_LABELS},
            "per_label_support": {label: 1 for label in CANONICAL_LABELS},
            "num_records": 2,
        }

    monkeypatch.setattr(baseline, "evaluate", fake_evaluate)

    for dataset_name in datasets:
        split_dir = tmp_path / "splits" / dataset_name
        split_dir.mkdir(parents=True)
        _write_split_manifest(split_dir / "split_manifest.csv")
        dataset_dir = tmp_path / "dataset_configs"
        dataset_dir.mkdir(exist_ok=True)
        config_path = dataset_dir / f"{dataset_name}.yaml"
        config_path.write_text("name: fake\nroot: /tmp/fake\n", encoding="utf-8")

    experiment_config = {
        "experiment": "resnet1d-internal-dataset-baseline",
        "method": "source_only",
        "source_datasets": ["PTBXL", "CODE15", "CHAPMAN", "SPH"],
        "target_datasets": [],
        "canonical_labels": list(CANONICAL_LABELS),
        "model": {"name": "resnet1d", "in_channels": 12, "width": 4},
        "data": {"input_length": 64, "target_sampling_rate": 500, "target_length": 64, "unit": "mV"},
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "workers": 0,
            "optimizer": "adamw",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "seed": 42,
            "amp": False,
        },
        "evaluation": {
            "selection_metric": "validation_macro_auprc",
            "metrics": [
                "macro_auroc",
                "micro_auroc",
                "macro_auprc",
                "micro_auprc",
                "per_label_auroc",
                "per_label_auprc",
                "per_label_support",
            ],
        },
        "baseline_results": {
            "output_root": str(tmp_path / "outputs"),
            "datasets": {
                name: {
                    "dataset": name,
                    "dataset_config": str(tmp_path / "dataset_configs" / f"{name}.yaml"),
                    "split_manifest": str(tmp_path / "splits" / name / "split_manifest.csv"),
                }
                for name in datasets
            },
        },
    }
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(json.dumps(experiment_config), encoding="utf-8")

    status = baseline.run_internal_dataset_baseline(
        experiment_config=experiment_config,
        experiment_config_path=config_path,
        requested_device="cpu",
        command="python scripts/train.py --config baseline.yaml",
    )

    assert status["status"] == "completed"
    output_root = Path(experiment_config["baseline_results"]["output_root"])
    assert (output_root / "results_summary.csv").is_file()
    assert (output_root / "per_class_summary.csv").is_file()
    assert (output_root / "README.md").is_file()
    for name in datasets:
        dataset_output = output_root / name
        assert (dataset_output / "run_status.json").is_file()
        assert (dataset_output / "validation_metrics.json").is_file()
        assert (dataset_output / "test_metrics.json").is_file()
        assert (dataset_output / "best_checkpoint.pt").is_file()
        assert (dataset_output / "split_manifest.csv").is_file()


def test_internal_baseline_runner_rejects_missing_dataset_config(tmp_path: Path) -> None:
    from ecg_shift_bench.training import internal_dataset_baseline as baseline

    config = {
        "experiment": "resnet1d-internal-dataset-baseline",
        "method": "source_only",
        "source_datasets": ["PTBXL", "CODE15", "CHAPMAN", "SPH"],
        "target_datasets": [],
        "canonical_labels": list(CANONICAL_LABELS),
        "model": {"name": "resnet1d", "in_channels": 12, "width": 4},
        "data": {"input_length": 64, "target_sampling_rate": 500, "target_length": 64, "unit": "mV"},
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "workers": 0,
            "optimizer": "adamw",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "seed": 42,
            "amp": False,
        },
        "evaluation": {"selection_metric": "validation_macro_auprc", "metrics": []},
        "baseline_results": {"output_root": str(tmp_path / "outputs"), "datasets": {}},
    }

    try:
        baseline.run_internal_dataset_baseline(
            experiment_config=config,
            experiment_config_path=tmp_path / "baseline.yaml",
            requested_device="cpu",
            command="python scripts/train.py --config baseline.yaml",
        )
    except ValueError as error:
        assert "dataset" in str(error).lower()
    else:
        raise AssertionError("Expected missing-dataset config to fail")
