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
    summary = pd.read_csv(output_root / "results_summary.csv")
    assert summary["output_dir"].tolist() == list(datasets)
    assert not summary["output_dir"].astype(str).str.startswith("/").any()
    run_status = json.loads((output_root / "run_status.json").read_text(encoding="utf-8"))
    assert run_status["artifact_paths"]["results_summary"] == "results_summary.csv"
    for name in datasets:
        dataset_output = output_root / name
        assert (dataset_output / "run_status.json").is_file()
        assert (dataset_output / "validation_metrics.json").is_file()
        assert (dataset_output / "test_metrics.json").is_file()
        assert (dataset_output / "internal_baseline_predictions.npz").is_file()
        assert (dataset_output / "generate_report" / f"{name}_validation_report.json").is_file()
        assert (dataset_output / "generate_report" / f"{name}_test_report.json").is_file()
        assert (dataset_output / "best_checkpoint.pt").is_file()
        assert (dataset_output / "split_manifest.csv").is_file()
        dataset_status = json.loads((dataset_output / "run_status.json").read_text(encoding="utf-8"))
        assert not any(str(value).startswith("/") for value in dataset_status["artifact_paths"].values())


def test_rebuild_internal_baseline_results_from_completed_run(
    tmp_path: Path, monkeypatch
) -> None:
    from ecg_shift_bench.training import internal_dataset_baseline as baseline

    source_root = tmp_path / "source"
    output_root = tmp_path / "outputs" / "resnet1d_internal_dataset_baseline_results"
    source_root.mkdir(parents=True)

    record_ids = ["r1", "r2", "r3", "r4", "r5", "r6"]
    labels = {
        record_id: {
            label: int((index + offset) % 2 == 0)
            for offset, label in enumerate(CANONICAL_LABELS)
        }
        for index, record_id in enumerate(record_ids, start=1)
    }
    datasets = {
        name: FakeDataset(record_ids, labels, input_length=64)
        for name in ("ptbxl", "code15", "chapman", "sph")
    }

    completed_datasets = list(datasets)
    source_status = {
        "experiment_id": "resnet1d-internal-dataset-baseline",
        "status": "completed",
        "completed_datasets": completed_datasets,
    }
    (source_root / "run_status.json").write_text(json.dumps(source_status), encoding="utf-8")

    experiment_config = {
        "experiment": "resnet1d-internal-dataset-baseline",
        "model": {"name": "resnet1d", "in_channels": 12, "width": 4},
        "data": {"input_length": 64, "target_sampling_rate": 500, "target_length": 64, "unit": "mV"},
        "training": {
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
    }
    dataset_config = {"name": "fake", "root": str(tmp_path / "fake-data")}

    config_root = tmp_path / "configs"
    config_root.mkdir()
    split_root = tmp_path / "splits"
    split_root.mkdir()

    def _write_split_manifest(path: Path) -> None:
        pd.DataFrame(
            {
                "record_id": record_ids,
                "patient_id": [f"p{index}" for index in range(1, 7)],
                "split": ["train", "train", "validation", "validation", "test", "test"],
                "domain": ["fake_domain"] * 6,
            }
        ).to_csv(path, index=False)

    completed_runs: dict[str, baseline.InternalBaselineCompletedRun] = {}
    for dataset_key in completed_datasets:
        dataset_dir = source_root / dataset_key
        dataset_dir.mkdir()
        experiment_config_path = config_root / f"{dataset_key}_experiment.yaml"
        dataset_config_path = config_root / f"{dataset_key}_dataset.yaml"
        split_manifest_path = split_root / f"{dataset_key}_split_manifest.csv"
        checkpoint_path = dataset_dir / "best_checkpoint.pt"
        experiment_config_path.write_text(json.dumps(experiment_config), encoding="utf-8")
        dataset_config_path.write_text(json.dumps(dataset_config), encoding="utf-8")
        _write_split_manifest(split_manifest_path)
        checkpoint_path.write_bytes(b"checkpoint-not-used-in-test")
        completed_runs[dataset_key] = baseline.InternalBaselineCompletedRun(
            dataset_key=dataset_key,
            dataset_name=dataset_key,
            source_output_dir=dataset_dir,
            experiment_config_path=experiment_config_path,
            dataset_config_path=dataset_config_path,
            split_manifest_path=split_manifest_path,
            checkpoint_path=checkpoint_path,
            output_dir=output_root / dataset_key,
        )

    split_predictions = {
        "train": (
            np.array(
                [
                    [1, 0, 1, 0, 1, 0],
                    [0, 1, 0, 1, 0, 1],
                    [1, 1, 0, 0, 1, 1],
                    [0, 0, 1, 1, 0, 0],
                ]
            ),
            np.array(
                [
                    [0.9, 0.1, 0.9, 0.1, 0.9, 0.1],
                    [0.1, 0.9, 0.1, 0.9, 0.1, 0.9],
                    [0.9, 0.9, 0.1, 0.1, 0.9, 0.9],
                    [0.1, 0.1, 0.9, 0.9, 0.1, 0.1],
                ]
            ),
        ),
        "validation": (
            np.array(
                [
                    [0, 0, 1, 0, 1, 0],
                    [0, 1, 0, 1, 0, 1],
                    [0, 1, 0, 0, 1, 1],
                    [0, 0, 1, 1, 0, 0],
                ]
            ),
            np.array(
                [
                    [0.1, 0.1, 0.9, 0.1, 0.9, 0.1],
                    [0.1, 0.9, 0.1, 0.9, 0.1, 0.9],
                    [0.1, 0.9, 0.1, 0.1, 0.9, 0.9],
                    [0.1, 0.1, 0.9, 0.9, 0.1, 0.1],
                ]
            ),
        ),
        "test": (
            np.array(
                [
                    [0, 0, 1, 0, 1, 0],
                    [0, 1, 0, 1, 0, 1],
                    [0, 1, 0, 0, 1, 1],
                    [0, 0, 1, 1, 0, 0],
                ]
            ),
            np.array(
                [
                    [0.1, 0.1, 0.9, 0.1, 0.9, 0.1],
                    [0.1, 0.9, 0.1, 0.9, 0.1, 0.9],
                    [0.1, 0.9, 0.1, 0.1, 0.9, 0.9],
                    [0.1, 0.1, 0.9, 0.9, 0.1, 0.1],
                ]
            ),
        ),
    }

    monkeypatch.setattr(
        baseline,
        "_load_completed_internal_baseline_run",
        lambda *args, **kwargs: completed_runs[args[1]],
    )
    monkeypatch.setattr(baseline, "create_dataset", lambda name, root, config=None: datasets[name])
    monkeypatch.setattr(
        baseline,
        "_load_model_checkpoint",
        lambda **kwargs: (baseline.nn.Module(), {"epoch": 4, "validation_macro_auprc": 0.91}),
    )
    monkeypatch.setattr(
        baseline,
        "_evaluate_internal_baseline_splits",
        lambda **kwargs: split_predictions,
    )

    result = baseline.rebuild_internal_dataset_baseline_results(
        source_root=source_root,
        output_root=output_root,
        requested_device="cpu",
        command="python scripts/train.py --rebuild-results-from source",
    )

    assert result["status"]["status"] == "completed"
    assert (output_root / "results_summary.csv").is_file()
    assert (output_root / "per_class_summary.csv").is_file()
    assert (output_root / "README.md").is_file()
    assert (output_root / "resnet1d_internal_dataset_baseline_overall_metrics.png").is_file()
    assert (output_root / "resnet1d_internal_dataset_baseline_per_label_metrics.png").is_file()
    assert Path(result["status"]["source_root"]).name == "source"
    assert result["status"]["artifact_paths"]["results_summary"] == "results_summary.csv"
    for dataset_key in completed_datasets:
        dataset_output = output_root / dataset_key
        assert (dataset_output / "internal_baseline_predictions.npz").is_file()
        assert (dataset_output / "generate_report" / f"{dataset_key}_validation_report.json").is_file()
        assert (dataset_output / "generate_report" / f"{dataset_key}_test_report.json").is_file()

    summary = pd.read_csv(output_root / "results_summary.csv")
    assert list(summary["dataset"]) == completed_datasets
    assert "validation_f1_score" in summary.columns
    assert "validation_auprc" in summary.columns
    assert "validation_aprec" in summary.columns
    assert "test_spec" in summary.columns
    assert "test_br_score" in summary.columns

    per_class = pd.read_csv(output_root / "per_class_summary.csv")
    chapman_nan = per_class.loc[
        (per_class["dataset"] == "chapman")
        & (per_class["split"] == "validation")
        & (per_class["label"] == "AF"),
        "auroc",
    ].iloc[0]
    assert np.isnan(chapman_nan)
    assert "prec" in per_class.columns
    assert "rec" in per_class.columns
    assert "aprec" in per_class.columns
    assert "br_score" in per_class.columns

    readme = (output_root / "README.md").read_text(encoding="utf-8")
    assert "generate_report.py" in readme


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
