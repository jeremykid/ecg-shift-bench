"""Internal supervised baseline runner for the four dataset adapters."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from ecg_shift_bench.datasets.base import BaseECGDataset
from ecg_shift_bench.datasets.registry import create_dataset
from ecg_shift_bench.internal_dataset_baseline_reporting import (
    write_internal_dataset_baseline_result_figures,
)
from ecg_shift_bench.labels.canonical import CANONICAL_LABELS
from ecg_shift_bench.labels.harmonize import labels_to_vector
from ecg_shift_bench.models.resnet1d import ResNet1D
from ecg_shift_bench.training.ptbxl_baseline import (
    _git_state,
    _make_loader,
    _resolve_device,
    evaluate,
    preflight_forward_backward,
    train_epoch,
    write_json,
)
from ecg_shift_bench.utils.config import load_yaml
from ecg_shift_bench.utils.seed import seed_everything


@dataclass(frozen=True)
class Issue11Run:
    """Resolved paths and dataset settings for one internal baseline run."""

    dataset_key: str
    dataset_name: str
    dataset_config_path: Path
    split_manifest_path: Path
    output_dir: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class Issue11ClassificationDataset(Dataset[tuple[Tensor, Tensor]]):
    """Load aligned waveforms and canonical multilabel targets."""

    def __init__(
        self,
        dataset: BaseECGDataset,
        metadata: pd.DataFrame,
        input_length: int,
    ) -> None:
        self.dataset = dataset
        self.metadata = metadata.reset_index(drop=True)
        self.input_length = int(input_length)
        record_column = "record_id"
        self.record_ids = self.metadata[record_column].astype(str).tolist()
        self.targets = np.asarray(
            [labels_to_vector(self.dataset.get_labels(record_id)) for record_id in self.record_ids],
            dtype=np.float32,
        )

    def __len__(self) -> int:
        return len(self.record_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record_id = self.record_ids[index]
        signal = self.dataset.load_aligned_signal(record_id)
        expected_shape = (12, self.input_length)
        if signal.shape != expected_shape:
            raise ValueError(
                f"{self.dataset.name} record {record_id!r} has shape {signal.shape}, "
                f"expected {expected_shape}"
            )
        if not np.isfinite(signal).all():
            raise ValueError(f"{self.dataset.name} record {record_id!r} contains non-finite values")
        return torch.from_numpy(signal), torch.from_numpy(self.targets[index].copy())


def _load_split_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    required = {"record_id", "split"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Split manifest {path} is missing columns: {sorted(missing)}")
    return manifest


def _prepare_split_frames(manifest: pd.DataFrame) -> dict[str, pd.DataFrame]:
    splits = {name: frame.copy().reset_index(drop=True) for name, frame in manifest.groupby("split")}
    for name in ("train", "validation", "test"):
        if name not in splits:
            raise ValueError(f"Split manifest is missing required split {name!r}")
    return splits


def _flatten_metric_rows(
    *,
    dataset_key: str,
    split_name: str,
    metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    per_label_auroc = metrics.get("per_label_auroc", {})
    per_label_auprc = metrics.get("per_label_auprc", {})
    support = metrics.get("per_label_support", {})
    for label in CANONICAL_LABELS:
        rows.append(
            {
                "dataset": dataset_key,
                "split": split_name,
                "label": label,
                "auroc": float(per_label_auroc.get(label, float("nan"))),
                "auprc": float(per_label_auprc.get(label, float("nan"))),
                "support": int(support.get(label, 0)),
            }
        )
    return rows


def _artifact_paths(output_dir: Path) -> dict[str, str]:
    return {
        "experiment_config": str(output_dir / "experiment_config.yaml"),
        "dataset_config": str(output_dir / "dataset_config.yaml"),
        "split_manifest": str(output_dir / "split_manifest.csv"),
        "reproducibility": str(output_dir / "reproducibility.json"),
        "history": str(output_dir / "history.json"),
        "checkpoint": str(output_dir / "best_checkpoint.pt"),
        "validation_metrics": str(output_dir / "validation_metrics.json"),
        "test_metrics": str(output_dir / "test_metrics.json"),
    }


def _copy_text_config(source: Path, destination: Path) -> None:
    destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def _resolve_dataset_run(dataset_key: str, dataset_run: dict[str, Any], output_root: Path) -> Issue11Run:
    return Issue11Run(
        dataset_key=dataset_key,
        dataset_name=str(dataset_run["dataset"]),
        dataset_config_path=Path(dataset_run["dataset_config"]).expanduser().resolve(),
        split_manifest_path=Path(dataset_run["split_manifest"]).expanduser().resolve(),
        output_dir=output_root / dataset_key,
    )


def _build_reproducibility_note(
    *,
    run: Issue11Run,
    experiment_config: dict[str, Any],
    dataset_config: dict[str, Any],
    split_manifest: pd.DataFrame,
) -> dict[str, Any]:
    data_config = experiment_config["data"]
    training_config = experiment_config["training"]
    evaluation_config = experiment_config["evaluation"]
    return {
        "dataset": run.dataset_key,
        "dataset_name": run.dataset_name,
        "split_manifest_path": str(run.split_manifest_path),
        "split_manifest_sha256": _sha256_file(run.split_manifest_path),
        "dataset_config_path": str(run.dataset_config_path),
        "records": {
            "train": int((split_manifest["split"] == "train").sum()),
            "validation": int((split_manifest["split"] == "validation").sum()),
            "test": int((split_manifest["split"] == "test").sum()),
        },
        "preprocessing": {
            "input_length": int(data_config["input_length"]),
            "target_sampling_rate": int(data_config["target_sampling_rate"]),
            "target_length": int(data_config["target_length"]),
            "unit": str(data_config.get("unit", "mV")),
        },
        "model": {
            "name": str(experiment_config["model"]["name"]),
            "in_channels": int(experiment_config["model"]["in_channels"]),
            "width": int(experiment_config["model"]["width"]),
            "num_labels": len(CANONICAL_LABELS),
        },
        "training": {
            "optimizer": str(training_config["optimizer"]),
            "learning_rate": float(training_config["learning_rate"]),
            "weight_decay": float(training_config["weight_decay"]),
            "batch_size": int(training_config["batch_size"]),
            "epochs": int(training_config["epochs"]),
            "workers": int(training_config["workers"]),
            "seed": int(training_config["seed"]),
        },
        "evaluation": {
            "selection_metric": str(evaluation_config["selection_metric"]),
            "metrics": list(evaluation_config["metrics"]),
        },
    }


def _run_single_dataset(
    *,
    run: Issue11Run,
    experiment_config: dict[str, Any],
    experiment_config_path: Path,
    requested_device: str,
    command: str,
    preflight_only: bool = False,
) -> dict[str, Any]:
    run.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_config = load_yaml(run.dataset_config_path)
    split_manifest = _load_split_manifest(run.split_manifest_path)
    splits = _prepare_split_frames(split_manifest)
    dataset = create_dataset(run.dataset_name, dataset_config.get("root", "."), dataset_config)

    seed = int(experiment_config["training"]["seed"])
    commit, dirty = _git_state()
    status_path = run.output_dir / "run_status.json"
    artifact_paths = _artifact_paths(run.output_dir)
    status: dict[str, Any] = {
        "experiment_id": experiment_config["experiment"],
        "dataset": run.dataset_key,
        "dataset_name": run.dataset_name,
        "status": "running",
        "command": command,
        "git_commit": commit,
        "git_dirty": dirty,
        "seed": seed,
        "requested_device": requested_device,
        "started_at": _utc_now(),
        "finished_at": None,
        "artifact_paths": artifact_paths,
        "selection_metric": experiment_config["evaluation"]["selection_metric"],
    }
    write_json(status_path, status)

    _copy_text_config(experiment_config_path, Path(artifact_paths["experiment_config"]))
    _copy_text_config(run.dataset_config_path, Path(artifact_paths["dataset_config"]))
    split_manifest.to_csv(artifact_paths["split_manifest"], index=False)

    reproducibility = _build_reproducibility_note(
        run=run,
        experiment_config=experiment_config,
        dataset_config=dataset_config,
        split_manifest=split_manifest,
    )
    write_json(Path(artifact_paths["reproducibility"]), reproducibility)

    device = _resolve_device(requested_device)
    amp_enabled = device.type == "cuda" and bool(experiment_config["training"].get("amp", False))
    if amp_enabled:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    status["resolved_device"] = str(device)
    status["device_name"] = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    status["amp_fp16"] = amp_enabled
    write_json(status_path, status)

    input_length = int(experiment_config["data"]["input_length"])
    datasets = {
        split_name: Issue11ClassificationDataset(dataset, frame, input_length)
        for split_name, frame in splits.items()
    }
    batch_size = int(experiment_config["training"]["batch_size"])
    workers = int(experiment_config["training"]["workers"])

    def make_loaders() -> dict[str, DataLoader[tuple[Tensor, Tensor]]]:
        return {
            split_name: _make_loader(
                split_dataset,
                batch_size=batch_size,
                workers=workers,
                shuffle=split_name == "train",
                seed=seed,
                pin_memory=device.type == "cuda",
            )
            for split_name, split_dataset in datasets.items()
        }

    seed_everything(seed)
    model = ResNet1D(
        in_channels=int(experiment_config["model"]["in_channels"]),
        num_labels=len(CANONICAL_LABELS),
        width=int(experiment_config["model"]["width"]),
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()

    preflight_loaders = make_loaders()
    preflight_loss = preflight_forward_backward(
        model,
        next(iter(preflight_loaders["train"])),
        criterion,
        device,
        amp_enabled,
    )
    status["preflight"] = {"status": "passed", "loss": preflight_loss}
    write_json(status_path, status)

    if preflight_only:
        status["status"] = "preflight_completed"
        status["finished_at"] = _utc_now()
        write_json(status_path, status)
        return {
            "dataset": run.dataset_key,
            "dataset_name": run.dataset_name,
            "output_dir": str(run.output_dir),
            "preflight_loss": preflight_loss,
            "split_counts": {split_name: int(len(frame)) for split_name, frame in splits.items()},
            "status": status,
        }

    loaders = make_loaders()
    optimizer_name = str(experiment_config["training"]["optimizer"]).lower()
    if optimizer_name != "adamw":
        raise ValueError(f"Expected optimizer 'adamw', got {optimizer_name!r}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(experiment_config["training"]["learning_rate"]),
        weight_decay=float(experiment_config["training"]["weight_decay"]),
    )

    history: list[dict[str, Any]] = []
    best_score = float("-inf")
    checkpoint_path = Path(artifact_paths["checkpoint"])
    epochs = int(experiment_config["training"]["epochs"])
    for epoch in range(1, epochs + 1):
        train_summary = train_epoch(
            model,
            loaders["train"],
            optimizer,
            criterion,
            device,
            amp_enabled,
            description=f"{run.dataset_key} train {epoch}/{epochs}",
        )
        validation_metrics = evaluate(
            model,
            loaders["validation"],
            device,
            amp_enabled,
            description=f"{run.dataset_key} validation {epoch}/{epochs}",
        )
        history.append(
            {
                "epoch": epoch,
                "train": train_summary,
                "validation": validation_metrics,
            }
        )
        write_json(Path(artifact_paths["history"]), {"epochs": history})
        score = float(validation_metrics["macro_auprc"])
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "experiment_id": experiment_config["experiment"],
                    "dataset": run.dataset_key,
                    "model_name": "resnet1d",
                    "canonical_labels": CANONICAL_LABELS,
                    "epoch": epoch,
                    "validation_macro_auprc": score,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    validation_metrics = evaluate(
        model,
        loaders["validation"],
        device,
        amp_enabled,
        description=f"{run.dataset_key} validation best",
    )
    test_metrics = evaluate(
        model,
        loaders["test"],
        device,
        amp_enabled,
        description=f"{run.dataset_key} test best",
    )
    write_json(Path(artifact_paths["validation_metrics"]), validation_metrics)
    write_json(Path(artifact_paths["test_metrics"]), test_metrics)

    status["status"] = "completed"
    status["finished_at"] = _utc_now()
    status["best_epoch"] = int(checkpoint["epoch"])
    status["best_validation_macro_auprc"] = float(checkpoint["validation_macro_auprc"])
    write_json(status_path, status)

    return {
        "dataset": run.dataset_key,
        "dataset_name": run.dataset_name,
        "output_dir": str(run.output_dir),
        "best_epoch": int(checkpoint["epoch"]),
        "best_validation_macro_auprc": float(checkpoint["validation_macro_auprc"]),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "split_counts": {split_name: int(len(frame)) for split_name, frame in splits.items()},
        "status": status,
    }


def run_internal_dataset_baseline(
    *,
    experiment_config: dict[str, Any],
    experiment_config_path: Path,
    requested_device: str,
    command: str,
    output_root_override: str | Path | None = None,
    preflight_only: bool = False,
) -> dict[str, Any]:
    """Run the internal baselines and write a combined summary."""
    baseline_config = experiment_config.get("baseline_results")
    if not baseline_config:
        raise ValueError("Baseline config is missing the 'baseline_results' section")
    dataset_runs = baseline_config.get("datasets", {})
    if not dataset_runs:
        raise ValueError("Baseline config must define at least one dataset run")

    output_root = (
        Path(output_root_override).expanduser().resolve()
        if output_root_override is not None
        else Path(
            baseline_config.get("output_root", "outputs/resnet1d_internal_dataset_baseline_results")
        ).expanduser().resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)
    root_status_path = output_root / "run_status.json"
    commit, dirty = _git_state()
    root_status = {
        "experiment_id": experiment_config["experiment"],
        "status": "running",
        "command": command,
        "git_commit": commit,
        "git_dirty": dirty,
        "requested_device": requested_device,
        "started_at": _utc_now(),
        "finished_at": None,
        "artifact_paths": {
            "results_summary": str(output_root / "results_summary.csv"),
            "per_class_summary": str(output_root / "per_class_summary.csv"),
            "run_status": str(root_status_path),
        },
    }
    write_json(root_status_path, root_status)

    dataset_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    completed_runs: dict[str, Any] = {}
    try:
        for dataset_key, dataset_run in dataset_runs.items():
            resolved_run = _resolve_dataset_run(dataset_key, dataset_run, output_root)
            result = _run_single_dataset(
                run=resolved_run,
                experiment_config=experiment_config,
                experiment_config_path=experiment_config_path,
                requested_device=requested_device,
                command=command,
                preflight_only=preflight_only,
            )
            completed_runs[dataset_key] = result
            if preflight_only:
                continue
            dataset_rows.append(
                {
                    "dataset": dataset_key,
                    "dataset_name": result["dataset_name"],
                    "output_dir": str(
                        Path(baseline_config.get("output_root", "outputs/resnet1d_internal_dataset_baseline_results"))
                        / dataset_key
                    ),
                    "train_records": result["split_counts"]["train"],
                    "validation_records": result["split_counts"]["validation"],
                    "test_records": result["split_counts"]["test"],
                    "best_epoch": result["best_epoch"],
                    "best_validation_macro_auprc": result["best_validation_macro_auprc"],
                    "validation_macro_auroc": float(result["validation_metrics"]["macro_auroc"]),
                    "validation_micro_auroc": float(result["validation_metrics"]["micro_auroc"]),
                    "validation_macro_auprc": float(result["validation_metrics"]["macro_auprc"]),
                    "validation_micro_auprc": float(result["validation_metrics"]["micro_auprc"]),
                    "test_macro_auroc": float(result["test_metrics"]["macro_auroc"]),
                    "test_micro_auroc": float(result["test_metrics"]["micro_auroc"]),
                    "test_macro_auprc": float(result["test_metrics"]["macro_auprc"]),
                    "test_micro_auprc": float(result["test_metrics"]["micro_auprc"]),
                }
            )
            per_class_rows.extend(
                _flatten_metric_rows(
                    dataset_key=dataset_key,
                    split_name="validation",
                    metrics=result["validation_metrics"],
                )
            )
            per_class_rows.extend(
                _flatten_metric_rows(
                    dataset_key=dataset_key,
                    split_name="test",
                    metrics=result["test_metrics"],
                )
            )

        if preflight_only:
            root_status["status"] = "preflight_completed"
            root_status["finished_at"] = _utc_now()
            root_status["completed_datasets"] = list(completed_runs)
            write_json(root_status_path, root_status)
            return root_status
        summary_frame = pd.DataFrame(dataset_rows)
        summary_frame.to_csv(output_root / "results_summary.csv", index=False)
        per_class_frame = pd.DataFrame(per_class_rows)
        per_class_frame.to_csv(output_root / "per_class_summary.csv", index=False)
        figure_paths = write_internal_dataset_baseline_result_figures(output_root)
        root_status["status"] = "completed"
        root_status["finished_at"] = _utc_now()
        root_status["artifact_paths"]["results_summary"] = str(output_root / "results_summary.csv")
        root_status["artifact_paths"]["per_class_summary"] = str(output_root / "per_class_summary.csv")
        root_status["artifact_paths"].update(figure_paths)
        root_status["completed_datasets"] = list(completed_runs)
        write_json(root_status_path, root_status)
        return root_status
    except BaseException as error:
        root_status["status"] = "failed"
        root_status["finished_at"] = _utc_now()
        root_status["failure"] = {"type": type(error).__name__, "message": str(error)}
        write_json(root_status_path, root_status)
        raise
