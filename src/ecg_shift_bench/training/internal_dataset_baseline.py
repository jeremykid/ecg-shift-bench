"""Internal supervised baseline runner for the four dataset adapters."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
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
from ecg_shift_bench.evaluation.metrics import optimal_multilabel_thresholds, source_script_multilabel_report
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


@dataclass(frozen=True)
class Issue11CompletedRun:
    """Resolved paths for rebuilding results from a completed issue 11 run."""

    dataset_key: str
    dataset_name: str
    source_output_dir: Path
    experiment_config_path: Path
    dataset_config_path: Path
    split_manifest_path: Path
    checkpoint_path: Path
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
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    per_label_reports = report.get("per_label_reports", {})
    per_label_thresholds = report.get("thresholds", {})
    support = report.get("per_label_support", {})
    for label in CANONICAL_LABELS:
        label_report = per_label_reports[label]
        rows.append(
            {
                "dataset": dataset_key,
                "split": split_name,
                "label": label,
                "threshold": float(per_label_thresholds.get(label, float("nan"))),
                "accuracy": float(label_report["accuracy"]),
                "auroc": float(label_report["auroc"]),
                "auprc": float(label_report["auprc"]),
                "f1_score": float(label_report["f1_score"]),
                "prec": float(label_report["prec"]),
                "rec": float(label_report["rec"]),
                "sensitivity": float(label_report["sensitivity"]),
                "spec": float(label_report["spec"]),
                "aprec": float(label_report["aprec"]),
                "br_score": float(label_report["br_score"]),
                "tn": int(label_report["tn"]),
                "fp": int(label_report["fp"]),
                "fn": int(label_report["fn"]),
                "tp": int(label_report["tp"]),
                "support": int(support.get(label, 0)),
            }
        )
    return rows


@torch.no_grad()
def _collect_predictions(
    model: nn.Module,
    batches: Iterable[tuple[Tensor, Tensor]],
    device: torch.device,
    amp_enabled: bool,
    *,
    description: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    targets_all: list[np.ndarray] = []
    scores_all: list[np.ndarray] = []
    for inputs, targets in batches:
        inputs = inputs.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(inputs)
        scores_all.append(torch.sigmoid(logits).float().cpu().numpy())
        targets_all.append(targets.numpy().astype(np.int64, copy=False))
    if not targets_all:
        raise ValueError(f"{description} produced no batches")
    return np.concatenate(targets_all), np.concatenate(scores_all)


def _artifact_paths(output_dir: Path) -> dict[str, str]:
    return {
        "experiment_config": str(output_dir / "experiment_config.yaml"),
        "dataset_config": str(output_dir / "dataset_config.yaml"),
        "split_manifest": str(output_dir / "split_manifest.csv"),
        "reproducibility": str(output_dir / "reproducibility.json"),
        "history": str(output_dir / "history.json"),
        "checkpoint": str(output_dir / "best_checkpoint.pt"),
        "predictions": str(output_dir / "issue11_predictions.npz"),
        "validation_metrics": str(output_dir / "validation_metrics.json"),
        "test_metrics": str(output_dir / "test_metrics.json"),
    }


def _save_issue11_predictions(
    *,
    output_dir: Path,
    split_predictions: dict[str, tuple[np.ndarray, np.ndarray]],
) -> Path:
    payload: dict[str, np.ndarray] = {
        "label_names": np.asarray(CANONICAL_LABELS, dtype="U32"),
    }
    for split_name, (truth, scores) in split_predictions.items():
        payload[f"{split_name}_y_true"] = np.asarray(truth)
        payload[f"{split_name}_y_score"] = np.asarray(scores)
    path = output_dir / "issue11_predictions.npz"
    np.savez_compressed(path, **payload)
    return path


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


def _resolve_artifact_path(path_value: str | Path, *, base_dir: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _load_completed_issue11_run(source_root: Path, dataset_key: str, output_root: Path) -> Issue11CompletedRun:
    source_output_dir = source_root / dataset_key
    status_path = source_output_dir / "run_status.json"
    if not status_path.is_file():
        raise FileNotFoundError(f"Missing completed run status for {dataset_key!r}: {status_path}")
    with status_path.open(encoding="utf-8") as handle:
        dataset_status = json.load(handle)
    artifact_paths = dataset_status.get("artifact_paths", {})
    if not artifact_paths:
        raise ValueError(f"Completed run status for {dataset_key!r} does not include artifact paths")
    return Issue11CompletedRun(
        dataset_key=dataset_key,
        dataset_name=str(dataset_status.get("dataset_name", dataset_key)),
        source_output_dir=source_output_dir,
        experiment_config_path=_resolve_artifact_path(
            artifact_paths["experiment_config"], base_dir=source_output_dir
        ),
        dataset_config_path=_resolve_artifact_path(artifact_paths["dataset_config"], base_dir=source_output_dir),
        split_manifest_path=_resolve_artifact_path(artifact_paths["split_manifest"], base_dir=source_output_dir),
        checkpoint_path=_resolve_artifact_path(artifact_paths["checkpoint"], base_dir=source_output_dir),
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


def _build_model_from_config(experiment_config: dict[str, Any], device: torch.device) -> nn.Module:
    model = ResNet1D(
        in_channels=int(experiment_config["model"]["in_channels"]),
        num_labels=len(CANONICAL_LABELS),
        width=int(experiment_config["model"]["width"]),
    ).to(device)
    return model


def _load_model_checkpoint(
    *,
    experiment_config: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    model = _build_model_from_config(experiment_config, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


def _make_issue11_loaders(
    *,
    dataset: BaseECGDataset,
    splits: dict[str, pd.DataFrame],
    input_length: int,
    batch_size: int,
    workers: int,
    seed: int,
    pin_memory: bool,
    shuffle_train: bool,
) -> dict[str, DataLoader[tuple[Tensor, Tensor]]]:
    datasets = {
        split_name: Issue11ClassificationDataset(dataset, frame, input_length)
        for split_name, frame in splits.items()
    }
    return {
        split_name: _make_loader(
            split_dataset,
            batch_size=batch_size,
            workers=workers,
            shuffle=shuffle_train and split_name == "train",
            seed=seed,
            pin_memory=pin_memory,
        )
        for split_name, split_dataset in datasets.items()
    }


def _evaluate_issue11_splits(
    *,
    model: nn.Module,
    loaders: dict[str, DataLoader[tuple[Tensor, Tensor]]],
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    split_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split_name in ("train", "validation", "test"):
        truth, scores = _collect_predictions(
            model,
            loaders[split_name],
            device,
            amp_enabled,
            description=f"{split_name} evaluation",
        )
        split_predictions[split_name] = (truth, scores)
    return split_predictions


def _summarize_issue11_predictions(
    *,
    dataset_key: str,
    dataset_name: str,
    output_dir: Path,
    split_counts: dict[str, int],
    best_epoch: int,
    best_validation_macro_auprc: float,
    split_predictions: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    train_truth, train_scores = split_predictions["train"]
    thresholds = optimal_multilabel_thresholds(train_truth, train_scores, CANONICAL_LABELS)

    validation_truth, validation_scores = split_predictions["validation"]
    validation_metrics = source_script_multilabel_report(
        validation_truth,
        validation_scores,
        CANONICAL_LABELS,
        thresholds,
    )
    validation_metrics["num_records"] = int(validation_truth.shape[0])

    test_truth, test_scores = split_predictions["test"]
    test_metrics = source_script_multilabel_report(
        test_truth,
        test_scores,
        CANONICAL_LABELS,
        thresholds,
    )
    test_metrics["num_records"] = int(test_truth.shape[0])

    def _count_sum(report: dict[str, Any], key: str) -> int:
        per_label_reports = report["per_label_reports"]
        return int(sum(int(per_label_reports[label][key]) for label in CANONICAL_LABELS))

    def _macro(report: dict[str, Any], key: str) -> float:
        return float(report[f"macro_{key}"])

    summary_row = {
        "dataset": dataset_key,
        "dataset_name": dataset_name,
        "output_dir": str(output_dir),
        "train_records": int(split_counts["train"]),
        "validation_records": int(split_counts["validation"]),
        "test_records": int(split_counts["test"]),
        "best_epoch": int(best_epoch),
        "best_validation_macro_auprc": float(best_validation_macro_auprc),
        "validation_accuracy": _macro(validation_metrics, "accuracy"),
        "validation_auroc": _macro(validation_metrics, "auroc"),
        "validation_auprc": _macro(validation_metrics, "auprc"),
        "validation_f1_score": _macro(validation_metrics, "f1_score"),
        "validation_prec": _macro(validation_metrics, "prec"),
        "validation_rec": _macro(validation_metrics, "rec"),
        "validation_sensitivity": _macro(validation_metrics, "sensitivity"),
        "validation_spec": _macro(validation_metrics, "spec"),
        "validation_aprec": _macro(validation_metrics, "aprec"),
        "validation_br_score": _macro(validation_metrics, "br_score"),
        "validation_tn": _count_sum(validation_metrics, "tn"),
        "validation_fp": _count_sum(validation_metrics, "fp"),
        "validation_fn": _count_sum(validation_metrics, "fn"),
        "validation_tp": _count_sum(validation_metrics, "tp"),
        "test_accuracy": _macro(test_metrics, "accuracy"),
        "test_auroc": _macro(test_metrics, "auroc"),
        "test_auprc": _macro(test_metrics, "auprc"),
        "test_f1_score": _macro(test_metrics, "f1_score"),
        "test_prec": _macro(test_metrics, "prec"),
        "test_rec": _macro(test_metrics, "rec"),
        "test_sensitivity": _macro(test_metrics, "sensitivity"),
        "test_spec": _macro(test_metrics, "spec"),
        "test_aprec": _macro(test_metrics, "aprec"),
        "test_br_score": _macro(test_metrics, "br_score"),
        "test_tn": _count_sum(test_metrics, "tn"),
        "test_fp": _count_sum(test_metrics, "fp"),
        "test_fn": _count_sum(test_metrics, "fn"),
        "test_tp": _count_sum(test_metrics, "tp"),
    }

    per_class_rows = []
    per_class_rows.extend(
        _flatten_metric_rows(dataset_key=dataset_key, split_name="validation", report=validation_metrics)
    )
    per_class_rows.extend(_flatten_metric_rows(dataset_key=dataset_key, split_name="test", report=test_metrics))

    return summary_row, per_class_rows, validation_metrics, test_metrics


def _finalize_internal_dataset_baseline_outputs(
    *,
    output_root: Path,
    command: str,
    requested_device: str,
    experiment_config: dict[str, Any],
    completed_runs: list[dict[str, Any]],
    dataset_rows: list[dict[str, Any]],
    per_class_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    summary_frame = pd.DataFrame(dataset_rows)
    summary_frame.to_csv(output_root / "results_summary.csv", index=False)
    per_class_frame = pd.DataFrame(per_class_rows)
    per_class_frame.to_csv(output_root / "per_class_summary.csv", index=False)
    figure_paths = write_internal_dataset_baseline_result_figures(output_root)
    root_status = {
        "experiment_id": experiment_config["experiment"],
        "status": "completed",
        "command": command,
        "git_commit": _git_state()[0],
        "git_dirty": _git_state()[1],
        "requested_device": requested_device,
        "started_at": None,
        "finished_at": _utc_now(),
        "artifact_paths": {
            "results_summary": str(output_root / "results_summary.csv"),
            "per_class_summary": str(output_root / "per_class_summary.csv"),
            "run_status": str(output_root / "run_status.json"),
            **figure_paths,
        },
        "completed_datasets": [run["dataset"] for run in completed_runs],
        "source_root": completed_runs[0].get("source_root") if completed_runs else None,
    }
    return root_status


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
    model = _build_model_from_config(experiment_config, device)
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
    eval_loaders = _make_issue11_loaders(
        dataset=dataset,
        splits=splits,
        input_length=input_length,
        batch_size=batch_size,
        workers=workers,
        seed=seed,
        pin_memory=device.type == "cuda",
        shuffle_train=False,
    )
    split_predictions = _evaluate_issue11_splits(
        model=model,
        loaders=eval_loaders,
        device=device,
        amp_enabled=amp_enabled,
    )
    _save_issue11_predictions(output_dir=run.output_dir, split_predictions=split_predictions)
    summary_row, per_class_rows, validation_metrics, test_metrics = _summarize_issue11_predictions(
        dataset_key=run.dataset_key,
        dataset_name=run.dataset_name,
        output_dir=run.output_dir,
        split_counts={split_name: int(len(frame)) for split_name, frame in splits.items()},
        best_epoch=int(checkpoint["epoch"]),
        best_validation_macro_auprc=float(checkpoint["validation_macro_auprc"]),
        split_predictions=split_predictions,
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
        "summary_row": summary_row,
        "per_class_rows": per_class_rows,
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
    completed_runs: list[dict[str, Any]] = []
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
            completed_runs.append(result)
            if preflight_only:
                continue
            dataset_rows.append(result["summary_row"])
            per_class_rows.extend(result["per_class_rows"])

        if preflight_only:
            root_status["status"] = "preflight_completed"
            root_status["finished_at"] = _utc_now()
            root_status["completed_datasets"] = [result["dataset"] for result in completed_runs]
            write_json(root_status_path, root_status)
            return root_status
        pd.DataFrame(dataset_rows).to_csv(output_root / "results_summary.csv", index=False)
        pd.DataFrame(per_class_rows).to_csv(output_root / "per_class_summary.csv", index=False)
        figure_paths = write_internal_dataset_baseline_result_figures(output_root)
        root_status["status"] = "completed"
        root_status["finished_at"] = _utc_now()
        root_status["artifact_paths"]["results_summary"] = str(output_root / "results_summary.csv")
        root_status["artifact_paths"]["per_class_summary"] = str(output_root / "per_class_summary.csv")
        root_status["artifact_paths"].update(figure_paths)
        root_status["completed_datasets"] = [result["dataset"] for result in completed_runs]
        write_json(root_status_path, root_status)
        return root_status
    except BaseException as error:
        root_status["status"] = "failed"
        root_status["finished_at"] = _utc_now()
        root_status["failure"] = {"type": type(error).__name__, "message": str(error)}
        write_json(root_status_path, root_status)
        raise


def rebuild_internal_dataset_baseline_results(
    *,
    source_root: str | Path = "outputs/issue-11",
    output_root: str | Path = "outputs/resnet1d_internal_dataset_baseline_results",
    requested_device: str = "cpu",
    command: str = "scripts/train.py --rebuild-results-from outputs/issue-11",
) -> dict[str, Any]:
    """Rebuild the issue 11 result tables from an already completed run."""
    source_root_path = Path(source_root).expanduser().resolve()
    output_root_path = Path(output_root).expanduser().resolve()
    source_status_path = source_root_path / "run_status.json"
    if not source_status_path.is_file():
        raise FileNotFoundError(f"Missing completed-run root status: {source_status_path}")
    with source_status_path.open(encoding="utf-8") as handle:
        source_status = json.load(handle)
    if str(source_status.get("status")) != "completed":
        raise ValueError(f"Source run root must be completed, got {source_status.get('status')!r}")

    completed_dataset_keys = list(source_status.get("completed_datasets") or [])
    if not completed_dataset_keys:
        completed_dataset_keys = sorted(
            path.name for path in source_root_path.iterdir() if (path / "run_status.json").is_file()
        )
    if not completed_dataset_keys:
        raise ValueError(f"No completed dataset runs found under {source_root_path}")

    output_root_path.mkdir(parents=True, exist_ok=True)
    root_status_path = output_root_path / "run_status.json"
    commit, dirty = _git_state()
    root_status = {
        "experiment_id": source_status.get("experiment_id", "issue11-internal-resnet1d-baseline"),
        "status": "running",
        "command": command,
        "git_commit": commit,
        "git_dirty": dirty,
        "requested_device": requested_device,
        "started_at": _utc_now(),
        "finished_at": None,
        "artifact_paths": {
            "results_summary": str(output_root_path / "results_summary.csv"),
            "per_class_summary": str(output_root_path / "per_class_summary.csv"),
            "run_status": str(root_status_path),
        },
        "source_root": str(source_root_path),
        "completed_datasets": completed_dataset_keys,
    }
    write_json(root_status_path, root_status)

    dataset_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    try:
        for dataset_key in completed_dataset_keys:
            print(f"Rebuilding results for {dataset_key}...", flush=True)
            completed_run = _load_completed_issue11_run(source_root_path, dataset_key, output_root_path)
            completed_run.output_dir.mkdir(parents=True, exist_ok=True)
            experiment_config = load_yaml(completed_run.experiment_config_path)
            dataset_config = load_yaml(completed_run.dataset_config_path)
            split_manifest = _load_split_manifest(completed_run.split_manifest_path)
            splits = _prepare_split_frames(split_manifest)
            dataset = create_dataset(completed_run.dataset_name, dataset_config.get("root", "."), dataset_config)

            device = _resolve_device(requested_device)
            amp_enabled = device.type == "cuda" and bool(experiment_config["training"].get("amp", False))
            if amp_enabled:
                os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            seed = int(experiment_config["training"]["seed"])
            seed_everything(seed)
            input_length = int(experiment_config["data"]["input_length"])
            evaluation_batch_size = max(int(experiment_config["training"]["batch_size"]), 256)
            loaders = _make_issue11_loaders(
                dataset=dataset,
                splits=splits,
                input_length=input_length,
                batch_size=evaluation_batch_size,
                workers=0,
                seed=seed,
                pin_memory=device.type == "cuda",
                shuffle_train=False,
            )
            model, checkpoint = _load_model_checkpoint(
                experiment_config=experiment_config,
                checkpoint_path=completed_run.checkpoint_path,
                device=device,
            )
            split_predictions = _evaluate_issue11_splits(
                model=model,
                loaders=loaders,
                device=device,
                amp_enabled=amp_enabled,
            )
            _save_issue11_predictions(
                output_dir=completed_run.output_dir,
                split_predictions=split_predictions,
            )
            summary_row, per_class_result_rows, validation_metrics, test_metrics = _summarize_issue11_predictions(
                dataset_key=dataset_key,
                dataset_name=completed_run.dataset_name,
                output_dir=completed_run.output_dir,
                split_counts={split_name: int(len(frame)) for split_name, frame in splits.items()},
                best_epoch=int(checkpoint["epoch"]),
                best_validation_macro_auprc=float(checkpoint["validation_macro_auprc"]),
                split_predictions=split_predictions,
            )
            dataset_rows.append(summary_row)
            per_class_rows.extend(per_class_result_rows)
            print(f"Finished {dataset_key}", flush=True)
        pd.DataFrame(dataset_rows).to_csv(output_root_path / "results_summary.csv", index=False)
        pd.DataFrame(per_class_rows).to_csv(output_root_path / "per_class_summary.csv", index=False)
        figure_paths = write_internal_dataset_baseline_result_figures(output_root_path)
        root_status["status"] = "completed"
        root_status["finished_at"] = _utc_now()
        root_status["artifact_paths"].update(figure_paths)
        write_json(root_status_path, root_status)
        return {
            "output_root": str(output_root_path),
            "results_summary": str(output_root_path / "results_summary.csv"),
            "per_class_summary": str(output_root_path / "per_class_summary.csv"),
            "figures": figure_paths,
            "status": root_status,
        }
    except BaseException as error:
        root_status["status"] = "failed"
        root_status["finished_at"] = _utc_now()
        root_status["failure"] = {"type": type(error).__name__, "message": str(error)}
        write_json(root_status_path, root_status)
        raise
