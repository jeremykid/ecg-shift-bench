"""Generic UDA training and reporting workflow."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ecg_shift_bench.datasets.audit import build_split_manifest
from ecg_shift_bench.datasets.base import BaseECGDataset
from ecg_shift_bench.datasets.registry import create_dataset
from ecg_shift_bench.evaluation.metrics import (
    multilabel_metrics,
    optimal_multilabel_thresholds,
    source_script_multilabel_report,
)
from ecg_shift_bench.labels.canonical import CANONICAL_LABELS
from ecg_shift_bench.labels.harmonize import labels_to_vector
from ecg_shift_bench.methods.uda import UdaMethod, build_uda_method
from ecg_shift_bench.models.registry import canonical_model_name, create_model
from ecg_shift_bench.training.optim import create_optimizer
from ecg_shift_bench.training.ptbxl_baseline import _git_state, _resolve_device
from ecg_shift_bench.utils.config import load_yaml
from ecg_shift_bench.utils.seed import seed_everything

SPLIT_ORDER = ("source_train", "source_validation", "target_validation", "target_test")
REPORT_METRICS = (
    "macro_accuracy",
    "macro_auroc",
    "macro_auprc",
    "macro_f1_score",
    "macro_prec",
    "macro_rec",
    "macro_sensitivity",
    "macro_spec",
    "macro_aprec",
    "macro_br_score",
)
SCORE_METRICS = ("macro_auroc", "micro_auroc", "macro_auprc", "micro_auprc")
EVALUATION_METRICS = [
    *REPORT_METRICS,
    "score_macro_auroc",
    "score_micro_auroc",
    "score_macro_auprc",
    "score_micro_auprc",
]


@dataclass(frozen=True)
class DatasetSpec:
    """Resolved dataset configuration for one UDA run."""

    name: str
    root: Path
    config: dict[str, Any]
    config_path: Path


class AlignedClassificationDataset(Dataset[tuple[Tensor, Tensor]]):
    """Load aligned ECGs and canonical multi-label targets from one dataset."""

    def __init__(
        self,
        dataset: BaseECGDataset,
        metadata: pd.DataFrame,
        input_length: int,
    ) -> None:
        self.dataset = dataset
        self.metadata = metadata.reset_index(drop=True).copy()
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
        return torch.from_numpy(signal.copy()), torch.from_numpy(self.targets[index].copy())


class UnlabeledAlignedDataset(Dataset[Tensor]):
    """Load aligned ECGs without exposing labels to the training step."""

    def __init__(
        self,
        dataset: BaseECGDataset,
        metadata: pd.DataFrame,
        input_length: int,
    ) -> None:
        self.dataset = dataset
        self.metadata = metadata.reset_index(drop=True).copy()
        self.input_length = int(input_length)
        self.record_ids = self.metadata["record_id"].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.record_ids)

    def __getitem__(self, index: int) -> Tensor:
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
        return torch.from_numpy(signal.copy())


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _copy_text_config(source: Path, destination: Path) -> None:
    destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def _write_yaml_config(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _split_frames(manifest: pd.DataFrame) -> dict[str, pd.DataFrame]:
    splits = {
        name: frame.copy().reset_index(drop=True) for name, frame in manifest.groupby("split")
    }
    for name in ("train", "validation", "test"):
        if name not in splits:
            raise ValueError(f"Split manifest is missing required split {name!r}")
    return splits


def _split_version(policy: dict[str, Any]) -> str:
    method = str(policy.get("method", "unknown"))
    split_source = str(policy.get("split_source", "unknown"))
    seed = policy.get("seed")
    if split_source == "generated" and seed is not None:
        return f"{method}_seed{seed}"
    return method


def _combined_split_version(source_policy: dict[str, Any], target_policy: dict[str, Any]) -> str:
    return f"source:{_split_version(source_policy)}|target:{_split_version(target_policy)}"


def _safe_float(value: Any) -> float:
    if value is None:
        return float("nan")
    number = float(value)
    return number if math.isfinite(number) else float("nan")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write standards-compliant JSON."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _save_numpy_bundle(
    path: Path,
    *,
    label_names: list[str],
    split_predictions: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    payload: dict[str, np.ndarray] = {
        "label_names": np.asarray(label_names, dtype="U32"),
    }
    for split_name, (truth, scores) in split_predictions.items():
        payload[f"{split_name}_y_true"] = np.asarray(truth)
        payload[f"{split_name}_y_score"] = np.asarray(scores)
    np.savez_compressed(path, **payload)


def _load_numpy_bundle(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as bundle:
        if "label_names" not in bundle.files:
            raise ValueError(f"{path} is missing label_names")
        label_names = [
            str(label) for label in np.asarray(bundle["label_names"], dtype=object).tolist()
        ]
        split_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for split_name in SPLIT_ORDER:
            truth_key = f"{split_name}_y_true"
            score_key = f"{split_name}_y_score"
            if truth_key not in bundle.files or score_key not in bundle.files:
                raise ValueError(f"{path} is missing {truth_key!r} or {score_key!r}")
            split_predictions[split_name] = (
                np.asarray(bundle[truth_key]),
                np.asarray(bundle[score_key]),
            )
    return {"label_names": label_names, "split_predictions": split_predictions}


def _report_payload(
    *,
    split_name: str,
    report: dict[str, Any],
    score_metrics: dict[str, Any],
    metadata: dict[str, Any],
    num_records: int,
) -> dict[str, Any]:
    payload = dict(metadata)
    payload["split"] = split_name
    payload["num_records"] = int(num_records)
    for key in REPORT_METRICS:
        payload[key] = _safe_float(report[key])
    for key in SCORE_METRICS:
        payload[f"score_{key}"] = _safe_float(score_metrics[key])
    payload["thresholds"] = _json_safe(report["thresholds"])
    payload["per_label_reports"] = _json_safe(report["per_label_reports"])
    payload["per_label_support"] = _json_safe(report["per_label_support"])
    return payload


def _build_summary_row(
    *,
    metadata: dict[str, Any],
    split_counts: dict[str, int],
    split_payloads: dict[str, dict[str, Any]],
    best_epoch: int,
    best_selection_score: float,
    selection_metric: str,
) -> dict[str, Any]:
    row = dict(metadata)
    row.update(
        {
            "source_train_records": int(split_counts["source_train"]),
            "source_validation_records": int(split_counts["source_validation"]),
            "target_validation_records": int(split_counts["target_validation"]),
            "target_test_records": int(split_counts["target_test"]),
            "best_epoch": int(best_epoch),
            "selection_metric": selection_metric,
            "best_selection_score": _safe_float(best_selection_score),
        }
    )
    for prefix, payload in split_payloads.items():
        for key in REPORT_METRICS:
            row[f"{prefix}_{key}"] = _safe_float(payload[key])
        for key in SCORE_METRICS:
            row[f"{prefix}_score_{key}"] = _safe_float(payload[f"score_{key}"])
    return row


def _build_per_class_rows(
    *,
    metadata: dict[str, Any],
    payloads: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name in SPLIT_ORDER:
        payload = payloads[split_name]
        thresholds = payload["thresholds"]
        per_label_reports = payload["per_label_reports"]
        per_label_support = payload["per_label_support"]
        for label in CANONICAL_LABELS:
            label_report = per_label_reports[label]
            rows.append(
                {
                    **metadata,
                    "split": split_name,
                    "label": label,
                    "threshold": _safe_float(thresholds[label]),
                    "accuracy": _safe_float(label_report["accuracy"]),
                    "auroc": _safe_float(label_report["auroc"]),
                    "auprc": _safe_float(label_report["auprc"]),
                    "f1_score": _safe_float(label_report["f1_score"]),
                    "prec": _safe_float(label_report["prec"]),
                    "rec": _safe_float(label_report["rec"]),
                    "sensitivity": _safe_float(label_report["sensitivity"]),
                    "spec": _safe_float(label_report["spec"]),
                    "aprec": _safe_float(label_report["aprec"]),
                    "br_score": _safe_float(label_report["br_score"]),
                    "tn": int(label_report["tn"]),
                    "fp": int(label_report["fp"]),
                    "fn": int(label_report["fn"]),
                    "tp": int(label_report["tp"]),
                    "support": int(per_label_support[label]),
                }
            )
    return rows


def _write_tables(
    *,
    output_dir: Path,
    summary_row: dict[str, Any],
    per_class_rows: list[dict[str, Any]],
) -> dict[str, str]:
    summary_path = output_dir / "results_summary.csv"
    per_class_path = output_dir / "per_class_summary.csv"
    pd.DataFrame([summary_row]).to_csv(summary_path, index=False)
    pd.DataFrame(per_class_rows).to_csv(per_class_path, index=False)
    return {
        "results_summary": str(summary_path),
        "per_class_summary": str(per_class_path),
    }


def _write_result_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_training_log(output_dir: Path, rows: list[dict[str, Any]]) -> Path:
    path = output_dir / "training_log.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "experiment_config": output_dir / "experiment_config.yaml",
        "source_dataset_config": output_dir / "source_dataset_config.yaml",
        "target_dataset_config": output_dir / "target_dataset_config.yaml",
        "source_split_manifest": output_dir / "source_split_manifest.csv",
        "target_split_manifest": output_dir / "target_split_manifest.csv",
        "best_checkpoint": output_dir / "best_checkpoint.pt",
        "predictions": output_dir / "predictions.npz",
        "train_metrics": output_dir / "train_metrics.json",
        "validation_metrics": output_dir / "validation_metrics.json",
        "target_validation_metrics": output_dir / "target_validation_metrics.json",
        "test_metrics": output_dir / "test_metrics.json",
        "training_log": output_dir / "training_log.csv",
        "results_summary": output_dir / "results_summary.csv",
        "per_class_summary": output_dir / "per_class_summary.csv",
        "run_status": output_dir / "run_status.json",
    }


def _resolve_root(root_override: str | Path | None, config: dict[str, Any]) -> Path:
    if root_override is not None:
        return Path(root_override).expanduser().resolve()
    return Path(str(config.get("root", "."))).expanduser().resolve()


def _prepare_dataset_spec(
    *,
    dataset_name: str,
    dataset_config_path: Path,
    root_override: str | Path | None,
) -> DatasetSpec:
    dataset_config = load_yaml(dataset_config_path)
    root = _resolve_root(root_override, dataset_config)
    dataset_config = dict(dataset_config)
    dataset_config["root"] = str(root)
    return DatasetSpec(
        name=dataset_name,
        root=root,
        config=dataset_config,
        config_path=dataset_config_path,
    )


def _selection_metric_key(selection_metric: str) -> tuple[str, str]:
    normalized = str(selection_metric).strip().lower().replace("-", "_")
    if normalized in {"source_validation_macro_auroc", "macro_auroc"}:
        return "macro_auroc", "source_validation_macro_auroc"
    if normalized in {"source_validation_macro_auprc", "macro_auprc"}:
        return "macro_auprc", "source_validation_macro_auprc"
    raise ValueError(
        "Unsupported UDA selection metric; expected source_validation_macro_auroc or "
        "source_validation_macro_auprc"
    )


def _source_target_contract(
    *,
    experiment_config: dict[str, Any],
    source_dataset_spec: DatasetSpec,
    target_dataset_spec: DatasetSpec,
) -> None:
    source_datasets = list(experiment_config.get("source_datasets") or [])
    target_datasets = list(experiment_config.get("target_datasets") or [])
    if len(source_datasets) != 1 or len(target_datasets) != 1:
        raise ValueError("UDA runs require exactly one source and one target dataset")
    if str(source_datasets[0]).upper() != source_dataset_spec.name.upper():
        raise ValueError("Source dataset config does not match the experiment config")
    if str(target_datasets[0]).upper() != target_dataset_spec.name.upper():
        raise ValueError("Target dataset config does not match the experiment config")


def _model_supports_features(model: nn.Module) -> None:
    if not hasattr(model, "forward_features"):
        raise TypeError(f"{type(model).__name__} must define forward_features() for UDA training")
    if not hasattr(model, "head"):
        raise TypeError(f"{type(model).__name__} must expose a head module for UDA training")


def _model_forward_features(model: nn.Module, inputs: Tensor) -> Tensor:
    features = model.forward_features(inputs)  # type: ignore[attr-defined]
    if not isinstance(features, Tensor):
        raise TypeError("forward_features() must return a torch.Tensor")
    if features.ndim != 2:
        raise ValueError(f"forward_features() must return a 2D tensor, got {features.shape}")
    return features


def _model_forward_logits(model: nn.Module, features: Tensor) -> Tensor:
    logits = model.head(features)  # type: ignore[attr-defined]
    if not isinstance(logits, Tensor):
        raise TypeError("model.head(...) must return a torch.Tensor")
    return logits


def _uda_run_metadata(
    *,
    experiment_config: dict[str, Any],
    source_dataset_spec: DatasetSpec,
    target_dataset_spec: DatasetSpec,
    source_policy: dict[str, Any],
    target_policy: dict[str, Any],
    input_length: int,
    method: UdaMethod,
) -> dict[str, Any]:
    return {
        "experiment_id": str(experiment_config["experiment"]),
        "method": method.method_name,
        "method_params": _json_safe(method.method_params),
        "source_dataset": source_dataset_spec.name,
        "target_dataset": target_dataset_spec.name,
        "source_domain": source_dataset_spec.config.get("domain"),
        "target_domain": target_dataset_spec.config.get("domain"),
        "label_space": "canonical_six_label",
        "canonical_labels": "|".join(CANONICAL_LABELS),
        "model_architecture": canonical_model_name(str(experiment_config["model"]["name"])),
        "feature_extraction": "model.forward_features",
        "evaluation_metrics": "|".join(EVALUATION_METRICS),
        "random_seed": int(experiment_config["training"]["seed"]),
        "source_split_version": _split_version(source_policy),
        "target_split_version": _split_version(target_policy),
        "split_version": _combined_split_version(source_policy, target_policy),
        "preprocessing_version": str(experiment_config["data"]["preprocessing_version"]),
        "input_length": int(input_length),
        "source_root": str(source_dataset_spec.root),
        "target_root": str(target_dataset_spec.root),
    }


def _zero_like(reference: Tensor) -> Tensor:
    return reference.new_zeros(())


def _aggregate_scalar_metrics(
    accumulator: dict[str, float],
    counts: dict[str, int],
    metrics: dict[str, Any],
) -> None:
    for key, value in metrics.items():
        if isinstance(value, (bool, str)):
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            accumulator[key] = accumulator.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1


def _finalize_scalar_metrics(
    accumulator: dict[str, float],
    counts: dict[str, int],
) -> dict[str, float]:
    return {key: (total / counts[key]) for key, total in accumulator.items() if counts.get(key, 0)}


def _cycled_batches(loader: DataLoader[Any]) -> Iterator[Any]:
    while True:
        yield from loader


def _uda_step(
    *,
    model: nn.Module,
    method: UdaMethod,
    source_batch: tuple[Tensor, Tensor],
    target_batch: Tensor,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    epoch: int,
    step: int,
) -> tuple[Tensor, Tensor, Tensor, dict[str, Any]]:
    source_inputs, source_targets = source_batch
    source_inputs = source_inputs.to(device, non_blocking=True)
    source_targets = source_targets.to(device, non_blocking=True).float()
    target_inputs = target_batch.to(device, non_blocking=True)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
        source_features = _model_forward_features(model, source_inputs)
        source_logits = _model_forward_logits(model, source_features)
        source_loss = criterion(source_logits, source_targets)
        target_features = _model_forward_features(model, target_inputs)
        target_logits = _model_forward_logits(model, target_features)
        adaptation_loss, method_metrics = method.adaptation_loss(
            source_features=source_features,
            target_features=target_features,
            source_logits=source_logits,
            target_logits=target_logits,
            source_targets=source_targets,
            epoch=epoch,
            step=step,
        )
        if adaptation_loss.ndim != 0:
            raise ValueError("Adaptation loss must be a scalar tensor")
        total_loss = source_loss + adaptation_loss
    return total_loss, source_loss, adaptation_loss, method_metrics


def preflight_forward_backward(
    model: nn.Module,
    method: UdaMethod,
    source_batch: tuple[Tensor, Tensor],
    target_batch: Tensor,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, Any]:
    """Check one paired source/target batch without changing the model state."""
    original_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    model.train()
    model.zero_grad(set_to_none=True)
    total_loss, source_loss, adaptation_loss, method_metrics = _uda_step(
        model=model,
        method=method,
        source_batch=source_batch,
        target_batch=target_batch,
        criterion=criterion,
        device=device,
        amp_enabled=amp_enabled,
        epoch=1,
        step=1,
    )
    if not torch.isfinite(total_loss):
        raise FloatingPointError(f"Preflight loss is non-finite: {float(total_loss.detach())}")
    total_loss.backward()
    if not any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("Preflight backward pass produced no gradients")
    model.zero_grad(set_to_none=True)
    model.load_state_dict(original_state)
    return {
        "status": "passed",
        "source_loss": float(source_loss.detach()),
        "adaptation_loss": float(adaptation_loss.detach()),
        "loss": float(total_loss.detach()),
        "method_metrics": _json_safe(method_metrics),
    }


def train_uda_epoch(
    model: nn.Module,
    source_batches: DataLoader[tuple[Tensor, Tensor]],
    target_batches: DataLoader[Tensor],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    method: UdaMethod,
    device: torch.device,
    amp_enabled: bool,
    *,
    description: str,
    epoch: int,
) -> dict[str, float | int | dict[str, float]]:
    """Train one epoch with paired source and target batches."""
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    total_loss = 0.0
    total_source_loss = 0.0
    total_adaptation_loss = 0.0
    total_samples = 0
    minimum = float("inf")
    maximum = float("-inf")
    steps = 0
    extra_totals: dict[str, float] = {}
    extra_counts: dict[str, int] = {}
    progress = tqdm(source_batches, desc=description, leave=False)
    target_iter = _cycled_batches(target_batches)
    for step, source_batch in enumerate(progress, start=1):
        target_batch = next(target_iter)
        optimizer.zero_grad(set_to_none=True)
        total_batch_loss, source_loss, adaptation_loss, method_metrics = _uda_step(
            model=model,
            method=method,
            source_batch=source_batch,
            target_batch=target_batch,
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
            epoch=epoch,
            step=step,
        )
        if not torch.isfinite(total_batch_loss):
            raise FloatingPointError(
                f"Non-finite training loss at step {step}: {total_batch_loss.item()}"
            )
        scaler.scale(total_batch_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = source_batch[0].shape[0]
        loss_value = float(total_batch_loss.detach())
        source_loss_value = float(source_loss.detach())
        adaptation_loss_value = float(adaptation_loss.detach())
        total_loss += loss_value * batch_size
        total_source_loss += source_loss_value * batch_size
        total_adaptation_loss += adaptation_loss_value * batch_size
        total_samples += batch_size
        minimum = min(minimum, loss_value)
        maximum = max(maximum, loss_value)
        steps += 1
        _aggregate_scalar_metrics(extra_totals, extra_counts, method_metrics)
        progress.set_postfix(loss=f"{loss_value:.5f}")
    if total_samples == 0:
        raise ValueError("Training loader produced no batches")
    summary: dict[str, float | int | dict[str, float]] = {
        "loss": total_loss / total_samples,
        "source_loss": total_source_loss / total_samples,
        "adaptation_loss": total_adaptation_loss / total_samples,
        "min_batch_loss": minimum,
        "max_batch_loss": maximum,
        "steps": steps,
        "samples": total_samples,
    }
    extra_metrics = _finalize_scalar_metrics(extra_totals, extra_counts)
    if extra_metrics:
        summary["method_metrics"] = extra_metrics
    return summary


@torch.no_grad()
def _collect_predictions(
    model: nn.Module,
    batches: DataLoader[tuple[Tensor, Tensor]],
    device: torch.device,
    amp_enabled: bool,
    *,
    description: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    truth_all: list[np.ndarray] = []
    score_all: list[np.ndarray] = []
    for inputs, targets in batches:
        inputs = inputs.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            features = _model_forward_features(model, inputs)
            logits = _model_forward_logits(model, features)
        score_all.append(torch.sigmoid(logits).float().cpu().numpy())
        truth_all.append(targets.numpy().astype(np.int64, copy=False))
    if not truth_all:
        raise ValueError(f"{description} produced no batches")
    return np.concatenate(truth_all), np.concatenate(score_all)


def _build_model(model_config: dict[str, Any], num_labels: int, device: torch.device) -> nn.Module:
    model_name = canonical_model_name(str(model_config["name"]))
    if model_name != "resnet1d":
        raise ValueError(
            f"UDA runs require resnet1d for the current feature hook, got {model_name!r}"
        )
    return create_model(model_config, num_labels=num_labels).to(device)


def run_uda_cross_domain(
    *,
    experiment_config: dict[str, Any],
    experiment_config_path: Path,
    source_dataset_spec: DatasetSpec,
    target_dataset_spec: DatasetSpec,
    output_dir: Path,
    requested_device: str,
    command: str,
    preflight_only: bool = False,
) -> dict[str, Any]:
    """Train on labeled source batches and unlabeled target batches."""
    _source_target_contract(
        experiment_config=experiment_config,
        source_dataset_spec=source_dataset_spec,
        target_dataset_spec=target_dataset_spec,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = _artifact_paths(output_dir)
    status_path = artifact_paths["run_status"]
    commit, dirty = _git_state()
    seed = int(experiment_config["training"]["seed"])
    input_length = int(experiment_config["data"]["input_length"])
    method = build_uda_method(
        str(experiment_config.get("method", "source_only")),
        method_params=experiment_config.get("method_params") or {},
    )
    selection_metric = str(
        (experiment_config.get("evaluation") or {}).get(
            "selection_metric", "source_validation_macro_auroc"
        )
    )
    selection_report_key, selection_metric_name = _selection_metric_key(selection_metric)
    metadata = _uda_run_metadata(
        experiment_config=experiment_config,
        source_dataset_spec=source_dataset_spec,
        target_dataset_spec=target_dataset_spec,
        source_policy={"method": "unknown"},
        target_policy={"method": "unknown"},
        input_length=input_length,
        method=method,
    )
    status: dict[str, Any] = {
        **metadata,
        "status": "running",
        "command": command,
        "git_commit": commit,
        "git_dirty": dirty,
        "requested_device": requested_device,
        "started_at": _utc_now(),
        "finished_at": None,
        "artifact_paths": {key: str(path) for key, path in artifact_paths.items()},
        "protocol": {
            "source_labels_available_during_training": True,
            "target_labels_available_during_training": False,
            "target_inputs_available_during_training": True,
            "model_updates_during_testing": False,
            "normalization": str(experiment_config["data"].get("normalization", "none")),
        },
        "selection_metric": selection_metric_name,
        "selection_report_key": selection_report_key,
        "method_metadata": method.metadata(),
        "recovery": {
            "resume_supported": False,
            "action": "Rerun the recorded command; incomplete epochs are not checkpointed.",
        },
    }
    write_json(status_path, status)

    _copy_text_config(experiment_config_path, artifact_paths["experiment_config"])
    _write_yaml_config(artifact_paths["source_dataset_config"], source_dataset_spec.config)
    _write_yaml_config(artifact_paths["target_dataset_config"], target_dataset_spec.config)

    device = _resolve_device(requested_device)
    amp_enabled = device.type == "cuda" and bool(experiment_config["training"].get("amp", False))
    if amp_enabled:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    status["resolved_device"] = str(device)
    status["device_name"] = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    status["amp_fp16"] = amp_enabled
    write_json(status_path, status)

    source_dataset = create_dataset(
        source_dataset_spec.name, source_dataset_spec.root, source_dataset_spec.config
    )
    target_dataset = create_dataset(
        target_dataset_spec.name, target_dataset_spec.root, target_dataset_spec.config
    )
    source_metadata = source_dataset.load_metadata()
    target_metadata = target_dataset.load_metadata()
    source_manifest, source_policy = build_split_manifest(source_dataset, source_metadata)
    target_manifest, target_policy = build_split_manifest(target_dataset, target_metadata)
    source_manifest.to_csv(artifact_paths["source_split_manifest"], index=False)
    target_manifest.to_csv(artifact_paths["target_split_manifest"], index=False)
    source_splits = _split_frames(source_manifest)
    target_splits = _split_frames(target_manifest)

    source_train_dataset = AlignedClassificationDataset(
        source_dataset, source_splits["train"], input_length
    )
    source_train_eval_dataset = AlignedClassificationDataset(
        source_dataset,
        source_splits["train"],
        input_length,
    )
    source_validation_dataset = AlignedClassificationDataset(
        source_dataset,
        source_splits["validation"],
        input_length,
    )
    target_validation_dataset = AlignedClassificationDataset(
        target_dataset,
        target_splits["validation"],
        input_length,
    )
    target_test_dataset = AlignedClassificationDataset(
        target_dataset,
        target_splits["test"],
        input_length,
    )
    target_train_unlabeled_dataset = UnlabeledAlignedDataset(
        target_dataset,
        target_splits["train"],
        input_length,
    )

    batch_size = int(experiment_config["training"]["batch_size"])
    workers = int(experiment_config["training"]["workers"])
    source_train_loader = DataLoader(
        source_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=None,
    )
    source_train_eval_loader = DataLoader(
        source_train_eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    source_validation_loader = DataLoader(
        source_validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    target_validation_loader = DataLoader(
        target_validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    target_test_loader = DataLoader(
        target_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    target_train_loader = DataLoader(
        target_train_unlabeled_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )

    status["source_split_policy"] = source_policy
    status["target_split_policy"] = target_policy
    status["source_split_version"] = _split_version(source_policy)
    status["target_split_version"] = _split_version(target_policy)
    write_json(status_path, status)

    seed_everything(seed)
    model = _build_model(experiment_config["model"], len(CANONICAL_LABELS), device)
    _model_supports_features(model)
    criterion = nn.BCEWithLogitsLoss()
    preflight = preflight_forward_backward(
        model,
        method,
        next(iter(source_train_loader)),
        next(iter(target_train_loader)),
        criterion,
        device,
        amp_enabled,
    )
    status["preflight"] = preflight
    write_json(status_path, status)

    if preflight_only:
        status["status"] = "preflight_completed"
        status["finished_at"] = _utc_now()
        write_json(status_path, status)
        return status

    optimizer = create_optimizer(
        model.parameters(),
        str(experiment_config["training"]["optimizer"]),
        float(experiment_config["training"]["learning_rate"]),
        float(experiment_config["training"].get("weight_decay", 0.0)),
    )
    epochs = int(experiment_config["training"]["epochs"])
    best_epoch = 0
    best_score = float("-inf")
    best_checkpoint = artifact_paths["best_checkpoint"]
    training_log_rows: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        train_summary = train_uda_epoch(
            model,
            source_train_loader,
            target_train_loader,
            optimizer,
            criterion,
            method,
            device,
            amp_enabled,
            description=f"source-target train {epoch}/{epochs}",
            epoch=epoch,
        )
        train_truth, train_scores = _collect_predictions(
            model,
            source_train_eval_loader,
            device,
            amp_enabled,
            description=f"source train eval {epoch}/{epochs}",
        )
        thresholds = optimal_multilabel_thresholds(train_truth, train_scores, CANONICAL_LABELS)
        validation_truth, validation_scores = _collect_predictions(
            model,
            source_validation_loader,
            device,
            amp_enabled,
            description=f"source validation {epoch}/{epochs}",
        )
        train_report = source_script_multilabel_report(
            train_truth,
            train_scores,
            CANONICAL_LABELS,
            thresholds,
        )
        train_score_metrics = multilabel_metrics(train_truth, train_scores, CANONICAL_LABELS)
        validation_report = source_script_multilabel_report(
            validation_truth,
            validation_scores,
            CANONICAL_LABELS,
            thresholds,
        )
        validation_score_metrics = multilabel_metrics(
            validation_truth,
            validation_scores,
            CANONICAL_LABELS,
        )

        train_payload = _report_payload(
            split_name="source_train",
            report=train_report,
            score_metrics=train_score_metrics,
            metadata=metadata,
            num_records=train_truth.shape[0],
        )
        validation_payload = _report_payload(
            split_name="source_validation",
            report=validation_report,
            score_metrics=validation_score_metrics,
            metadata=metadata,
            num_records=validation_truth.shape[0],
        )
        score = _safe_float(validation_report[selection_report_key])
        checkpoint_updated = score > best_score
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(
                {
                    "experiment_id": experiment_config["experiment"],
                    "model_name": "resnet1d",
                    "canonical_labels": CANONICAL_LABELS,
                    "epoch": epoch,
                    "validation_macro_auroc": score,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "method": method.method_name,
                    "method_params": dict(method.method_params),
                },
                best_checkpoint,
            )
        training_log_row = {
            "phase": "epoch",
            "epoch": int(epoch),
            "source_classification_loss": _safe_float(train_summary["source_loss"]),
            "adaptation_loss": _safe_float(train_summary["adaptation_loss"]),
            "total_loss": _safe_float(train_summary["loss"]),
            "source_train_macro_auroc": _safe_float(train_payload["macro_auroc"]),
            "source_train_macro_auprc": _safe_float(train_payload["macro_auprc"]),
            "source_validation_macro_auroc": _safe_float(validation_payload["macro_auroc"]),
            "source_validation_macro_auprc": _safe_float(validation_payload["macro_auprc"]),
            "selection_metric": selection_metric_name,
            "selection_score": _safe_float(score),
            "best_checkpoint_updated": bool(checkpoint_updated),
            "best_epoch_so_far": int(best_epoch),
            "checkpoint_path": str(best_checkpoint) if checkpoint_updated else "",
        }
        for key, value in dict(train_summary.get("method_metrics") or {}).items():
            training_log_row[f"method_{key}"] = _safe_float(value)
        training_log_rows.append(training_log_row)

    checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    train_truth, train_scores = _collect_predictions(
        model,
        source_train_eval_loader,
        device,
        amp_enabled,
        description="source train best",
    )
    thresholds = optimal_multilabel_thresholds(train_truth, train_scores, CANONICAL_LABELS)
    validation_truth, validation_scores = _collect_predictions(
        model,
        source_validation_loader,
        device,
        amp_enabled,
        description="source validation best",
    )
    target_validation_truth, target_validation_scores = _collect_predictions(
        model,
        target_validation_loader,
        device,
        amp_enabled,
        description="target validation best",
    )
    target_test_truth, target_test_scores = _collect_predictions(
        model,
        target_test_loader,
        device,
        amp_enabled,
        description="target test best",
    )

    train_report = source_script_multilabel_report(
        train_truth, train_scores, CANONICAL_LABELS, thresholds
    )
    train_score_metrics = multilabel_metrics(train_truth, train_scores, CANONICAL_LABELS)
    validation_report = source_script_multilabel_report(
        validation_truth,
        validation_scores,
        CANONICAL_LABELS,
        thresholds,
    )
    validation_score_metrics = multilabel_metrics(
        validation_truth, validation_scores, CANONICAL_LABELS
    )
    target_validation_report = source_script_multilabel_report(
        target_validation_truth,
        target_validation_scores,
        CANONICAL_LABELS,
        thresholds,
    )
    target_validation_score_metrics = multilabel_metrics(
        target_validation_truth,
        target_validation_scores,
        CANONICAL_LABELS,
    )
    target_test_report = source_script_multilabel_report(
        target_test_truth,
        target_test_scores,
        CANONICAL_LABELS,
        thresholds,
    )
    target_test_score_metrics = multilabel_metrics(
        target_test_truth, target_test_scores, CANONICAL_LABELS
    )

    train_payload = _report_payload(
        split_name="source_train",
        report=train_report,
        score_metrics=train_score_metrics,
        metadata=metadata,
        num_records=train_truth.shape[0],
    )
    validation_payload = _report_payload(
        split_name="source_validation",
        report=validation_report,
        score_metrics=validation_score_metrics,
        metadata=metadata,
        num_records=validation_truth.shape[0],
    )
    target_validation_payload = _report_payload(
        split_name="target_validation",
        report=target_validation_report,
        score_metrics=target_validation_score_metrics,
        metadata=metadata,
        num_records=target_validation_truth.shape[0],
    )
    target_test_payload = _report_payload(
        split_name="target_test",
        report=target_test_report,
        score_metrics=target_test_score_metrics,
        metadata=metadata,
        num_records=target_test_truth.shape[0],
    )

    payloads_by_split = {
        "source_train": train_payload,
        "source_validation": validation_payload,
        "target_validation": target_validation_payload,
        "target_test": target_test_payload,
    }

    _save_numpy_bundle(
        artifact_paths["predictions"],
        label_names=list(CANONICAL_LABELS),
        split_predictions={
            "source_train": (train_truth, train_scores),
            "source_validation": (validation_truth, validation_scores),
            "target_validation": (target_validation_truth, target_validation_scores),
            "target_test": (target_test_truth, target_test_scores),
        },
    )
    _write_result_json(artifact_paths["train_metrics"], train_payload)
    _write_result_json(artifact_paths["validation_metrics"], validation_payload)
    _write_result_json(artifact_paths["target_validation_metrics"], target_validation_payload)
    _write_result_json(artifact_paths["test_metrics"], target_test_payload)

    split_counts = {
        "source_train": int(len(source_splits["train"])),
        "source_validation": int(len(source_splits["validation"])),
        "target_validation": int(len(target_splits["validation"])),
        "target_test": int(len(target_splits["test"])),
    }
    summary_row = _build_summary_row(
        metadata=metadata,
        split_counts=split_counts,
        split_payloads=payloads_by_split,
        best_epoch=best_epoch,
        best_selection_score=best_score,
        selection_metric=selection_metric_name,
    )
    per_class_rows = _build_per_class_rows(metadata=metadata, payloads=payloads_by_split)
    summary_paths = _write_tables(
        output_dir=output_dir,
        summary_row=summary_row,
        per_class_rows=per_class_rows,
    )
    training_log_rows.append(
        {
            "phase": "final_evaluation",
            "epoch": int(best_epoch),
            "source_classification_loss": float("nan"),
            "adaptation_loss": float("nan"),
            "total_loss": float("nan"),
            "source_train_macro_auroc": _safe_float(train_payload["macro_auroc"]),
            "source_train_macro_auprc": _safe_float(train_payload["macro_auprc"]),
            "source_validation_macro_auroc": _safe_float(validation_payload["macro_auroc"]),
            "source_validation_macro_auprc": _safe_float(validation_payload["macro_auprc"]),
            "target_validation_macro_auroc": _safe_float(target_validation_payload["macro_auroc"]),
            "target_validation_macro_auprc": _safe_float(target_validation_payload["macro_auprc"]),
            "target_test_macro_auroc": _safe_float(target_test_payload["macro_auroc"]),
            "target_test_macro_auprc": _safe_float(target_test_payload["macro_auprc"]),
            "selection_metric": selection_metric_name,
            "selection_score": _safe_float(best_score),
            "best_checkpoint_updated": False,
            "best_epoch_so_far": int(best_epoch),
            "checkpoint_path": str(best_checkpoint),
        }
    )
    _write_training_log(output_dir, training_log_rows)

    status["status"] = "completed"
    status["finished_at"] = _utc_now()
    status["best_epoch"] = int(best_epoch)
    status["best_selection_score"] = _safe_float(best_score)
    status["best_validation_macro_auroc"] = (
        _safe_float(best_score) if selection_report_key == "macro_auroc" else float("nan")
    )
    status["best_validation_macro_auprc"] = (
        _safe_float(best_score) if selection_report_key == "macro_auprc" else float("nan")
    )
    status["selection_metric"] = selection_metric_name
    status["source_split_counts"] = {key: int(len(frame)) for key, frame in source_splits.items()}
    status["target_split_counts"] = {key: int(len(frame)) for key, frame in target_splits.items()}
    status["source_train_metrics"] = {"macro_auroc": train_payload["macro_auroc"]}
    status["source_validation_metrics"] = {"macro_auroc": validation_payload["macro_auroc"]}
    status["target_validation_metrics"] = {"macro_auroc": target_validation_payload["macro_auroc"]}
    status["test_metrics"] = {"macro_auroc": target_test_payload["macro_auroc"]}
    status["artifact_paths"].update(summary_paths)
    write_json(status_path, status)
    return status


def rebuild_uda_cross_domain_results(
    *,
    run_dir: str | Path,
    requested_device: str = "cpu",
    command: str = "scripts/evaluate.py --run-dir <run_dir>",
) -> dict[str, Any]:
    """Rebuild the standard UDA tables from a completed run directory."""
    output_dir = Path(run_dir).expanduser().resolve()
    status_path = output_dir / "run_status.json"
    if not status_path.is_file():
        raise FileNotFoundError(f"Missing run status: {status_path}")
    with status_path.open(encoding="utf-8") as handle:
        status = json.load(handle)
    if str(status.get("status")) not in {"completed", "preflight_completed"}:
        raise ValueError(f"Run directory must be completed, got {status.get('status')!r}")
    bundle_path = output_dir / "predictions.npz"
    if not bundle_path.is_file():
        raise FileNotFoundError(f"Missing predictions bundle: {bundle_path}")
    bundle = _load_numpy_bundle(bundle_path)
    label_names = list(bundle["label_names"])
    split_predictions = bundle["split_predictions"]
    metadata = {
        key: status[key]
        for key in (
            "experiment_id",
            "method",
            "method_params",
            "source_dataset",
            "target_dataset",
            "source_domain",
            "target_domain",
            "label_space",
            "canonical_labels",
            "model_architecture",
            "feature_extraction",
            "evaluation_metrics",
            "random_seed",
            "source_split_version",
            "target_split_version",
            "split_version",
            "preprocessing_version",
            "input_length",
        )
        if key in status
    }

    train_truth, train_scores = split_predictions["source_train"]
    thresholds = optimal_multilabel_thresholds(train_truth, train_scores, label_names)
    train_report = source_script_multilabel_report(
        train_truth, train_scores, label_names, thresholds
    )
    train_score_metrics = multilabel_metrics(train_truth, train_scores, label_names)
    validation_truth, validation_scores = split_predictions["source_validation"]
    validation_report = source_script_multilabel_report(
        validation_truth,
        validation_scores,
        label_names,
        thresholds,
    )
    validation_score_metrics = multilabel_metrics(validation_truth, validation_scores, label_names)
    target_validation_truth, target_validation_scores = split_predictions["target_validation"]
    target_validation_report = source_script_multilabel_report(
        target_validation_truth,
        target_validation_scores,
        label_names,
        thresholds,
    )
    target_validation_score_metrics = multilabel_metrics(
        target_validation_truth,
        target_validation_scores,
        label_names,
    )
    target_test_truth, target_test_scores = split_predictions["target_test"]
    target_test_report = source_script_multilabel_report(
        target_test_truth,
        target_test_scores,
        label_names,
        thresholds,
    )
    target_test_score_metrics = multilabel_metrics(
        target_test_truth, target_test_scores, label_names
    )

    train_payload = _report_payload(
        split_name="source_train",
        report=train_report,
        score_metrics=train_score_metrics,
        metadata=metadata,
        num_records=train_truth.shape[0],
    )
    validation_payload = _report_payload(
        split_name="source_validation",
        report=validation_report,
        score_metrics=validation_score_metrics,
        metadata=metadata,
        num_records=validation_truth.shape[0],
    )
    target_validation_payload = _report_payload(
        split_name="target_validation",
        report=target_validation_report,
        score_metrics=target_validation_score_metrics,
        metadata=metadata,
        num_records=target_validation_truth.shape[0],
    )
    target_test_payload = _report_payload(
        split_name="target_test",
        report=target_test_report,
        score_metrics=target_test_score_metrics,
        metadata=metadata,
        num_records=target_test_truth.shape[0],
    )
    payloads_by_split = {
        "source_train": train_payload,
        "source_validation": validation_payload,
        "target_validation": target_validation_payload,
        "target_test": target_test_payload,
    }
    _write_result_json(output_dir / "train_metrics.json", train_payload)
    _write_result_json(output_dir / "validation_metrics.json", validation_payload)
    _write_result_json(output_dir / "target_validation_metrics.json", target_validation_payload)
    _write_result_json(output_dir / "test_metrics.json", target_test_payload)
    summary_row = _build_summary_row(
        metadata=metadata,
        split_counts=status.get("source_split_counts", {}),
        split_payloads=payloads_by_split,
        best_epoch=int(status.get("best_epoch", 0)),
        best_selection_score=_safe_float(status.get("best_selection_score", float("nan"))),
        selection_metric=str(status.get("selection_metric", "source_validation_macro_auroc")),
    )
    per_class_rows = _build_per_class_rows(metadata=metadata, payloads=payloads_by_split)
    summary_paths = _write_tables(
        output_dir=output_dir,
        summary_row=summary_row,
        per_class_rows=per_class_rows,
    )
    status["artifact_paths"] = {
        **status.get("artifact_paths", {}),
        **summary_paths,
    }
    status["rebuild"] = {
        "status": "completed",
        "requested_device": requested_device,
        "command": command,
        "rebuilt_at": _utc_now(),
    }
    with status_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(status), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return {
        "status": status,
        "results_summary": str(output_dir / "results_summary.csv"),
        "per_class_summary": str(output_dir / "per_class_summary.csv"),
        "train_metrics": train_payload,
        "validation_metrics": validation_payload,
        "target_validation_metrics": target_validation_payload,
        "test_metrics": target_test_payload,
    }
