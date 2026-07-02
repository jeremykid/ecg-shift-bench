"""Source-only cross-domain training and report generation."""

from __future__ import annotations

import json
import math
import os
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
from ecg_shift_bench.models.registry import canonical_model_name, create_model
from ecg_shift_bench.training.optim import create_optimizer
from ecg_shift_bench.training.ptbxl_baseline import (
    _git_state,
    _make_loader,
    _resolve_device,
    preflight_forward_backward,
    train_epoch,
    write_json,
)
from ecg_shift_bench.utils.seed import seed_everything

SPLIT_ORDER = ("source_train", "source_validation", "target_test")
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
    """Resolved dataset configuration for one source-only run."""

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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _copy_text_config(source: Path, destination: Path) -> None:
    destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def _write_yaml_config(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _split_frames(manifest: pd.DataFrame) -> dict[str, pd.DataFrame]:
    splits = {name: frame.copy().reset_index(drop=True) for name, frame in manifest.groupby("split")}
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
        label_names = [str(label) for label in np.asarray(bundle["label_names"], dtype=object).tolist()]
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
    train_metrics: dict[str, Any],
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    best_epoch: int,
    best_validation_macro_auprc: float,
) -> dict[str, Any]:
    row = dict(metadata)
    row.update(
        {
            "source_train_records": int(split_counts["source_train"]),
            "source_validation_records": int(split_counts["source_validation"]),
            "target_test_records": int(split_counts["target_test"]),
            "best_epoch": int(best_epoch),
            "selection_metric": "source_validation_macro_auprc",
            "best_validation_macro_auprc": _safe_float(best_validation_macro_auprc),
        }
    )
    for prefix, payload in (
        ("source_train", train_metrics),
        ("source_validation", validation_metrics),
        ("target_test", test_metrics),
    ):
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
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _build_model(model_config: dict[str, Any], num_labels: int, device: torch.device) -> nn.Module:
    model_name = canonical_model_name(str(model_config["name"]))
    if model_name != "resnet1d":
        raise ValueError(f"Source-only cross-domain runs require resnet1d, got {model_name!r}")
    return create_model(model_config, num_labels=num_labels).to(device)


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
        "test_metrics": output_dir / "test_metrics.json",
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
    from ecg_shift_bench.utils.config import load_yaml

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


def _source_only_run_metadata(
    *,
    experiment_config: dict[str, Any],
    source_dataset_spec: DatasetSpec,
    target_dataset_spec: DatasetSpec,
    source_policy: dict[str, Any],
    target_policy: dict[str, Any],
    input_length: int,
) -> dict[str, Any]:
    return {
        "experiment_id": str(experiment_config["experiment"]),
        "method": str(experiment_config.get("method", "source_only")),
        "source_dataset": source_dataset_spec.name,
        "target_dataset": target_dataset_spec.name,
        "source_domain": source_dataset_spec.config.get("domain"),
        "target_domain": target_dataset_spec.config.get("domain"),
        "label_space": "canonical_six_label",
        "canonical_labels": "|".join(CANONICAL_LABELS),
        "model_architecture": canonical_model_name(str(experiment_config["model"]["name"])),
        "evaluation_metrics": "|".join(EVALUATION_METRICS),
        "random_seed": int(experiment_config["training"]["seed"]),
        "source_split_version": _split_version(source_policy),
        "target_split_version": _split_version(target_policy),
        "split_version": _combined_split_version(source_policy, target_policy),
        "preprocessing_version": "shared_alignment_v1",
        "input_length": int(input_length),
        "source_root": str(source_dataset_spec.root),
        "target_root": str(target_dataset_spec.root),
    }


def _source_only_contract(
    *,
    experiment_config: dict[str, Any],
    source_dataset_spec: DatasetSpec,
    target_dataset_spec: DatasetSpec,
) -> None:
    source_datasets = list(experiment_config.get("source_datasets") or [])
    target_datasets = list(experiment_config.get("target_datasets") or [])
    if len(source_datasets) != 1 or len(target_datasets) != 1:
        raise ValueError("Source-only cross-domain runs require exactly one source and one target dataset")
    if str(source_datasets[0]).upper() != source_dataset_spec.name.upper():
        raise ValueError("Source dataset config does not match the experiment config")
    if str(target_datasets[0]).upper() != target_dataset_spec.name.upper():
        raise ValueError("Target dataset config does not match the experiment config")


def run_source_only_cross_domain(
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
    """Train on the source dataset and evaluate directly on the target test split."""
    _source_only_contract(
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
    metadata = _source_only_run_metadata(
        experiment_config=experiment_config,
        source_dataset_spec=source_dataset_spec,
        target_dataset_spec=target_dataset_spec,
        source_policy={"method": "unknown"},
        target_policy={"method": "unknown"},
        input_length=input_length,
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
            "target_unlabeled_data_available_during_training": False,
            "model_updates_during_testing": False,
            "normalization": "none",
        },
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

    source_dataset = create_dataset(source_dataset_spec.name, source_dataset_spec.root, source_dataset_spec.config)
    source_metadata = source_dataset.load_metadata()
    source_manifest, source_policy = build_split_manifest(source_dataset, source_metadata)
    source_manifest.to_csv(artifact_paths["source_split_manifest"], index=False)
    source_splits = _split_frames(source_manifest)

    source_train_dataset = AlignedClassificationDataset(source_dataset, source_splits["train"], input_length)
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

    batch_size = int(experiment_config["training"]["batch_size"])
    workers = int(experiment_config["training"]["workers"])
    source_train_loader = _make_loader(
        source_train_dataset,
        batch_size=batch_size,
        workers=workers,
        shuffle=True,
        seed=seed,
        pin_memory=device.type == "cuda",
    )
    source_train_eval_loader = _make_loader(
        source_train_eval_dataset,
        batch_size=batch_size,
        workers=workers,
        shuffle=False,
        seed=seed,
        pin_memory=device.type == "cuda",
    )
    source_validation_loader = _make_loader(
        source_validation_dataset,
        batch_size=batch_size,
        workers=workers,
        shuffle=False,
        seed=seed,
        pin_memory=device.type == "cuda",
    )

    source_policy = source_policy if source_policy else {"method": "unknown"}
    status["source_split_policy"] = source_policy
    status["source_split_version"] = _split_version(source_policy)
    write_json(status_path, status)

    seed_everything(seed)
    model = _build_model(experiment_config["model"], len(CANONICAL_LABELS), device)
    criterion = nn.BCEWithLogitsLoss()
    preflight_loss = preflight_forward_backward(
        model,
        next(iter(source_train_loader)),
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
    payloads_by_split: dict[str, dict[str, Any]] = {}

    for epoch in range(1, epochs + 1):
        train_epoch(
            model,
            source_train_loader,
            optimizer,
            criterion,
            device,
            amp_enabled,
            description=f"source train {epoch}/{epochs}",
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
        payloads_by_split["source_train"] = train_payload
        payloads_by_split["source_validation"] = validation_payload
        score = _safe_float(validation_report["macro_auprc"])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(
                {
                    "experiment_id": experiment_config["experiment"],
                    "model_name": "resnet1d",
                    "canonical_labels": CANONICAL_LABELS,
                    "epoch": epoch,
                    "validation_macro_auprc": score,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                best_checkpoint,
            )

    checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    target_dataset = create_dataset(target_dataset_spec.name, target_dataset_spec.root, target_dataset_spec.config)
    target_metadata = target_dataset.load_metadata()
    target_manifest, target_policy = build_split_manifest(target_dataset, target_metadata)
    target_manifest.to_csv(artifact_paths["target_split_manifest"], index=False)
    target_splits = _split_frames(target_manifest)
    target_test_dataset = AlignedClassificationDataset(target_dataset, target_splits["test"], input_length)
    target_test_loader = _make_loader(
        target_test_dataset,
        batch_size=batch_size,
        workers=workers,
        shuffle=False,
        seed=seed,
        pin_memory=device.type == "cuda",
    )
    target_policy = target_policy if target_policy else {"method": "unknown"}
    metadata = _source_only_run_metadata(
        experiment_config=experiment_config,
        source_dataset_spec=source_dataset_spec,
        target_dataset_spec=target_dataset_spec,
        source_policy=source_policy,
        target_policy=target_policy,
        input_length=input_length,
    )
    status.update(metadata)
    status["source_split_policy"] = source_policy
    status["target_split_policy"] = target_policy
    write_json(status_path, status)

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
    target_test_truth, target_test_scores = _collect_predictions(
        model,
        target_test_loader,
        device,
        amp_enabled,
        description="target test best",
    )

    train_report = source_script_multilabel_report(train_truth, train_scores, CANONICAL_LABELS, thresholds)
    train_score_metrics = multilabel_metrics(train_truth, train_scores, CANONICAL_LABELS)
    validation_report = source_script_multilabel_report(
        validation_truth,
        validation_scores,
        CANONICAL_LABELS,
        thresholds,
    )
    validation_score_metrics = multilabel_metrics(validation_truth, validation_scores, CANONICAL_LABELS)
    target_test_report = source_script_multilabel_report(
        target_test_truth,
        target_test_scores,
        CANONICAL_LABELS,
        thresholds,
    )
    target_test_score_metrics = multilabel_metrics(target_test_truth, target_test_scores, CANONICAL_LABELS)

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
        "target_test": target_test_payload,
    }

    _save_numpy_bundle(
        artifact_paths["predictions"],
        label_names=list(CANONICAL_LABELS),
        split_predictions={
            "source_train": (train_truth, train_scores),
            "source_validation": (validation_truth, validation_scores),
            "target_test": (target_test_truth, target_test_scores),
        },
    )
    _write_result_json(artifact_paths["train_metrics"], train_payload)
    _write_result_json(artifact_paths["validation_metrics"], validation_payload)
    _write_result_json(artifact_paths["test_metrics"], target_test_payload)

    split_counts = {
        "source_train": int(len(source_splits["train"])),
        "source_validation": int(len(source_splits["validation"])),
        "target_test": int(len(target_splits["test"])),
    }
    summary_row = _build_summary_row(
        metadata=metadata,
        split_counts=split_counts,
        train_metrics=train_payload,
        validation_metrics=validation_payload,
        test_metrics=target_test_payload,
        best_epoch=best_epoch,
        best_validation_macro_auprc=best_score,
    )
    per_class_rows = _build_per_class_rows(metadata=metadata, payloads=payloads_by_split)
    summary_paths = _write_tables(
        output_dir=output_dir,
        summary_row=summary_row,
        per_class_rows=per_class_rows,
    )

    status["status"] = "completed"
    status["finished_at"] = _utc_now()
    status["best_epoch"] = int(best_epoch)
    status["best_validation_macro_auprc"] = _safe_float(best_score)
    status["selection_metric"] = "source_validation_macro_auprc"
    status["thresholds"] = _json_safe(thresholds)
    status["train_metrics"] = {"macro_auprc": train_payload["macro_auprc"]}
    status["validation_metrics"] = {"macro_auprc": validation_payload["macro_auprc"]}
    status["test_metrics"] = {"macro_auprc": target_test_payload["macro_auprc"]}
    status["source_split_counts"] = {key: int(len(frame)) for key, frame in source_splits.items()}
    status["target_split_counts"] = {key: int(len(frame)) for key, frame in target_splits.items()}
    status["artifact_paths"].update(summary_paths)
    write_json(status_path, status)
    return status


def rebuild_source_only_cross_domain_results(
    *,
    run_dir: str | Path,
    requested_device: str = "cpu",
    command: str = "scripts/evaluate.py --run-dir <run_dir>",
) -> dict[str, Any]:
    """Rebuild the standard source-only tables from a completed run directory."""
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
            "source_dataset",
            "target_dataset",
            "source_domain",
            "target_domain",
            "label_space",
            "canonical_labels",
            "model_architecture",
            "evaluation_metrics",
            "random_seed",
            "source_split_version",
            "target_split_version",
            "split_version",
            "preprocessing_version",
            "input_length",
        )
    }

    train_truth, train_scores = split_predictions["source_train"]
    thresholds = optimal_multilabel_thresholds(train_truth, train_scores, label_names)
    train_report = source_script_multilabel_report(train_truth, train_scores, label_names, thresholds)
    train_score_metrics = multilabel_metrics(train_truth, train_scores, label_names)
    validation_truth, validation_scores = split_predictions["source_validation"]
    validation_report = source_script_multilabel_report(
        validation_truth,
        validation_scores,
        label_names,
        thresholds,
    )
    validation_score_metrics = multilabel_metrics(validation_truth, validation_scores, label_names)
    target_test_truth, target_test_scores = split_predictions["target_test"]
    target_test_report = source_script_multilabel_report(
        target_test_truth,
        target_test_scores,
        label_names,
        thresholds,
    )
    target_test_score_metrics = multilabel_metrics(target_test_truth, target_test_scores, label_names)

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
        "target_test": target_test_payload,
    }
    _write_result_json(output_dir / "train_metrics.json", train_payload)
    _write_result_json(output_dir / "validation_metrics.json", validation_payload)
    _write_result_json(output_dir / "test_metrics.json", target_test_payload)
    summary_row = _build_summary_row(
        metadata=metadata,
        split_counts=status.get("source_split_counts", {}),
        train_metrics=train_payload,
        validation_metrics=validation_payload,
        test_metrics=target_test_payload,
        best_epoch=int(status.get("best_epoch", 0)),
        best_validation_macro_auprc=_safe_float(status.get("best_validation_macro_auprc", float("nan"))),
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
        "test_metrics": target_test_payload,
    }
