"""Dataset discriminator training workflow."""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import subprocess
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from ecg_shift_bench.datasets.discriminator import (
    DiscriminatorECGDataset,
    add_target_labels,
    build_discriminator_cohort,
    select_cohort_subset,
)
from ecg_shift_bench.datasets.registry import create_dataset
from ecg_shift_bench.evaluation.discriminator import dataset_classification_metrics
from ecg_shift_bench.labels.dataset_ids import (
    DATASET_ID_ORDER,
    canonical_dataset_name,
    selected_dataset_names,
)
from ecg_shift_bench.models.resnet1d_wang import resnet1d_wang
from ecg_shift_bench.training.optim import create_optimizer


@dataclass(frozen=True)
class DatasetSpec:
    """Dataset source configuration for one discriminator run."""

    name: str
    root: Path
    config: dict[str, Any]
    config_path: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _git_state() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _make_loader(
    dataset: DiscriminatorECGDataset,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
) -> DataLoader[tuple[Tensor, Tensor]]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        generator=generator,
        persistent_workers=workers > 0,
    )


def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled)


def _resolve_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type != "cuda":
        raise ValueError("Dataset discriminator runs on CUDA only")
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {requested!r} requested, but CUDA is unavailable")
    return device


def preflight_forward_backward(
    model: nn.Module,
    batch: tuple[Tensor, Tensor],
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> float:
    """Check one real batch without changing the model state."""
    original_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    inputs, targets = batch
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True).long()
    model.train()
    model.zero_grad(set_to_none=True)
    with _autocast(device, amp_enabled):
        logits = model(inputs)
        loss = criterion(logits, targets)
    if logits.ndim != 2 or logits.shape[0] != targets.shape[0]:
        raise ValueError(f"Preflight logits shape {logits.shape} does not match {targets.shape}")
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Preflight loss is non-finite: {float(loss.detach())}")
    loss.backward()
    if not any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("Preflight backward pass produced no gradients")
    model.zero_grad(set_to_none=True)
    model.load_state_dict(original_state)
    return float(loss.detach())


def train_epoch(
    model: nn.Module,
    batches: DataLoader[tuple[Tensor, Tensor]],
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    *,
    description: str,
) -> dict[str, float | int]:
    """Train one epoch with finite-loss enforcement and optional FP16 AMP."""
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    total_loss = 0.0
    total_samples = 0
    minimum = float("inf")
    maximum = float("-inf")
    steps = 0
    progress = tqdm(batches, desc=description, leave=False)
    for inputs, targets in progress:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).long()
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp_enabled):
            logits = model(inputs)
            loss = criterion(logits, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at step {steps}: {loss.item()}")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = inputs.shape[0]
        loss_value = float(loss.detach())
        total_loss += loss_value * batch_size
        total_samples += batch_size
        minimum = min(minimum, loss_value)
        maximum = max(maximum, loss_value)
        steps += 1
        progress.set_postfix(loss=f"{loss_value:.5f}")
    if total_samples == 0:
        raise ValueError("Training loader produced no batches")
    return {
        "loss": total_loss / total_samples,
        "min_batch_loss": minimum,
        "max_batch_loss": maximum,
        "steps": steps,
        "samples": total_samples,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    batches: DataLoader[tuple[Tensor, Tensor]],
    device: torch.device,
    amp_enabled: bool,
    *,
    label_names: list[str],
    description: str,
) -> dict[str, object]:
    """Evaluate dataset-ID predictions and compute classification metrics."""
    model.eval()
    truth_all: list[np.ndarray] = []
    score_all: list[np.ndarray] = []
    for inputs, targets in tqdm(batches, desc=description, leave=False):
        inputs = inputs.to(device, non_blocking=True)
        with _autocast(device, amp_enabled):
            logits = model(inputs)
        score_all.append(torch.softmax(logits, dim=1).float().cpu().numpy())
        truth_all.append(targets.numpy().astype(np.int64, copy=False))
    if not truth_all:
        raise ValueError("Evaluation loader produced no batches")
    return dataset_classification_metrics(
        np.concatenate(truth_all),
        np.concatenate(score_all),
        label_names,
    )


def _cohort_for_study(
    *,
    datasets: dict[str, Any],
    mode: str,
    subset: str,
    pair: tuple[str, str] | None,
    seed: int,
) -> pd.DataFrame:
    cohort = build_discriminator_cohort(datasets)
    selected = (
        DATASET_ID_ORDER if mode == "multiclass" else selected_dataset_names(list(pair or ()))
    )
    cohort = select_cohort_subset(cohort, selected_datasets=selected, subset=subset, seed=seed)
    cohort = add_target_labels(
        cohort,
        selected_datasets=selected,
        random_label=subset == "random_label",
        seed=seed,
    )
    return cohort.reset_index(drop=True)


def run_dataset_discriminator(
    *,
    experiment_config: dict[str, Any],
    experiment_config_path: Path,
    dataset_specs: list[DatasetSpec],
    output_dir: Path,
    requested_device: str,
    command: str,
    mode: str,
    subset: str,
    pair: tuple[str, str] | None = None,
    preflight_only: bool = False,
) -> dict[str, Any]:
    """Run the dataset discriminator and persist reconstruction artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "run_status.json"
    commit, dirty = _git_state()
    seed = int(experiment_config["training"]["seed"])
    started_at = _utc_now()
    pair_names = list(pair) if pair else None
    artifact_paths = {
        "experiment_config": str(output_dir / "experiment_config.yaml"),
        "cohort_manifest": str(output_dir / "cohort_manifest.csv"),
        "split_manifest": str(output_dir / "split_manifest.csv"),
        "history": str(output_dir / "history.json"),
        "checkpoint": str(output_dir / "best_checkpoint.pt"),
        "validation_metrics": str(output_dir / "validation_metrics.json"),
        "test_metrics": str(output_dir / "test_metrics.json"),
        "confusion_matrix": str(output_dir / "confusion_matrix.csv"),
    }
    status: dict[str, Any] = {
        "experiment_id": experiment_config["experiment"],
        "status": "running",
        "command": command,
        "git_commit": commit,
        "git_dirty": dirty,
        "seed": seed,
        "requested_device": requested_device,
        "mode": mode,
        "subset": subset,
        "pair": pair_names,
        "started_at": started_at,
        "finished_at": None,
        "artifact_paths": artifact_paths,
        "recovery": {
            "resume_supported": False,
            "action": "Rerun the recorded command; incomplete epochs are not checkpointed.",
        },
    }
    write_json(status_path, status)

    try:
        shutil.copy2(experiment_config_path, artifact_paths["experiment_config"])

        datasets = {
            canonical_dataset_name(spec.name): create_dataset(spec.name, spec.root, spec.config)
            for spec in dataset_specs
        }
        cohort = _cohort_for_study(
            datasets=datasets,
            mode=mode,
            subset=subset,
            pair=pair,
            seed=seed,
        )
        if cohort.empty:
            raise ValueError("Discriminator cohort is empty after filtering")

        selected_names = (
            DATASET_ID_ORDER if mode == "multiclass" else selected_dataset_names(list(pair or ()))
        )
        label_names = list(selected_names)
        cohort["dataset_id"] = cohort["dataset_name"].map(
            {name: index for index, name in enumerate(label_names)}
        )
        if cohort["dataset_id"].isna().any():
            missing = sorted(cohort.loc[cohort["dataset_id"].isna(), "dataset_name"].unique())
            raise KeyError(f"Missing discriminator labels for: {missing}")
        cohort["dataset_id"] = cohort["dataset_id"].astype(int)
        cohort["target_dataset_id"] = cohort["target_dataset_id"].astype(int)

        split_manifest = cohort[
            ["dataset_name", "record_id", "patient_id", "split", "dataset_id"]
        ].copy()
        cohort.to_csv(artifact_paths["cohort_manifest"], index=False)
        split_manifest.to_csv(artifact_paths["split_manifest"], index=False)

        device = _resolve_device(requested_device)
        amp_enabled = device.type == "cuda"
        if amp_enabled:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        status["resolved_device"] = str(device)
        status["device_name"] = (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        )
        status["amp_fp16"] = amp_enabled
        status["label_names"] = label_names
        status["determinism"] = {
            "seeded": True,
            "deterministic_algorithms": "warn_only",
        }
        write_json(status_path, status)

        training_config = experiment_config["training"]
        batch_size = int(training_config["batch_size"])
        workers = int(training_config["workers"])
        pin_memory = True

        train_manifest = cohort.loc[cohort["split"] == "train"].reset_index(drop=True)
        validation_manifest = cohort.loc[cohort["split"] == "validation"].reset_index(drop=True)
        test_manifest = cohort.loc[cohort["split"] == "test"].reset_index(drop=True)
        if train_manifest.empty or validation_manifest.empty or test_manifest.empty:
            raise ValueError(
                "Discriminator cohort must contain train, validation, and test records"
            )

        train_dataset = DiscriminatorECGDataset(train_manifest, datasets)
        validation_dataset = DiscriminatorECGDataset(validation_manifest, datasets)
        test_dataset = DiscriminatorECGDataset(test_manifest, datasets)
        train_loader = _make_loader(
            train_dataset,
            batch_size=batch_size,
            workers=workers,
            shuffle=True,
            seed=seed,
            pin_memory=pin_memory,
        )
        validation_loader = _make_loader(
            validation_dataset,
            batch_size=batch_size,
            workers=workers,
            shuffle=False,
            seed=seed,
            pin_memory=pin_memory,
        )
        test_loader = _make_loader(
            test_dataset,
            batch_size=batch_size,
            workers=workers,
            shuffle=False,
            seed=seed,
            pin_memory=pin_memory,
        )

        model = resnet1d_wang(
            in_channels=int(experiment_config["model"]["in_channels"]),
            num_labels=len(label_names),
            channels=int(experiment_config["model"].get("channels", 128)),
            dropout=float(experiment_config["model"].get("dropout", 0.5)),
        ).to(device)
        optimizer = create_optimizer(
            model.parameters(),
            experiment_config["training"]["optimizer"],
            float(experiment_config["training"]["learning_rate"]),
            float(experiment_config["training"].get("weight_decay", 0.0)),
        )
        criterion = nn.CrossEntropyLoss()

        preflight_loss = preflight_forward_backward(
            model,
            next(iter(train_loader)),
            criterion,
            device,
            amp_enabled,
        )
        status["preflight_loss"] = preflight_loss
        status["preflight_completed_at"] = _utc_now()
        if preflight_only:
            status["status"] = "preflight_completed"
            status["finished_at"] = _utc_now()
            write_json(status_path, status)
            return status

        epochs = int(experiment_config["training"]["epochs"])
        print(
            "Training start: "
            f"experiment={experiment_config['experiment']} "
            f"mode={mode} subset={subset} device={device} epochs={epochs}"
        )

        history: list[dict[str, Any]] = []
        best_metric = float("-inf")
        best_state: dict[str, Any] | None = None
        best_epoch = -1

        for epoch in range(int(experiment_config["training"]["epochs"])):
            train_stats = train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                amp_enabled,
                description=f"train epoch {epoch + 1}",
            )
            validation_metrics = evaluate(
                model,
                validation_loader,
                device,
                amp_enabled,
                label_names=label_names,
                description=f"validation epoch {epoch + 1}",
            )
            epoch_summary = {
                "epoch": epoch + 1,
                "train_loss": train_stats["loss"],
                "validation_metrics": validation_metrics,
            }
            history.append(epoch_summary)
            current_metric = float(validation_metrics["balanced_accuracy"])
            if current_metric >= best_metric:
                best_metric = current_metric
                best_epoch = epoch + 1
                best_state = {
                    "model_state_dict": deepcopy(model.state_dict()),
                    "epoch": best_epoch,
                    "label_names": label_names,
                }

        if best_state is None:
            raise RuntimeError("Training finished without a checkpoint")
        torch.save(best_state, artifact_paths["checkpoint"])
        write_json(Path(artifact_paths["history"]), {"epochs": history, "best_epoch": best_epoch})

        state = torch.load(artifact_paths["checkpoint"], map_location=device, weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        validation_metrics = evaluate(
            model,
            validation_loader,
            device,
            amp_enabled,
            label_names=label_names,
            description="validation final",
        )
        test_metrics = evaluate(
            model,
            test_loader,
            device,
            amp_enabled,
            label_names=label_names,
            description="test final",
        )
        write_json(Path(artifact_paths["validation_metrics"]), validation_metrics)
        write_json(Path(artifact_paths["test_metrics"]), test_metrics)
        pd.DataFrame(
            test_metrics["confusion_matrix"], index=label_names, columns=label_names
        ).to_csv(artifact_paths["confusion_matrix"])

        status.update(
            {
                "status": "completed",
                "finished_at": _utc_now(),
                "best_epoch": best_epoch,
                "best_validation_balanced_accuracy": best_metric,
                "artifact_paths": artifact_paths,
            }
        )
        write_json(status_path, status)
        return status
    except Exception as error:
        status.update(
            {
                "status": "failed",
                "finished_at": _utc_now(),
                "error": repr(error),
            }
        )
        write_json(status_path, status)
        raise
