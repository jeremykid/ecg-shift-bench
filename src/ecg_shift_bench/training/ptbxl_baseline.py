"""Reproducible source-only PTB-XL baseline training workflow."""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import subprocess
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from ecg_shift_bench.datasets.ptbxl_baseline import (
    PreparedPTBXLSnapshot,
    PTBXLClassificationDataset,
    canonical_split_manifest_bytes,
    prepare_ptbxl_snapshot,
)
from ecg_shift_bench.evaluation.metrics import multilabel_metrics
from ecg_shift_bench.labels.canonical import CANONICAL_LABELS
from ecg_shift_bench.models.resnet1d_wang import resnet1d_wang
from ecg_shift_bench.utils.seed import seed_everything


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
    dataset: PTBXLClassificationDataset,
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


def preflight_forward_backward(
    model: nn.Module,
    batch: tuple[Tensor, Tensor],
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> float:
    """Check one real batch without changing the model's initial state."""
    original_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    inputs, targets = batch
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True).float()
    model.train()
    model.zero_grad(set_to_none=True)
    with _autocast(device, amp_enabled):
        logits = model(inputs)
        loss = criterion(logits, targets)
    if logits.shape != targets.shape:
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
    batches: Iterable[tuple[Tensor, Tensor]],
    optimizer: AdamW,
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
        targets = targets.to(device, non_blocking=True).float()
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp_enabled):
            loss = criterion(model(inputs), targets)
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
    batches: Iterable[tuple[Tensor, Tensor]],
    device: torch.device,
    amp_enabled: bool,
    *,
    description: str,
) -> dict[str, Any]:
    """Evaluate all six labels and include positive support counts."""
    model.eval()
    targets_all: list[np.ndarray] = []
    scores_all: list[np.ndarray] = []
    for inputs, targets in tqdm(batches, desc=description, leave=False):
        inputs = inputs.to(device, non_blocking=True)
        with _autocast(device, amp_enabled):
            logits = model(inputs)
        scores_all.append(torch.sigmoid(logits).float().cpu().numpy())
        targets_all.append(targets.numpy().astype(np.int64, copy=False))
    if not targets_all:
        raise ValueError("Evaluation loader produced no batches")
    truth = np.concatenate(targets_all)
    scores = np.concatenate(scores_all)
    metrics = multilabel_metrics(truth, scores, CANONICAL_LABELS)
    metrics["per_label_support"] = {
        label: int(truth[:, index].sum()) for index, label in enumerate(CANONICAL_LABELS)
    }
    metrics["num_records"] = int(truth.shape[0])
    return metrics


def _save_split_manifest(snapshot: PreparedPTBXLSnapshot, path: Path) -> None:
    path.write_bytes(canonical_split_manifest_bytes(snapshot.split_manifest))


def _resolve_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {requested!r} requested, but CUDA is unavailable")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("This baseline supports only CPU and CUDA devices")
    return device


def run_ptbxl_baseline(
    *,
    experiment_config: dict[str, Any],
    experiment_config_path: Path,
    dataset_config: dict[str, Any],
    dataset_config_path: Path,
    snapshot_manifest: dict[str, Any],
    snapshot_manifest_path: Path,
    root: Path,
    output_dir: Path,
    requested_device: str,
    command: str,
    preflight_only: bool = False,
) -> dict[str, Any]:
    """Run the fixed PTB-XL baseline and persist reconstruction artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "run_status.json"
    commit, dirty = _git_state()
    seed = int(experiment_config["training"]["seed"])
    started_at = _utc_now()
    artifact_paths = {
        "experiment_config": str(output_dir / "experiment_config.yaml"),
        "dataset_config": str(output_dir / "dataset_config.yaml"),
        "snapshot_manifest": str(output_dir / "dataset_snapshot_manifest.yaml"),
        "snapshot_identity": str(output_dir / "snapshot_identity.json"),
        "split_manifest": str(output_dir / "split_manifest.csv"),
        "history": str(output_dir / "history.json"),
        "checkpoint": str(output_dir / "best_checkpoint.pt"),
        "validation_metrics": str(output_dir / "validation_metrics.json"),
        "test_metrics": str(output_dir / "test_metrics.json"),
    }
    status: dict[str, Any] = {
        "experiment_id": experiment_config["experiment"],
        "snapshot_id": snapshot_manifest["snapshot_id"],
        "status": "running",
        "command": command,
        "git_commit": commit,
        "git_dirty": dirty,
        "seed": seed,
        "requested_device": requested_device,
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
        shutil.copy2(dataset_config_path, artifact_paths["dataset_config"])
        shutil.copy2(snapshot_manifest_path, artifact_paths["snapshot_manifest"])
        snapshot = prepare_ptbxl_snapshot(root, dataset_config, snapshot_manifest)
        write_json(Path(artifact_paths["snapshot_identity"]), snapshot.summary)
        _save_split_manifest(snapshot, Path(artifact_paths["split_manifest"]))

        device = _resolve_device(requested_device)
        amp_enabled = device.type == "cuda"
        if amp_enabled:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        status["resolved_device"] = str(device)
        status["device_name"] = (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        )
        status["amp_fp16"] = amp_enabled
        status["determinism"] = {
            "seeded": True,
            "deterministic_algorithms": "warn_only",
            "known_limitation": (
                "CUDA adaptive max-pool backward may not be bitwise deterministic."
                if amp_enabled
                else None
            ),
        }
        write_json(status_path, status)

        data_config = experiment_config["data"]
        training_config = experiment_config["training"]
        input_length = int(data_config["input_length"])
        datasets = {
            name: PTBXLClassificationDataset(root, dataset_config, frame, input_length)
            for name, frame in snapshot.splits.items()
        }
        batch_size = int(training_config["batch_size"])
        workers = int(training_config["workers"])

        def make_loaders() -> dict[str, DataLoader[tuple[Tensor, Tensor]]]:
            return {
                name: _make_loader(
                    dataset,
                    batch_size=batch_size,
                    workers=workers,
                    shuffle=name == "train",
                    seed=seed,
                    pin_memory=device.type == "cuda",
                )
                for name, dataset in datasets.items()
            }

        seed_everything(seed)
        model = resnet1d_wang(
            in_channels=int(experiment_config["model"]["in_channels"]),
            num_labels=len(CANONICAL_LABELS),
            channels=int(experiment_config["model"]["channels"]),
            dropout=float(experiment_config["model"]["dropout"]),
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
        del preflight_loaders

        if preflight_only:
            status["status"] = "preflight_completed"
            status["finished_at"] = _utc_now()
            write_json(status_path, status)
            return status

        seed_everything(seed)
        loaders = make_loaders()
        optimizer_name = str(training_config["optimizer"]).lower()
        if optimizer_name != "adamw":
            raise ValueError(f"Expected optimizer 'adamw', got {optimizer_name!r}")
        optimizer = AdamW(
            model.parameters(),
            lr=float(training_config["learning_rate"]),
            weight_decay=float(training_config["weight_decay"]),
        )

        history: list[dict[str, Any]] = []
        best_score = float("-inf")
        checkpoint_path = Path(artifact_paths["checkpoint"])
        epochs = int(training_config["epochs"])
        for epoch in range(1, epochs + 1):
            train_summary = train_epoch(
                model,
                loaders["train"],
                optimizer,
                criterion,
                device,
                amp_enabled,
                description=f"train {epoch}/{epochs}",
            )
            validation_metrics = evaluate(
                model,
                loaders["validation"],
                device,
                amp_enabled,
                description=f"validation {epoch}/{epochs}",
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
                        "snapshot_id": snapshot.snapshot_id,
                        "model_name": "resnet1d_wang",
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
            description="validation best",
        )
        test_metrics = evaluate(
            model,
            loaders["test"],
            device,
            amp_enabled,
            description="test best",
        )
        write_json(Path(artifact_paths["validation_metrics"]), validation_metrics)
        write_json(Path(artifact_paths["test_metrics"]), test_metrics)
        status["status"] = "completed"
        status["finished_at"] = _utc_now()
        status["best_epoch"] = int(checkpoint["epoch"])
        status["selection_metric"] = "validation_macro_auprc"
        status["best_validation_macro_auprc"] = float(
            checkpoint["validation_macro_auprc"]
        )
        write_json(status_path, status)
        return status
    except BaseException as error:
        status["status"] = "failed"
        status["finished_at"] = _utc_now()
        status["failure"] = {"type": type(error).__name__, "message": str(error)}
        write_json(status_path, status)
        raise
