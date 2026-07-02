#!/usr/bin/env python3
"""Run a configured baseline or a synthetic source-only smoke test."""

import argparse
import shlex
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ecg_shift_bench.models.resnet1d import ResNet1D
from ecg_shift_bench.training.optim import create_optimizer
from ecg_shift_bench.training.source_only_cross_domain import (
    _prepare_dataset_spec,
    run_source_only_cross_domain,
)
from ecg_shift_bench.training.ptbxl_baseline import run_ptbxl_baseline
from ecg_shift_bench.training.trainer import train_one_epoch
from ecg_shift_bench.utils.config import load_yaml, require_keys
from ecg_shift_bench.utils.paths import resolve_project_path
from ecg_shift_bench.utils.seed import seed_everything


def _dataset_config_path(dataset_name: str) -> Path:
    return resolve_project_path(Path("configs/datasets") / f"{dataset_name.lower().replace('-', '')}.yaml")


def _apply_pair_overrides(config: dict, args: argparse.Namespace) -> dict:
    updated = dict(config)
    source_datasets = list(updated.get("source_datasets") or [])
    target_datasets = list(updated.get("target_datasets") or [])
    if args.source_dataset:
        source_datasets = [args.source_dataset]
    if args.target_dataset:
        target_datasets = [args.target_dataset]
    if source_datasets:
        updated["source_datasets"] = source_datasets
    if target_datasets:
        updated["target_datasets"] = target_datasets

    dataset_configs = dict(updated.get("dataset_configs") or {})
    if args.source_dataset_config:
        dataset_configs["source"] = str(resolve_project_path(args.source_dataset_config))
    elif args.source_dataset:
        dataset_configs["source"] = str(_dataset_config_path(args.source_dataset))
    if args.target_dataset_config:
        dataset_configs["target"] = str(resolve_project_path(args.target_dataset_config))
    elif args.target_dataset:
        dataset_configs["target"] = str(_dataset_config_path(args.target_dataset))
    if dataset_configs:
        updated["dataset_configs"] = dataset_configs
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-config", help="Dataset YAML for a real-data run")
    parser.add_argument("--root", help="Local dataset release root (never copied into config)")
    parser.add_argument("--source-dataset", help="Override the source dataset name for pair-shaped runs")
    parser.add_argument("--target-dataset", help="Override the target dataset name for pair-shaped runs")
    parser.add_argument(
        "--source-dataset-config",
        help="Override the source dataset config path for pair-shaped runs",
    )
    parser.add_argument(
        "--target-dataset-config",
        help="Override the target dataset config path for pair-shaped runs",
    )
    parser.add_argument("--source-root", help="Override the source dataset root for pair-shaped runs")
    parser.add_argument("--target-root", help="Override the target dataset root for pair-shaped runs")
    parser.add_argument("--device", default="cpu", help="Torch device, for example cuda:0")
    parser.add_argument("--output-dir", help="Artifact directory for a real-data run")
    parser.add_argument(
        "--snapshot-manifest",
        help="Override the snapshot manifest path declared by the experiment config",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate the snapshot and run one real forward/backward batch, then stop",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Run one tiny synthetic epoch")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    require_keys(config, ["method", "source_datasets", "target_datasets", "model", "training"])
    config = _apply_pair_overrides(config, args)
    training = config["training"]
    seed_everything(int(training["seed"]))
    print(f"Validated experiment: {config.get('experiment', 'unnamed')}")
    if args.smoke_test:
        run_smoke_test(config)
        return
    source_datasets = list(config.get("source_datasets") or [])
    target_datasets = list(config.get("target_datasets") or [])
    pair_shaped = len(source_datasets) == 1 and len(target_datasets) == 1 and bool(
        config.get("dataset_configs")
    )
    if pair_shaped:
        if not args.output_dir:
            parser.error("--output-dir is required for source-target runs")
        dataset_configs = config["dataset_configs"]
        if "source" not in dataset_configs or "target" not in dataset_configs:
            raise ValueError("Pair-shaped source-target configs must declare dataset_configs.source and .target")
        source_dataset_config_path = resolve_project_path(dataset_configs["source"])
        target_dataset_config_path = resolve_project_path(dataset_configs["target"])
        source_dataset_spec = _prepare_dataset_spec(
            dataset_name=source_datasets[0],
            dataset_config_path=source_dataset_config_path,
            root_override=args.source_root,
        )
        target_dataset_spec = _prepare_dataset_spec(
            dataset_name=target_datasets[0],
            dataset_config_path=target_dataset_config_path,
            root_override=args.target_root,
        )
        command = shlex.join([sys.executable, *sys.argv])
        status = run_source_only_cross_domain(
            experiment_config=config,
            experiment_config_path=config_path,
            source_dataset_spec=source_dataset_spec,
            target_dataset_spec=target_dataset_spec,
            output_dir=Path(args.output_dir).expanduser().resolve(),
            requested_device=args.device,
            command=command,
            preflight_only=args.preflight_only,
        )
        print(f"Run status: {status['status']}")
        return
    if config.get("experiment") != "ptbxl-source-only-resnet1d-wang-500hz-e1-seed42":
        raise NotImplementedError("The real-data CLI currently supports the fixed PTB-XL baseline")
    if not args.root:
        print(
            "No data loaded. Pass --smoke-test, or provide --dataset-config, --root, "
            "and --output-dir for a real run."
        )
        return
    if not args.dataset_config or not args.output_dir:
        parser.error("--dataset-config and --output-dir are required with --root")

    dataset_config_path = Path(args.dataset_config).resolve()
    dataset_config = load_yaml(dataset_config_path)
    snapshot_path_value = args.snapshot_manifest or config.get("dataset_snapshot_manifest")
    if not snapshot_path_value:
        raise ValueError("Experiment config must declare dataset_snapshot_manifest")
    snapshot_path = Path(snapshot_path_value)
    if not snapshot_path.is_absolute():
        snapshot_path = (Path.cwd() / snapshot_path).resolve()
    snapshot_manifest = load_yaml(snapshot_path)
    command = shlex.join([sys.executable, *sys.argv])
    status = run_ptbxl_baseline(
        experiment_config=config,
        experiment_config_path=config_path,
        dataset_config=dataset_config,
        dataset_config_path=dataset_config_path,
        snapshot_manifest=snapshot_manifest,
        snapshot_manifest_path=snapshot_path,
        root=Path(args.root).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        requested_device=args.device,
        command=command,
        preflight_only=args.preflight_only,
    )
    print(f"Run status: {status['status']}")


def run_smoke_test(config: dict) -> None:
    """Preserve the original data-free source-only optimizer check."""
    if config["method"] != "source_only":
        raise NotImplementedError("Only source_only has a runnable baseline in this initialization")

    training = config["training"]
    inputs = torch.randn(4, 12, 256)
    targets = torch.randint(0, 2, (4, 6)).float()
    batches = DataLoader(TensorDataset(inputs, targets), batch_size=2)
    model = ResNet1D()
    optimizer = create_optimizer(
        model.parameters(),
        training["optimizer"],
        float(training["learning_rate"]),
        float(training.get("weight_decay", 0.0)),
    )
    loss = train_one_epoch(model, batches, optimizer, nn.BCEWithLogitsLoss(), "cpu")
    print(f"Synthetic smoke-test loss: {loss:.6f}")


if __name__ == "__main__":
    main()
