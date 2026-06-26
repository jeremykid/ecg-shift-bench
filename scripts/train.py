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
from ecg_shift_bench.training.ptbxl_baseline import run_ptbxl_baseline
from ecg_shift_bench.training.trainer import train_one_epoch
from ecg_shift_bench.utils.config import load_yaml, require_keys
from ecg_shift_bench.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-config", help="Dataset YAML for a real-data run")
    parser.add_argument("--root", help="Local dataset release root (never copied into config)")
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
    training = config["training"]
    seed_everything(int(training["seed"]))
    print(f"Validated experiment: {config.get('experiment', 'unnamed')}")
    if args.smoke_test:
        run_smoke_test(config)
        return
    if not args.root:
        print(
            "No data loaded. Pass --smoke-test, or provide --dataset-config, --root, "
            "and --output-dir for a real run."
        )
        return
    if not args.dataset_config or not args.output_dir:
        parser.error("--dataset-config and --output-dir are required with --root")
    if config.get("experiment") != "ptbxl-source-only-resnet1d-wang-500hz-e1-seed42":
        raise NotImplementedError("The real-data CLI currently supports the fixed PTB-XL baseline")

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
