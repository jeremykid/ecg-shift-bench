#!/usr/bin/env python3
"""Validate training configuration; optionally run a synthetic source-only smoke test."""

import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ecg_shift_bench.models.resnet1d import ResNet1D
from ecg_shift_bench.training.optim import create_optimizer
from ecg_shift_bench.training.trainer import train_one_epoch
from ecg_shift_bench.utils.config import load_yaml, require_keys
from ecg_shift_bench.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke-test", action="store_true", help="Run one tiny synthetic epoch")
    args = parser.parse_args()

    config = load_yaml(args.config)
    require_keys(config, ["method", "source_datasets", "target_datasets", "model", "training"])
    training = config["training"]
    seed_everything(int(training["seed"]))
    print(f"Validated experiment: {config.get('experiment', 'unnamed')}")
    if not args.smoke_test:
        print(
            "No data loaded. Pass --smoke-test or prepare dataset indexes "
            "and a project DataLoader."
        )
        return
    if config["method"] != "source_only":
        raise NotImplementedError("Only source_only has a runnable baseline in this initialization")

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
