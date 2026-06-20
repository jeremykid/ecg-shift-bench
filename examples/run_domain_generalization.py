"""Example: Domain Generalisation (Leave-One-Dataset-Out).

This script demonstrates the Leave-One-Dataset-Out (LODO) domain
generalisation benchmark.  For each dataset, a model is trained on all
other datasets and evaluated zero-shot on the held-out dataset.

Usage::

    python examples/run_domain_generalization.py \
        --ptbxl /data/ptbxl \
        --chapman /data/chapman \
        --cpsc2018 /data/cpsc2018 \
        --g12ec /data/g12ec \
        --epochs 10

Requires the datasets to be downloaded and structured as documented in the
ECGShiftBench README.
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ecgshiftbench.benchmarks.domain_generalization import DomainGeneralizationBenchmark
from ecgshiftbench.datasets import (
    ChapmanDataset,
    CPSC2018Dataset,
    G12ECDataset,
    PTBXLDataset,
)
from ecgshiftbench.models import resnet1d_18
from ecgshiftbench.utils.preprocessing import bandpass_filter, normalize_signal, pad_or_truncate

SHARED_LABELS = ["AF"]
TARGET_LENGTH = 5000


def preprocess(signal):
    import numpy as np
    signal = bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=500.0)
    signal = normalize_signal(signal, method="zscore")
    signal = pad_or_truncate(signal, TARGET_LENGTH)
    return torch.from_numpy(signal)


def build_datasets(args, split):
    return {
        "ptbxl": PTBXLDataset(root=args.ptbxl, split=split, transform=preprocess,
                               labels=SHARED_LABELS),
        "chapman": ChapmanDataset(root=args.chapman, split=split, transform=preprocess,
                                  labels=SHARED_LABELS),
        "cpsc2018": CPSC2018Dataset(root=args.cpsc2018, split=split, transform=preprocess,
                                    labels=SHARED_LABELS),
        "g12ec": G12ECDataset(root=args.g12ec, split=split, transform=preprocess,
                              labels=SHARED_LABELS),
    }


def train_model(train_datasets, epochs, batch_size, lr, device):
    combined = ConcatDataset(list(train_datasets.values()))
    loader = DataLoader(combined, batch_size=batch_size, shuffle=True,
                        num_workers=4, pin_memory=True)
    model = resnet1d_18(num_leads=12, num_classes=len(SHARED_LABELS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for signals, labels in loader:
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(signals), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * signals.size(0)
        print(f"    epoch {epoch + 1}/{epochs}  loss={total_loss / len(combined):.4f}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Domain generalisation LODO example")
    parser.add_argument("--ptbxl", required=True)
    parser.add_argument("--chapman", required=True)
    parser.add_argument("--cpsc2018", required=True)
    parser.add_argument("--g12ec", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    train_datasets = build_datasets(args, split="train")
    test_datasets = build_datasets(args, split="test")

    all_names = list(train_datasets.keys())

    benchmark = DomainGeneralizationBenchmark(datasets=test_datasets, labels=SHARED_LABELS)

    all_results = []
    for target_name in all_names:
        print(f"\n--- LODO: held-out = {target_name} ---")
        sources = {k: v for k, v in train_datasets.items() if k != target_name}
        model = train_model(sources, args.epochs, args.batch_size, args.lr, device)
        result = benchmark.evaluate_lodo(model_for=lambda _name, m=model: m)
        # filter to the current target
        r = [x for x in result if x.target_dataset == target_name][0]
        all_results.append(r)

    print("\n=== LODO Domain Generalisation Results ===")
    for r in all_results:
        print(r)


if __name__ == "__main__":
    main()
