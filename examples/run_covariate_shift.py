"""Example: Covariate Shift Evaluation.

This script trains a 1D-ResNet on PTB-XL and evaluates it on Chapman and
CPSC-2018 to measure performance degradation due to covariate shift.

Usage::

    python examples/run_covariate_shift.py \
        --ptbxl /data/ptbxl \
        --chapman /data/chapman \
        --cpsc2018 /data/cpsc2018 \
        --epochs 20 \
        --batch-size 32

Requires the datasets to be downloaded and structured as documented in the
ECGShiftBench README.
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ecgshiftbench.benchmarks.covariate_shift import CovariateShiftBenchmark
from ecgshiftbench.datasets import ChapmanDataset, CPSC2018Dataset, PTBXLDataset
from ecgshiftbench.models import resnet1d_18
from ecgshiftbench.utils.preprocessing import bandpass_filter, normalize_signal, pad_or_truncate


# Shared labels available across all three datasets
SHARED_LABELS = ["AF"]

# Target signal length at 500 Hz for 10 seconds
TARGET_LENGTH = 5000


def preprocess(signal):
    """Apply standard ECG preprocessing pipeline."""
    signal = bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=500.0)
    signal = normalize_signal(signal, method="zscore")
    signal = pad_or_truncate(signal, TARGET_LENGTH)
    return torch.from_numpy(signal)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for signals, labels in loader:
        signals, labels = signals.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(signals)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * signals.size(0)
    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser(description="Covariate shift benchmark example")
    parser.add_argument("--ptbxl", required=True, help="Path to PTB-XL root directory")
    parser.add_argument("--chapman", required=True, help="Path to Chapman root directory")
    parser.add_argument("--cpsc2018", required=True, help="Path to CPSC-2018 root directory")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load datasets
    print("Loading datasets …")
    source_train = PTBXLDataset(root=args.ptbxl, split="train", transform=preprocess,
                                labels=SHARED_LABELS)
    chapman_test = ChapmanDataset(root=args.chapman, split="test", transform=preprocess,
                                  labels=SHARED_LABELS)
    cpsc_test = CPSC2018Dataset(root=args.cpsc2018, split="test", transform=preprocess,
                                labels=SHARED_LABELS)

    train_loader = DataLoader(source_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)

    # Build model
    model = resnet1d_18(num_leads=12, num_classes=len(SHARED_LABELS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # Train on source
    print(f"\nTraining on PTB-XL for {args.epochs} epochs …")
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"  Epoch {epoch + 1}/{args.epochs}  loss={loss:.4f}")

    # Evaluate covariate shift
    print("\nEvaluating covariate shift …")
    benchmark = CovariateShiftBenchmark(
        source=source_train,
        targets=[chapman_test, cpsc_test],
        labels=SHARED_LABELS,
    )
    results = benchmark.evaluate(model)

    print("\n=== Results ===")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
