"""Example: Unsupervised Domain Adaptation (UDA).

This script demonstrates the UDA benchmark: a model is first trained on
PTB-XL, then adapted using unlabelled Chapman training data (e.g. via
simple fine-tuning on pseudo-labels), and finally evaluated on the
labelled Chapman test split.

Usage::

    python examples/run_uda.py \
        --ptbxl /data/ptbxl \
        --chapman /data/chapman \
        --epochs-source 10 \
        --epochs-adapt 5

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

from ecgshiftbench.benchmarks.uda import UDABenchmark
from ecgshiftbench.datasets import ChapmanDataset, PTBXLDataset
from ecgshiftbench.models import resnet1d_18
from ecgshiftbench.utils.preprocessing import bandpass_filter, normalize_signal, pad_or_truncate

SHARED_LABELS = ["AF"]
TARGET_LENGTH = 5000


def preprocess(signal):
    signal = bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=500.0)
    signal = normalize_signal(signal, method="zscore")
    signal = pad_or_truncate(signal, TARGET_LENGTH)
    return torch.from_numpy(signal)


def train_supervised(model, loader, optimizer, criterion, epochs, device):
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
        print(f"  [supervised] epoch {epoch + 1}/{epochs}  "
              f"loss={total_loss / len(loader.dataset):.4f}")


def adapt_pseudo_label(model, unlabelled_loader, optimizer, criterion, epochs, device,
                       threshold=0.8):
    """Simple pseudo-label adaptation: use confident predictions as supervision."""
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n = 0
        for signals, _ in unlabelled_loader:
            signals = signals.to(device)
            with torch.no_grad():
                probs = torch.sigmoid(model(signals))
            # Only use samples where model is confident (> threshold or < 1-threshold)
            pseudo_labels = (probs > threshold).float()
            confident = ((probs > threshold) | (probs < 1 - threshold)).all(dim=1)
            if confident.sum() == 0:
                continue
            optimizer.zero_grad()
            logits = model(signals[confident])
            loss = criterion(logits, pseudo_labels[confident])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * confident.sum().item()
            n += confident.sum().item()
        avg_loss = total_loss / max(n, 1)
        print(f"  [adapt]      epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}  "
              f"n_confident={n}")


def main():
    parser = argparse.ArgumentParser(description="UDA benchmark example")
    parser.add_argument("--ptbxl", required=True)
    parser.add_argument("--chapman", required=True)
    parser.add_argument("--epochs-source", type=int, default=10)
    parser.add_argument("--epochs-adapt", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Datasets
    source_train = PTBXLDataset(root=args.ptbxl, split="train", transform=preprocess,
                                labels=SHARED_LABELS)
    target_train = ChapmanDataset(root=args.chapman, split="train", transform=preprocess,
                                  labels=SHARED_LABELS)
    target_test = ChapmanDataset(root=args.chapman, split="test", transform=preprocess,
                                 labels=SHARED_LABELS)

    source_loader = DataLoader(source_train, batch_size=args.batch_size, shuffle=True,
                               num_workers=4, pin_memory=True)
    target_loader = DataLoader(target_train, batch_size=args.batch_size, shuffle=True,
                               num_workers=4, pin_memory=True)

    model = resnet1d_18(num_leads=12, num_classes=len(SHARED_LABELS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    print("Phase 1: Supervised training on PTB-XL …")
    train_supervised(model, source_loader, optimizer, criterion, args.epochs_source, device)

    print("\nPhase 2: Pseudo-label adaptation on Chapman (unlabelled) …")
    adapt_pseudo_label(model, target_loader, optimizer, criterion, args.epochs_adapt, device)

    # Evaluate with the benchmark
    benchmark = UDABenchmark(
        source=source_train,
        target_unlabelled=target_train,
        target_test=target_test,
        labels=SHARED_LABELS,
    )
    results = benchmark.evaluate(model)

    print("\n=== UDA Results ===")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
