"""Tests for the fixed PTB-XL source-only baseline."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from ecg_shift_bench.datasets.ptbxl_baseline import (
    PTBXLClassificationDataset,
    prepare_ptbxl_snapshot,
)
from ecg_shift_bench.models.resnet1d_wang import resnet1d_wang
from ecg_shift_bench.training.ptbxl_baseline import preflight_forward_backward
from ecg_shift_bench.utils.config import load_yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_PTBXL_ROOT = os.environ.get("PTBXL_ROOT")


def _synthetic_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ecg_id": [1, 2],
            "patient_id": [10, 20],
            "scp_codes": ["{'AFIB': 0.0}", "{}"],
            "filename_hr": ["one", "two"],
        }
    )


def test_resnet1d_wang_legacy_parameter_parity_and_six_label_shape() -> None:
    legacy_model = resnet1d_wang(num_labels=5)
    assert sum(parameter.numel() for parameter in legacy_model.parameters()) == 440_837

    model = resnet1d_wang(num_labels=6).eval()
    with torch.no_grad():
        logits = model(torch.randn(1, 12, 64))
    assert logits.shape == (1, 6)


def test_dataset_batch_model_and_loss_uses_raw_waveforms(monkeypatch: pytest.MonkeyPatch) -> None:
    base = np.arange(12 * 64, dtype=np.float32).reshape(12, 64)
    monkeypatch.setattr(
        "ecg_shift_bench.datasets.ptbxl.PTBXLDataset.load_signal",
        lambda self, record_id: base + int(record_id),
    )
    dataset = PTBXLClassificationDataset(
        ".",
        {"sampling_rate": 500},
        _synthetic_metadata(),
        input_length=64,
    )
    inputs, targets = next(iter(DataLoader(dataset, batch_size=2)))
    torch.testing.assert_close(inputs[0], torch.from_numpy(base + 1))
    torch.testing.assert_close(inputs[1], torch.from_numpy(base + 2))
    model = resnet1d_wang(num_labels=6)
    loss = nn.BCEWithLogitsLoss()(model(inputs), targets)
    assert torch.isfinite(loss)
    preflight_loss = preflight_forward_backward(
        model,
        (inputs, targets),
        nn.BCEWithLogitsLoss(),
        torch.device("cpu"),
        False,
    )
    assert np.isfinite(preflight_loss)


def test_checkpoint_reload_preserves_logits(tmp_path: Path) -> None:
    torch.manual_seed(7)
    model = resnet1d_wang(num_labels=6).eval()
    inputs = torch.randn(2, 12, 64)
    with torch.no_grad():
        expected = model(inputs)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint)
    reloaded = resnet1d_wang(num_labels=6).eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    reloaded.load_state_dict(state["model_state_dict"])
    with torch.no_grad():
        observed = reloaded(inputs)
    torch.testing.assert_close(observed, expected)


@pytest.mark.skipif(not REAL_PTBXL_ROOT, reason="set PTBXL_ROOT for release integration checks")
def test_real_ptbxl_snapshot_counts_folds_support_and_leakage() -> None:
    dataset_config = load_yaml(PROJECT_ROOT / "configs/datasets/ptbxl.yaml")
    manifest = load_yaml(
        PROJECT_ROOT / "configs/datasets/snapshots/ptbxl-1.0.3-six-label-v1.yaml"
    )
    snapshot = prepare_ptbxl_snapshot(REAL_PTBXL_ROOT, dataset_config, manifest)
    assert snapshot.split_manifest_sha256 == manifest["split_policy"]["manifest_sha256"]
    assert snapshot.summary["records_total"] == 21_799
    assert not any(snapshot.summary["patient_overlap"].values())
    for split in ("train", "validation", "test"):
        assert all(snapshot.summary["splits"][split]["positive_support"].values())
