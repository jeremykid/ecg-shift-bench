"""Tests for the main training CLI."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("train", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cli_defaults_to_cuda_zero_for_issue11_run(monkeypatch: pytest.MonkeyPatch) -> None:
    script = _load_script_module()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        script,
        "load_yaml",
        lambda path: {
            "experiment": "resnet1d-internal-dataset-baseline",
            "method": "source_only",
            "source_datasets": ["PTBXL", "CODE15", "CHAPMAN", "SPH"],
            "target_datasets": [],
            "model": {"name": "resnet1d", "in_channels": 12, "width": 32},
            "training": {
                "seed": 42,
                "batch_size": 2,
                "epochs": 1,
                "optimizer": "adamw",
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "workers": 0,
            },
        },
    )
    monkeypatch.setattr(script, "seed_everything", lambda seed: None)
    monkeypatch.setattr(
        script,
        "run_internal_dataset_baseline",
        lambda **kwargs: captured.update(kwargs) or {"status": "preflight_completed"},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/train.py",
            "--config",
            "configs/experiments/resnet1d_internal_dataset_baseline.yaml",
            "--preflight-only",
        ],
    )

    script.main()

    assert captured["requested_device"] == "cuda:0"

