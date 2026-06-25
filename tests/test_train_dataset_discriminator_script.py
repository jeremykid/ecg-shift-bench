"""Tests for the dataset-discriminator CLI."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "train_dataset_discriminator.py"
    spec = importlib.util.spec_from_file_location("train_dataset_discriminator", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cli_builds_dataset_specs_and_calls_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    script = _load_script_module()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        script,
        "load_yaml",
        lambda path: {
            "experiment": "dataset_discriminator_resnet1d_wang_v1",
            "model": {"in_channels": 12},
            "training": {
                "seed": 42,
                "batch_size": 2,
                "epochs": 1,
                "optimizer": "adamw",
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "workers": 0,
            },
            "dataset_configs": {
                "PTBXL": "configs/datasets/ptbxl.yaml",
                "CODE15": "configs/datasets/code15.yaml",
                "CHAPMAN": "configs/datasets/chapman.yaml",
                "SPH": "configs/datasets/sph.yaml",
            },
        },
    )
    monkeypatch.setattr(
        script,
        "run_dataset_discriminator",
        lambda **kwargs: captured.update(kwargs) or {"status": "preflight_completed"},
    )
    argv = [
        "scripts/train_dataset_discriminator.py",
        "--config",
        "configs/experiments/dataset_discriminator.yaml",
        "--ptbxl-root",
        str(tmp_path / "ptbxl"),
        "--code15-root",
        str(tmp_path / "code15"),
        "--chapman-root",
        str(tmp_path / "chapman"),
        "--sph-root",
        str(tmp_path / "sph"),
        "--output-dir",
        str(tmp_path / "outputs"),
        "--device",
        "cuda:0",
        "--mode",
        "multiclass",
        "--subset",
        "uncontrolled",
        "--preflight-only",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    script.main()

    assert captured["mode"] == "multiclass"
    assert captured["subset"] == "uncontrolled"
    assert len(captured["dataset_specs"]) == 4
    assert [spec.name for spec in captured["dataset_specs"]] == [
        "PTBXL",
        "CODE15",
        "CHAPMAN",
        "SPH",
    ]
    assert captured["requested_device"] == "cuda:0"
