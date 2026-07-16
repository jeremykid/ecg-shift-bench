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


def test_cli_defaults_to_cuda_zero_for_internal_baseline_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_cli_routes_uda_pair_config_to_uda_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    script = _load_script_module()
    captured: dict[str, object] = {}
    source_dataset_config = tmp_path / "source.yaml"
    target_dataset_config = tmp_path / "target.yaml"
    source_dataset_config.write_text("name: SOURCE\nroot: /placeholder/source\n", encoding="utf-8")
    target_dataset_config.write_text("name: TARGET\nroot: /placeholder/target\n", encoding="utf-8")

    monkeypatch.setattr(
        script,
        "load_yaml",
        lambda path: {
            "experiment": "uda_source_only_source_to_target",
            "method": "source_only",
            "source_datasets": ["SOURCE"],
            "target_datasets": ["TARGET"],
            "dataset_configs": {
                "source": "configs/datasets/source.yaml",
                "target": "configs/datasets/target.yaml",
            },
            "model": {"name": "resnet1d", "width": 4},
            "data": {
                "input_length": 64,
                "preprocessing_version": "shared_alignment_v1",
                "sampling_rate": 500,
                "target_sampling_rate": 500,
                "source_unit": "mV",
                "target_unit": "mV",
                "normalization": "none",
            },
            "training": {
                "seed": 7,
                "batch_size": 2,
                "workers": 0,
                "epochs": 1,
                "optimizer": "adamw",
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "amp": False,
            },
            "protocol": {
                "target_inputs_available_during_training": True,
                "target_labels_available_during_training": False,
            },
        },
    )
    monkeypatch.setattr(
        script,
        "resolve_project_path",
        lambda path: source_dataset_config if "source" in str(path) else target_dataset_config,
    )
    monkeypatch.setattr(
        script,
        "run_uda_cross_domain",
        lambda **kwargs: captured.update(kwargs) or {"status": "completed"},
        raising=False,
    )
    monkeypatch.setattr(
        script,
        "run_source_only_cross_domain",
        lambda **kwargs: pytest.fail(
            "legacy source-only runner should not be used for UDA configs"
        ),
    )
    argv = [
        "scripts/train.py",
        "--config",
        "configs/experiments/uda_source_only.yaml",
        "--source-root",
        str(tmp_path / "source"),
        "--target-root",
        str(tmp_path / "target"),
        "--output-dir",
        str(tmp_path / "outputs"),
        "--device",
        "cpu",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    script.main()

    assert captured["requested_device"] == "cpu"
    assert captured["output_dir"] == tmp_path / "outputs"
    assert captured["source_dataset_spec"].name == "SOURCE"
    assert captured["target_dataset_spec"].name == "TARGET"


def test_cli_routes_coral_pair_config_to_uda_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    script = _load_script_module()
    captured: dict[str, object] = {}
    source_dataset_config = tmp_path / "source.yaml"
    target_dataset_config = tmp_path / "target.yaml"
    source_dataset_config.write_text("name: SOURCE\nroot: /placeholder/source\n", encoding="utf-8")
    target_dataset_config.write_text("name: TARGET\nroot: /placeholder/target\n", encoding="utf-8")

    monkeypatch.setattr(
        script,
        "load_yaml",
        lambda path: {
            "experiment": "uda_coral_source_to_target",
            "method": "coral",
            "source_datasets": ["SOURCE"],
            "target_datasets": ["TARGET"],
            "dataset_configs": {
                "source": "configs/datasets/source.yaml",
                "target": "configs/datasets/target.yaml",
            },
            "model": {"name": "resnet1d", "width": 4},
            "data": {
                "input_length": 64,
                "preprocessing_version": "shared_alignment_v1",
                "sampling_rate": 500,
                "target_sampling_rate": 500,
                "source_unit": "mV",
                "target_unit": "mV",
                "normalization": "none",
            },
            "training": {
                "seed": 7,
                "batch_size": 2,
                "workers": 0,
                "epochs": 1,
                "optimizer": "adamw",
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "amp": False,
            },
            "method_params": {"lambda": 0.1},
            "protocol": {
                "target_inputs_available_during_training": True,
                "target_labels_available_during_training": False,
            },
        },
    )
    monkeypatch.setattr(
        script,
        "resolve_project_path",
        lambda path: source_dataset_config if "source" in str(path) else target_dataset_config,
    )
    monkeypatch.setattr(
        script,
        "run_uda_cross_domain",
        lambda **kwargs: captured.update(kwargs) or {"status": "completed"},
        raising=False,
    )
    monkeypatch.setattr(
        script,
        "run_source_only_cross_domain",
        lambda **kwargs: pytest.fail(
            "legacy source-only runner should not be used for CORAL configs"
        ),
    )
    argv = [
        "scripts/train.py",
        "--config",
        "configs/experiments/uda_coral.yaml",
        "--source-root",
        str(tmp_path / "source"),
        "--target-root",
        str(tmp_path / "target"),
        "--output-dir",
        str(tmp_path / "outputs"),
        "--device",
        "cpu",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    script.main()

    assert captured["experiment_config"]["method"] == "coral"
    assert captured["experiment_config"]["method_params"] == {"lambda": 0.1}
    assert captured["requested_device"] == "cpu"
