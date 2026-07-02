"""CLI tests for the source-only cross-domain workflow."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_script_module(script_name: str):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.removesuffix(".py"), script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_train_cli_routes_pair_config_to_source_only_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    script = _load_script_module("train.py")
    captured: dict[str, object] = {}
    source_dataset_config = tmp_path / "source.yaml"
    target_dataset_config = tmp_path / "target.yaml"
    source_dataset_config.write_text("name: SOURCE\nroot: /placeholder/source\n", encoding="utf-8")
    target_dataset_config.write_text("name: TARGET\nroot: /placeholder/target\n", encoding="utf-8")

    monkeypatch.setattr(
        script,
        "load_yaml",
        lambda path: {
            "experiment": "source_only_source_to_target",
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
        },
    )
    monkeypatch.setattr(
        script,
        "resolve_project_path",
        lambda path: source_dataset_config if "source" in str(path) else target_dataset_config,
    )
    monkeypatch.setattr(
        script,
        "run_source_only_cross_domain",
        lambda **kwargs: captured.update(kwargs) or {"status": "completed"},
        raising=False,
    )
    monkeypatch.setattr(
        script,
        "run_ptbxl_baseline",
        lambda **kwargs: pytest.fail("legacy PTB-XL path should not be used for pair-shaped configs"),
    )
    argv = [
        "scripts/train.py",
        "--config",
        "configs/experiments/source_only.yaml",
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
    assert captured["source_dataset_spec"].root == (tmp_path / "source").resolve()
    assert captured["target_dataset_spec"].root == (tmp_path / "target").resolve()


def test_train_cli_can_override_template_pair_for_any_source_target_pair(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    script = _load_script_module("train.py")
    captured: dict[str, object] = {}
    source_dataset_config = tmp_path / "source.yaml"
    target_dataset_config = tmp_path / "target.yaml"
    source_dataset_config.write_text("name: SOURCE\nroot: /placeholder/source\n", encoding="utf-8")
    target_dataset_config.write_text("name: TARGET\nroot: /placeholder/target\n", encoding="utf-8")
    experiment_config = tmp_path / "experiment.yaml"
    experiment_config.write_text(
        "\n".join(
            [
                "experiment: source_only_template",
                "method: source_only",
                "source_datasets: [PTBXL]",
                "target_datasets: [CHAPMAN]",
                "dataset_configs:",
                "  source: configs/datasets/ptbxl.yaml",
                "  target: configs/datasets/chapman.yaml",
                "model:",
                "  name: resnet1d",
                "  width: 4",
                "training:",
                "  seed: 7",
                "  batch_size: 2",
                "  workers: 0",
                "  epochs: 1",
                "  optimizer: adamw",
                "  learning_rate: 0.001",
                "  weight_decay: 0.0",
                "  amp: false",
                "data:",
                "  input_length: 64",
                "  preprocessing_version: shared_alignment_v1",
                "  sampling_rate: 500",
                "  target_sampling_rate: 500",
                "  source_unit: mV",
                "  target_unit: mV",
                "  normalization: none",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        script,
        "run_source_only_cross_domain",
        lambda **kwargs: captured.update(kwargs) or {"status": "completed"},
        raising=False,
    )
    monkeypatch.setattr(
        script,
        "run_ptbxl_baseline",
        lambda **kwargs: pytest.fail("legacy PTB-XL path should not be used for overridden pair configs"),
    )
    argv = [
        "scripts/train.py",
        "--config",
        str(experiment_config),
        "--source-dataset",
        "SOURCE",
        "--target-dataset",
        "TARGET",
        "--source-dataset-config",
        str(source_dataset_config),
        "--target-dataset-config",
        str(target_dataset_config),
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

    assert captured["source_dataset_spec"].name == "SOURCE"
    assert captured["target_dataset_spec"].name == "TARGET"
    assert captured["source_dataset_spec"].config_path == source_dataset_config.resolve()
    assert captured["target_dataset_spec"].config_path == target_dataset_config.resolve()
    assert captured["experiment_config"]["source_datasets"] == ["SOURCE"]
    assert captured["experiment_config"]["target_datasets"] == ["TARGET"]


def test_evaluate_cli_rebuilds_completed_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    script = _load_script_module("evaluate.py")
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        script,
        "rebuild_source_only_cross_domain_results",
        lambda **kwargs: captured.update(kwargs) or {"status": {"status": "completed"}},
        raising=False,
    )
    argv = [
        "scripts/evaluate.py",
        "--run-dir",
        str(tmp_path / "outputs"),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    script.main()

    assert captured["run_dir"] == tmp_path / "outputs"
