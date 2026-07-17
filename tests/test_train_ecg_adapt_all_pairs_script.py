"""CLI tests for the ECG-Adapt-inspired multi-label all-pairs helper."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "train_ecg_adapt_all_pairs.py"
    spec = importlib.util.spec_from_file_location("train_ecg_adapt_all_pairs", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_ecg_adapt_pairs_script_runs_all_pairs_with_root_overrides(
    monkeypatch, tmp_path: Path
) -> None:
    script = _load_script_module()
    captured: list[list[str]] = []

    template_config = {
        "experiment": "uda_ecg_adapt_ptbxl_to_chapman",
        "method": "ecg_adapt_multilabel",
        "source_datasets": ["PTBXL"],
        "target_datasets": ["CHAPMAN"],
        "dataset_configs": {
            "source": "configs/datasets/ptbxl.yaml",
            "target": "configs/datasets/chapman.yaml",
        },
        "training": {"seed": 42},
    }

    def fake_run(command, check, env):
        del check, env
        captured.append(list(command))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(script.subprocess, "run", fake_run)
    monkeypatch.setattr(script.yaml, "safe_load", lambda text: template_config)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/train_ecg_adapt_all_pairs.py",
            "--output-root",
            str(tmp_path / "runs"),
            "--root",
            "PTBXL=/data/ptbxl",
            "--root",
            "CHAPMAN=/data/chapman",
            "--root",
            "SPH=/data/sph",
            "--root",
            "CODE15=/data/code15",
        ],
    )

    exit_code = script.main()

    assert exit_code == 0
    assert len(captured) == 12
    assert captured[0].count("--config") == 1
    assert captured[0].count("--source-dataset") == 1
    assert captured[0].count("--target-dataset") == 1
    assert captured[0].count("--source-root") == 1
    assert captured[0].count("--target-root") == 1

    expected_pairs = [
        ("CHAPMAN", "PTBXL"),
        ("CHAPMAN", "SPH"),
        ("CHAPMAN", "CODE15"),
        ("PTBXL", "CHAPMAN"),
        ("PTBXL", "SPH"),
        ("PTBXL", "CODE15"),
        ("SPH", "CHAPMAN"),
        ("SPH", "PTBXL"),
        ("SPH", "CODE15"),
        ("CODE15", "CHAPMAN"),
        ("CODE15", "PTBXL"),
        ("CODE15", "SPH"),
    ]
    actual_pairs = []
    for command in captured:
        source_index = command.index("--source-dataset") + 1
        target_index = command.index("--target-dataset") + 1
        config_index = command.index("--config") + 1
        actual_pairs.append((command[source_index], command[target_index]))

        pair_slug = f"{command[source_index].lower()}_to_{command[target_index].lower()}"
        pair_config_path = Path(command[config_index])
        assert pair_config_path.name == "launch_config.yaml"
        assert pair_config_path.parent.name == pair_slug
        assert pair_config_path.is_file()
        pair_config = yaml.full_load(pair_config_path.read_text(encoding="utf-8"))
        assert pair_config["experiment"] == f"uda_ecg_adapt_{pair_slug}"
        assert pair_config["source_datasets"] == [command[source_index]]
        assert pair_config["target_datasets"] == [command[target_index]]
        assert pair_config["dataset_configs"]["source"] == (
            f"configs/datasets/{command[source_index].lower()}.yaml"
        )
        assert pair_config["dataset_configs"]["target"] == (
            f"configs/datasets/{command[target_index].lower()}.yaml"
        )

    assert actual_pairs == expected_pairs
