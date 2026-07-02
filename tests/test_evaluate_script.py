"""Tests for the evaluation CLI."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("evaluate", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "canonical_labels": ["AF"],
                "evaluation": {"metrics": ["macro_auroc"]},
            }
        ),
        encoding="utf-8",
    )


def test_cli_keeps_multilabel_metrics_without_thresholds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _load_script_module()
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)
    predictions_path = tmp_path / "predictions.npz"
    np.savez_compressed(
        predictions_path,
        y_true=np.array([[1], [0], [1]]),
        y_score=np.array([[0.9], [0.2], [0.8]]),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/evaluate.py",
            "--config",
            str(config_path),
            "--predictions",
            str(predictions_path),
        ],
    )

    script.main()
    output = json.loads(capsys.readouterr().out)

    assert "macro_auroc" in output
    assert "micro_auprc" in output


def test_cli_uses_source_report_when_thresholds_are_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _load_script_module()
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)
    predictions_path = tmp_path / "predictions.npz"
    np.savez_compressed(
        predictions_path,
        label_names=np.array(["AF"], dtype="U32"),
        y_true=np.array([[1], [0], [1]]),
        y_score=np.array([[0.9], [0.2], [0.8]]),
        thresholds=np.array([0.5], dtype=float),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/evaluate.py",
            "--config",
            str(config_path),
            "--predictions",
            str(predictions_path),
        ],
    )

    script.main()
    output = json.loads(capsys.readouterr().out)

    assert "per_label_reports" in output
    assert "thresholds" in output
    assert output["per_label_reports"]["AF"]["prec"] >= 0.0
