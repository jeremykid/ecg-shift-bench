"""Tests for the dataset-statistics visualization script."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


def _load_script_module():
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "visualize_dataset_statistics.py"
    )
    spec = importlib.util.spec_from_file_location("visualize_dataset_statistics", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_fixture(stats_dir: Path, dataset: str = "chapman") -> Path:
    dataset_dir = stats_dir / dataset
    dataset_dir.mkdir(parents=True)
    audit = {
        "dataset": "CHAPMAN",
        "records_total": 100,
        "records_usable": 99,
        "records_excluded": 1,
        "patients_total": None,
        "split_policy": {"split_source": "generated", "split_level": "record"},
    }
    reproducibility = {
        "target_sampling_rate": 500,
        "target_length": 5000,
        "lead_order": ["I", "II"],
        "source_unit": "uV",
        "target_unit": "mV",
        "unit_converted": True,
    }
    (dataset_dir / "audit.json").write_text(json.dumps(audit), encoding="utf-8")
    (dataset_dir / "reproducibility.json").write_text(
        json.dumps(reproducibility), encoding="utf-8"
    )
    pd.DataFrame(
        {"label": ["AF", "LBBB"], "positive_rate": [0.10, 0.02]}
    ).to_csv(dataset_dir / "positive_rate.csv", index=False)
    pd.DataFrame(
        {"label": ["AF", "LBBB"], "positive_count": [10, 2]}
    ).to_csv(dataset_dir / "label_distribution.csv", index=False)
    pd.DataFrame(
        {
            "split": ["train", "validation", "test"],
            "records": [70, 10, 20],
            "patients": ["n/a", "n/a", "n/a"],
        }
    ).to_csv(dataset_dir / "split_summary.csv", index=False)
    pd.DataFrame(
        {
            "split": ["train", "train", "validation", "validation", "test", "test"],
            "label": ["AF", "LBBB"] * 3,
            "positive_rate": [0.10, 0.02, 0.13, 0.02, 0.09, 0.02],
        }
    ).to_csv(dataset_dir / "split_positive_rate.csv", index=False)
    return dataset_dir


def test_generates_all_figures_and_summary_without_raw_data(tmp_path: Path) -> None:
    module = _load_script_module()
    stats_dir = tmp_path / "statistics"
    output_dir = tmp_path / "figures"
    _write_fixture(stats_dir)

    outputs = module.generate_visualizations(stats_dir, output_dir, "png")

    expected = {
        "positive_rate_heatmap.png",
        "positive_count_heatmap.png",
        "label_prior_range.png",
        "split_label_gap_heatmap.png",
        "dataset_statistics_summary.md",
    }
    assert {path.name for path in outputs} == expected
    assert all((output_dir / name).is_file() for name in expected)

    report = (output_dir / "dataset_statistics_summary.md").read_text(encoding="utf-8")
    assert "N/A patients" in report
    assert "train 70.0%" in report
    assert "Chapman validation" in report
    assert "0.0300 exceeds 0.01" in report
    assert "no raw ECG waveforms were read" in report


def test_missing_csv_skips_only_affected_figure_and_reports_warning(tmp_path: Path) -> None:
    module = _load_script_module()
    stats_dir = tmp_path / "statistics"
    output_dir = tmp_path / "figures"
    dataset_dir = _write_fixture(stats_dir)
    (dataset_dir / "label_distribution.csv").unlink()

    outputs = module.generate_visualizations(stats_dir, output_dir, "png")

    assert "positive_count_heatmap.png" not in {path.name for path in outputs}
    assert (output_dir / "positive_rate_heatmap.png").is_file()
    report = (output_dir / "dataset_statistics_summary.md").read_text(encoding="utf-8")
    assert "positive-count heatmap skipped this dataset" in report
    assert "figure not generated because no usable data were available" in report
