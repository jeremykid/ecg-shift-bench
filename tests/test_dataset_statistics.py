"""Tests for dataset statistics export."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from ecg_shift_bench.datasets.audit import DatasetAuditResult
from ecg_shift_bench.datasets.statistics import (
    render_batch_dataset_statistics_report,
    write_batch_dataset_statistics_report,
    write_dataset_statistics_outputs,
)


def _fake_audit_result(dataset: str = "code15", title: str = "CODE15") -> DatasetAuditResult:
    manifest = pd.DataFrame(
        {
            "record_id": ["1", "2", "3"],
            "patient_id": ["p1", "p1", "p2"],
            "split": ["train", "validation", "test"],
            "domain": [f"{dataset}_domain"] * 3,
        }
    )
    audit = {
        "dataset": title,
        "domain": f"{dataset}_domain",
        "records_total": 3,
        "records_usable": 3,
        "records_excluded": 0,
        "patients_total": 2,
        "available_labels": ["AF", "RBBB", "LBBB", "1dAVB", "SB", "ST"],
        "missing_labels": {label: 0 for label in ["AF", "RBBB", "LBBB", "1dAVB", "SB", "ST"]},
        "positive_counts": {
            "AF": 1,
            "RBBB": 2,
            "LBBB": 0,
            "1dAVB": 1,
            "SB": 0,
            "ST": 1,
        },
        "positive_prevalence": {
            "AF": 1 / 3,
            "RBBB": 2 / 3,
            "LBBB": 0.0,
            "1dAVB": 1 / 3,
            "SB": 0.0,
            "ST": 1 / 3,
        },
        "waveform_check": {
            "checked_records": 0,
            "mode": "metadata_only",
            "target_sampling_rate": 500,
            "target_length": 5000,
            "lead_order": ["I", "II", "III"],
            "source_unit": "mV",
            "target_unit": "mV",
            "unit_converted": False,
        },
        "split_policy": {
            "split_source": "generated",
            "split_level": "patient",
            "split_algorithm": "patient_level_random",
            "train_fraction": 0.7,
            "validation_fraction": 0.1,
            "test_fraction": 0.2,
            "seed": 42,
        },
        "split_label_summary": {
            "train": {
                "records": 1,
                "patients": 1,
                "positive_counts": {
                    "AF": 1,
                    "RBBB": 0,
                    "LBBB": 0,
                    "1dAVB": 0,
                    "SB": 0,
                    "ST": 0,
                },
                "positive_prevalence": {
                    "AF": 1.0,
                    "RBBB": 0.0,
                    "LBBB": 0.0,
                    "1dAVB": 0.0,
                    "SB": 0.0,
                    "ST": 0.0,
                },
            },
            "validation": {
                "records": 1,
                "patients": 1,
                "positive_counts": {
                    "AF": 0,
                    "RBBB": 1,
                    "LBBB": 0,
                    "1dAVB": 0,
                    "SB": 0,
                    "ST": 0,
                },
                "positive_prevalence": {
                    "AF": 0.0,
                    "RBBB": 1.0,
                    "LBBB": 0.0,
                    "1dAVB": 0.0,
                    "SB": 0.0,
                    "ST": 0.0,
                },
            },
            "test": {
                "records": 1,
                "patients": 1,
                "positive_counts": {
                    "AF": 0,
                    "RBBB": 0,
                    "LBBB": 1,
                    "1dAVB": 0,
                    "SB": 0,
                    "ST": 0,
                },
                "positive_prevalence": {
                    "AF": 0.0,
                    "RBBB": 0.0,
                    "LBBB": 1.0,
                    "1dAVB": 0.0,
                    "SB": 0.0,
                    "ST": 0.0,
                },
            },
        },
    }
    exclusions = pd.DataFrame({"record_id": [], "reason": []})
    reproducibility = {
        "dataset": title,
        "domain": f"{dataset}_domain",
        "source_sampling_rate": 500,
        "target_sampling_rate": 500,
        "target_length": 5000,
        "source_unit": "mV",
        "target_unit": "mV",
        "unit_converted": False,
    }
    return DatasetAuditResult(
        dataset=dataset,
        audit=audit,
        split_manifest=manifest,
        exclusions=exclusions,
        reproducibility=reproducibility,
    )


def test_write_dataset_statistics_outputs_writes_expected_artifacts(tmp_path: Path) -> None:
    result = _fake_audit_result()

    written = write_dataset_statistics_outputs(result, tmp_path / "outputs")

    dataset_dir = tmp_path / "outputs" / "code15"
    assert (dataset_dir / "audit.json").is_file()
    assert (dataset_dir / "split_manifest.csv").is_file()
    assert (dataset_dir / "train.csv").is_file()
    assert (dataset_dir / "validation.csv").is_file()
    assert (dataset_dir / "test.csv").is_file()
    assert (dataset_dir / "exclusions.csv").is_file()
    assert (dataset_dir / "reproducibility.json").is_file()
    assert (dataset_dir / "label_distribution.csv").is_file()
    assert (dataset_dir / "positive_rate.csv").is_file()
    assert (dataset_dir / "split_label_distribution.csv").is_file()
    assert (dataset_dir / "split_positive_rate.csv").is_file()
    assert (dataset_dir / "summary.md").is_file()
    assert written["summary"].endswith("summary.md")

    summary = (dataset_dir / "summary.md").read_text()
    assert "CODE15" in summary
    assert "Records total" in summary
    assert "Positive rate" in summary

    distribution = pd.read_csv(dataset_dir / "label_distribution.csv")
    assert list(distribution.columns) == ["label", "positive_count"]
    positive_rate = pd.read_csv(dataset_dir / "positive_rate.csv")
    assert list(positive_rate.columns) == ["label", "positive_rate"]
    split_distribution = pd.read_csv(dataset_dir / "split_label_distribution.csv")
    assert list(split_distribution.columns) == ["split", "label", "positive_count"]
    split_rate = pd.read_csv(dataset_dir / "split_positive_rate.csv")
    assert list(split_rate.columns) == ["split", "label", "positive_rate"]
    assert pd.read_csv(dataset_dir / "train.csv").shape[0] == 1
    assert pd.read_csv(dataset_dir / "validation.csv").shape[0] == 1
    assert pd.read_csv(dataset_dir / "test.csv").shape[0] == 1


def test_batch_statistics_report_includes_all_datasets(tmp_path: Path) -> None:
    results = [_fake_audit_result("code15", "CODE15"), _fake_audit_result("sph", "SPH")]

    report_path = write_batch_dataset_statistics_report(results, tmp_path / "outputs")
    report = Path(report_path).read_text()

    assert "Dataset Statistics Report" in report
    assert "CODE15" in report
    assert "SPH" in report
    assert "| code15 |" in report
    assert "| sph |" in report

    rendered = render_batch_dataset_statistics_report(results)
    assert rendered.startswith("# Dataset Statistics Report")


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "export_dataset_statistics.py"
    spec = importlib.util.spec_from_file_location("export_dataset_statistics", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cli_exports_statistics_for_all_datasets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    script = _load_script_module()
    captured: list[str] = []
    waveform_limits: list[object] = []

    monkeypatch.setattr(
        script,
        "load_yaml",
        lambda path: {
            "root": str(tmp_path / Path(path).stem),
            "metadata_file": "metadata.csv",
            "records_root": "records",
        },
    )
    monkeypatch.setattr(
        script,
        "create_dataset",
        lambda name, root, config: SimpleNamespace(name=name.upper(), domain=f"{name}_domain", config=config),
    )
    monkeypatch.setattr(
        script,
        "audit_dataset",
        lambda dataset, waveform_check_limit=None: (
            waveform_limits.append(waveform_check_limit)
            or _fake_audit_result(dataset.name.lower(), dataset.name)
        ),
    )
    monkeypatch.setattr(
        script,
        "write_dataset_statistics_outputs",
        lambda result, output_dir: captured.append(result.dataset) or {"summary": str(Path(output_dir) / result.dataset / "summary.md")},
    )
    monkeypatch.setattr(
        script,
        "write_batch_dataset_statistics_report",
        lambda results, output_dir: str(Path(output_dir) / "report.md"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/export_dataset_statistics.py",
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
        ],
    )

    script.main()

    assert captured == ["ptbxl", "code15", "chapman", "sph"]
    assert waveform_limits == [None, None, None, None]
