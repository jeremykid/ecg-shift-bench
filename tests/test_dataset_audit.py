"""Tests for issue #7 dataset audit and split standardization."""

import subprocess
import sys
from pathlib import Path

import pandas as pd

from ecg_shift_bench.datasets.audit import (
    audit_dataset,
    build_split_manifest,
    write_alignment_audit_outputs,
)
from ecg_shift_bench.datasets.chapman import ChapmanDataset
from ecg_shift_bench.datasets.code15 import CODE15Dataset
from ecg_shift_bench.datasets.ptbxl import PTBXLDataset
from ecg_shift_bench.datasets.sph import SPHDataset


def test_audit_counts_records_patients_labels_and_prevalence(tmp_path: Path) -> None:
    metadata = pd.DataFrame(
        {
            "exam_id": [1, 2, 3],
            "patient_id": ["p1", "p2", "p2"],
            "trace_file": ["part.h5", "part.h5", "part.h5"],
            "AF": [1, 0, 1],
            "RBBB": [0, 1, 0],
            "LBBB": [0, 0, 0],
            "1dAVb": [0, 0, 1],
            "SB": [0, 0, 0],
            "ST": [1, 0, 0],
        }
    )
    metadata.to_csv(tmp_path / "exams.csv", index=False)
    dataset = CODE15Dataset(tmp_path)

    result = audit_dataset(dataset)

    assert result.audit["records_total"] == 3
    assert result.audit["records_usable"] == 3
    assert result.audit["patients_total"] == 2
    assert result.audit["missing_labels"] == {
        "AF": 0,
        "RBBB": 0,
        "LBBB": 0,
        "1dAVB": 0,
        "SB": 0,
        "ST": 0,
    }
    assert result.audit["positive_counts"]["AF"] == 2
    assert result.audit["positive_prevalence"]["AF"] == 2 / 3
    assert result.exclusions.empty


def test_ptbxl_official_split_manifest_uses_strat_fold(tmp_path: Path) -> None:
    dataset = PTBXLDataset(tmp_path)
    metadata = pd.DataFrame(
        {
            "ecg_id": [1, 2, 3],
            "patient_id": ["p1", "p2", "p3"],
            "scp_codes": ["{'AFIB': 0.0}", "{}", "{}"],
            "filename_hr": ["a", "b", "c"],
            "strat_fold": [1, 9, 10],
        }
    )

    manifest, policy = build_split_manifest(dataset, metadata)

    assert dict(zip(manifest["record_id"], manifest["split"], strict=True)) == {
        "1": "train",
        "2": "validation",
        "3": "test",
    }
    assert policy["split_source"] == "official"
    assert policy["leakage_check"]["status"] == "passed"


def test_patient_level_split_manifest_has_no_patient_overlap(tmp_path: Path) -> None:
    dataset = SPHDataset(tmp_path)
    metadata = pd.DataFrame(
        {
            "ECG_ID": [f"A{i:05d}" for i in range(12)],
            "Patient_ID": [f"S{i // 2:05d}" for i in range(12)],
            "AHA_Code": ["22"] * 12,
        }
    )

    manifest, policy = build_split_manifest(dataset, metadata)

    assert set(manifest["split"]) == {"train", "validation", "test"}
    assert policy["split_level"] == "patient"
    assert policy["leakage_check"]["status"] == "passed"


def test_chapman_uses_record_level_split_without_patient_leakage_check(tmp_path: Path) -> None:
    dataset = ChapmanDataset(tmp_path, {"metadata_file": "Diagnostics.csv"})
    metadata = pd.DataFrame(
        {
            "FileName": [f"MUSE_{index}" for index in range(10)],
            "Rhythm": ["AFIB"] * 10,
        }
    )

    manifest, policy = build_split_manifest(dataset, metadata)

    assert set(manifest["split"]) == {"train", "validation", "test"}
    assert "patient_id" not in manifest.columns
    assert policy["split_level"] == "record"
    assert policy["leakage_check"]["status"] == "unavailable"


def test_write_alignment_audit_outputs_creates_expected_files(tmp_path: Path) -> None:
    dataset = CODE15Dataset(tmp_path)
    pd.DataFrame(
        {
            "exam_id": [1],
            "patient_id": ["p1"],
            "trace_file": ["part.h5"],
            "AF": [0],
            "RBBB": [0],
            "LBBB": [0],
            "1dAVb": [0],
            "SB": [0],
            "ST": [1],
        }
    ).to_csv(tmp_path / "exams.csv", index=False)

    result = audit_dataset(dataset)
    written = write_alignment_audit_outputs(result, tmp_path / "outputs")

    assert (tmp_path / "outputs" / "code15" / "audit.json").is_file()
    assert (tmp_path / "outputs" / "code15" / "split_manifest.csv").is_file()
    assert (tmp_path / "outputs" / "code15" / "exclusions.csv").is_file()
    assert (tmp_path / "outputs" / "code15" / "reproducibility.json").is_file()
    assert set(written) == {"audit", "split_manifest", "exclusions", "reproducibility"}
    assert result.reproducibility["source_sampling_rate"] == 400
    assert result.reproducibility["target_sampling_rate"] == 500
    assert result.reproducibility["resampling_method"] == "polyphase_resample_signal"


def test_audit_alignment_cli_writes_expected_files(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "ECG_ID": ["A00001", "A00002"],
            "AHA_Code": ["22", "50"],
            "Patient_ID": ["S00001", "S00002"],
        }
    ).to_csv(tmp_path / "metadata.csv", index=False)
    output_dir = tmp_path / "outputs"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/audit_alignment.py",
            "--dataset",
            "sph",
            "--config",
            "configs/datasets/sph.yaml",
            "--root",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Dataset: SPH" in completed.stdout
    assert (output_dir / "sph" / "audit.json").is_file()
    assert (output_dir / "sph" / "split_manifest.csv").is_file()
    assert (output_dir / "sph" / "exclusions.csv").is_file()
    assert (output_dir / "sph" / "reproducibility.json").is_file()
