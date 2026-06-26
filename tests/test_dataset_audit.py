"""Tests for dataset audit and split manifests."""

import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
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
    assert result.audit["waveform_check"] == {
        "checked_records": 0,
        "mode": "metadata_only",
        "target_sampling_rate": 500,
        "target_length": 5000,
        "lead_order": [
            "I",
            "II",
            "III",
            "aVR",
            "aVL",
            "aVF",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
        ],
        "source_unit": "mV",
        "target_unit": "mV",
        "unit_converted": False,
    }
    assert result.reproducibility["source_unit"] == "mV"
    assert result.reproducibility["target_unit"] == "mV"
    assert result.reproducibility["unit_converted"] is False
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
    metadata.to_csv(tmp_path / "metadata.csv", index=False)

    manifest, policy = build_split_manifest(dataset, metadata)

    assert set(manifest["split"]) == {"train", "validation", "test"}
    assert policy["split_level"] == "patient"
    assert policy["leakage_check"]["status"] == "passed"


def test_patient_level_split_prefers_label_aware_strategy_when_possible(tmp_path: Path) -> None:
    dataset = CODE15Dataset(tmp_path)
    rows = []
    labels = [
        {"AF": 1, "RBBB": 0, "LBBB": 0, "1dAVb": 0, "SB": 0, "ST": 0},
        {"AF": 0, "RBBB": 1, "LBBB": 0, "1dAVb": 0, "SB": 0, "ST": 0},
    ]
    for index in range(30):
        patient_index = index
        label_block = labels[index % len(labels)]
        rows.append(
            {
                "exam_id": index + 1,
                "patient_id": f"p{patient_index:02d}",
                "trace_file": "part.h5",
                **label_block,
            }
        )
    metadata = pd.DataFrame(rows)
    metadata.to_csv(tmp_path / "exams.csv", index=False)

    manifest, policy = build_split_manifest(dataset, metadata)

    assert policy["split_level"] == "patient"
    assert policy["split_algorithm"] == "patient_level_stratified_signature"
    assert set(manifest["split"]) == {"train", "validation", "test"}
    assert policy["leakage_check"]["status"] == "passed"
    assert manifest.groupby("split")["patient_id"].nunique().sum() == 30


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
    assert result.reproducibility["source_unit"] == "mV"
    assert result.reproducibility["target_unit"] == "mV"


def test_waveform_audit_validates_aligned_contract(tmp_path: Path) -> None:
    records = tmp_path / "ECGData" / "ECGData"
    records.mkdir(parents=True)
    leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    stored = pd.DataFrame(
        np.tile(np.linspace(-1.0, 1.0, 4000, dtype=np.float32), (12, 1)).T,
        columns=leads,
    )
    stored.to_csv(records / "MUSE_TEST.csv", index=False)
    pd.DataFrame({"FileName": ["MUSE_TEST"], "Rhythm": ["AFIB"], "Beat": ["RBBB"]}).to_csv(
        tmp_path / "Diagnostics.csv", index=False
    )
    dataset = ChapmanDataset(
        tmp_path,
        {
            "metadata_file": "Diagnostics.csv",
            "records_root": "ECGData",
            "record_id_column": "FileName",
            "label_column": "Rhythm",
            "beat_label_column": "Beat",
            "sampling_rate": 500,
            "target_sampling_rate": 500,
            "target_length": 5000,
            "source_unit": "uV",
            "target_unit": "mV",
        },
    )

    result = audit_dataset(dataset, waveform_check_limit=1)

    assert result.audit["records_usable"] == 1
    assert result.audit["waveform_check"] == {
        "checked_records": 1,
        "mode": "sampled",
        "target_sampling_rate": 500,
        "target_length": 5000,
        "lead_order": [
            "I",
            "II",
            "III",
            "aVR",
            "aVL",
            "aVF",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
        ],
        "source_unit": "uV",
        "target_unit": "mV",
        "unit_converted": True,
    }
    assert result.exclusions.empty


def test_full_waveform_audit_records_corrupted_records(tmp_path: Path) -> None:
    records = tmp_path / "records"
    records.mkdir(parents=True)
    with h5py.File(records / "A00001.h5", "w") as handle:
        handle.create_dataset("not_ecg", data=np.zeros((12, 5), dtype=np.float32))
    pd.DataFrame({"ECG_ID": ["A00001"], "Patient_ID": ["S00001"], "AHA_Code": ["22"]}).to_csv(
        tmp_path / "metadata.csv",
        index=False,
    )
    dataset = SPHDataset(
        tmp_path,
        {
            "metadata_file": "metadata.csv",
            "records_root": "records",
            "record_id_column": "ECG_ID",
            "patient_id_column": "Patient_ID",
            "label_column": "AHA_Code",
        },
    )

    result = audit_dataset(dataset, waveform_check_limit=None)

    assert result.audit["waveform_check"]["mode"] == "full"
    assert result.audit["records_usable"] == 0
    assert result.audit["records_excluded"] == 1
    assert not result.exclusions.empty
    assert "missing dataset" in result.exclusions.iloc[0]["reason"].lower()


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
