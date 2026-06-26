"""Data-free tests for dataset adapters and preprocessing."""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from ecg_shift_bench.datasets.base import BaseECGDataset
from ecg_shift_bench.datasets.alignment import convert_signal_units
from ecg_shift_bench.datasets.code15 import CODE15Dataset
from ecg_shift_bench.datasets.ptbxl import CANONICAL_LEAD_ORDER, PTBXLDataset
from ecg_shift_bench.datasets.registry import DATASET_REGISTRY, create_dataset
from ecg_shift_bench.preprocessing.signal import crop_or_pad, resample_signal


def test_registered_datasets_implement_base_interface(tmp_path: Path) -> None:
    assert set(DATASET_REGISTRY) == {"ptbxl", "chapman", "sph", "code15"}
    for name in DATASET_REGISTRY:
        dataset = create_dataset(name, tmp_path)
        assert isinstance(dataset, BaseECGDataset)
        assert dataset.name
        assert dataset.domain
        with pytest.raises(FileNotFoundError):
            dataset.load_metadata()


def test_crop_pad_and_resample() -> None:
    signal = np.vstack([np.arange(10), np.arange(10) * 2]).astype(np.float32)
    cropped = crop_or_pad(signal, 6)
    padded = crop_or_pad(cropped, 10)
    downsampled = resample_signal(signal, source_rate=10, target_rate=5)
    assert cropped.shape == (2, 6)
    assert padded.shape == (2, 10)
    assert downsampled.shape == (2, 5)


def test_convert_signal_units_preserves_millivolt_inputs() -> None:
    signal = np.arange(12, dtype=np.float32).reshape(3, 4)
    converted = convert_signal_units(signal, source_unit="mV", target_unit="mV")
    np.testing.assert_array_equal(converted, signal)
    assert converted.dtype == np.float32


def test_code15_reads_boolean_labels_and_hdf5_signal(tmp_path: Path) -> None:
    metadata = pd.DataFrame(
        {
            "exam_id": [101, 102],
            "patient_id": [11, 12],
            "trace_file": ["exams_part0.hdf5", "exams_part1.hdf5"],
            "AF": [True, False],
            "RBBB": [False, True],
            "LBBB": [False, False],
            "1dAVb": [False, False],
            "SB": [False, False],
            "ST": [True, False],
        }
    )
    metadata.to_csv(tmp_path / "exams.csv", index=False)
    stored = np.arange(48, dtype=np.float32).reshape(4, 12)
    with h5py.File(tmp_path / "exams_part0.hdf5", "w") as handle:
        handle.create_dataset("exam_id", data=np.array([101, 0]))
        handle.create_dataset("tracings", data=np.stack([stored, np.zeros_like(stored)]))

    dataset = CODE15Dataset(tmp_path)
    assert dataset.get_labels("101") == {
        "AF": 1,
        "RBBB": 0,
        "LBBB": 0,
        "1dAVB": 0,
        "SB": 0,
        "ST": 1,
    }
    np.testing.assert_array_equal(dataset.load_signal("101"), stored.T)
    with pytest.raises(FileNotFoundError, match="exams_part1.hdf5"):
        dataset.load_signal("102")


def test_ptbxl_reorders_wfdb_signal_and_checks_rate(tmp_path: Path, monkeypatch) -> None:
    dataset = PTBXLDataset(tmp_path)
    dataset._metadata = pd.DataFrame(
        {
            "ecg_id": [1],
            "patient_id": [10],
            "scp_codes": ["{'AFIB': 0.0}"],
            "filename_hr": ["records500/00000/00001_hr"],
        }
    )
    reversed_leads = list(reversed(CANONICAL_LEAD_ORDER))
    stored = np.arange(120, dtype=np.float32).reshape(10, 12)
    monkeypatch.setattr(
        "ecg_shift_bench.datasets.ptbxl.wfdb.rdsamp",
        lambda path: (stored, {"sig_name": reversed_leads, "fs": 500}),
    )

    signal = dataset.load_signal("1")
    assert signal.shape == (12, 10)
    np.testing.assert_array_equal(signal[0], stored[:, -1])
    assert dataset.get_labels("1")["AF"] == 1
