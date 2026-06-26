"""Tests for ECG alignment helpers and dataset loaders."""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from ecg_shift_bench.datasets.alignment import align_ecg_signal, convert_signal_units
from ecg_shift_bench.datasets.base import BaseECGDataset
from ecg_shift_bench.datasets.chapman import ChapmanDataset
from ecg_shift_bench.datasets.sph import SPHDataset


class TinyDataset(BaseECGDataset):
    name = "TINY"
    domain = "tiny_domain"

    def load_metadata(self) -> pd.DataFrame:
        return pd.DataFrame({"record_id": ["r1"], "patient_id": ["p1"]})

    def load_signal(self, record_id: str) -> np.ndarray:
        if record_id != "r1":
            raise KeyError(record_id)
        return np.tile(np.linspace(-2.0, 2.0, 4096, dtype=np.float32), (12, 1))

    def get_labels(self, record_id: str) -> dict[str, int]:
        if record_id != "r1":
            raise KeyError(record_id)
        return {"AF": 1, "RBBB": 0, "LBBB": 0, "1dAVB": 0, "SB": 0, "ST": 0}


def test_align_ecg_signal_resamples_sizes_and_preserves_units() -> None:
    signal = np.tile(np.linspace(-1.0, 1.0, 4096, dtype=np.float32), (12, 1))

    aligned = align_ecg_signal(
        signal,
        source_rate=400,
        target_rate=500,
        target_length=5000,
        source_unit="mV",
        target_unit="mV",
    )

    assert aligned.shape == (12, 5000)
    assert aligned.dtype == np.float32
    assert np.isfinite(aligned).all()


def test_convert_signal_units_scales_microvolt_to_millivolt() -> None:
    signal = np.full((12, 8), 2500.0, dtype=np.float32)

    converted = convert_signal_units(signal, source_unit="uV", target_unit="mV")

    np.testing.assert_allclose(converted, signal * 1e-3)
    assert converted.dtype == np.float32


def test_align_ecg_signal_converts_microvolt_inputs_to_millivolts() -> None:
    signal = np.full((12, 5000), 2500.0, dtype=np.float32)

    aligned = align_ecg_signal(
        signal,
        source_rate=500,
        target_rate=500,
        target_length=5000,
        source_unit="uV",
        target_unit="mV",
    )

    np.testing.assert_allclose(aligned, 2.5)
    assert aligned.dtype == np.float32


def test_dataset_load_aligned_signal_uses_shared_contract() -> None:
    dataset = TinyDataset(
        ".",
        {
            "sampling_rate": 400,
            "target_sampling_rate": 500,
            "target_length": 5000,
            "record_id_column": "record_id",
            "patient_id_column": "patient_id",
            "source_unit": "mV",
            "target_unit": "mV",
        },
    )

    aligned = dataset.load_aligned_signal("r1")
    sample = dataset.load_aligned_sample("r1")

    assert aligned.shape == (12, 5000)
    assert aligned.dtype == np.float32
    np.testing.assert_array_equal(sample.signal, aligned)
    assert sample.record_id == "r1"
    assert sample.patient_id == "p1"
    assert sample.sampling_rate == 500
    assert sample.source_sampling_rate == 400
    assert sample.source_length == 4096
    assert sample.labels["AF"] == 1
    assert sample.meta["source_unit"] == "mV"
    assert sample.meta["target_unit"] == "mV"
    assert sample.meta["unit_converted"] is False


def test_chapman_loader_reads_csv_waveform_as_lead_first(tmp_path: Path) -> None:
    records = tmp_path / "ECGData" / "ECGData"
    records.mkdir(parents=True)
    leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    stored = pd.DataFrame(np.arange(60, dtype=np.float32).reshape(5, 12), columns=leads)
    stored.to_csv(records / "MUSE_TEST.csv", index=False)
    pd.DataFrame({"FileName": ["MUSE_TEST"], "Rhythm": ["AFIB"]}).to_csv(
        tmp_path / "Diagnostics.csv", index=False
    )

    dataset = ChapmanDataset(
        tmp_path,
        {
            "metadata_file": "Diagnostics.csv",
            "records_root": "ECGData",
            "record_id_column": "FileName",
            "label_column": "Rhythm",
            "source_unit": "uV",
            "target_unit": "mV",
        },
    )

    signal = dataset.load_signal("MUSE_TEST")

    assert signal.shape == (12, 5)
    np.testing.assert_array_equal(signal[0], stored["I"].to_numpy(dtype=np.float32))
    assert dataset.get_labels("MUSE_TEST")["AF"] == 1


def test_chapman_labels_combine_rhythm_and_beat_columns(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "FileName": ["MUSE_TEST"],
            "Rhythm": ["AFIB"],
            "Beat": ["RBBB LBBB 1AVB"],
        }
    ).to_csv(tmp_path / "Diagnostics.csv", index=False)

    dataset = ChapmanDataset(
        tmp_path,
        {
            "metadata_file": "Diagnostics.csv",
            "record_id_column": "FileName",
            "label_column": "Rhythm",
            "beat_label_column": "Beat",
        },
    )

    assert dataset.get_labels("MUSE_TEST") == {
        "AF": 1,
        "RBBB": 1,
        "LBBB": 1,
        "1dAVB": 1,
        "SB": 0,
        "ST": 0,
    }


def test_sph_loader_reads_hdf5_waveform_and_metadata_columns(tmp_path: Path) -> None:
    records = tmp_path / "records" / "records"
    records.mkdir(parents=True)
    with h5py.File(records / "A00001.h5", "w") as handle:
        handle.create_dataset("ecg", data=np.arange(60, dtype=np.float32).reshape(12, 5))
    pd.DataFrame(
        {
            "ECG_ID": ["A00001"],
            "AHA_Code": ["50+347;82"],
            "Patient_ID": ["S00001"],
            "N": [5],
        }
    ).to_csv(tmp_path / "metadata.csv", index=False)

    dataset = SPHDataset(tmp_path)
    signal = dataset.load_signal("A00001")

    assert signal.shape == (12, 5)
    assert signal.dtype == np.float32
    assert dataset.get_labels("A00001")["AF"] == 1
    assert dataset.get_labels("A00001")["1dAVB"] == 1
