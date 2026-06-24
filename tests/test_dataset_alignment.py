"""Tests for the issue #7 ECG alignment contract and pending loaders."""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from ecg_shift_bench.datasets.alignment import align_ecg_signal
from ecg_shift_bench.datasets.chapman import ChapmanDataset
from ecg_shift_bench.datasets.sph import SPHDataset


def test_align_ecg_signal_resamples_sizes_and_normalizes() -> None:
    signal = np.tile(np.linspace(-1.0, 1.0, 4096, dtype=np.float32), (12, 1))

    aligned = align_ecg_signal(signal, source_rate=400, target_rate=500, target_length=5000)

    assert aligned.shape == (12, 5000)
    assert aligned.dtype == np.float32
    np.testing.assert_allclose(aligned.mean(axis=-1), 0.0, atol=1e-5)
    np.testing.assert_allclose(aligned.std(axis=-1), 1.0, atol=1e-5)


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
        },
    )

    signal = dataset.load_signal("MUSE_TEST")

    assert signal.shape == (12, 5)
    np.testing.assert_array_equal(signal[0], stored["I"].to_numpy(dtype=np.float32))
    assert dataset.get_labels("MUSE_TEST")["AF"] == 1


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
