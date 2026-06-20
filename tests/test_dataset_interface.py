"""Data-free tests for dataset adapters and preprocessing."""

from pathlib import Path

import numpy as np
import pytest

from ecg_shift_bench.datasets.base import BaseECGDataset
from ecg_shift_bench.datasets.registry import DATASET_REGISTRY, create_dataset
from ecg_shift_bench.preprocessing.signal import crop_or_pad, per_lead_zscore, resample_signal


def test_registered_datasets_implement_base_interface(tmp_path: Path) -> None:
    assert set(DATASET_REGISTRY) == {"ptbxl", "chapman", "sph", "code15"}
    for name in DATASET_REGISTRY:
        dataset = create_dataset(name, tmp_path)
        assert isinstance(dataset, BaseECGDataset)
        assert dataset.name
        assert dataset.domain
        with pytest.raises(FileNotFoundError):
            dataset.load_metadata()


def test_crop_pad_normalize_and_resample() -> None:
    signal = np.vstack([np.arange(10), np.arange(10) * 2]).astype(np.float32)
    cropped = crop_or_pad(signal, 6)
    padded = crop_or_pad(cropped, 10)
    normalized = per_lead_zscore(signal)
    downsampled = resample_signal(signal, source_rate=10, target_rate=5)
    assert cropped.shape == (2, 6)
    assert padded.shape == (2, 10)
    np.testing.assert_allclose(normalized.mean(axis=1), 0.0, atol=1e-6)
    np.testing.assert_allclose(normalized.std(axis=1), 1.0, atol=1e-6)
    assert downsampled.shape == (2, 5)


def test_constant_lead_normalizes_to_zero() -> None:
    normalized = per_lead_zscore(np.ones((12, 20), dtype=np.float32))
    assert np.count_nonzero(normalized) == 0
