"""Tests for ECGDataset base class behaviour."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from ecgshiftbench.datasets.base import DatasetInfo, ECGDataset


class _MinimalDataset(ECGDataset):
    """Minimal concrete dataset that works without real files."""

    LABELS = ["AF", "LBBB", "RBBB"]
    info = DatasetInfo(
        name="Minimal",
        num_leads=12,
        sampling_rate=500,
        duration_seconds=10.0,
        labels=LABELS,
    )

    def __init__(self, root: str, split: str = "train", **kwargs) -> None:
        super().__init__(root=root, split=split, **kwargs)

    def load_metadata(self) -> None:
        rng = np.random.default_rng(0)
        n = {"train": 30, "val": 10, "test": 10}[self.split]
        data = {"record_id": [str(i) for i in range(n)]}
        for lbl in self.LABELS:
            data[lbl] = rng.integers(0, 2, size=n).tolist()
        self._metadata = pd.DataFrame(data)
        self.info.num_samples = n

    def load_signal(self, index: int) -> np.ndarray:
        rng = np.random.default_rng(index)
        return rng.standard_normal((12, 5000)).astype(np.float32)


@pytest.fixture
def tmpdir_path(tmp_path):
    return str(tmp_path)


@pytest.fixture
def train_dataset(tmpdir_path):
    return _MinimalDataset(root=tmpdir_path, split="train")


class TestDatasetInfo:
    def test_fields(self):
        info = DatasetInfo(
            name="Test",
            num_leads=12,
            sampling_rate=500,
            duration_seconds=10.0,
            labels=["AF", "LBBB"],
        )
        assert info.name == "Test"
        assert info.num_leads == 12
        assert info.sampling_rate == 500
        assert info.labels == ["AF", "LBBB"]


class TestECGDatasetBase:
    def test_len(self, train_dataset):
        assert len(train_dataset) == 30

    def test_split_val(self, tmpdir_path):
        ds = _MinimalDataset(root=tmpdir_path, split="val")
        assert len(ds) == 10

    def test_split_test(self, tmpdir_path):
        ds = _MinimalDataset(root=tmpdir_path, split="test")
        assert len(ds) == 10

    def test_invalid_split(self, tmpdir_path):
        with pytest.raises(ValueError, match="split must be"):
            _MinimalDataset(root=tmpdir_path, split="invalid")

    def test_missing_root(self):
        with pytest.raises(FileNotFoundError):
            _MinimalDataset(root="/non/existent/path", split="train")

    def test_getitem_shapes(self, train_dataset):
        signal, labels = train_dataset[0]
        assert signal.shape == (12, 5000)
        assert labels.shape == (3,)

    def test_getitem_dtype(self, train_dataset):
        signal, labels = train_dataset[0]
        assert signal.dtype == np.float32
        assert labels.dtype == np.float32

    def test_active_labels_default(self, train_dataset):
        assert train_dataset.active_labels == _MinimalDataset.LABELS

    def test_set_labels(self, train_dataset):
        train_dataset.set_labels(["AF"])
        assert train_dataset.active_labels == ["AF"]
        _, labels = train_dataset[0]
        assert labels.shape == (1,)

    def test_set_labels_invalid(self, train_dataset):
        with pytest.raises(ValueError, match="Unknown labels"):
            train_dataset.set_labels(["UnknownLabel"])

    def test_get_labels_shape(self, train_dataset):
        labels = train_dataset.get_labels()
        assert labels.shape == (30, 3)

    def test_label_counts(self, train_dataset):
        counts = train_dataset.label_counts()
        assert set(counts.keys()) == set(_MinimalDataset.LABELS)
        for v in counts.values():
            assert 0 <= v <= 30

    def test_transform_applied(self, tmpdir_path):
        def double(x):
            return x * 2

        ds = _MinimalDataset(root=tmpdir_path, split="train", transform=double)
        signal_transformed, _ = ds[0]

        ds_no_transform = _MinimalDataset(root=tmpdir_path, split="train")
        signal_raw, _ = ds_no_transform[0]

        np.testing.assert_allclose(signal_transformed, signal_raw * 2)

    def test_repr(self, train_dataset):
        r = repr(train_dataset)
        assert "train" in r
        assert "30" in r

    def test_load_dataset_registry(self, tmpdir_path):
        """load_dataset raises ValueError for unknown dataset names."""
        from ecgshiftbench.datasets import load_dataset

        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset("nonexistent", root=tmpdir_path)
