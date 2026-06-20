"""Tests for benchmark protocol classes (no real data required)."""

from __future__ import annotations

from typing import Any, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ecgshiftbench.benchmarks.base import BenchmarkResult
from ecgshiftbench.benchmarks.covariate_shift import CovariateShiftBenchmark
from ecgshiftbench.benchmarks.domain_generalization import DomainGeneralizationBenchmark
from ecgshiftbench.benchmarks.uda import UDABenchmark
from ecgshiftbench.datasets.base import DatasetInfo, ECGDataset


# ---------------------------------------------------------------------------
# Minimal stub dataset for testing (no files on disk needed)
# ---------------------------------------------------------------------------

class _StubDataset(ECGDataset):
    """In-memory stub dataset returning random signals and labels."""

    _LABELS = ["AF", "LBBB", "RBBB"]

    info = DatasetInfo(
        name="Stub",
        num_leads=12,
        sampling_rate=500,
        duration_seconds=10.0,
        labels=_LABELS,
        num_samples=20,
    )

    def __init__(self, name: str = "Stub", n: int = 20, split: str = "train") -> None:
        self._name = name
        self._n = n
        self.info = DatasetInfo(
            name=name,
            num_leads=12,
            sampling_rate=500,
            duration_seconds=10.0,
            labels=self._LABELS,
            num_samples=n,
        )
        # Skip the parent __init__ that checks for a real directory
        self.root = "/non/existent"
        self.split = split
        self.transform = None
        self.target_transform = None
        self._active_labels = None
        self._metadata = None
        self.load_metadata()

    def load_metadata(self) -> None:
        import pandas as pd
        rng = np.random.default_rng(42)
        data = {
            "record_id": [str(i) for i in range(self._n)],
        }
        for lbl in self._LABELS:
            data[lbl] = rng.integers(0, 2, size=self._n).tolist()
        self._metadata = pd.DataFrame(data)

    def load_signal(self, index: int) -> np.ndarray:
        rng = np.random.default_rng(index)
        return rng.standard_normal((12, 5000)).astype(np.float32)


@pytest.fixture
def stub_source():
    return _StubDataset("Source", n=30)


@pytest.fixture
def stub_targets():
    return [_StubDataset("Target1", n=20), _StubDataset("Target2", n=20)]


@pytest.fixture
def stub_model():
    """Tiny stub model that returns constant predictions."""
    import torch
    import torch.nn as nn

    class _ConstantModel(nn.Module):
        def __init__(self, num_classes: int = 3):
            super().__init__()
            self._fc = nn.Linear(1, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch = x.shape[0]
            return torch.zeros(batch, 3)

    return _ConstantModel()


class TestBenchmarkResult:
    def test_str(self):
        r = BenchmarkResult(
            protocol="covariate_shift",
            source_dataset="PTB-XL",
            target_dataset="Chapman",
            metrics={"macro_auroc": 0.75, "macro_f1": 0.60},
        )
        s = str(r)
        assert "covariate_shift" in s
        assert "PTB-XL" in s
        assert "Chapman" in s

    def test_defaults(self):
        r = BenchmarkResult(
            protocol="test",
            source_dataset="A",
            target_dataset="B",
        )
        assert r.metrics == {}
        assert r.extra == {}


class TestCovariateShiftBenchmark:
    def test_get_source(self, stub_source, stub_targets):
        bench = CovariateShiftBenchmark(source=stub_source, targets=stub_targets)
        sources = bench.get_source_datasets()
        assert len(sources) == 1
        assert sources[0] is stub_source

    def test_get_targets(self, stub_source, stub_targets):
        bench = CovariateShiftBenchmark(source=stub_source, targets=stub_targets)
        assert bench.get_target_datasets() == stub_targets

    def test_evaluate_returns_correct_count(self, stub_source, stub_targets, stub_model):
        bench = CovariateShiftBenchmark(source=stub_source, targets=stub_targets)
        results = bench.evaluate(stub_model)
        assert len(results) == len(stub_targets)

    def test_evaluate_result_protocol(self, stub_source, stub_targets, stub_model):
        bench = CovariateShiftBenchmark(source=stub_source, targets=stub_targets)
        results = bench.evaluate(stub_model)
        for r in results:
            assert r.protocol == "covariate_shift"
            assert r.source_dataset == stub_source.info.name

    def test_evaluate_metrics_keys(self, stub_source, stub_targets, stub_model):
        bench = CovariateShiftBenchmark(source=stub_source, targets=stub_targets)
        results = bench.evaluate(stub_model)
        for r in results:
            assert "macro_auroc" in r.metrics or np.isnan(r.metrics.get("macro_auroc", 0))


class TestDomainGeneralizationBenchmark:
    def test_evaluate_all_datasets(self, stub_targets, stub_model):
        datasets = {d.info.name: d for d in stub_targets}
        bench = DomainGeneralizationBenchmark(datasets=datasets)
        results = bench.evaluate(stub_model)
        assert len(results) == len(datasets)

    def test_evaluate_protocol_name(self, stub_targets, stub_model):
        datasets = {d.info.name: d for d in stub_targets}
        bench = DomainGeneralizationBenchmark(datasets=datasets)
        results = bench.evaluate(stub_model)
        for r in results:
            assert r.protocol == "domain_generalization"

    def test_evaluate_lodo(self, stub_targets, stub_model):
        datasets = {d.info.name: d for d in stub_targets}
        bench = DomainGeneralizationBenchmark(datasets=datasets)
        results = bench.evaluate_lodo(model_for=lambda name: stub_model)
        assert len(results) == len(datasets)
        for r in results:
            assert r.target_dataset in datasets


class TestUDABenchmark:
    def test_evaluate_returns_one_result(self, stub_source, stub_targets, stub_model):
        bench = UDABenchmark(
            source=stub_source,
            target_unlabelled=stub_targets[0],
            target_test=stub_targets[1],
        )
        results = bench.evaluate(stub_model)
        assert len(results) == 1

    def test_evaluate_protocol_name(self, stub_source, stub_targets, stub_model):
        bench = UDABenchmark(
            source=stub_source,
            target_unlabelled=stub_targets[0],
            target_test=stub_targets[1],
        )
        results = bench.evaluate(stub_model)
        assert results[0].protocol == "uda"

    def test_target_unlabelled_property(self, stub_source, stub_targets):
        bench = UDABenchmark(
            source=stub_source,
            target_unlabelled=stub_targets[0],
            target_test=stub_targets[1],
        )
        assert bench.target_unlabelled is stub_targets[0]
