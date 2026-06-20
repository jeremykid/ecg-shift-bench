"""Covariate Shift benchmark protocol.

In this setting a model is trained on a *single* source dataset and then
evaluated — without any adaptation — on the test splits of one or more
target datasets.  This measures how much performance degrades due to
covariate shift across institutions, countries, or device manufacturers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ecgshiftbench.benchmarks.base import BenchmarkProtocol, BenchmarkResult
from ecgshiftbench.datasets.base import ECGDataset
from ecgshiftbench.metrics.classification import compute_metrics


class CovariateShiftBenchmark(BenchmarkProtocol):
    """Evaluate covariate shift by testing a model trained on one dataset
    on the test splits of other datasets.

    Example usage::

        benchmark = CovariateShiftBenchmark(
            source=ptbxl_train,
            targets=[chapman_test, cpsc_test, g12ec_test],
            labels=["AF", "LBBB", "RBBB"],
        )
        results = benchmark.evaluate(model)
        for r in results:
            print(r)

    Args:
        source: Training dataset (source domain).
        targets: List of test datasets (target domains).
        labels: Shared label names.  Defaults to labels from *source*.
        seed: Random seed.
    """

    name = "covariate_shift"

    def __init__(
        self,
        source: ECGDataset,
        targets: List[ECGDataset],
        labels: Optional[List[str]] = None,
        seed: int = 42,
    ) -> None:
        super().__init__(labels=labels, seed=seed)
        self._source = source
        self._targets = targets

    def get_source_datasets(self) -> List[ECGDataset]:
        return [self._source]

    def get_target_datasets(self) -> List[ECGDataset]:
        return self._targets

    def evaluate(self, model: Any) -> List[BenchmarkResult]:
        """Evaluate *model* on all target datasets.

        Args:
            model: Trained PyTorch model (or any callable that accepts a
                batch tensor and returns logits).

        Returns:
            One :class:`~ecgshiftbench.benchmarks.base.BenchmarkResult`
            per target dataset.
        """
        results: List[BenchmarkResult] = []
        for target in self._targets:
            preds, targets_arr = self._collect_predictions(model, target)
            metrics = compute_metrics(targets_arr, preds)
            result = BenchmarkResult(
                protocol=self.name,
                source_dataset=self._source.info.name,
                target_dataset=target.info.name,
                metrics=metrics,
            )
            results.append(result)
        return results
