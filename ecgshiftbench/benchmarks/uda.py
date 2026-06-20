"""Unsupervised Domain Adaptation (UDA) benchmark protocol.

In this setting a model may use *unlabelled* target-domain data during
training (e.g. for feature alignment or pseudo-labelling).  The evaluation
uses the same labelled test split as in the covariate-shift setting.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ecgshiftbench.benchmarks.base import BenchmarkProtocol, BenchmarkResult
from ecgshiftbench.datasets.base import ECGDataset
from ecgshiftbench.metrics.classification import compute_metrics


class UDABenchmark(BenchmarkProtocol):
    """Evaluate unsupervised domain adaptation from a source to a target dataset.

    The benchmark accepts a *pre-adapted* model — i.e. a model that has
    already been fine-tuned or aligned using the unlabelled target training
    split — and measures performance on the labelled target test split.

    Example usage::

        benchmark = UDABenchmark(
            source=ptbxl_train,
            target_unlabelled=chapman_train_unlabelled,
            target_test=chapman_test,
            labels=["AF", "LBBB", "RBBB"],
        )
        results = benchmark.evaluate(adapted_model)

    Args:
        source: Labelled source training dataset.
        target_unlabelled: Unlabelled target training dataset (used
            externally by the adaptation algorithm).
        target_test: Labelled target test dataset used for evaluation.
        labels: Shared label names.
        seed: Random seed.
    """

    name = "uda"

    def __init__(
        self,
        source: ECGDataset,
        target_unlabelled: ECGDataset,
        target_test: ECGDataset,
        labels: Optional[List[str]] = None,
        seed: int = 42,
    ) -> None:
        super().__init__(labels=labels, seed=seed)
        self._source = source
        self._target_unlabelled = target_unlabelled
        self._target_test = target_test

    def get_source_datasets(self) -> List[ECGDataset]:
        return [self._source]

    def get_target_datasets(self) -> List[ECGDataset]:
        return [self._target_test]

    @property
    def target_unlabelled(self) -> ECGDataset:
        """Return the unlabelled target dataset for adaptation."""
        return self._target_unlabelled

    def evaluate(self, model: Any) -> List[BenchmarkResult]:
        """Evaluate the adapted *model* on the target test split.

        Args:
            model: Adapted model callable.

        Returns:
            A list containing a single
            :class:`~ecgshiftbench.benchmarks.base.BenchmarkResult`.
        """
        preds, targets_arr = self._collect_predictions(model, self._target_test)
        metrics = compute_metrics(targets_arr, preds)
        return [
            BenchmarkResult(
                protocol=self.name,
                source_dataset=self._source.info.name,
                target_dataset=self._target_test.info.name,
                metrics=metrics,
                extra={"unlabelled_target": self._target_unlabelled.info.name},
            )
        ]
