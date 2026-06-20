"""Domain Generalisation benchmark protocol.

In this setting a model is trained on *multiple* source datasets
simultaneously, and then evaluated zero-shot on a held-out target dataset.
This is the standard Leave-One-Dataset-Out (LODO) evaluation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from torch.utils.data import ConcatDataset, DataLoader

from ecgshiftbench.benchmarks.base import BenchmarkProtocol, BenchmarkResult
from ecgshiftbench.datasets.base import ECGDataset
from ecgshiftbench.metrics.classification import compute_metrics


class DomainGeneralizationBenchmark(BenchmarkProtocol):
    """Leave-One-Dataset-Out domain generalisation evaluation.

    All datasets are provided up-front.  For each dataset *d*, the model
    is evaluated on *d* after being trained on all other datasets.

    Because ECGShiftBench benchmarks are evaluation-only tools
    (the user provides a pre-trained model), the "training on sources"
    is performed externally.  This class handles the evaluation loop.

    Example usage::

        benchmark = DomainGeneralizationBenchmark(
            datasets={
                "ptbxl": ptbxl_test,
                "chapman": chapman_test,
                "cpsc2018": cpsc_test,
                "g12ec": g12ec_test,
            },
            labels=["AF", "LBBB", "RBBB"],
        )
        # model_for is a callable that returns the model trained on all
        # datasets except the one with the given name.
        results = benchmark.evaluate_lodo(model_for)

    Args:
        datasets: Mapping of dataset name → test :class:`ECGDataset`.
        labels: Shared label names.
        seed: Random seed.
    """

    name = "domain_generalization"

    def __init__(
        self,
        datasets: Dict[str, ECGDataset],
        labels: Optional[List[str]] = None,
        seed: int = 42,
    ) -> None:
        super().__init__(labels=labels, seed=seed)
        self._datasets = datasets

    def get_source_datasets(self) -> List[ECGDataset]:
        return list(self._datasets.values())

    def get_target_datasets(self) -> List[ECGDataset]:
        return list(self._datasets.values())

    def evaluate(self, model: Any) -> List[BenchmarkResult]:
        """Evaluate a single *model* on every dataset in the collection.

        Use this when a single model has been trained on all-but-one
        dataset and you want to evaluate on the held-out dataset.

        Args:
            model: Trained model callable.

        Returns:
            One :class:`~ecgshiftbench.benchmarks.base.BenchmarkResult`
            per dataset.
        """
        results: List[BenchmarkResult] = []
        for name, dataset in self._datasets.items():
            preds, targets_arr = self._collect_predictions(model, dataset)
            metrics = compute_metrics(targets_arr, preds)
            results.append(
                BenchmarkResult(
                    protocol=self.name,
                    source_dataset="multi-source",
                    target_dataset=name,
                    metrics=metrics,
                )
            )
        return results

    def evaluate_lodo(self, model_for) -> List[BenchmarkResult]:
        """Full Leave-One-Dataset-Out evaluation loop.

        Args:
            model_for: Callable ``model_for(target_name) → model`` that
                returns the model trained on all datasets except
                *target_name*.

        Returns:
            One :class:`~ecgshiftbench.benchmarks.base.BenchmarkResult`
            per dataset (the dataset used as the held-out target).
        """
        results: List[BenchmarkResult] = []
        for target_name, target_dataset in self._datasets.items():
            model = model_for(target_name)
            preds, targets_arr = self._collect_predictions(model, target_dataset)
            metrics = compute_metrics(targets_arr, preds)
            source_names = [n for n in self._datasets if n != target_name]
            results.append(
                BenchmarkResult(
                    protocol=self.name,
                    source_dataset="+".join(source_names),
                    target_dataset=target_name,
                    metrics=metrics,
                )
            )
        return results
