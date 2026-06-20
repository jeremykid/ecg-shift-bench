"""Base classes for ECGShiftBench benchmark protocols."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ecgshiftbench.datasets.base import ECGDataset


@dataclass
class BenchmarkResult:
    """Container for benchmark evaluation results.

    Attributes:
        protocol: Name of the benchmark protocol (e.g. ``'covariate_shift'``).
        source_dataset: Name of the source / training dataset.
        target_dataset: Name of the target / evaluation dataset.
        metrics: Dictionary mapping metric names to scalar values.
        extra: Optional dictionary for additional metadata.
    """

    protocol: str
    source_dataset: str
    target_dataset: str
    metrics: Dict[str, float] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in self.metrics.items())
        return (
            f"BenchmarkResult("
            f"protocol={self.protocol!r}, "
            f"{self.source_dataset!r} → {self.target_dataset!r}, "
            f"{metrics_str})"
        )


class BenchmarkProtocol(ABC):
    """Abstract base class for ECGShiftBench benchmark protocols.

    Each subclass implements a specific evaluation scenario and exposes a
    common :meth:`evaluate` interface that accepts a trained model and
    returns a :class:`BenchmarkResult`.
    """

    #: Short identifier for this protocol.
    name: str = "base"

    def __init__(
        self,
        labels: Optional[List[str]] = None,
        seed: int = 42,
    ) -> None:
        """Initialise the benchmark.

        Args:
            labels: Shared label names to use across all datasets.  When
                ``None``, each dataset uses its own default labels.
            seed: Random seed for reproducible splits.
        """
        self.labels = labels
        self.seed = seed

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def get_source_datasets(self) -> List[ECGDataset]:
        """Return the list of datasets used for training."""

    @abstractmethod
    def get_target_datasets(self) -> List[ECGDataset]:
        """Return the list of datasets used for evaluation."""

    @abstractmethod
    def evaluate(self, model: Any) -> List[BenchmarkResult]:
        """Evaluate *model* under this benchmark protocol.

        Args:
            model: A callable ``model(x) → logits`` or any object with a
                ``predict`` method returning per-sample label probabilities.

        Returns:
            List of :class:`BenchmarkResult` objects, one per
            source–target dataset pair.
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_predictions(model: Any, dataset: ECGDataset) -> tuple[np.ndarray, np.ndarray]:
        """Run *model* on every sample in *dataset* and collect outputs.

        Returns:
            Tuple of ``(predictions, targets)`` where each is a float32
            array of shape ``(N, num_labels)``.
        """
        import torch
        from torch.utils.data import DataLoader

        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
        all_preds: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []

        device = next(model.parameters()).device if hasattr(model, "parameters") else "cpu"

        with torch.no_grad():
            for signals, targets in loader:
                signals = signals.to(device)
                outputs = model(signals)
                if isinstance(outputs, torch.Tensor):
                    probs = torch.sigmoid(outputs).cpu().numpy()
                else:
                    probs = np.array(outputs)
                all_preds.append(probs)
                all_targets.append(targets.numpy())

        predictions = np.concatenate(all_preds, axis=0)
        targets_arr = np.concatenate(all_targets, axis=0)
        return predictions, targets_arr
