"""Model inference and metric aggregation."""

from collections.abc import Iterable

import numpy as np
import torch
from torch import Tensor, nn

from ecg_shift_bench.evaluation.metrics import multilabel_metrics


def evaluate_model(
    model: nn.Module,
    batches: Iterable[tuple[Tensor, Tensor]],
    device: torch.device | str,
    label_names: list[str] | None = None,
) -> dict[str, float | dict[str, float]]:
    """Run inference and compute canonical metrics."""
    model.eval()
    targets: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, batch_targets in batches:
            logits = model(inputs.to(device))
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(batch_targets.cpu().numpy())
    if not targets:
        raise ValueError("Evaluation iterator produced no batches")
    return multilabel_metrics(np.concatenate(targets), np.concatenate(probabilities), label_names)
