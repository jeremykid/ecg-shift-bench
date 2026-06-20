"""Minimal source-only supervised baseline."""

import torch
from torch import nn


class SourceOnly:
    """Optimize multi-label BCE using labeled source batches only."""

    def __init__(self, model: nn.Module, pos_weight: torch.Tensor | None = None) -> None:
        self.model = model
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def training_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute supervised source loss."""
        return self.criterion(self.model(inputs), targets.float())
