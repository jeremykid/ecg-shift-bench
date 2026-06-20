"""Loss factory for benchmark methods."""

import torch
from torch import nn


def multilabel_bce(pos_weight: torch.Tensor | None = None) -> nn.BCEWithLogitsLoss:
    """Construct the canonical multi-label binary cross-entropy loss."""
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
