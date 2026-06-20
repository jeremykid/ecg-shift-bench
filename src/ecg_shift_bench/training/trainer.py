"""Small, explicit source-only training loop."""

from collections.abc import Iterable

import torch
from torch import Tensor, nn
from torch.optim import Optimizer


def train_one_epoch(
    model: nn.Module,
    batches: Iterable[tuple[Tensor, Tensor]],
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device | str,
) -> float:
    """Train for one epoch and return sample-weighted mean loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    for inputs, targets in batches:
        inputs = inputs.to(device)
        targets = targets.to(device).float()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()
        batch_size = inputs.shape[0]
        total_loss += float(loss.detach()) * batch_size
        total_samples += batch_size
    if total_samples == 0:
        raise ValueError("Training iterator produced no batches")
    return total_loss / total_samples
