"""Optimizer construction from research configs."""

from collections.abc import Iterable

from torch import Tensor
from torch.optim import SGD, Adam, AdamW, Optimizer


def create_optimizer(
    parameters: Iterable[Tensor],
    name: str,
    learning_rate: float,
    weight_decay: float = 0.0,
) -> Optimizer:
    """Construct a supported optimizer with explicit hyperparameters."""
    options = {"lr": learning_rate, "weight_decay": weight_decay}
    key = name.lower()
    if key == "adam":
        return Adam(parameters, **options)
    if key == "adamw":
        return AdamW(parameters, **options)
    if key == "sgd":
        return SGD(parameters, momentum=0.9, **options)
    raise ValueError(f"Unsupported optimizer: {name}")
