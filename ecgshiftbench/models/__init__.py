"""Baseline models for ECG classification."""

from ecgshiftbench.models.resnet1d import ResNet1D, resnet1d_18, resnet1d_34

__all__ = [
    "ResNet1D",
    "resnet1d_18",
    "resnet1d_34",
]
