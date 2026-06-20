"""Compact neural network baselines for 12-lead inputs."""

from ecg_shift_bench.models.inception_time import InceptionTime1D
from ecg_shift_bench.models.resnet1d import ResNet1D

__all__ = ["ResNet1D", "InceptionTime1D"]
