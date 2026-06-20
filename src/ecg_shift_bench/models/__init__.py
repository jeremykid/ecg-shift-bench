"""Compact neural network baselines for 12-lead inputs."""

from ecg_shift_bench.models.inception_time import InceptionTime1D
from ecg_shift_bench.models.resnet1d import ResNet1D
from ecg_shift_bench.models.resnet1d_wang import ResNet1DWang, resnet1d_wang

__all__ = ["ResNet1D", "ResNet1DWang", "InceptionTime1D", "resnet1d_wang"]
