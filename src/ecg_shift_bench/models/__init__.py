"""Compact neural network baselines for 12-lead inputs, including XResNet-style variants."""

from ecg_shift_bench.models.inception_time import InceptionTime1D
from ecg_shift_bench.models.resnet1d import ResNet1D
from ecg_shift_bench.models.resnet1d_wang import ResNet1DWang, resnet1d_wang
from ecg_shift_bench.models.registry import MODEL_REGISTRY, canonical_model_name, create_model
from ecg_shift_bench.models.xresnet1d import XResNet1D, xresnet1d

__all__ = [
    "ResNet1D",
    "ResNet1DWang",
    "InceptionTime1D",
    "XResNet1D",
    "MODEL_REGISTRY",
    "canonical_model_name",
    "create_model",
    "resnet1d_wang",
    "xresnet1d",
]
