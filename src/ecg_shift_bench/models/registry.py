"""Model selection helpers for ECG benchmarks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch.nn as nn

from ecg_shift_bench.models.inception_time import InceptionTime1D
from ecg_shift_bench.models.resnet1d import ResNet1D
from ecg_shift_bench.models.resnet1d_wang import ResNet1DWang
from ecg_shift_bench.models.xresnet1d import XResNet1D

MODEL_NAME_ALIASES = {
    "inception-time": "inception_time",
    "resnet1d": "resnet1d",
    "resnet1d-wang": "resnet1d_wang",
    "wang-resnet1d": "resnet1d_wang",
    "xresnet1d": "xresnet1d",
    "xresnet-1d": "xresnet1d",
    "xresnet": "xresnet1d",
}

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "inception_time": InceptionTime1D,
    "resnet1d": ResNet1D,
    "resnet1d_wang": ResNet1DWang,
    "xresnet1d": XResNet1D,
}


def canonical_model_name(name: str) -> str:
    """Return the canonical model identifier used across configs and statuses."""
    key = str(name).strip().lower().replace("_", "-")
    try:
        return MODEL_NAME_ALIASES[key]
    except KeyError as error:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"Unknown model {name!r}; supported models: {supported}") from error


def _require_no_unknown_keys(name: str, config: Mapping[str, Any], allowed: Sequence[str]) -> None:
    extra = sorted(key for key in config if key not in allowed)
    if extra:
        supported = ", ".join(allowed)
        raise ValueError(f"Unsupported options for {name}: {extra}. Allowed keys: {supported}")


def create_model(model_config: Mapping[str, Any], *, num_labels: int) -> nn.Module:
    """Instantiate one model from a named config mapping."""
    if "name" not in model_config:
        raise ValueError("model config is missing required key: name")
    config = dict(model_config)
    name = canonical_model_name(str(config.pop("name")))
    in_channels = int(config.pop("in_channels", 12))

    if name == "resnet1d":
        width = int(config.pop("width", 32))
        _require_no_unknown_keys(name, config, ())
        return ResNet1D(in_channels=in_channels, num_labels=num_labels, width=width)
    if name == "resnet1d_wang":
        channels = int(config.pop("channels", 128))
        dropout = float(config.pop("dropout", 0.5))
        _require_no_unknown_keys(name, config, ())
        return ResNet1DWang(
            in_channels=in_channels,
            num_labels=num_labels,
            channels=channels,
            dropout=dropout,
        )
    if name == "inception_time":
        width = int(config.pop("width", 32))
        _require_no_unknown_keys(name, config, ())
        return InceptionTime1D(in_channels=in_channels, num_labels=num_labels, width=width)
    if name == "xresnet1d":
        channels = int(config.pop("channels", 64))
        layers = config.pop("layers", (2, 2, 2, 2))
        expansion = int(config.pop("expansion", 4))
        dropout = float(config.pop("dropout", 0.0))
        _require_no_unknown_keys(name, config, ())
        if isinstance(layers, list):
            layers = tuple(int(item) for item in layers)
        elif isinstance(layers, tuple):
            layers = tuple(int(item) for item in layers)
        else:
            raise TypeError("layers must be a list or tuple of four integers")
        return XResNet1D(
            in_channels=in_channels,
            num_labels=num_labels,
            channels=channels,
            layers=layers,
            expansion=expansion,
            dropout=dropout,
        )

    supported = ", ".join(sorted(MODEL_REGISTRY))
    raise KeyError(f"Unknown model {name!r}; supported models: {supported}")
