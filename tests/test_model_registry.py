"""Tests for configurable ECG backbone selection."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ecg_shift_bench.models.registry import create_model
from ecg_shift_bench.utils.config import load_yaml


@pytest.mark.parametrize(
    ("model_config", "num_labels"),
    [
        ({"name": "xresnet1d", "channels": 32, "dropout": 0.0}, 4),
        ({"name": "resnet1d_wang", "channels": 32, "dropout": 0.0}, 4),
        ({"name": "resnet1d", "width": 16}, 4),
        ({"name": "inception_time", "width": 16}, 4),
    ],
)
def test_create_model_returns_logits_for_supported_backbones(
    model_config: dict[str, object], num_labels: int
) -> None:
    model = create_model(model_config, num_labels=num_labels)
    inputs = torch.randn(2, 12, 256)

    with torch.no_grad():
        logits = model(inputs)

    assert logits.shape == (2, num_labels)


def test_xresnet1d_supports_single_item_train_batches() -> None:
    model = create_model({"name": "xresnet1d"}, num_labels=4).train()
    inputs = torch.randn(1, 12, 256)

    logits = model(inputs)

    assert logits.shape == (1, 4)


def test_create_model_rejects_unknown_backbone() -> None:
    with pytest.raises(KeyError, match="supported models"):
        create_model({"name": "unknown_backbone"}, num_labels=4)


def test_dataset_discriminator_config_defaults_to_xresnet1d() -> None:
    config = load_yaml(Path("configs/experiments/dataset_discriminator.yaml"))

    assert config["experiment"] == "dataset_discriminator_xresnet1d_v1"
    assert config["model"]["name"] == "xresnet1d"
