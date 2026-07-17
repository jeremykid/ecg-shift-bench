"""Tests for the CORAL UDA method."""

from __future__ import annotations

import pytest
import torch

from ecg_shift_bench.methods.coral import CoralUdaMethod, coral_loss, feature_covariance
from ecg_shift_bench.methods.uda import SourceOnlyUdaMethod, build_uda_method
from ecg_shift_bench.utils.config import load_yaml


def test_feature_covariance_matches_unbiased_batch_covariance() -> None:
    features = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 8.0],
        ]
    )

    covariance = feature_covariance(features)

    centered = features - features.mean(dim=0, keepdim=True)
    expected = centered.T @ centered / 2.0
    torch.testing.assert_close(covariance, expected)


def test_coral_loss_is_zero_for_identical_features() -> None:
    features = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [4.0, 8.0, 16.0],
        ]
    )

    loss = coral_loss(features, features.clone())

    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_coral_loss_is_positive_for_different_covariances() -> None:
    source = torch.tensor(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ]
    )
    target = torch.tensor(
        [
            [0.0, 1.0],
            [0.0, 3.0],
            [0.0, 7.0],
        ]
    )

    loss = coral_loss(source, target)

    assert loss.item() > 0.0


@pytest.mark.parametrize(
    ("source", "target", "message"),
    [
        (torch.randn(2, 3, 1), torch.randn(2, 3, 1), "2D"),
        (torch.randn(1, 3), torch.randn(1, 3), "at least two"),
        (torch.randn(2, 3), torch.randn(2, 4), "same feature dimension"),
    ],
)
def test_coral_loss_validates_feature_shape(
    source: torch.Tensor,
    target: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        coral_loss(source, target)


def test_coral_method_scales_loss_and_reports_raw_metric() -> None:
    method = CoralUdaMethod(method_params={"lambda": 0.25})
    source_features = torch.tensor(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ]
    )
    target_features = torch.tensor(
        [
            [0.0, 1.0],
            [0.0, 3.0],
            [0.0, 7.0],
        ]
    )
    raw_loss = coral_loss(source_features, target_features)

    adaptation_loss, metrics = method.adaptation_loss(
        source_features=source_features,
        target_features=target_features,
        source_logits=torch.zeros(3, 6),
        target_logits=torch.zeros(3, 6),
        source_targets=torch.zeros(3, 6),
        epoch=1,
        step=1,
    )

    torch.testing.assert_close(adaptation_loss, raw_loss * 0.25)
    assert metrics["coral_loss"] == pytest.approx(float(raw_loss))
    assert metrics["coral_lambda"] == pytest.approx(0.25)


def test_uda_method_registry_builds_coral_and_keeps_source_only() -> None:
    coral = build_uda_method("coral", method_params={"lambda": 0.1})
    source_only = build_uda_method("source_only")

    assert isinstance(coral, CoralUdaMethod)
    assert coral.method_name == "coral"
    assert coral.method_params == {"lambda": 0.1}
    assert isinstance(source_only, SourceOnlyUdaMethod)


def test_coral_experiment_config_and_registry_entry_match() -> None:
    config = load_yaml("configs/experiments/uda_coral.yaml")
    registry = load_yaml("configs/experiments/registry.yaml")
    entry = registry["experiments"]["uda-coral-ptbxl-to-chapman"]

    assert config["experiment"] == "uda_coral_ptbxl_to_chapman"
    assert config["method"] == "coral"
    assert config["source_datasets"] == ["PTBXL"]
    assert config["target_datasets"] == ["CHAPMAN"]
    assert config["method_params"] == {"lambda": 0.1}
    assert config["protocol"]["target_labels_available_during_training"] is False
    assert config["protocol"]["target_inputs_available_during_training"] is True
    assert entry["config"] == "configs/experiments/uda_coral.yaml"
    assert entry["artifact_root"] == "outputs/uda/coral_ptbxl_to_chapman"
