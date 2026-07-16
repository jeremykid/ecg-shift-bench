"""CORAL domain adaptation method."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import Tensor

from ecg_shift_bench.methods.uda import UdaMethod


def _validate_feature_batch(features: Tensor, *, name: str) -> None:
    if features.ndim != 2:
        raise ValueError(f"{name} features must be a 2D tensor")
    if features.shape[0] < 2:
        raise ValueError(f"{name} features must contain at least two samples")


def feature_covariance(features: Tensor) -> Tensor:
    """Return unbiased batch covariance over feature columns."""
    _validate_feature_batch(features, name="CORAL")
    centered = features - features.mean(dim=0, keepdim=True)
    return centered.transpose(0, 1).matmul(centered) / (features.shape[0] - 1)


def coral_loss(source_features: Tensor, target_features: Tensor) -> Tensor:
    """Return Deep CORAL covariance alignment loss."""
    _validate_feature_batch(source_features, name="source")
    _validate_feature_batch(target_features, name="target")
    if source_features.shape[1] != target_features.shape[1]:
        raise ValueError("source and target features must have the same feature dimension")
    source_covariance = feature_covariance(source_features)
    target_covariance = feature_covariance(target_features)
    feature_dim = source_features.shape[1]
    covariance_delta = source_covariance - target_covariance
    return covariance_delta.pow(2).sum() / (4.0 * feature_dim * feature_dim)


class CoralUdaMethod(UdaMethod):
    """Deep CORAL objective over source and target feature covariance."""

    method_name = "coral"

    def __init__(self, method_params: Mapping[str, Any] | None = None) -> None:
        params = dict(method_params or {})
        self.lambda_weight = float(params.get("lambda", 1.0))
        params["lambda"] = self.lambda_weight
        super().__init__(method_params=params)

    def adaptation_loss(
        self,
        *,
        source_features: Tensor,
        target_features: Tensor,
        source_logits: Tensor,
        target_logits: Tensor,
        source_targets: Tensor,
        epoch: int,
        step: int,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Return weighted CORAL loss and raw diagnostics."""
        del source_logits, target_logits, source_targets, epoch, step
        raw_loss = coral_loss(source_features, target_features)
        return raw_loss * self.lambda_weight, {
            "coral_loss": float(raw_loss.detach()),
            "coral_lambda": self.lambda_weight,
        }
