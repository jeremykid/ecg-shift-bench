"""UDA method extension points."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn


class UdaMethod(nn.Module):
    """Base interface for a generic UDA objective."""

    method_name = "source_only"

    def __init__(self, method_params: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        self.method_params = dict(method_params or {})

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
        """Return the adaptation loss and any scalar diagnostics."""
        del source_features, target_features, target_logits, source_targets, epoch, step
        return torch.zeros((), device=source_logits.device), {}

    def metadata(self) -> dict[str, Any]:
        """Return method metadata for run status and artifacts."""
        return {"name": self.method_name, "params": dict(self.method_params)}


class SourceOnlyUdaMethod(UdaMethod):
    """Neutral UDA objective that only optimizes source classification loss."""

    method_name = "source_only"


def build_uda_method(
    method_name: str,
    *,
    method_params: Mapping[str, Any] | None = None,
) -> UdaMethod:
    """Construct the configured UDA method implementation."""
    key = str(method_name).strip().lower().replace("-", "_")
    if key == "source_only":
        return SourceOnlyUdaMethod(method_params=method_params)
    raise NotImplementedError(
        f"UDA method {method_name!r} is not implemented yet; add it under ecg_shift_bench.methods"
    )
