"""ECG-Adapt-inspired class-conditional UDA for multi-label ECGs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ecg_shift_bench.methods.uda import UdaMethod


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, inputs: Tensor, lambda_weight: float) -> Tensor:
        ctx.lambda_weight = float(lambda_weight)
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None]:
        return -ctx.lambda_weight * grad_output, None


def gradient_reverse(inputs: Tensor, lambda_weight: float) -> Tensor:
    """Return inputs unchanged while reversing/scaling their backward gradient."""
    return _GradientReverse.apply(inputs, float(lambda_weight))


def _as_hidden_dims(value: Any) -> list[int]:
    if value is None:
        return [512, 128]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("hidden_dims must be a sequence of positive integers")
    if not value or any(not isinstance(item, Integral) or isinstance(item, bool) for item in value):
        raise ValueError("hidden_dims must contain at least one positive integer")
    hidden_dims = [int(item) for item in value]
    if any(dim <= 0 for dim in hidden_dims):
        raise ValueError("hidden_dims must contain at least one positive integer")
    return hidden_dims


def _per_label_masked_mean(
    losses: Tensor,
    mask: Tensor,
    *,
    weights: Tensor | None = None,
) -> Tensor:
    """Average selected losses within labels, then equally across active labels."""
    effective_weights = mask.to(dtype=losses.dtype)
    if weights is not None:
        effective_weights = effective_weights * weights.to(dtype=losses.dtype)
    per_label_denominator = effective_weights.sum(dim=0)
    active_labels = per_label_denominator > 0
    per_label_loss = (losses * effective_weights).sum(dim=0) / per_label_denominator.clamp_min(1.0)
    active_weights = active_labels.to(dtype=losses.dtype)
    return (per_label_loss * active_weights).sum() / active_weights.sum().clamp_min(1.0)


def _mean_available(components: Sequence[tuple[Tensor, Tensor]]) -> Tensor:
    """Average loss components that contain at least one selected entry."""
    losses = torch.stack([loss for loss, _ in components])
    active = torch.stack([count > 0 for _, count in components]).to(dtype=losses.dtype)
    return (losses * active).sum() / active.sum().clamp_min(1.0)


def _selected_accuracy(domain_logits: Tensor, mask: Tensor, domain_target: int) -> float:
    selected_count = int(mask.sum().detach())
    if selected_count == 0:
        return 0.0
    correct = (domain_logits.argmax(dim=-1) == domain_target) & mask
    return float(correct.sum().detach() / selected_count)


class FineGrainedDiscriminator(nn.Module):
    """Predict source-versus-target logits independently for each ECG label."""

    def __init__(
        self,
        *,
        feature_dim: int,
        num_labels: int,
        hidden_dims: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_labels = int(num_labels)
        layers: list[nn.Module] = []
        previous_dim = self.feature_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, int(hidden_dim)))
            layers.append(nn.LeakyReLU(negative_slope=0.2))
            if dropout > 0.0:
                layers.append(nn.Dropout(p=float(dropout)))
            previous_dim = int(hidden_dim)
        layers.append(nn.Linear(previous_dim, self.num_labels * 2))
        self.network = nn.Sequential(*layers)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2:
            raise ValueError("discriminator features must be a 2D tensor")
        if features.shape[1] != self.feature_dim:
            raise ValueError(
                f"discriminator features must have feature dimension {self.feature_dim}"
            )
        raw_logits = self.network(features)
        return raw_logits.reshape(features.shape[0], self.num_labels, 2)


class ECGAdaptUdaMethod(UdaMethod):
    """ECG-Adapt-inspired multi-label class-conditional domain alignment.

    The original ECG-Adapt objective is single-label and uses a unified one-hot
    class-domain discriminator. ECGShiftBench instead treats each disease label
    independently and trains a two-class source-versus-target discriminator for
    every label. Positive alignment is the primary objective. The optional early
    negative phase aligns disease-absence cohorts and is a benchmark-specific
    experiment, not the paper's random complementary-label procedure.
    """

    method_name = "ecg_adapt_multilabel"

    def __init__(self, method_params: Mapping[str, Any] | None = None) -> None:
        params = dict(method_params or {})
        self.lambda_weight = float(params.get("lambda", 0.001))
        self.pseudo_label_threshold = float(params.get("pseudo_label_threshold", 0.7))
        self.negative_label_threshold = float(params.get("negative_label_threshold", 0.3))
        self.negative_learning_epochs = int(params.get("negative_learning_epochs", 0))
        self.hidden_dims = _as_hidden_dims(params.get("hidden_dims", [512, 128]))
        self.dropout = float(params.get("dropout", 0.2))
        if self.lambda_weight < 0.0:
            raise ValueError("lambda must be non-negative")
        if not 0.0 <= self.negative_label_threshold <= 1.0:
            raise ValueError("negative_label_threshold must be in [0, 1]")
        if not 0.0 <= self.pseudo_label_threshold <= 1.0:
            raise ValueError("pseudo_label_threshold must be in [0, 1]")
        if self.negative_learning_epochs < 0:
            raise ValueError("negative_learning_epochs must be non-negative")
        if (
            self.negative_learning_epochs > 0
            and self.negative_label_threshold >= self.pseudo_label_threshold
        ):
            raise ValueError(
                "negative_label_threshold must be less than pseudo_label_threshold "
                "when negative learning is enabled"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        params.update(
            {
                "lambda": self.lambda_weight,
                "pseudo_label_threshold": self.pseudo_label_threshold,
                "negative_label_threshold": self.negative_label_threshold,
                "negative_learning_epochs": self.negative_learning_epochs,
                "hidden_dims": list(self.hidden_dims),
                "dropout": self.dropout,
            }
        )
        super().__init__(method_params=params)
        self.feature_dim: int | None = None
        self.num_labels: int | None = None
        self.label_names: tuple[str, ...] = ()
        self.discriminator: FineGrainedDiscriminator | None = None

    def initialize(
        self,
        *,
        feature_dim: int,
        num_labels: int,
        device: torch.device,
        label_names: Sequence[str] | None = None,
    ) -> None:
        """Create the discriminator once pooled-feature dimensions are known."""
        if not isinstance(feature_dim, Integral) or isinstance(feature_dim, bool):
            raise TypeError("feature_dim must be a positive integer")
        if not isinstance(num_labels, Integral) or isinstance(num_labels, bool):
            raise TypeError("num_labels must be a positive integer")
        feature_dim = int(feature_dim)
        num_labels = int(num_labels)
        if feature_dim <= 0 or num_labels <= 0:
            raise ValueError("feature_dim and num_labels must be positive")
        resolved_names = tuple(
            str(name) for name in (label_names or [f"label_{index}" for index in range(num_labels)])
        )
        if len(resolved_names) != num_labels or any(not name for name in resolved_names):
            raise ValueError("label_names must contain one non-empty name per label")
        if len(set(resolved_names)) != num_labels:
            raise ValueError("label_names must be unique")
        self.label_names = resolved_names
        if (
            self.discriminator is not None
            and self.feature_dim == feature_dim
            and self.num_labels == num_labels
        ):
            self.to(device)
            return
        self.feature_dim = feature_dim
        self.num_labels = num_labels
        self.discriminator = FineGrainedDiscriminator(
            feature_dim=feature_dim,
            num_labels=num_labels,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        ).to(device)
        self.to(device)

    def _require_initialized(self) -> FineGrainedDiscriminator:
        if self.discriminator is None or self.num_labels is None or self.feature_dim is None:
            raise RuntimeError("ECGAdaptUdaMethod.initialize() must be called before training")
        return self.discriminator

    def _validate_inputs(
        self,
        *,
        source_features: Tensor,
        target_features: Tensor,
        source_logits: Tensor,
        target_logits: Tensor,
        source_targets: Tensor,
        epoch: int,
    ) -> None:
        assert self.feature_dim is not None
        assert self.num_labels is not None
        for name, features in (
            ("source_features", source_features),
            ("target_features", target_features),
        ):
            if features.ndim != 2:
                raise ValueError(f"{name} must be a 2D pooled-feature tensor")
            if features.shape[1] != self.feature_dim:
                raise ValueError(f"{name} must have feature dimension {self.feature_dim}")
        expected_source_shape = (source_features.shape[0], self.num_labels)
        expected_target_shape = (target_features.shape[0], self.num_labels)
        if source_logits.shape != expected_source_shape:
            raise ValueError(
                f"source_logits must have shape {expected_source_shape}, "
                f"got {tuple(source_logits.shape)}"
            )
        if target_logits.shape != expected_target_shape:
            raise ValueError(
                f"target_logits must have shape {expected_target_shape}, "
                f"got {tuple(target_logits.shape)}"
            )
        if source_targets.shape != expected_source_shape:
            raise ValueError(
                "source_targets must have shape "
                f"{expected_source_shape}, got {tuple(source_targets.shape)}"
            )
        if not bool(torch.isfinite(source_targets).all()):
            raise ValueError("source_targets must be finite")
        if not bool(((source_targets == 0) | (source_targets == 1)).all()):
            raise ValueError("source_targets must contain only binary values 0 or 1")
        tensors = (source_features, target_features, source_logits, target_logits, source_targets)
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError("features, logits, and source_targets must be on the same device")
        if not isinstance(epoch, Integral) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("epoch must be a non-negative zero-based integer")

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
        """Return per-label source-versus-target adversarial alignment loss."""
        del step
        discriminator = self._require_initialized()
        self._validate_inputs(
            source_features=source_features,
            target_features=target_features,
            source_logits=source_logits,
            target_logits=target_logits,
            source_targets=source_targets,
            epoch=epoch,
        )
        source_domain_logits = discriminator(gradient_reverse(source_features, self.lambda_weight))
        target_domain_logits = discriminator(gradient_reverse(target_features, self.lambda_weight))
        source_domain_targets = torch.zeros_like(source_targets, dtype=torch.long)
        target_domain_targets = torch.ones_like(target_logits, dtype=torch.long)
        source_domain_losses = F.cross_entropy(
            source_domain_logits.reshape(-1, 2),
            source_domain_targets.reshape(-1),
            reduction="none",
        ).reshape_as(source_targets)
        target_domain_losses = F.cross_entropy(
            target_domain_logits.reshape(-1, 2),
            target_domain_targets.reshape(-1),
            reduction="none",
        ).reshape_as(target_logits)

        source_positive_mask = source_targets == 1
        target_probabilities = torch.sigmoid(target_logits.detach())
        target_positive_mask = target_probabilities >= self.pseudo_label_threshold
        source_positive_loss = _per_label_masked_mean(source_domain_losses, source_positive_mask)
        target_positive_loss = _per_label_masked_mean(
            target_domain_losses,
            target_positive_mask,
            weights=target_probabilities,
        )
        source_positive_count = source_positive_mask.sum()
        target_positive_count = target_positive_mask.sum()
        positive_loss = _mean_available(
            (
                (source_positive_loss, source_positive_count),
                (target_positive_loss, target_positive_count),
            )
        )

        negative_enabled = epoch < self.negative_learning_epochs
        if negative_enabled:
            source_negative_mask = source_targets == 0
            target_negative_mask = target_probabilities <= self.negative_label_threshold
            source_negative_loss = _per_label_masked_mean(
                source_domain_losses, source_negative_mask
            )
            target_negative_loss = _per_label_masked_mean(
                target_domain_losses,
                target_negative_mask,
                weights=1.0 - target_probabilities,
            )
            negative_loss = _mean_available(
                (
                    (source_negative_loss, source_negative_mask.sum()),
                    (target_negative_loss, target_negative_mask.sum()),
                )
            )
            stage = "positive_plus_negative"
        else:
            source_negative_loss = source_domain_losses.sum() * 0.0
            target_negative_loss = target_domain_losses.sum() * 0.0
            negative_loss = source_negative_loss + target_negative_loss
            stage = "positive_only"

        adaptation_loss = positive_loss + negative_loss
        source_batch_entries = source_positive_mask.numel()
        target_batch_entries = target_positive_mask.numel()
        metrics: dict[str, Any] = {
            "ecg_adapt_stage": stage,
            "source_positive_domain_loss": float(source_positive_loss.detach()),
            "target_positive_domain_loss": float(target_positive_loss.detach()),
            "source_negative_domain_loss": float(source_negative_loss.detach()),
            "target_negative_domain_loss": float(target_negative_loss.detach()),
            "discriminator_loss": float(adaptation_loss.detach()),
            "selected_source_positive_count": int(source_positive_count.detach()),
            "selected_target_pseudo_positive_count": int(target_positive_count.detach()),
            "source_positive_selection_rate": float(
                source_positive_count.detach() / source_batch_entries
            ),
            "target_pseudo_positive_selection_rate": float(
                target_positive_count.detach() / target_batch_entries
            ),
            "source_domain_accuracy": _selected_accuracy(
                source_domain_logits, source_positive_mask, 0
            ),
            "target_domain_accuracy": _selected_accuracy(
                target_domain_logits, target_positive_mask, 1
            ),
            "lambda": self.lambda_weight,
        }
        for label_index, label_name in enumerate(self.label_names):
            metrics[f"source_positive_{label_name}_count"] = int(
                source_positive_mask[:, label_index].sum().detach()
            )
            metrics[f"target_pseudo_positive_{label_name}_rate"] = float(
                target_positive_mask[:, label_index].float().mean().detach()
            )
        return adaptation_loss, metrics
