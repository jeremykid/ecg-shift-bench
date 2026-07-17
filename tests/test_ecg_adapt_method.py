"""Tests for ECG-Adapt-inspired multi-label UDA."""

from __future__ import annotations

import copy
import math
import re
import unittest

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ecg_shift_bench.methods.coral import CoralUdaMethod
from ecg_shift_bench.methods.ecg_adapt import ECGAdaptUdaMethod, gradient_reverse
from ecg_shift_bench.methods.uda import SourceOnlyUdaMethod, build_uda_method
from ecg_shift_bench.utils.config import load_yaml


class FixedDomainDiscriminator(nn.Module):
    """Return prescribed per-label domain logits while retaining autograd."""

    def __init__(self, outputs: Tensor) -> None:
        super().__init__()
        self.outputs = nn.Parameter(outputs.clone())

    def forward(self, features: Tensor) -> Tensor:
        if features.shape[0] != self.outputs.shape[0]:
            raise ValueError("fixed discriminator batch mismatch")
        return self.outputs + features.sum(dim=1, keepdim=True).unsqueeze(-1) * 0.0


def _method(
    *,
    num_labels: int = 2,
    lambda_weight: float = 1.0,
    negative_learning_epochs: int = 0,
) -> ECGAdaptUdaMethod:
    method = ECGAdaptUdaMethod(
        method_params={
            "lambda": lambda_weight,
            "hidden_dims": [4],
            "dropout": 0.0,
            "negative_learning_epochs": negative_learning_epochs,
            "negative_label_threshold": 0.3,
            "pseudo_label_threshold": 0.7,
        }
    )
    method.initialize(
        feature_dim=3,
        num_labels=num_labels,
        device=torch.device("cpu"),
        label_names=[f"label_{index}" for index in range(num_labels)],
    )
    return method


def _adaptation_loss(
    method: ECGAdaptUdaMethod,
    *,
    source_features: Tensor,
    target_features: Tensor,
    source_targets: Tensor,
    target_logits: Tensor,
    epoch: int = 0,
) -> tuple[Tensor, dict[str, object]]:
    num_labels = source_targets.shape[1]
    return method.adaptation_loss(
        source_features=source_features,
        target_features=target_features,
        source_logits=torch.zeros(source_features.shape[0], num_labels),
        target_logits=target_logits,
        source_targets=source_targets,
        epoch=epoch,
        step=1,
    )


class ECGAdaptMethodTests(unittest.TestCase):
    def test_discriminator_returns_two_domain_logits_per_label(self) -> None:
        method = _method(num_labels=3)

        logits = method.discriminator(torch.randn(2, 3))

        self.assertEqual(logits.shape, (2, 3, 2))

    def test_source_and_target_domain_logits_compete_with_cross_entropy(self) -> None:
        method = _method(num_labels=1)
        source_features = torch.zeros(1, 3)
        target_features = torch.zeros(1, 3)

        method.discriminator = FixedDomainDiscriminator(torch.tensor([[[5.0, -5.0]]]))
        source_correct_loss, _ = _adaptation_loss(
            method,
            source_features=source_features,
            target_features=target_features,
            source_targets=torch.ones(1, 1),
            target_logits=torch.full((1, 1), -10.0),
        )
        method.discriminator = FixedDomainDiscriminator(torch.tensor([[[-5.0, 5.0]]]))
        source_wrong_loss, _ = _adaptation_loss(
            method,
            source_features=source_features,
            target_features=target_features,
            source_targets=torch.ones(1, 1),
            target_logits=torch.full((1, 1), -10.0),
        )
        target_correct_loss, _ = _adaptation_loss(
            method,
            source_features=source_features,
            target_features=target_features,
            source_targets=torch.zeros(1, 1),
            target_logits=torch.full((1, 1), 10.0),
        )
        method.discriminator = FixedDomainDiscriminator(torch.tensor([[[5.0, -5.0]]]))
        target_wrong_loss, _ = _adaptation_loss(
            method,
            source_features=source_features,
            target_features=target_features,
            source_targets=torch.zeros(1, 1),
            target_logits=torch.full((1, 1), 10.0),
        )

        self.assertGreater(float(source_wrong_loss), float(source_correct_loss))
        self.assertGreater(float(target_wrong_loss), float(target_correct_loss))

    def test_source_alignment_uses_only_positive_labels(self) -> None:
        method = _method(num_labels=2)
        source_features = torch.zeros(1, 3)
        target_features = torch.zeros(1, 3)
        source_targets = torch.tensor([[1.0, 0.0]])
        target_logits = torch.full((1, 2), -10.0)
        baseline = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
        changed_unselected = torch.tensor([[[0.0, 0.0], [-100.0, 100.0]]])

        method.discriminator = FixedDomainDiscriminator(baseline)
        baseline_loss, _ = _adaptation_loss(
            method,
            source_features=source_features,
            target_features=target_features,
            source_targets=source_targets,
            target_logits=target_logits,
        )
        method.discriminator = FixedDomainDiscriminator(changed_unselected)
        changed_loss, _ = _adaptation_loss(
            method,
            source_features=source_features,
            target_features=target_features,
            source_targets=source_targets,
            target_logits=target_logits,
        )

        torch.testing.assert_close(changed_loss, baseline_loss)

    def test_multi_label_source_positives_contribute_independently(self) -> None:
        method = _method(num_labels=2)
        method.discriminator = FixedDomainDiscriminator(torch.tensor([[[2.0, 0.0], [0.0, 2.0]]]))

        loss, _ = _adaptation_loss(
            method,
            source_features=torch.zeros(1, 3),
            target_features=torch.zeros(1, 3),
            source_targets=torch.ones(1, 2),
            target_logits=torch.full((1, 2), -10.0),
        )

        expected = torch.stack(
            [
                F.cross_entropy(torch.tensor([[2.0, 0.0]]), torch.tensor([0])),
                F.cross_entropy(torch.tensor([[0.0, 2.0]]), torch.tensor([0])),
            ]
        ).mean()
        torch.testing.assert_close(loss, expected)

    def test_selected_entries_are_normalized_per_label(self) -> None:
        method = _method(num_labels=2)
        outputs = torch.tensor(
            [
                [[0.0, 0.0], [-2.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        )
        method.discriminator = FixedDomainDiscriminator(outputs)

        loss, _ = _adaptation_loss(
            method,
            source_features=torch.zeros(3, 3),
            target_features=torch.zeros(3, 3),
            source_targets=torch.tensor([[1.0, 1.0], [1.0, 0.0], [1.0, 0.0]]),
            target_logits=torch.full((3, 2), -10.0),
        )

        label_zero_loss = F.cross_entropy(torch.zeros(3, 2), torch.zeros(3, dtype=torch.long))
        label_one_loss = F.cross_entropy(
            torch.tensor([[-2.0, 0.0]]), torch.zeros(1, dtype=torch.long)
        )
        torch.testing.assert_close(loss, (label_zero_loss + label_one_loss) / 2.0)

    def test_target_confidence_weights_loss_but_is_not_domain_target(self) -> None:
        method = _method(num_labels=1)
        probabilities = torch.tensor([[0.8], [0.9]])
        target_logits = torch.logit(probabilities)
        outputs = torch.tensor([[[0.0, 0.0]], [[2.0, 0.0]]])
        method.discriminator = FixedDomainDiscriminator(outputs)

        loss, _ = _adaptation_loss(
            method,
            source_features=torch.zeros(2, 3),
            target_features=torch.zeros(2, 3),
            source_targets=torch.zeros(2, 1),
            target_logits=target_logits,
        )

        per_entry = F.cross_entropy(
            outputs[:, 0, :], torch.ones(2, dtype=torch.long), reduction="none"
        )
        expected = (per_entry * probabilities[:, 0]).sum() / probabilities.sum()
        torch.testing.assert_close(loss, expected)

    def test_empty_masks_return_differentiable_zero(self) -> None:
        method = _method(num_labels=2)
        source_features = torch.randn(2, 3, requires_grad=True)
        target_features = torch.randn(2, 3, requires_grad=True)

        loss, metrics = _adaptation_loss(
            method,
            source_features=source_features,
            target_features=target_features,
            source_targets=torch.zeros(2, 2),
            target_logits=torch.full((2, 2), -10.0),
        )
        loss.backward()

        self.assertTrue(loss.requires_grad)
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(metrics["selected_source_positive_count"], 0)
        self.assertEqual(metrics["selected_target_pseudo_positive_count"], 0)
        self.assertIsNotNone(source_features.grad)
        self.assertIsNotNone(target_features.grad)

    def test_gradient_reversal_scales_feature_gradient_once_only(self) -> None:
        base_method = _method(num_labels=1, lambda_weight=1.0)
        scaled_method = _method(num_labels=1, lambda_weight=0.25)
        zero_method = _method(num_labels=1, lambda_weight=0.0)
        scaled_method.load_state_dict(copy.deepcopy(base_method.state_dict()))
        zero_method.load_state_dict(copy.deepcopy(base_method.state_dict()))

        def backward(method: ECGAdaptUdaMethod) -> tuple[Tensor, Tensor]:
            source_features = torch.tensor([[1.0, -0.5, 0.25]], requires_grad=True)
            target_features = torch.tensor([[-0.5, 1.0, 0.5]], requires_grad=True)
            loss, _ = _adaptation_loss(
                method,
                source_features=source_features,
                target_features=target_features,
                source_targets=torch.ones(1, 1),
                target_logits=torch.full((1, 1), 10.0),
            )
            loss.backward()
            discriminator_gradient = next(method.discriminator.parameters()).grad.detach().clone()
            return source_features.grad.detach().clone(), discriminator_gradient

        full_feature_gradient, full_discriminator_gradient = backward(base_method)
        scaled_feature_gradient, scaled_discriminator_gradient = backward(scaled_method)
        zero_feature_gradient, zero_discriminator_gradient = backward(zero_method)

        torch.testing.assert_close(scaled_feature_gradient, full_feature_gradient * 0.25)
        torch.testing.assert_close(scaled_discriminator_gradient, full_discriminator_gradient)
        torch.testing.assert_close(zero_feature_gradient, torch.zeros_like(full_feature_gradient))
        torch.testing.assert_close(zero_discriminator_gradient, full_discriminator_gradient)

    def test_discriminator_gradient_step_reduces_domain_loss(self) -> None:
        method = _method(num_labels=1)
        method.discriminator = FixedDomainDiscriminator(torch.tensor([[[-2.0, 2.0]]]))
        optimizer = torch.optim.SGD(method.discriminator.parameters(), lr=0.1)
        arguments = {
            "source_features": torch.zeros(1, 3),
            "target_features": torch.zeros(1, 3),
            "source_targets": torch.ones(1, 1),
            "target_logits": torch.full((1, 1), -10.0),
        }

        before, _ = _adaptation_loss(method, **arguments)
        before.backward()
        optimizer.step()
        after, _ = _adaptation_loss(method, **arguments)

        self.assertLess(float(after), float(before))

    def test_gradient_reverse_reverses_backward_gradient(self) -> None:
        features = torch.tensor([[1.0, -2.0]], requires_grad=True)

        gradient_reverse(features, 0.5).sum().backward()

        torch.testing.assert_close(features.grad, torch.full_like(features, -0.5))

    def test_negative_learning_stage_uses_zero_based_boundary(self) -> None:
        method = _method(num_labels=1, negative_learning_epochs=2)
        method.discriminator = FixedDomainDiscriminator(torch.zeros(1, 1, 2))
        arguments = {
            "source_features": torch.zeros(1, 3),
            "target_features": torch.zeros(1, 3),
            "source_targets": torch.zeros(1, 1),
            "target_logits": torch.full((1, 1), -10.0),
        }

        _, first = _adaptation_loss(method, epoch=0, **arguments)
        _, second = _adaptation_loss(method, epoch=1, **arguments)
        _, after = _adaptation_loss(method, epoch=2, **arguments)

        self.assertEqual(first["ecg_adapt_stage"], "positive_plus_negative")
        self.assertEqual(second["ecg_adapt_stage"], "positive_plus_negative")
        self.assertEqual(after["ecg_adapt_stage"], "positive_only")
        self.assertGreater(first["source_negative_domain_loss"], 0.0)
        self.assertGreater(first["target_negative_domain_loss"], 0.0)

    def test_zero_negative_learning_epochs_disables_negative_stage(self) -> None:
        method = _method(num_labels=1, negative_learning_epochs=0)

        _, metrics = _adaptation_loss(
            method,
            source_features=torch.zeros(1, 3),
            target_features=torch.zeros(1, 3),
            source_targets=torch.zeros(1, 1),
            target_logits=torch.full((1, 1), -10.0),
            epoch=0,
        )

        self.assertEqual(metrics["ecg_adapt_stage"], "positive_only")
        self.assertEqual(metrics["source_negative_domain_loss"], 0.0)
        self.assertEqual(metrics["target_negative_domain_loss"], 0.0)

    def test_diagnostics_include_per_label_counts_rates_and_domain_accuracy(self) -> None:
        method = ECGAdaptUdaMethod(
            method_params={"hidden_dims": [4], "dropout": 0.0, "negative_learning_epochs": 0}
        )
        method.initialize(
            feature_dim=3,
            num_labels=2,
            device=torch.device("cpu"),
            label_names=["AF", "RBBB"],
        )

        _, metrics = _adaptation_loss(
            method,
            source_features=torch.zeros(2, 3),
            target_features=torch.zeros(2, 3),
            source_targets=torch.tensor([[1.0, 0.0], [1.0, 1.0]]),
            target_logits=torch.tensor([[10.0, -10.0], [10.0, 10.0]]),
        )

        self.assertEqual(metrics["source_positive_AF_count"], 2)
        self.assertEqual(metrics["source_positive_RBBB_count"], 1)
        self.assertEqual(metrics["target_pseudo_positive_AF_rate"], 1.0)
        self.assertEqual(metrics["target_pseudo_positive_RBBB_rate"], 0.5)
        self.assertIn("source_domain_accuracy", metrics)
        self.assertIn("target_domain_accuracy", metrics)

    def test_invalid_method_configuration_is_rejected(self) -> None:
        invalid_params = (
            ({"lambda": -0.1}, "lambda must be non-negative"),
            ({"dropout": 1.0}, "dropout must be in [0, 1)"),
            ({"pseudo_label_threshold": 1.1}, "pseudo_label_threshold must be in [0, 1]"),
            ({"negative_label_threshold": -0.1}, "negative_label_threshold must be in [0, 1]"),
            ({"hidden_dims": [0]}, "hidden_dims must contain"),
            ({"negative_learning_epochs": -1}, "negative_learning_epochs must be non-negative"),
            (
                {
                    "negative_learning_epochs": 1,
                    "negative_label_threshold": 0.8,
                    "pseudo_label_threshold": 0.7,
                },
                "negative_label_threshold must be less than pseudo_label_threshold",
            ),
        )
        for params, message in invalid_params:
            with (
                self.subTest(params=params),
                self.assertRaisesRegex(ValueError, re.escape(message)),
            ):
                ECGAdaptUdaMethod(method_params=params)

    def test_invalid_training_tensor_shapes_and_values_are_rejected(self) -> None:
        method = _method(num_labels=2)
        valid = {
            "source_features": torch.zeros(2, 3),
            "target_features": torch.zeros(2, 3),
            "source_logits": torch.zeros(2, 2),
            "target_logits": torch.zeros(2, 2),
            "source_targets": torch.zeros(2, 2),
            "epoch": 0,
            "step": 1,
        }
        invalid_cases = (
            ({"source_features": torch.zeros(2, 3, 1)}, "source_features must be a 2D"),
            (
                {"target_features": torch.zeros(2, 4)},
                "target_features must have feature dimension 3",
            ),
            ({"source_logits": torch.zeros(2, 3)}, "source_logits must have shape"),
            ({"target_logits": torch.zeros(1, 2)}, "target_logits must have shape"),
            ({"source_targets": torch.full((2, 2), 0.5)}, "source_targets must contain only"),
            ({"source_targets": torch.full((2, 2), math.nan)}, "source_targets must be finite"),
        )
        for replacement, message in invalid_cases:
            arguments = dict(valid)
            arguments.update(replacement)
            with self.subTest(replacement=replacement), self.assertRaisesRegex(ValueError, message):
                method.adaptation_loss(**arguments)

    def test_method_moves_device_and_restores_discriminator_state(self) -> None:
        method = _method(num_labels=2)
        saved = copy.deepcopy(method.state_dict())
        for parameter in method.parameters():
            parameter.data.add_(1.0)
        method.load_state_dict(saved)

        for key, value in method.state_dict().items():
            torch.testing.assert_close(value, saved[key])
        self.assertEqual(next(method.parameters()).device.type, "cpu")
        if torch.cuda.is_available():
            method.to(torch.device("cuda"))
            self.assertEqual(next(method.parameters()).device.type, "cuda")

    def test_registry_supports_canonical_name_and_legacy_alias(self) -> None:
        canonical = build_uda_method("ecg_adapt_multilabel")
        legacy = build_uda_method("ecg_adapt")
        coral = build_uda_method("coral", method_params={"lambda": 0.1})
        source_only = build_uda_method("source_only")

        self.assertIsInstance(canonical, ECGAdaptUdaMethod)
        self.assertIsInstance(legacy, ECGAdaptUdaMethod)
        self.assertEqual(canonical.method_name, "ecg_adapt_multilabel")
        self.assertEqual(legacy.method_name, "ecg_adapt_multilabel")
        self.assertIsInstance(coral, CoralUdaMethod)
        self.assertIsInstance(source_only, SourceOnlyUdaMethod)

    def test_experiment_config_uses_multilabel_name_and_safe_defaults(self) -> None:
        config = load_yaml("configs/experiments/uda_ecg_adapt.yaml")
        registry = load_yaml("configs/experiments/registry.yaml")
        entry = registry["experiments"]["uda-ecg-adapt-ptbxl-to-chapman"]

        self.assertEqual(config["method"], "ecg_adapt_multilabel")
        self.assertEqual(config["method_params"]["negative_learning_epochs"], 0)
        self.assertEqual(config["canonical_labels"], ["AF", "RBBB", "LBBB", "1dAVB", "SB", "ST"])
        self.assertFalse(config["protocol"]["target_labels_available_during_training"])
        self.assertEqual(entry["config"], "configs/experiments/uda_ecg_adapt.yaml")


if __name__ == "__main__":
    unittest.main()
