"""Tests for the 1D-ResNet baseline model."""

import numpy as np
import pytest
import torch

from ecgshiftbench.models.resnet1d import ResNet1D, resnet1d_18, resnet1d_34


@pytest.fixture
def batch():
    """Random batch: (4, 12, 5000) — batch × leads × samples."""
    rng = torch.Generator().manual_seed(0)
    return torch.randn(4, 12, 5000, generator=rng)


class TestResNet1D:
    def test_resnet18_output_shape(self, batch):
        model = resnet1d_18(num_leads=12, num_classes=5)
        model.eval()
        with torch.no_grad():
            out = model(batch)
        assert out.shape == (4, 5)

    def test_resnet34_output_shape(self, batch):
        model = resnet1d_34(num_leads=12, num_classes=5)
        model.eval()
        with torch.no_grad():
            out = model(batch)
        assert out.shape == (4, 5)

    def test_custom_num_classes(self, batch):
        model = resnet1d_18(num_leads=12, num_classes=9)
        model.eval()
        with torch.no_grad():
            out = model(batch)
        assert out.shape == (4, 9)

    def test_custom_num_leads(self):
        model = resnet1d_18(num_leads=8, num_classes=5)
        batch_8lead = torch.randn(2, 8, 5000)
        model.eval()
        with torch.no_grad():
            out = model(batch_8lead)
        assert out.shape == (2, 5)

    def test_output_dtype(self, batch):
        model = resnet1d_18(num_leads=12, num_classes=5)
        model.eval()
        with torch.no_grad():
            out = model(batch)
        assert out.dtype == torch.float32

    def test_training_mode_forward(self, batch):
        model = resnet1d_18(num_leads=12, num_classes=5)
        model.train()
        out = model(batch)
        assert out.shape == (4, 5)

    def test_gradients_flow(self, batch):
        model = resnet1d_18(num_leads=12, num_classes=5)
        model.train()
        out = model(batch)
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_dropout_zero(self, batch):
        """Eval mode should give deterministic outputs across two calls."""
        model = ResNet1D(num_leads=12, num_classes=5, layers=[2, 2, 2, 2], dropout=0.0)
        model.eval()
        with torch.no_grad():
            out1 = model(batch)
            out2 = model(batch)
        torch.testing.assert_close(out1, out2)

    def test_resnet1d_custom_layers(self, batch):
        model = ResNet1D(num_leads=12, num_classes=5, layers=[1, 1, 1, 1])
        model.eval()
        with torch.no_grad():
            out = model(batch)
        assert out.shape == (4, 5)
