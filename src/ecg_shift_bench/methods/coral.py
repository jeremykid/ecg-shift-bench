"""CORAL domain adaptation extension point."""

from torch import nn


class CORAL(nn.Module):
    """Placeholder for covariance alignment over source/target features."""

    def forward(self, *args, **kwargs):
        raise NotImplementedError("TODO: implement and validate CORAL feature alignment")
