"""DANN domain adaptation extension point."""

from torch import nn


class DANN(nn.Module):
    """Placeholder for gradient-reversal domain-adversarial training."""

    def forward(self, *args, **kwargs):
        raise NotImplementedError("TODO: implement DANN with a scheduled reversal coefficient")
