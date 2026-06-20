"""MMD domain adaptation extension point."""

from torch import nn


class MMD(nn.Module):
    """Placeholder for kernel mean matching over source/target features."""

    def forward(self, *args, **kwargs):
        raise NotImplementedError("TODO: implement and validate an MMD kernel and bandwidth policy")
