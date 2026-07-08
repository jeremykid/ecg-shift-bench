"""TENT test-time adaptation extension point."""

from torch import nn


class TENT(nn.Module):
    """Placeholder for entropy-minimizing test-time adaptation updates."""

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "TODO: define TENT reset and episodic/continual adaptation policy"
        )
