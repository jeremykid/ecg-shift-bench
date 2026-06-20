"""Compact InceptionTime-inspired alternative encoder."""

import torch
from torch import nn

from ecg_shift_bench.models.heads import MultiLabelHead


class InceptionTime1D(nn.Module):
    """Simple multi-kernel temporal convolution baseline."""

    def __init__(self, in_channels: int = 12, num_labels: int = 6, width: int = 32) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(in_channels, width, kernel, padding=kernel // 2)
                for kernel in (9, 19, 39)
            ]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = MultiLabelHead(width * len(self.branches), num_labels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return one logit per canonical label."""
        features = torch.cat([torch.relu(branch(inputs)) for branch in self.branches], dim=1)
        return self.head(self.pool(features).squeeze(-1))
