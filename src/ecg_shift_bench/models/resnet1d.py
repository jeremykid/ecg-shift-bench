"""Small ResNet-style baseline for lead-first ECG tensors."""

import torch
from torch import nn

from ecg_shift_bench.models.heads import MultiLabelHead


class ResidualBlock1D(nn.Module):
    """Two-convolution residual block with optional downsampling."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 7, stride=stride, padding=3, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, 5, padding=2, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False)
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply residual block."""
        return self.activation(self.body(inputs) + self.skip(inputs))


class ResNet1D(nn.Module):
    """Minimal source-only reference model, not a claimed tuned baseline."""

    def __init__(self, in_channels: int = 12, num_labels: int = 6, width: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, width, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
            ResidualBlock1D(width, width, stride=1),
            ResidualBlock1D(width, width * 2, stride=2),
            ResidualBlock1D(width * 2, width * 4, stride=2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.head = MultiLabelHead(width * 4, num_labels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return one logit per canonical label."""
        return self.head(self.encoder(inputs))
