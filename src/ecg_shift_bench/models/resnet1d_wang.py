"""Pure-PyTorch port of the legacy Wang-style 1D ResNet."""

from __future__ import annotations

import torch
from torch import nn


def _conv1d(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
) -> nn.Conv1d:
    return nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=(kernel_size - 1) // 2,
        bias=False,
    )


class WangResidualBlock1D(nn.Module):
    """Two-convolution residual stage used by the legacy benchmark."""

    def __init__(self, channels: int = 128, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = _conv1d(channels, channels, kernel_size=5, stride=stride)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv1d(channels, channels, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(channels)
        self.downsample = (
            nn.Identity()
            if stride == 1
            else nn.Sequential(
                _conv1d(channels, channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(channels),
            )
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the residual stage."""
        residual = self.downsample(inputs)
        output = self.relu(self.bn1(self.conv1(inputs)))
        output = self.bn2(self.conv2(output))
        return self.relu(output + residual)


class ResNet1DWang(nn.Module):
    """Wang-style ResNet with adaptive max+average pooling.

    The convolutional topology and initialization match the legacy fastai
    benchmark while the implementation has no fastai dependency.
    """

    def __init__(
        self,
        in_channels: int = 12,
        num_labels: int = 6,
        channels: int = 128,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            _conv1d(in_channels, channels, kernel_size=7),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.stages = nn.Sequential(
            WangResidualBlock1D(channels, stride=1),
            WangResidualBlock1D(channels, stride=2),
            WangResidualBlock1D(channels, stride=2),
        )
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.BatchNorm1d(channels * 2),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, num_labels),
        )
        self.apply(_legacy_weight_init)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return one logit per canonical label."""
        features = self.stages(self.stem(inputs))
        pooled = torch.cat([self.max_pool(features), self.avg_pool(features)], dim=1)
        return self.head(torch.flatten(pooled, 1))


def _legacy_weight_init(module: nn.Module) -> None:
    """Match the explicit initialization used by the legacy benchmark."""
    if isinstance(module, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def resnet1d_wang(
    *,
    in_channels: int = 12,
    num_labels: int = 6,
    channels: int = 128,
    dropout: float = 0.5,
) -> ResNet1DWang:
    """Construct the public ``resnet1d_wang`` baseline."""
    return ResNet1DWang(in_channels, num_labels, channels, dropout)
