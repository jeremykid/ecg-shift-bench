"""XResNet-style 1D backbone for lead-first ECG tensors."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _conv1d(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    *,
    stride: int = 1,
    groups: int = 1,
) -> nn.Conv1d:
    return nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=(kernel_size - 1) // 2,
        groups=groups,
        bias=False,
    )


class XResNetBottleneck1D(nn.Module):
    """Bottleneck residual block used by the XResNet-style encoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        expansion: int = 4,
    ) -> None:
        super().__init__()
        if out_channels % expansion != 0:
            raise ValueError("out_channels must be divisible by expansion")
        mid_channels = out_channels // expansion
        self.conv1 = _conv1d(in_channels, mid_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(mid_channels)
        self.conv2 = _conv1d(mid_channels, mid_channels, kernel_size=3, stride=stride)
        self.bn2 = nn.BatchNorm1d(mid_channels)
        self.conv3 = _conv1d(mid_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Sequential(
                _conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the residual bottleneck."""
        residual = self.shortcut(inputs)
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.relu(self.bn2(self.conv2(outputs)))
        outputs = self.bn3(self.conv3(outputs))
        return self.relu(outputs + residual)


def _make_stage(
    in_channels: int,
    out_channels: int,
    *,
    blocks: int,
    stride: int,
    expansion: int,
) -> nn.Sequential:
    if blocks <= 0:
        raise ValueError("blocks must be positive")
    stages: list[nn.Module] = [
        XResNetBottleneck1D(
            in_channels,
            out_channels,
            stride=stride,
            expansion=expansion,
        )
    ]
    for _ in range(blocks - 1):
        stages.append(
            XResNetBottleneck1D(
                out_channels,
                out_channels,
                stride=1,
                expansion=expansion,
            )
        )
    return nn.Sequential(*stages)


class XResNet1D(nn.Module):
    """Compact XResNet-style backbone for 12-lead ECG classification."""

    def __init__(
        self,
        in_channels: int = 12,
        num_labels: int = 6,
        channels: int = 64,
        layers: Sequence[int] = (2, 2, 2, 2),
        expansion: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if len(tuple(layers)) != 4:
            raise ValueError("layers must contain exactly four stage depths")
        stage_depths = tuple(int(layer) for layer in layers)
        stage_channels = (
            channels,
            channels * 2,
            channels * 4,
            channels * 8,
        )

        self.stem = nn.Sequential(
            _conv1d(in_channels, channels, kernel_size=7, stride=2),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            _conv1d(channels, channels, kernel_size=3),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            _conv1d(channels, channels, kernel_size=3),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.stages = nn.Sequential(
            _make_stage(
                stage_channels[0],
                stage_channels[0],
                blocks=stage_depths[0],
                stride=1,
                expansion=expansion,
            ),
            _make_stage(
                stage_channels[0],
                stage_channels[1],
                blocks=stage_depths[1],
                stride=2,
                expansion=expansion,
            ),
            _make_stage(
                stage_channels[1],
                stage_channels[2],
                blocks=stage_depths[2],
                stride=2,
                expansion=expansion,
            ),
            _make_stage(
                stage_channels[2],
                stage_channels[3],
                blocks=stage_depths[3],
                stride=2,
                expansion=expansion,
            ),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.LayerNorm(stage_channels[-1]),
            nn.Dropout(dropout),
            nn.Linear(stage_channels[-1], num_labels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return one logit per label."""
        features = self.stem(inputs)
        features = self.stages(features)
        pooled = self.pool(features).squeeze(-1)
        return self.head(pooled)


def xresnet1d(
    *,
    in_channels: int = 12,
    num_labels: int = 6,
    channels: int = 64,
    layers: Sequence[int] = (2, 2, 2, 2),
    expansion: int = 4,
    dropout: float = 0.0,
) -> XResNet1D:
    """Construct the public XResNet-style 1D backbone."""
    return XResNet1D(
        in_channels=in_channels,
        num_labels=num_labels,
        channels=channels,
        layers=layers,
        expansion=expansion,
        dropout=dropout,
    )
