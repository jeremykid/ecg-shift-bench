"""1D ResNet baseline for 12-lead ECG classification.

Architecture adapted from:
    He et al. "Deep Residual Learning for Image Recognition." CVPR 2016.
    https://arxiv.org/abs/1512.03385

Modified for 1-D (temporal) convolutions suitable for ECG signals.
"""

from __future__ import annotations

from typing import List, Optional, Type, Union

import torch
import torch.nn as nn


class BasicBlock1D(nn.Module):
    """Basic residual block for 1-D signals.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Convolution stride.
        downsample: Optional downsampling layer applied to the shortcut.
        dropout: Dropout probability (applied between the two convolutions).
    """

    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=7, stride=1, padding=3, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class ResNet1D(nn.Module):
    """1-D ResNet for multi-label 12-lead ECG classification.

    Args:
        num_leads: Number of ECG input channels (typically 12).
        num_classes: Number of output classes.
        layers: Number of residual blocks per stage, e.g. ``[2, 2, 2, 2]``
            for ResNet-18 or ``[3, 4, 6, 3]`` for ResNet-34.
        base_channels: Width of the first convolutional stage.
        dropout: Dropout probability applied inside residual blocks.
    """

    def __init__(
        self,
        num_leads: int = 12,
        num_classes: int = 5,
        layers: Optional[List[int]] = None,
        base_channels: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if layers is None:
            layers = [2, 2, 2, 2]

        self.in_channels = base_channels

        self.stem = nn.Sequential(
            nn.Conv1d(num_leads, base_channels, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(base_channels, layers[0], stride=1, dropout=dropout)
        self.layer2 = self._make_layer(base_channels * 2, layers[1], stride=2, dropout=dropout)
        self.layer3 = self._make_layer(base_channels * 4, layers[2], stride=2, dropout=dropout)
        self.layer4 = self._make_layer(base_channels * 8, layers[3], stride=2, dropout=dropout)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(base_channels * 8, num_classes)

        self._init_weights()

    def _make_layer(
        self,
        out_channels: int,
        num_blocks: int,
        stride: int = 1,
        dropout: float = 0.0,
    ) -> nn.Sequential:
        downsample: Optional[nn.Module] = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

        blocks = [
            BasicBlock1D(self.in_channels, out_channels, stride=stride, downsample=downsample, dropout=dropout)
        ]
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            blocks.append(BasicBlock1D(out_channels, out_channels, dropout=dropout))

        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(batch, num_leads, signal_length)``.

        Returns:
            Logits tensor of shape ``(batch, num_classes)``.
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


def resnet1d_18(num_leads: int = 12, num_classes: int = 5, **kwargs) -> ResNet1D:
    """Construct a 1-D ResNet-18 model.

    Args:
        num_leads: Number of ECG input channels.
        num_classes: Number of output classes.
        **kwargs: Additional arguments forwarded to :class:`ResNet1D`.
    """
    return ResNet1D(num_leads=num_leads, num_classes=num_classes, layers=[2, 2, 2, 2], **kwargs)


def resnet1d_34(num_leads: int = 12, num_classes: int = 5, **kwargs) -> ResNet1D:
    """Construct a 1-D ResNet-34 model.

    Args:
        num_leads: Number of ECG input channels.
        num_classes: Number of output classes.
        **kwargs: Additional arguments forwarded to :class:`ResNet1D`.
    """
    return ResNet1D(num_leads=num_leads, num_classes=num_classes, layers=[3, 4, 6, 3], **kwargs)
