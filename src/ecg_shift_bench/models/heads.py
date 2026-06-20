"""Prediction heads shared by ECG encoders."""

from torch import nn


class MultiLabelHead(nn.Module):
    """Linear logit head for independent binary labels."""

    def __init__(self, input_features: int, num_labels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_features, num_labels)

    def forward(self, features):  # type annotation omitted for TorchScript friendliness
        """Return unnormalized logits."""
        return self.linear(features)
