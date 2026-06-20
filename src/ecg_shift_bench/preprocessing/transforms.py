"""Composable signal transforms."""

from collections.abc import Callable, Sequence

import numpy as np


class Compose:
    """Apply signal transforms sequentially."""

    def __init__(self, transforms: Sequence[Callable[[np.ndarray], np.ndarray]]) -> None:
        self.transforms = list(transforms)

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        for transform in self.transforms:
            signal = transform(signal)
        return signal
