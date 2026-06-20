"""Utilities for converting native labels into the benchmark label space."""

from collections.abc import Iterable

from ecg_shift_bench.labels.canonical import CANONICAL_LABELS
from ecg_shift_bench.labels.mappings import LABEL_MAP


def harmonize_labels(raw_labels: list[str], dataset_name: str) -> dict[str, int]:
    """Return canonical multi-hot labels for one record.

    Matching is case-sensitive because native code capitalization may carry
    meaning. Surrounding whitespace is ignored and duplicate codes are harmless.

    Args:
        raw_labels: Native diagnostic/rhythm codes for a record.
        dataset_name: Dataset key, matched case-insensitively.

    Raises:
        KeyError: If no mapping is registered for ``dataset_name``.
    """
    key = dataset_name.strip().upper()
    if key not in LABEL_MAP:
        supported = ", ".join(sorted(LABEL_MAP))
        raise KeyError(f"Unknown dataset {dataset_name!r}; supported datasets: {supported}")

    observed = {str(label).strip() for label in raw_labels}
    mapping = LABEL_MAP[key]
    return {
        label: int(bool(observed.intersection(mapping[label])))
        for label in CANONICAL_LABELS
    }


def labels_to_vector(labels: dict[str, int], order: Iterable[str] = CANONICAL_LABELS) -> list[int]:
    """Convert a canonical label dictionary to an ordered multi-hot vector."""
    return [int(labels[label]) for label in order]
