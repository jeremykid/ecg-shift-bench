"""Canonical label definitions and dataset harmonization."""

from ecg_shift_bench.labels.canonical import CANONICAL_LABELS, LABEL_DESCRIPTIONS
from ecg_shift_bench.labels.harmonize import harmonize_labels

__all__ = ["CANONICAL_LABELS", "LABEL_DESCRIPTIONS", "harmonize_labels"]
