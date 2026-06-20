"""Tests for the canonical label contract."""

import pytest

from ecg_shift_bench.labels.canonical import CANONICAL_LABELS, LABEL_DESCRIPTIONS
from ecg_shift_bench.labels.harmonize import harmonize_labels
from ecg_shift_bench.labels.mappings import LABEL_MAP


def test_canonical_labels_are_stable() -> None:
    assert CANONICAL_LABELS == ["AF", "RBBB", "LBBB", "1dAVB", "SB", "ST"]
    assert set(LABEL_DESCRIPTIONS) == set(CANONICAL_LABELS)
    assert LABEL_DESCRIPTIONS["ST"] == "Sinus tachycardia"


@pytest.mark.parametrize(
    ("dataset", "raw", "expected"),
    [
        ("PTBXL", ["AFIB", "IRBBB", "STACH"], {"AF", "RBBB", "ST"}),
        ("CHAPMAN", ["LBBB", "SB"], {"LBBB", "SB"}),
        ("SPH", ["50+347", "82", "21"], {"AF", "1dAVB", "ST"}),
        ("CODE15", ["1dAVb", "RBBB"], {"1dAVB", "RBBB"}),
    ],
)
def test_mapping_for_every_dataset(dataset: str, raw: list[str], expected: set[str]) -> None:
    result = harmonize_labels(raw, dataset)
    assert set(result) == set(CANONICAL_LABELS)
    assert {label for label, value in result.items() if value} == expected
    assert set(result.values()) <= {0, 1}


def test_every_mapping_has_all_canonical_labels() -> None:
    for mapping in LABEL_MAP.values():
        assert list(mapping) == CANONICAL_LABELS


def test_unknown_dataset_is_rejected() -> None:
    with pytest.raises(KeyError, match="Unknown dataset"):
        harmonize_labels(["AF"], "unknown")
