"""Tests for patient and domain leakage controls."""

import pandas as pd
import pytest

from ecg_shift_bench.splits.domain_split import split_by_domain
from ecg_shift_bench.splits.leave_one_domain_out import leave_one_domain_out
from ecg_shift_bench.splits.patient_split import assert_no_patient_overlap, patient_level_split


def test_patient_split_has_no_leakage_and_is_deterministic() -> None:
    metadata = pd.DataFrame(
        {
            "record_id": [f"r{i}" for i in range(20)],
            "patient_id": [f"p{i // 2}" for i in range(20)],
        }
    )
    first = patient_level_split(metadata, seed=7)
    second = patient_level_split(metadata, seed=7)
    assert_no_patient_overlap(first)
    assert sum(len(frame) for frame in first.values()) == len(metadata)
    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name])


def test_domain_split_and_leave_one_out() -> None:
    metadata = pd.DataFrame(
        {
            "record_id": ["a", "b", "c", "d"],
            "patient_id": ["p1", "p2", "p3", "p4"],
            "domain": ["A", "A", "B", "C"],
        }
    )
    split = split_by_domain(metadata, ["A", "B"], ["C"])
    assert set(split["source"]["domain"]) == {"A", "B"}
    assert set(split["target"]["domain"]) == {"C"}
    lodo = leave_one_domain_out(metadata, "B")
    assert set(lodo["train"]["domain"]) == {"A", "C"}
    assert set(lodo["test"]["domain"]) == {"B"}


def test_overlapping_source_target_domains_are_rejected() -> None:
    metadata = pd.DataFrame({"domain": ["A", "B"]})
    with pytest.raises(ValueError, match="overlap"):
        split_by_domain(metadata, ["A"], ["A"])
