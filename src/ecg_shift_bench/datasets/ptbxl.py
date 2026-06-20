"""PTB-XL loader skeleton.

Expected: ``ptbxl_database.csv`` plus WFDB records in ``records500/`` (or
``records100/``). Signal loading remains a TODO until lead ordering and units are
validated against an explicitly supported PTB-XL release.
"""

from ecg_shift_bench.datasets._tabular import TabularSkeletonDataset


class PTBXLDataset(TabularSkeletonDataset):
    """Metadata-aware PTB-XL dataset adapter."""

    name = "PTBXL"
    domain = "ptbxl_germany"
    metadata_file = "ptbxl_database.csv"
