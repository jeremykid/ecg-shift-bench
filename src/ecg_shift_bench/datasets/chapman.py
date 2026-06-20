"""Chapman-Shaoxing/Ningbo loader skeleton.

Expected: ``Diagnostics.xlsx`` and waveform CSV files under ``ECGData/``.
The public release layout should be verified before signal I/O is finalized.
"""

from ecg_shift_bench.datasets._tabular import TabularSkeletonDataset


class ChapmanDataset(TabularSkeletonDataset):
    """Metadata-aware Chapman-Shaoxing/Ningbo adapter."""

    name = "CHAPMAN"
    domain = "chapman_ningbo_china"
    metadata_file = "Diagnostics.xlsx"
