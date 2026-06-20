"""SPH ECG loader skeleton.

Expected: a normalized ``metadata.csv`` index and waveform files under
``records/``. The raw SPH release may require an indexing conversion step.
"""

from ecg_shift_bench.datasets._tabular import TabularSkeletonDataset


class SPHDataset(TabularSkeletonDataset):
    """Metadata-aware SPH dataset adapter."""

    name = "SPH"
    domain = "sph_china"
    metadata_file = "metadata.csv"
