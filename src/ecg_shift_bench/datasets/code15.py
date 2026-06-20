"""CODE-15% loader skeleton.

Expected: ``annotations.csv`` and HDF5 tracings under ``tracings/``. Release-
specific HDF5 key and lead conventions must be validated before signal I/O.
"""

from ecg_shift_bench.datasets._tabular import TabularSkeletonDataset


class CODE15Dataset(TabularSkeletonDataset):
    """Metadata-aware CODE-15% dataset adapter."""

    name = "CODE15"
    domain = "code15_brazil"
    metadata_file = "annotations.csv"
