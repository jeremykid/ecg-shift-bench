"""Multi-source supervised training extension point."""

from ecg_shift_bench.methods.source_only import SourceOnly


class MultiSource(SourceOnly):
    """Source-only loss applied to domain-balanced batches.

    TODO: implement and validate domain-balanced sampling and per-domain logging.
    """
