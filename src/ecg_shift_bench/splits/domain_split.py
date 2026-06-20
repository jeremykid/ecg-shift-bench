"""Domain-based source/target partitioning."""

import pandas as pd


def split_by_domain(
    metadata: pd.DataFrame,
    source_domains: list[str],
    target_domains: list[str],
    *,
    domain_column: str = "domain",
) -> dict[str, pd.DataFrame]:
    """Partition records into disjoint source and target domains."""
    if domain_column not in metadata:
        raise KeyError(f"Missing domain column: {domain_column}")
    overlap = set(source_domains).intersection(target_domains)
    if overlap:
        raise ValueError(f"Source and target domains overlap: {sorted(overlap)}")
    observed = set(metadata[domain_column].unique())
    missing = (set(source_domains) | set(target_domains)).difference(observed)
    if missing:
        raise ValueError(f"Requested domains are absent from metadata: {sorted(missing)}")
    return {
        "source": metadata.loc[metadata[domain_column].isin(source_domains)].copy(),
        "target": metadata.loc[metadata[domain_column].isin(target_domains)].copy(),
    }
