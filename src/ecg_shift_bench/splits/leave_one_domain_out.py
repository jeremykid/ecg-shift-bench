"""Leave-one-domain-out protocol construction."""

import pandas as pd

from ecg_shift_bench.splits.domain_split import split_by_domain


def leave_one_domain_out(
    metadata: pd.DataFrame,
    held_out_domain: str,
    *,
    domain_column: str = "domain",
) -> dict[str, pd.DataFrame]:
    """Use all non-held-out domains as sources and one domain as target test data.

    The returned target frame contains labels for evaluation storage. Training
    code is responsible for never exposing these labels to the model or method.
    """
    if domain_column not in metadata:
        raise KeyError(f"Missing domain column: {domain_column}")
    domains = list(metadata[domain_column].drop_duplicates())
    if held_out_domain not in domains:
        raise ValueError(f"Unknown held-out domain {held_out_domain!r}")
    source_domains = [domain for domain in domains if domain != held_out_domain]
    split = split_by_domain(
        metadata,
        source_domains,
        [held_out_domain],
        domain_column=domain_column,
    )
    return {"train": split["source"], "test": split["target"]}
