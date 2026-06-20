"""Domain-shift summary metrics."""


def domain_gap(in_domain_score: float, out_domain_score: float) -> float:
    """Return performance degradation: in-domain minus out-of-domain score."""
    return float(in_domain_score - out_domain_score)
