"""Stable dataset identifiers for the discriminator study."""

from __future__ import annotations

DATASET_ID_ORDER = ["PTBXL", "CODE15", "CHAPMAN", "SPH"]

DATASET_ID_TO_INDEX = {name: index for index, name in enumerate(DATASET_ID_ORDER)}

DATASET_NAME_ALIASES = {
    "PTB-XL": "PTBXL",
    "PTBXL": "PTBXL",
    "CODE-15": "CODE15",
    "CODE15": "CODE15",
    "CHAPMAN": "CHAPMAN",
    "SPH": "SPH",
}


def canonical_dataset_name(name: str) -> str:
    """Return the canonical dataset name used by the discriminator study."""
    key = str(name).strip().upper().replace("_", "-")
    if key not in DATASET_NAME_ALIASES:
        supported = ", ".join(DATASET_ID_ORDER)
        raise KeyError(f"Unknown dataset {name!r}; supported datasets: {supported}")
    return DATASET_NAME_ALIASES[key]


def dataset_index(name: str) -> int:
    """Return the integer dataset ID for one canonical dataset name."""
    canonical = canonical_dataset_name(name)
    return DATASET_ID_TO_INDEX[canonical]


def selected_dataset_names(names: list[str] | tuple[str, ...]) -> list[str]:
    """Return canonical dataset names in the shared study order."""
    canonical = [canonical_dataset_name(name) for name in names]
    ordered = [name for name in DATASET_ID_ORDER if name in canonical]
    if len(ordered) != len(set(canonical)):
        supported = ", ".join(DATASET_ID_ORDER)
        raise KeyError(f"Expected unique datasets drawn from: {supported}")
    return ordered
