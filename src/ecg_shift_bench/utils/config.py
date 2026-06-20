"""Small YAML configuration helpers."""

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping and reject empty or non-mapping configs."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")
    return config


def require_keys(config: dict[str, Any], keys: list[str], context: str = "config") -> None:
    """Validate required top-level keys."""
    missing = [key for key in keys if key not in config]
    if missing:
        raise ValueError(f"{context} is missing required keys: {missing}")
