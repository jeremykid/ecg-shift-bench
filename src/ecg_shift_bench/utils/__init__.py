"""Configuration, paths, logging, and reproducibility utilities."""

__all__ = ["load_yaml", "seed_everything"]


def __getattr__(name: str):
    """Lazily expose utility helpers with optional runtime dependencies."""
    if name == "load_yaml":
        from ecg_shift_bench.utils.config import load_yaml

        globals()["load_yaml"] = load_yaml
        return load_yaml
    if name == "seed_everything":
        from ecg_shift_bench.utils.seed import seed_everything

        globals()["seed_everything"] = seed_everything
        return seed_everything
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
