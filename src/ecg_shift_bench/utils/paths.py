"""Project-relative path resolution."""

from pathlib import Path


def project_root() -> Path:
    """Return repository root based on the installed source file location."""
    return Path(__file__).resolve().parents[3]


def resolve_project_path(path: str | Path) -> Path:
    """Resolve relative configuration paths against the repository root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root() / candidate
