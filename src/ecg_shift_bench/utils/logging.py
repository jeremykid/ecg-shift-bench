"""Consistent console logging setup."""

import logging


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the package logger."""
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("ecg_shift_bench")
