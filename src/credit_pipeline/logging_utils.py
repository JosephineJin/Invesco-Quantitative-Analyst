"""Centralised logging setup.

Call `setup_logging(cfg)` once at the entry point. Every module then
uses `logger = logging.getLogger(__name__)` and gets consistent
formatting and level. Idempotent: clears existing handlers so
re-running in a notebook doesn't produce duplicate log lines.
"""

from __future__ import annotations

import logging
import sys
from typing import Any


def setup_logging(cfg: dict[str, Any] | None = None) -> None:
    """Configure the root logger from a config dict.

    Expected keys (all optional):
        level  : "DEBUG" / "INFO" / "WARNING" (default "INFO")
        format : log line format string
    """
    cfg = cfg or {}
    level_name = cfg.get("level", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = cfg.get(
        "format",
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    root = logging.getLogger()
    # Remove any pre-existing handlers (notebook-friendly)
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(handler)
    root.setLevel(level)

    # Quiet very chatty third-party libraries
    for noisy in ("urllib3", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
