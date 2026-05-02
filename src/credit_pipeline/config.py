"""Configuration loading.

Single source of truth: every tunable parameter lives in YAML, not in
code. This module loads the YAML into a typed wrapper and resolves
relative paths against the project root.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Lightweight typed wrapper around the raw YAML dict.

    Storing the original `raw` dict alongside typed accessors lets new
    keys be added in YAML and read via `cfg.raw[...]` without breaking
    the typed interface. Avoids the maintenance burden of pydantic for
    a small project.
    """

    raw: dict[str, Any]
    project_root: Path

    @property
    def run_name(self) -> str:
        return self.raw.get("run_name", "default")

    @property
    def data(self) -> dict[str, Any]:
        return self.raw["data"]

    @property
    def backtest(self) -> dict[str, Any]:
        return self.raw["backtest"]

    @property
    def signal(self) -> dict[str, Any]:
        return self.raw["signal"]

    @property
    def portfolio(self) -> dict[str, Any]:
        return self.raw["portfolio"]

    @property
    def evaluation(self) -> dict[str, Any]:
        return self.raw["evaluation"]

    @property
    def logging_cfg(self) -> dict[str, Any]:
        return self.raw.get("logging", {"level": "INFO"})

    def resolve(self, relative_path: str) -> Path:
        """Resolve a path relative to the project root."""
        return (self.project_root / relative_path).resolve()


def find_project_root(start: Path | None = None) -> Path:
    """Walk up from `start` until we find configs/ + src/.

    Lets the pipeline run from any working directory or notebook
    without fragile relative paths.
    """
    here = (start or Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        if (parent / "configs").is_dir() and (parent / "src").is_dir():
            return parent
    # Fallback: assume two levels up from this file
    # (src/credit_pipeline/config.py -> project root)
    return Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path = "configs/default.yaml") -> Config:
    """Load YAML config, resolving paths against the project root."""
    project_root = find_project_root()
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = project_root / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r") as f:
        raw = yaml.safe_load(f)
    logger.info(
        "Loaded config from %s (run=%s)", cfg_path, raw.get("run_name")
    )
    return Config(raw=raw, project_root=project_root)
