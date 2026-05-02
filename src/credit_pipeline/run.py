"""End-to-end pipeline runner.

Single-command entry point:

    python -m credit_pipeline.run

Optional flags:

    python -m credit_pipeline.run --config configs/default.yaml
    python -m credit_pipeline.run --force-rebuild

Steps:
    1. Load config + setup logging
    2. Ingest bond panel (Parquet cache) + macro series (FRED, with fallback)
    3. Build signal + regime overlay
    4. Construct portfolio
    5. Compute returns + diagnostics
    6. Save outputs (CSVs + PNG plots)
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from .config import load_config
from .data import load_bond_panel, load_macro
from .evaluation import (
    compute_diagnostics,
    compute_strategy_returns,
    make_plots,
    save_results,
)
from .features import apply_regime_tilt, compute_regime, compute_signal
from .logging_utils import setup_logging
from .portfolio import PortfolioConstraints, construct_portfolio


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the systematic credit pipeline."
    )
    p.add_argument(
        "--config", default="configs/default.yaml",
        help="Path to YAML config (relative to project root).",
    )
    p.add_argument(
        "--force-rebuild", action="store_true",
        help="Ignore Parquet caches and re-read raw inputs.",
    )
    return p.parse_args()


def run(
    config_path: str = "configs/default.yaml",
    force_rebuild: bool = False,
) -> dict:
    """Run the full pipeline. Returns a summary dict."""
    cfg = load_config(config_path)
    setup_logging(cfg.logging_cfg)
    log = logging.getLogger("credit_pipeline.run")
    log.info("=== Pipeline run: %s ===", cfg.run_name)

    # --- 1. Ingest -------------------------------------------------------
    raw_files = [cfg.resolve(p) for p in cfg.data["raw_files"]]
    panel = load_bond_panel(
        raw_files=raw_files,
        processed_path=cfg.resolve(cfg.data["processed_path"]),
        force_rebuild=force_rebuild,
    )

    # Apply backtest date window
    bt = cfg.backtest
    start = pd.to_datetime(bt["start_date"])
    end = pd.to_datetime(bt["end_date"])
    panel = panel[(panel["Date"] >= start) & (panel["Date"] <= end)].copy()
    log.info(
        "Backtest window: %s to %s (%d months)",
        start.date(), end.date(), panel["Date"].nunique(),
    )

    macro = load_macro(
        series=cfg.data["fred_series"],
        macro_path=cfg.resolve(cfg.data["macro_path"]),
        force_rebuild=force_rebuild,
    )

    # --- 2. Signal + regime ----------------------------------------------
    panel = compute_signal(
        panel,
        group_by=cfg.signal["group_by"],
        winsorize_pct=cfg.signal["winsorize_pct"],
    )
    regime = compute_regime(
        macro, panel, threshold=cfg.signal["regime_threshold"]
    )
    panel = apply_regime_tilt(panel, regime)

    # --- 3. Portfolio ----------------------------------------------------
    constraints = PortfolioConstraints(
        top_quantile=cfg.portfolio["top_quantile"],
        weighting=cfg.portfolio["weighting"],
        max_issuer_weight=cfg.portfolio["max_issuer_weight"],
        max_sector_weight=cfg.portfolio["max_sector_weight"],
        duration_target=cfg.portfolio.get("duration_target"),
        duration_tolerance=cfg.portfolio.get("duration_tolerance", 0.5),
    )
    portfolio = construct_portfolio(panel, bt, constraints)

    # --- 4. Evaluation ---------------------------------------------------
    returns = compute_strategy_returns(portfolio, panel)
    diagnostics = compute_diagnostics(returns, portfolio)

    log.info("Headline diagnostics:")
    for k, v in diagnostics.items():
        if k == "top_sectors":
            continue
        log.info("  %-26s %s", k, v)

    out_dir = cfg.resolve(cfg.evaluation["outputs_dir"])
    saved = save_results(returns, portfolio, diagnostics, out_dir)
    if cfg.evaluation.get("save_plots", True):
        make_plots(returns, portfolio, out_dir)

    log.info("Outputs written to %s", out_dir)
    return {
        "panel_rows": len(panel),
        "portfolio_rows": len(portfolio),
        "returns_rows": len(returns),
        "diagnostics": diagnostics,
        "outputs": {k: str(v) for k, v in saved.items()},
    }


def main() -> None:
    args = parse_args()
    run(config_path=args.config, force_rebuild=args.force_rebuild)


if __name__ == "__main__":
    main()
