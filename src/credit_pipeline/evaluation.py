"""Backtesting and evaluation.

Three responsibilities:
  1. Align portfolio weights with NEXT MONTH's returns (no lookahead).
  2. Compute headline diagnostics: return, vol, Sharpe, drawdown, hit
     rate, turnover, sector exposures.
  3. Save plots and CSVs to outputs/.

Important convention: NO LOOKAHEAD
----------------------------------
The raw CSV reports `Total_Return_MTD` for the month *of* the snapshot
date. So a row with Date=2010-01-01 reports the return realised during
January 2010. To use a signal computed at the end of month T to weight
returns earned during month T+1 — the only honest way — we shift
returns by -1 within each Cusip's time series:

    realised_return_at_T  ==  panel[panel.Date == T+1].Total_Return_MTD

We do that explicitly in `_attach_forward_returns`. Bonds that drop
out of the index next month (matured, defaulted, upgraded out of HY)
get NaN for their forward return; the position contributes zero to
that month's portfolio return. This is conservative — it understates
the realised loss from defaults — and is flagged as a failure mode.

Benchmark
---------
Equal-weight return of the *eligible universe* each month — every bond
that passed our universe filters and has a forward return. Using the
panel as its own benchmark avoids needing external HY index data and
asks the right question: "did your signal beat random selection from
the same universe?"
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless / non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Forward-return alignment (no lookahead)
# ---------------------------------------------------------------------------
def _attach_forward_returns(
    portfolio: pd.DataFrame,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Attach next month's return for each (Date, Cusip) position.

    For each Cusip's time series we shift Total_Return_MTD and
    Excess_Return_MTD by -1, so the value at row T is the return that
    occurred between T and T+1. We then merge those forward returns
    onto the portfolio frame.

    Returns the portfolio with FwdTotalReturn, FwdExcessReturn, and
    NextDate columns added.
    """
    panel_keys = panel[
        ["Date", "Cusip", "Total_Return_MTD", "Excess_Return_MTD"]
    ].copy()
    panel_keys = panel_keys.sort_values(["Cusip", "Date"])
    panel_keys["NextDate"] = panel_keys.groupby("Cusip")["Date"].shift(-1)
    panel_keys["FwdTotalReturn"] = (
        panel_keys.groupby("Cusip")["Total_Return_MTD"].shift(-1)
    )
    panel_keys["FwdExcessReturn"] = (
        panel_keys.groupby("Cusip")["Excess_Return_MTD"].shift(-1)
    )

    merged = portfolio.merge(
        panel_keys[
            ["Date", "Cusip", "NextDate", "FwdTotalReturn", "FwdExcessReturn"]
        ],
        on=["Date", "Cusip"],
        how="left",
    )
    return merged


# ---------------------------------------------------------------------------
# Strategy + benchmark monthly returns
# ---------------------------------------------------------------------------
def compute_strategy_returns(
    portfolio: pd.DataFrame,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Compute monthly portfolio returns and a benchmark for comparison.

    Returns a DataFrame with one row per rebalance date:
        Date, RealizedDate,
        strat_total_ret, strat_excess_ret, invested_frac, n_holdings,
        bench_total_ret, bench_excess_ret, n_universe.

    `Date` is the snapshot date (when the signal was computed).
    `RealizedDate` is the month in which the return was actually
    earned (Date + 1 month). Plots and stats use RealizedDate.
    """
    pf = _attach_forward_returns(portfolio, panel)

    # Strategy: weighted forward return per Date
    pf["weighted_total"] = pf["weight"] * pf["FwdTotalReturn"].fillna(0.0)
    pf["weighted_excess"] = pf["weight"] * pf["FwdExcessReturn"].fillna(0.0)
    strat = (
        pf.groupby("Date")
          .agg(
              strat_total_ret=("weighted_total", "sum"),
              strat_excess_ret=("weighted_excess", "sum"),
              invested_frac=("weight", "sum"),
              n_holdings=("Cusip", "nunique"),
          )
          .reset_index()
    )

    # Benchmark: equal-weight all bonds with a forward return
    panel_sorted = panel.sort_values(["Cusip", "Date"]).copy()
    panel_sorted["FwdTotalReturn"] = (
        panel_sorted.groupby("Cusip")["Total_Return_MTD"].shift(-1)
    )
    panel_sorted["FwdExcessReturn"] = (
        panel_sorted.groupby("Cusip")["Excess_Return_MTD"].shift(-1)
    )
    bench = (
        panel_sorted.dropna(subset=["FwdTotalReturn"])
          .groupby("Date")
          .agg(
              bench_total_ret=("FwdTotalReturn", "mean"),
              bench_excess_ret=("FwdExcessReturn", "mean"),
              n_universe=("Cusip", "nunique"),
          )
          .reset_index()
    )

    out = strat.merge(bench, on="Date", how="left")
    # The return for "Date = T" is what was earned in month T+1.
    # Reindex so the row's date is when the return was realised.
    out["RealizedDate"] = out["Date"] + pd.offsets.MonthBegin(1)
    out = out.sort_values("RealizedDate").reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def compute_diagnostics(
    returns: pd.DataFrame,
    portfolio: pd.DataFrame,
) -> dict:
    """Compute headline summary stats and exposure decompositions.

    Returns a dict with annualised return/vol/Sharpe, max drawdown,
    hit rate, turnover, and a top-5 sector exposure breakdown.
    """
    r = returns.dropna(subset=["strat_total_ret"]).copy()
    if r.empty:
        return {}

    strat = r["strat_total_ret"].astype(float)
    bench = r["bench_total_ret"].astype(float)
    excess = strat - bench

    ann = 12  # monthly observations -> annualisation factor

    cum_strat = (1 + strat).cumprod()
    cum_bench = (1 + bench).cumprod()

    def _max_dd(cum: pd.Series) -> float:
        peak = cum.cummax()
        dd = cum / peak - 1.0
        return float(dd.min())

    summary = {
        "n_months": int(len(r)),
        "ann_return_strategy": float((1 + strat.mean()) ** ann - 1),
        "ann_return_benchmark": float((1 + bench.mean()) ** ann - 1),
        "ann_vol_strategy": float(strat.std(ddof=0) * np.sqrt(ann)),
        "ann_vol_benchmark": float(bench.std(ddof=0) * np.sqrt(ann)),
        "sharpe_strategy": float(
            (strat.mean() * ann)
            / (strat.std(ddof=0) * np.sqrt(ann) + 1e-12)
        ),
        "sharpe_benchmark": float(
            (bench.mean() * ann)
            / (bench.std(ddof=0) * np.sqrt(ann) + 1e-12)
        ),
        "max_dd_strategy": _max_dd(cum_strat),
        "max_dd_benchmark": _max_dd(cum_bench),
        "hit_rate_vs_bench": float((excess > 0).mean()),
        "avg_excess_ret_monthly": float(excess.mean()),
    }

    # Turnover: average L1 weight change between consecutive rebalances,
    # divided by 2 (so 100% turnover means selling everything and buying
    # a fresh portfolio).
    pf = portfolio[["Date", "Cusip", "weight"]].copy()
    wide = pf.pivot_table(
        index="Date", columns="Cusip", values="weight", fill_value=0.0
    ).sort_index()
    diffs = wide.diff().abs().sum(axis=1) / 2.0
    summary["avg_monthly_turnover"] = (
        float(diffs.iloc[1:].mean()) if len(diffs) > 1 else float("nan")
    )

    # Average sector exposure (top 5)
    sector_exposure = (
        portfolio.groupby("Class3")["weight"].sum()
        / portfolio["Date"].nunique()
    ).sort_values(ascending=False)
    summary["top_sectors"] = sector_exposure.head(5).to_dict()

    return summary


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def make_plots(
    returns: pd.DataFrame,
    portfolio: pd.DataFrame,
    outputs_dir: Path,
) -> list[Path]:
    """Save four diagnostic charts as PNGs. Returns the list of paths."""
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    r = returns.dropna(subset=["strat_total_ret"]).copy()
    r = r.set_index("RealizedDate")

    # 1. Cumulative return vs benchmark
    fig, ax = plt.subplots(figsize=(9, 4.5))
    (1 + r["strat_total_ret"]).cumprod().plot(ax=ax, label="Strategy")
    (1 + r["bench_total_ret"]).cumprod().plot(
        ax=ax, label="Equal-weight universe"
    )
    ax.set_title("Cumulative total return")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = outputs_dir / "01_cumulative_return.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)

    # 2. Drawdown
    fig, ax = plt.subplots(figsize=(9, 3.5))
    cum = (1 + r["strat_total_ret"]).cumprod()
    dd = cum / cum.cummax() - 1.0
    dd.plot(ax=ax, color="firebrick")
    ax.fill_between(dd.index, dd.values, 0, color="firebrick", alpha=0.25)
    ax.set_title("Strategy drawdown")
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = outputs_dir / "02_drawdown.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)

    # 3. Sector exposure over time (stacked area)
    sector_t = (
        portfolio.groupby(["Date", "Class3"])["weight"].sum()
        .unstack(fill_value=0.0)
        .sort_index()
    )
    top_sectors = sector_t.mean().sort_values(ascending=False).head(8).index
    plot_df = sector_t[top_sectors].copy()
    plot_df["Other"] = sector_t.drop(columns=top_sectors).sum(axis=1)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    plot_df.plot.area(ax=ax, alpha=0.8, linewidth=0)
    ax.set_title("Sector exposure over time (top 8 sectors + Other)")
    ax.set_ylabel("Weight")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = outputs_dir / "03_sector_exposure.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # 4. Turnover
    pf = portfolio[["Date", "Cusip", "weight"]].copy()
    wide = pf.pivot_table(
        index="Date", columns="Cusip", values="weight", fill_value=0.0
    ).sort_index()
    turnover = wide.diff().abs().sum(axis=1) / 2.0
    fig, ax = plt.subplots(figsize=(9, 3.5))
    turnover.iloc[1:].plot(ax=ax, color="navy")
    ax.set_title("Monthly turnover (one-way)")
    ax.set_ylabel("Turnover")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = outputs_dir / "04_turnover.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)

    return paths


# ---------------------------------------------------------------------------
# Save results to disk
# ---------------------------------------------------------------------------
def save_results(
    returns: pd.DataFrame,
    portfolio: pd.DataFrame,
    diagnostics: dict,
    outputs_dir: Path,
) -> dict[str, Path]:
    """Persist returns, latest holdings, and diagnostics summary."""
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    p_ret = outputs_dir / "results_returns.csv"
    returns.to_csv(p_ret, index=False)
    paths["returns_csv"] = p_ret

    # Latest holdings (more useful than the full panel for review)
    final_date = portfolio["Date"].max()
    p_hold = outputs_dir / "results_holdings_latest.csv"
    portfolio[portfolio["Date"] == final_date].sort_values(
        "weight", ascending=False
    ).to_csv(p_hold, index=False)
    paths["holdings_csv"] = p_hold

    # Diagnostics summary as a one-line CSV (nested top_sectors flattened)
    flat = {k: v for k, v in diagnostics.items() if k != "top_sectors"}
    flat.update({
        f"top_sector__{s}": w
        for s, w in (diagnostics.get("top_sectors") or {}).items()
    })
    p_diag = outputs_dir / "results_diagnostics.csv"
    pd.DataFrame([flat]).to_csv(p_diag, index=False)
    paths["diagnostics_csv"] = p_diag

    return paths
