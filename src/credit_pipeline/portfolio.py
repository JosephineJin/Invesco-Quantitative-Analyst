"""Portfolio construction.

Long-only, monthly rebalanced. For each rebalance date we:

  1. Filter the universe (maturity bounds, valid OAS/OAD, valid signal).
  2. Select the top-quantile bonds by tilted signal score.
  3. Assign initial weights (equal or signal-proportional).
  4. Apply issuer cap, sector cap, and (soft) duration target.
  5. Renormalise to sum to 1.

Why water-filling instead of an optimiser?
------------------------------------------
The cap step is the only non-trivial piece. The naive approach — cap
breaching groups, divide by sum, done — is wrong: dividing by a sub-1
total inflates the surviving weights, which can push them back over
the cap. Classic rebound problem.

Water-filling: cap breaching groups exactly to the cap, then
redistribute the freed weight ONLY to groups currently below the cap,
proportional to their existing weight. Repeat until stable. This
preserves the sum-to-1 invariant throughout and converges in 2-3
iterations for normal cases.

A QP solver (cvxpy, scipy.optimize) would solve all caps + duration
target jointly and give a marginally better solution. But it adds a
dependency, a solver-failure mode, and ~5x latency for a long-only
top-quintile portfolio with loose caps. Water-filling is the right
pragmatic choice at this scope.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constraints container
# ---------------------------------------------------------------------------
@dataclass
class PortfolioConstraints:
    """All portfolio-construction parameters in one place."""
    top_quantile: float = 0.20            # top 20% by signal
    weighting: str = "equal"              # "equal" or "signal"
    max_issuer_weight: float = 0.02       # 2% per Ticker
    max_sector_weight: float = 0.20       # 20% per Class3
    duration_target: float | None = 2.0       # matches universe; see MEMO
    duration_tolerance: float = 1.0       # +/- this many years from target


# ---------------------------------------------------------------------------
# Universe filter
# ---------------------------------------------------------------------------
def _filter_universe(
    snap: pd.DataFrame,
    min_ttm: float,
    max_ttm: float,
    drop_missing_oas: bool,
    drop_missing_oad: bool,
) -> pd.DataFrame:
    """Apply per-snapshot eligibility rules."""
    f = snap.copy()
    f = f[f["Years_To_Maturity"].between(min_ttm, max_ttm)]
    if drop_missing_oas:
        f = f[f["OAS"].notna()]
    if drop_missing_oad:
        f = f[f["OAD"].notna()]
    f = f[f["signal_tilted"].notna()]
    return f


# ---------------------------------------------------------------------------
# Initial weighting
# ---------------------------------------------------------------------------
def _initial_weights(selected: pd.DataFrame, weighting: str) -> pd.Series:
    """Return a weight Series indexed by Cusip, summing to 1."""
    if weighting == "equal":
        w = pd.Series(1.0 / len(selected), index=selected["Cusip"].values)
    elif weighting == "signal":
        # Translate signal so the smallest selected score is 0+epsilon,
        # then normalise. Avoids zero or negative weights.
        s = selected["signal_tilted"].values
        s_pos = s - s.min() + 1e-9
        w = pd.Series(s_pos / s_pos.sum(), index=selected["Cusip"].values)
    else:
        raise ValueError(f"Unknown weighting scheme: {weighting!r}")
    return w


# ---------------------------------------------------------------------------
# Water-filling cap
# ---------------------------------------------------------------------------
def _waterfill_cap(
    weights: pd.Series,
    group_key: pd.Series,
    cap: float,
) -> pd.Series:
    """Apply a per-group weight cap by water-filling.

    For any group whose total exceeds `cap`:
        - scale down its in-group weights so the group total equals `cap`
    Then redistribute the freed weight to the groups still below cap,
    pro-rata to their current weight. Repeat until no group exceeds cap.

    Preserves the invariant: weights sum to 1 throughout — UNLESS the
    cap is infeasible (number_of_distinct_groups * cap < 1.0). In that
    case we log a warning, return weights capped at `cap` per group
    (which sum to less than 1), and let the caller decide what to do.

    Infeasible example: 5 groups, cap=0.15 -> max achievable = 0.75.
    """
    if cap >= 1.0 or weights.empty:
        return weights

    # Check feasibility up front
    n_groups = group_key.nunique()
    if n_groups * cap < 1.0 - 1e-9:
        logger.warning(
            "Cap %.4f infeasible: only %d distinct groups (max achievable %.4f). "
            "Returning capped weights that sum to < 1.",
            cap, n_groups, n_groups * cap,
        )
        # Cap each group exactly to `cap`, accept partial investment
        group_tot = weights.groupby(group_key).transform("sum")
        scale = (cap / group_tot).clip(upper=1.0)
        return weights * scale

    w = weights.copy().astype(float)
    for _ in range(20):
        group_tot = w.groupby(group_key).transform("sum")
        over = group_tot > cap + 1e-12
        if not over.any():
            break

        # Step 1: scale down over-cap groups exactly to the cap
        scale = pd.Series(1.0, index=w.index)
        over_idx = over[over].index
        scale.loc[over_idx] = (cap / group_tot.loc[over_idx]).values
        w_capped = w * scale

        capped_total = w_capped.sum()
        deficit = 1.0 - capped_total
        if deficit <= 1e-12:
            w = w_capped
            break

        # Step 2: redistribute deficit to bonds in below-cap groups,
        # proportional to their current weight
        below_mask = ~over
        below_weight = w_capped[below_mask].sum()
        if below_weight <= 0:
            # Should be unreachable now that we check feasibility above,
            # but kept defensively.
            logger.warning("Water-fill cannot redistribute; capping in place")
            w = w_capped
            break

        boost = pd.Series(0.0, index=w.index)
        boost[below_mask] = w_capped[below_mask] * (deficit / below_weight)
        w = w_capped + boost
    return w


def _apply_caps(
    weights: pd.Series,
    selected: pd.DataFrame,
    max_issuer: float,
    max_sector: float,
) -> pd.Series:
    """Apply issuer cap then sector cap then issuer cap again.

    The second issuer pass cleans up cases where redistributing freed
    sector weight piled too much weight back onto a single-bond issuer.
    """
    cusip_to_issuer = dict(zip(selected["Cusip"], selected["Ticker"]))
    cusip_to_sector = dict(zip(selected["Cusip"], selected["Class3"]))
    issuer = pd.Series(
        weights.index.map(cusip_to_issuer).values, index=weights.index
    )
    sector = pd.Series(
        weights.index.map(cusip_to_sector).values, index=weights.index
    )

    w = _waterfill_cap(weights, issuer, max_issuer)
    w = _waterfill_cap(w, sector, max_sector)
    w = _waterfill_cap(w, issuer, max_issuer)
    return w


# ---------------------------------------------------------------------------
# Soft duration target
# ---------------------------------------------------------------------------
def _nudge_to_duration(
    weights: pd.Series,
    selected: pd.DataFrame,
    target: float | None,
    tol: float,
    max_iters: int = 10,
) -> pd.Series:
    """Soft duration target: tilt weights toward bonds nearer the target.

    If the weighted-average duration is already inside
    [target - tol, target + tol] we leave weights alone. Otherwise we
    apply a multiplicative tilt to bonds based on whether they push
    duration toward or away from the target. Renormalise. Repeat.

    Tilt strength is adaptive: starts gentle (±10%) and ramps up if the
    target is far. We cap iterations and log a warning if we can't
    reach tolerance — typically because the eligible universe doesn't
    span the target (e.g. all selected bonds have OAD < 3 but target is
    4). The honest behavior is to land as close as we can and report it.
    """
    if target is None:
        return weights
    w = weights.copy()
    cusip_to_oad = dict(zip(selected["Cusip"], selected["OAD"]))
    oad = pd.Series(
        w.index.map(cusip_to_oad).values, index=w.index, dtype=float
    )

    for i in range(max_iters):
        port_dur = float((w * oad).sum())
        gap = port_dur - target
        if abs(gap) <= tol:
            return w

        # Adaptive tilt: 1.10 / 0.90 on iter 0, 1.20 / 0.80 on iter 1, etc.
        boost = min(0.10 + 0.05 * i, 0.40)
        direction = 1.0 if gap > 0 else -1.0
        side = np.sign(oad - target) * direction
        adj = np.where(side > 0, 1.0 - boost, 1.0 + boost)
        w = w * adj
        w = w / w.sum()

    # Final check
    final_dur = float((w * oad).sum())
    if abs(final_dur - target) > tol:
        logger.warning(
            "Could not hit duration target %.2f +/- %.2f; landed at %.2f",
            target, tol, final_dur,
        )
    return w


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------
def construct_portfolio(
    panel_with_signal: pd.DataFrame,
    backtest_cfg: dict,
    constraints: PortfolioConstraints,
) -> pd.DataFrame:
    """Run the rebalancing loop over all dates in the panel.

    Parameters
    ----------
    panel_with_signal : output of features.apply_regime_tilt
        Must have columns: Date, Cusip, Ticker, Class3, Eff_Rating_Group,
        Years_To_Maturity, OAS, OAD, DTS, signal_tilted, Total_Return_MTD,
        Excess_Return_MTD.
    backtest_cfg : dict with keys
        min_years_to_maturity, max_years_to_maturity,
        drop_missing_oas, drop_missing_oad.
    constraints : PortfolioConstraints

    Returns
    -------
    DataFrame with one row per (Date, Cusip) holding:
        Date, Cusip, Ticker, Class3, OAD,
        weight, signal_tilted,
        Total_Return_MTD, Excess_Return_MTD.
    """
    rebalance_dates = sorted(panel_with_signal["Date"].unique())
    logger.info(
        "Constructing portfolio across %d rebalance dates",
        len(rebalance_dates),
    )

    out_rows: list[pd.DataFrame] = []
    for d in rebalance_dates:
        snap = panel_with_signal[panel_with_signal["Date"] == d]

        eligible = _filter_universe(
            snap,
            min_ttm=backtest_cfg["min_years_to_maturity"],
            max_ttm=backtest_cfg["max_years_to_maturity"],
            drop_missing_oas=backtest_cfg["drop_missing_oas"],
            drop_missing_oad=backtest_cfg["drop_missing_oad"],
        )
        if eligible.empty:
            logger.warning("No eligible bonds on %s — skipping", d)
            continue

        # Top-quantile selection
        cutoff = eligible["signal_tilted"].quantile(1 - constraints.top_quantile)
        selected = eligible[eligible["signal_tilted"] >= cutoff]
        if selected.empty:
            logger.warning("No bonds passed signal cutoff on %s — skipping", d)
            continue

        w = _initial_weights(selected, constraints.weighting)
        w = _apply_caps(
            w, selected,
            max_issuer=constraints.max_issuer_weight,
            max_sector=constraints.max_sector_weight,
        )
        w = _nudge_to_duration(
            w, selected,
            target=constraints.duration_target,
            tol=constraints.duration_tolerance,
        )
        # Re-apply caps: the duration nudge can push weights slightly
        # over cap. This second pass cleans up the small breaches.
        w = _apply_caps(
            w, selected,
            max_issuer=constraints.max_issuer_weight,
            max_sector=constraints.max_sector_weight,
        )

        snap_out = pd.DataFrame({
            "Date": d,
            "Cusip": w.index,
            "weight": w.values,
        }).merge(
            selected[[
                "Cusip", "Ticker", "Class3", "OAD",
                "signal_tilted", "Total_Return_MTD", "Excess_Return_MTD",
            ]],
            on="Cusip", how="left",
        )
        out_rows.append(snap_out)

    portfolio = pd.concat(out_rows, ignore_index=True)
    logger.info(
        "Portfolio built: %d position-rows across %d dates",
        len(portfolio), portfolio["Date"].nunique(),
    )
    return portfolio
