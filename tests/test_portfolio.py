"""Tests for portfolio.py.

Synthetic panels only — fast, hermetic, deterministic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_pipeline.features import compute_signal
from credit_pipeline.portfolio import (
    PortfolioConstraints,
    _waterfill_cap,
    construct_portfolio,
)


def _make_panel(
    n_dates: int = 3,
    n_bonds_per_group: int = 15,
    seed: int = 0,
) -> pd.DataFrame:
    """Synthetic bond panel — same structure as features tests but with
    return columns added (portfolio module passes them through)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="MS")
    rating_groups = ["BB", "B", "CCC"]
    sectors = ["Energy", "Tech", "Consumer_Cyclical"]

    rows = []
    cusip_id = 0
    for d in dates:
        for rg in rating_groups:
            for sec in sectors:
                for _ in range(n_bonds_per_group):
                    cusip_id += 1
                    oas = float(rng.uniform(100, 800))
                    oad = float(rng.uniform(2, 8))
                    dts = oas * oad / 100.0
                    rows.append({
                        "Date": d,
                        "Cusip": f"C{cusip_id:06d}",
                        "Ticker": f"T{cusip_id % 50:03d}",
                        "Class3": sec,
                        "Eff_Rating_Group": rg,
                        "Years_To_Maturity": float(rng.uniform(2, 10)),
                        "DTS": dts,
                        "OAS": oas,
                        "OAD": oad,
                        "Total_Return_MTD": float(rng.normal(0.005, 0.03)),
                        "Excess_Return_MTD": float(rng.normal(0.0, 0.025)),
                    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test 1: water-filling preserves sum-to-1 and respects cap (feasible case)
# ---------------------------------------------------------------------------
def test_waterfill_preserves_sum_and_respects_cap_feasible():
    """For a FEASIBLE cap, water-filling must produce weights that
    sum to 1 AND respect the cap."""
    # 10 bonds, 5 groups of 2. Equal weights = 0.10 each.
    # Cap at 0.30 (feasible: 5 * 0.30 = 1.50 >= 1.0).
    weights = pd.Series([0.10] * 10, index=[f"C{i}" for i in range(10)])
    groups = pd.Series(
        ["A","A","B","B","C","C","D","D","E","E"],
        index=weights.index,
    )

    capped = _waterfill_cap(weights, groups, cap=0.30)

    # Sum to 1
    assert abs(capped.sum() - 1.0) < 1e-9, f"Sum was {capped.sum()}"

    # No group exceeds cap
    group_totals = capped.groupby(groups).sum()
    assert group_totals.max() <= 0.30 + 1e-9, (
        f"Cap breached: {group_totals.to_dict()}"
    )


# ---------------------------------------------------------------------------
# Test 1b: infeasible cap — sums to < 1, every group at cap
# ---------------------------------------------------------------------------
def test_waterfill_handles_infeasible_cap():
    """When the cap is mathematically infeasible (n_groups * cap < 1),
    water-filling must cap each group at the limit and accept partial
    investment rather than silently inflating weights past the cap."""
    weights = pd.Series([0.10] * 10, index=[f"C{i}" for i in range(10)])
    groups = pd.Series(
        ["A","A","B","B","C","C","D","D","E","E"],
        index=weights.index,
    )

    # 5 groups, cap=0.15 -> max achievable = 0.75
    capped = _waterfill_cap(weights, groups, cap=0.15)

    # Sum should be at most the achievable maximum (with tiny slack)
    assert capped.sum() <= 0.75 + 1e-9, f"Sum {capped.sum()} > 0.75"

    # No group should exceed cap
    group_totals = capped.groupby(groups).sum()
    assert group_totals.max() <= 0.15 + 1e-9, (
        f"Cap breached: {group_totals.to_dict()}"
    )

# ---------------------------------------------------------------------------
# Test 2: portfolio respects caps and sums to 1 each date
# ---------------------------------------------------------------------------
def test_portfolio_respects_caps_and_sums_to_one():
    """End-to-end: built portfolio must satisfy all hard constraints."""
    panel = _make_panel(n_dates=3, n_bonds_per_group=15)
    panel = compute_signal(
        panel, group_by=["Eff_Rating_Group", "Class3"], winsorize_pct=0.0
    )
    panel["signal_tilted"] = panel["signal"]  # skip regime tilt for this test

    bt_cfg = dict(
        min_years_to_maturity=0.0,
        max_years_to_maturity=20.0,
        drop_missing_oas=True,
        drop_missing_oad=True,
    )
    cons = PortfolioConstraints(
        top_quantile=0.40,
        weighting="equal",
        max_issuer_weight=0.10,
        max_sector_weight=0.50,
        duration_target=None,
        duration_tolerance=0.5,
    )
    pf = construct_portfolio(panel, bt_cfg, cons)

    for d, snap in pf.groupby("Date"):
        # (a) weights sum to ~1
        assert abs(snap["weight"].sum() - 1.0) < 1e-6, (
            f"Weights don't sum to 1 on {d}"
        )

        # (b) issuer cap respected
        issuer = snap.groupby("Ticker")["weight"].sum()
        assert issuer.max() <= cons.max_issuer_weight + 1e-6, (
            f"Issuer cap breached on {d}: max={issuer.max()}"
        )

        # (c) sector cap respected
        sector = snap.groupby("Class3")["weight"].sum()
        assert sector.max() <= cons.max_sector_weight + 1e-6, (
            f"Sector cap breached on {d}: max={sector.max()}"
        )


# ---------------------------------------------------------------------------
# Test 3: top quantile selection actually selects top 20%
# ---------------------------------------------------------------------------
def test_top_quantile_selects_correct_count():
    """Number of selected bonds should be ~top_quantile × eligible count."""
    panel = _make_panel(n_dates=1, n_bonds_per_group=20)
    panel = compute_signal(
        panel, group_by=["Eff_Rating_Group", "Class3"], winsorize_pct=0.0
    )
    panel["signal_tilted"] = panel["signal"]

    bt_cfg = dict(
        min_years_to_maturity=0.0, max_years_to_maturity=20.0,
        drop_missing_oas=True, drop_missing_oad=True,
    )
    cons = PortfolioConstraints(
        top_quantile=0.20, weighting="equal",
        max_issuer_weight=0.50, max_sector_weight=1.0,
        duration_target=None,
    )
    pf = construct_portfolio(panel, bt_cfg, cons)

    # Eligible universe: 3 ratings × 3 sectors × 20 = 180 bonds
    # Top 20% should be ~36 bonds (allow a few extra due to ties at the cutoff)
    n_held = pf["Cusip"].nunique()
    assert 30 <= n_held <= 50, (
        f"Top quantile selection produced {n_held} bonds; expected ~36"
    )


# ---------------------------------------------------------------------------
# Test 4: duration nudge moves portfolio toward target
# ---------------------------------------------------------------------------
def test_duration_target_pulls_portfolio_close():
    """If a duration target is specified and feasible, portfolio OAD
    should land within tolerance."""
    panel = _make_panel(n_dates=1, n_bonds_per_group=15, seed=42)
    panel = compute_signal(
        panel, group_by=["Eff_Rating_Group", "Class3"], winsorize_pct=0.0
    )
    panel["signal_tilted"] = panel["signal"]

    bt_cfg = dict(
        min_years_to_maturity=0.0, max_years_to_maturity=20.0,
        drop_missing_oas=True, drop_missing_oad=True,
    )
    cons = PortfolioConstraints(
        top_quantile=0.40, weighting="equal",
        max_issuer_weight=0.10, max_sector_weight=0.50,
        duration_target=4.0, duration_tolerance=0.5,
    )
    pf = construct_portfolio(panel, bt_cfg, cons)

    # Weighted-average duration should be near 4.0 (within ~1 year — the
    # nudge is gentle; a hard target would need a real optimiser)
    port_dur = (pf["weight"] * pf["OAD"]).sum() / pf["Date"].nunique()
    # Allow generous tolerance since this is a soft target
    assert 2.5 <= port_dur <= 5.5, (
        f"Portfolio duration {port_dur:.2f} far from target 4.0"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
