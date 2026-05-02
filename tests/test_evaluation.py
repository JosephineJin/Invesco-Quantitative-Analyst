"""Tests for evaluation.py.

The most important test here is the no-lookahead test: we construct a
deliberate situation where lookahead would matter, and verify the
forward-return alignment ignores the contemporaneous return.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_pipeline.evaluation import (
    _attach_forward_returns,
    compute_diagnostics,
    compute_strategy_returns,
)


# ---------------------------------------------------------------------------
# Helper: tiny synthetic panel + portfolio with known returns
# ---------------------------------------------------------------------------
def _make_panel_and_portfolio() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a 3-month, 2-bond panel with hand-crafted returns.

    Bond A: returns 1%, 2%, 3% in months 1, 2, 3
    Bond B: returns 4%, 5%, 6% in months 1, 2, 3
    Portfolio: 100% Bond A in month 1, 100% Bond B in month 2.
    """
    panel = pd.DataFrame([
        # Date,        Cusip, Total_Return_MTD, Excess_Return_MTD
        ("2024-01-01", "A", 0.01, 0.005),
        ("2024-01-01", "B", 0.04, 0.025),
        ("2024-02-01", "A", 0.02, 0.010),
        ("2024-02-01", "B", 0.05, 0.030),
        ("2024-03-01", "A", 0.03, 0.015),
        ("2024-03-01", "B", 0.06, 0.035),
    ], columns=["Date", "Cusip", "Total_Return_MTD", "Excess_Return_MTD"])
    panel["Date"] = pd.to_datetime(panel["Date"])

    portfolio = pd.DataFrame([
        # Date,        Cusip, Class3, weight
        ("2024-01-01", "A", "Tech", 1.0),
        ("2024-02-01", "B", "Tech", 1.0),
    ], columns=["Date", "Cusip", "Class3", "weight"])
    portfolio["Date"] = pd.to_datetime(portfolio["Date"])

    return panel, portfolio


# ---------------------------------------------------------------------------
# Test 1: forward returns are NEXT month's returns, not this month's
# ---------------------------------------------------------------------------
def test_forward_returns_have_no_lookahead():
    """The cardinal sin of backtesting. _attach_forward_returns must
    pair Date=T's weight with the bond's return realised in T+1."""
    panel, portfolio = _make_panel_and_portfolio()
    out = _attach_forward_returns(portfolio, panel)

    # Bond A held on 2024-01-01 should have FwdTotalReturn = A's
    # return in 2024-02 = 0.02, NOT A's return in 2024-01 (=0.01)
    a_jan = out[(out["Cusip"] == "A") & (out["Date"] == "2024-01-01")].iloc[0]
    assert abs(a_jan["FwdTotalReturn"] - 0.02) < 1e-9, (
        f"Lookahead bias detected: FwdTotalReturn={a_jan['FwdTotalReturn']} "
        f"should be 0.02 (next month), not 0.01 (current month)"
    )

    # Bond B held on 2024-02-01 should have FwdTotalReturn = 0.06
    b_feb = out[(out["Cusip"] == "B") & (out["Date"] == "2024-02-01")].iloc[0]
    assert abs(b_feb["FwdTotalReturn"] - 0.06) < 1e-9


# ---------------------------------------------------------------------------
# Test 2: strategy return is the weighted forward return
# ---------------------------------------------------------------------------
def test_strategy_return_is_weighted_forward_return():
    """Strategy return on date T should equal sum(weight * fwd_return)."""
    panel, portfolio = _make_panel_and_portfolio()
    returns = compute_strategy_returns(portfolio, panel)

    # On Date=2024-01-01 (RealizedDate=2024-02-01), 100% in A:
    # strategy return = A's Feb return = 0.02
    jan_row = returns[returns["Date"] == "2024-01-01"].iloc[0]
    assert abs(jan_row["strat_total_ret"] - 0.02) < 1e-9

    # On Date=2024-02-01 (RealizedDate=2024-03-01), 100% in B:
    # strategy return = B's Mar return = 0.06
    feb_row = returns[returns["Date"] == "2024-02-01"].iloc[0]
    assert abs(feb_row["strat_total_ret"] - 0.06) < 1e-9


# ---------------------------------------------------------------------------
# Test 3: benchmark = equal-weight average of forward returns
# ---------------------------------------------------------------------------
def test_benchmark_is_equal_weight_universe():
    """Benchmark on date T = mean of all bonds' returns in T+1."""
    panel, portfolio = _make_panel_and_portfolio()
    returns = compute_strategy_returns(portfolio, panel)

    # On 2024-01-01: average of Feb returns = (0.02 + 0.05) / 2 = 0.035
    jan_row = returns[returns["Date"] == "2024-01-01"].iloc[0]
    assert abs(jan_row["bench_total_ret"] - 0.035) < 1e-9


# ---------------------------------------------------------------------------
# Test 4: diagnostics are positive numbers with sensible signs
# ---------------------------------------------------------------------------
def test_diagnostics_have_sensible_shape():
    """Smoke test: diagnostics dict should have all expected keys and
    at least produce numbers that have the right sign/range."""
    # Larger synthetic panel for stable stats
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=24, freq="MS")
    rows = []
    for d in dates:
        for cusip in ["A", "B", "C", "D"]:
            rows.append({
                "Date": d, "Cusip": cusip,
                "Total_Return_MTD": rng.normal(0.005, 0.02),
                "Excess_Return_MTD": rng.normal(0.0, 0.015),
            })
    panel = pd.DataFrame(rows)
    panel["Date"] = pd.to_datetime(panel["Date"])

    portfolio = pd.DataFrame([
        {"Date": d, "Cusip": "A", "Class3": "Tech", "weight": 0.5,
         "Ticker": "T1", "OAD": 4.0, "signal_tilted": 1.0,
         "Total_Return_MTD": 0.0, "Excess_Return_MTD": 0.0}
        for d in dates
    ] + [
        {"Date": d, "Cusip": "B", "Class3": "Tech", "weight": 0.5,
         "Ticker": "T2", "OAD": 4.0, "signal_tilted": 1.0,
         "Total_Return_MTD": 0.0, "Excess_Return_MTD": 0.0}
        for d in dates
    ])
    portfolio["Date"] = pd.to_datetime(portfolio["Date"])

    returns = compute_strategy_returns(portfolio, panel)
    diags = compute_diagnostics(returns, portfolio)

    # Required keys present
    expected = {
        "n_months", "ann_return_strategy", "ann_return_benchmark",
        "ann_vol_strategy", "ann_vol_benchmark",
        "sharpe_strategy", "sharpe_benchmark",
        "max_dd_strategy", "max_dd_benchmark",
        "hit_rate_vs_bench", "avg_excess_ret_monthly",
        "avg_monthly_turnover", "top_sectors",
    }
    assert expected.issubset(diags.keys()), (
        f"Missing keys: {expected - diags.keys()}"
    )

    # Shape sanity
    assert diags["ann_vol_strategy"] >= 0
    assert diags["max_dd_strategy"] <= 0
    assert 0.0 <= diags["hit_rate_vs_bench"] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
