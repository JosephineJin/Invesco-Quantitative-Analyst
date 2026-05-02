"""Tests for features.py.

We use small synthetic panels rather than real data so tests run fast
and don't depend on external state. The synthetic panels are
deterministic (seeded RNG) so failures reproduce.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_pipeline.features import (
    apply_regime_tilt,
    compute_regime,
    compute_signal,
)


# ---------------------------------------------------------------------------
# Helper: build a tiny synthetic panel
# ---------------------------------------------------------------------------
def _make_panel(
    n_dates: int = 4,
    n_bonds_per_group: int = 6,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a synthetic bond panel with predictable structure.

    n_dates × 3 ratings × 3 sectors × n_bonds_per_group rows.
    Default settings produce 4 × 3 × 3 × 6 = 216 rows.
    """
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
                    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test 1: signal is properly z-scored within groups
# ---------------------------------------------------------------------------
def test_signal_is_zscore_within_groups():
    """compute_signal must produce ~0-mean, ~1-std scores within each cohort."""
    panel = _make_panel(n_dates=2, n_bonds_per_group=10)
    out = compute_signal(
        panel, group_by=["Eff_Rating_Group", "Class3"], winsorize_pct=0.0
    )

    # For each (Date, rating, sector) cohort, the signal should be
    # mean ~ 0 and std ~ 1.
    grp = out.groupby(["Date", "Eff_Rating_Group", "Class3"])["signal"]
    means = grp.mean().abs()
    stds = grp.std(ddof=0)

    assert (means < 1e-9).all(), f"Group means not zero: max {means.max()}"
    assert ((stds - 1.0).abs() < 1e-6).all(), (
        f"Group stds not 1: {stds.describe()}"
    )


# ---------------------------------------------------------------------------
# Test 2: regime fallback when macro is empty
# ---------------------------------------------------------------------------
def test_regime_fallback_when_macro_missing():
    """compute_regime must produce a stress series even with empty macro."""
    panel = _make_panel(n_dates=12, n_bonds_per_group=5)
    empty_macro = pd.DataFrame(columns=["Date", "series", "value", "label"])

    regime = compute_regime(empty_macro, panel, threshold=1.0)

    # One row per panel date
    assert len(regime) == panel["Date"].nunique()
    # Required columns present
    assert {"Date", "stress_z", "in_stress"}.issubset(regime.columns)
    # No NaNs after fallback (everything should be filled)
    assert regime["stress_z"].notna().all()
    assert regime["in_stress"].dtype == bool


# ---------------------------------------------------------------------------
# Test 3: regime tilt only active in stress months
# ---------------------------------------------------------------------------
def test_regime_tilt_only_active_in_stress():
    """apply_regime_tilt must leave calm-month scores untouched."""
    panel = _make_panel(n_dates=2, n_bonds_per_group=5)
    panel = compute_signal(
        panel, group_by=["Eff_Rating_Group", "Class3"], winsorize_pct=0.0
    )

    # Build a manual regime: first date stress, second date calm.
    dates = sorted(panel["Date"].unique())
    regime = pd.DataFrame({
        "Date": dates,
        "stress_z": [2.0, 0.0],
        "in_stress": [True, False],
    })
    out = apply_regime_tilt(panel, regime)

    # Calm date: tilt factor = 1.0 -> signal_tilted == signal exactly
    calm = out[out["Date"] == dates[1]]
    assert np.allclose(
        calm["signal_tilted"], calm["signal"], equal_nan=True
    ), "Calm-month tilt should be a no-op"

    # Stress date: CCC bonds (tilt factor 0.85) must differ from raw signal
    stress = out[out["Date"] == dates[0]]
    ccc = stress[stress["Eff_Rating_Group"] == "CCC"]
    nonzero = ccc[ccc["signal"].abs() > 1e-9]
    assert (nonzero["signal_tilted"] != nonzero["signal"]).all(), (
        "Stress-month CCC tilt should change scores"
    )


# ---------------------------------------------------------------------------
# Test 4: tiny groups get NaN signal (not garbage)
# ---------------------------------------------------------------------------
def test_signal_returns_nan_for_tiny_groups():
    """Cohorts with <5 members can't produce a meaningful z-score."""
    # Make a panel with only 3 bonds per cohort
    panel = _make_panel(n_dates=1, n_bonds_per_group=3)
    out = compute_signal(
        panel, group_by=["Eff_Rating_Group", "Class3"], winsorize_pct=0.0
    )
    # Every signal should be NaN (every cohort has only 3 < 5 bonds)
    assert out["signal"].isna().all(), (
        "Signal must be NaN when cohort has fewer than 5 valid obs"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
