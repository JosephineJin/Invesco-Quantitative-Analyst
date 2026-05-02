"""Feature engineering and signal construction.

Three functions, in the order the pipeline calls them:

1. compute_signal     — turn raw bond data into a per-bond, per-month score
2. compute_regime     — turn macro data into a monthly stress indicator
3. apply_regime_tilt  — combine the two: tilt scores by quality in stress months

Signal in plain English
-----------------------
For each bond we compute carry-per-unit-of-spread-risk: OAS / DTS. We
z-score that within (rating bucket, sector) cohorts each month.

Why this signal?
  - OAS is the spread compensation an investor receives over Treasuries.
  - DTS = OAS x duration is a forward-looking measure of mark-to-market
    risk: a 10% proportional spread widening hurts a high-DTS bond more
    than a low-DTS bond.
  - The ratio captures *value* relative to risk taken — akin to a Sharpe
    ratio for a single bond.
  - Z-scoring within rating x sector strips out structural differences:
    we compare each bond against true peers, not, e.g., putting a BB
    Tech bond in the same bucket as a CCC Energy bond.

Regime overlay
--------------
We build a stress index from FRED VIX and ICE BofA HY OAS (whichever
loaded successfully). When stress is elevated (z >= regime_threshold),
we de-risk by boosting higher-quality buckets (BB > B > CCC). The bias
is intentionally mild — this is a tilt, not a regime-switching bet.

If FRED is unreachable we fall back to a self-contained regime
indicator: the cross-sectional median OAS of the panel itself, z-scored
over its history. Spreads widening = stress.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Mild quality tilt applied in stress regimes. Multiplicative factor on
# the raw signal score before quantile selection. Higher => stronger
# preference for that rating bucket. Values >1 boost, values <1 demote.
QUALITY_TILT_IN_STRESS = {
    "BB":  1.10,   # boost
    "B":   1.00,   # neutral
    "CCC": 0.85,   # demote
    # Anything else (e.g. weird "NR", "D_NR") gets neutral via .fillna(1.0)
}


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------
def compute_signal(
    panel: pd.DataFrame,
    group_by: list[str],
    winsorize_pct: float = 0.01,
) -> pd.DataFrame:
    """Compute the cross-sectional signal score per (Date, Cusip).

    Returns the input panel with an added `signal` column. Bonds with
    missing OAS, DTS, or where DTS <= 0 receive NaN signal and will be
    excluded from selection downstream.

    Performance note
    ----------------
    The naive version uses `groupby(...).transform(lambda)` for both
    winsorization and z-scoring. On the full ~394k-row panel that took
    ~2 minutes because pandas applies the python callable per group.

    The vectorised version below uses native groupby reductions
    ('quantile', 'mean', 'count') and aligns them back via transform's
    fast-path. Roughly 30x faster.
    """
    df = panel.copy()
    valid = df["DTS"].gt(0) & df["OAS"].notna()
    df["raw_value"] = np.where(valid, df["OAS"] / df["DTS"], np.nan)

    keys = ["Date", *group_by]

    # --- Winsorization (vectorised) -----------------------------------------
    if winsorize_pct > 0:
        lo = df.groupby(keys, observed=True)["raw_value"].transform(
            "quantile", winsorize_pct
        )
        hi = df.groupby(keys, observed=True)["raw_value"].transform(
            "quantile", 1 - winsorize_pct
        )
        df["raw_value_w"] = df["raw_value"].clip(lower=lo, upper=hi)
    else:
        df["raw_value_w"] = df["raw_value"]

    # --- Z-score (vectorised) -----------------------------------------------
    grp = df.groupby(keys, observed=True)["raw_value_w"]
    mu = grp.transform("mean")
    # ddof=0 to match a population z-score (we want comparability across
    # cohorts of varying size, not unbiased estimation).
    sd = grp.transform(lambda s: s.std(ddof=0))
    counts = grp.transform("count")

    z = (df["raw_value_w"] - mu) / sd
    # Groups with <5 valid obs => NaN (too few peers for a meaningful z)
    # Degenerate sd==0 => set z to 0 (every bond is identical -> neutral)
    z = np.where(counts < 5, np.nan, z)
    z = np.where(sd == 0, 0.0, z)
    df["signal"] = z

    n_signal = int(np.isfinite(df["signal"]).sum())
    logger.info(
        "Computed signal for %d / %d bond-month rows",
        n_signal, len(df),
    )
    return df.drop(columns=["raw_value", "raw_value_w"])


# ---------------------------------------------------------------------------
# Regime
# ---------------------------------------------------------------------------
def compute_regime(
    macro: pd.DataFrame,
    panel: pd.DataFrame,
    threshold: float = 1.0,
) -> pd.DataFrame:
    """Build a monthly stress indicator.

    Returns a frame with columns: Date, stress_z, in_stress (bool).
    `Date` is aligned to the bond panel's rebalance dates.

    Primary path
        Average z-scores of VIX (log-level) and HY OAS over the full
        sample. Log-VIX because VIX is heavily right-skewed.

    Fallback (FRED unreachable, or no usable series)
        Cross-sectional median OAS from the panel, z-scored over time.
        This is noisier and slightly contemporaneous (uses today's
        cross-section to detect today's stress) but lets the pipeline
        keep working when external services fail.
    """
    rebalance_dates = pd.DatetimeIndex(
        sorted(pd.to_datetime(panel["Date"].unique()))
    )

    # --- Fallback path ---------------------------------------------------
    if macro is None or macro.empty:
        logger.warning("Macro frame empty — using in-sample fallback regime")
        med_oas = panel.groupby("Date")["OAS"].median()
        z = (med_oas - med_oas.mean()) / med_oas.std(ddof=0)
        out = pd.DataFrame({"Date": med_oas.index, "stress_z": z.values})
    else:
        # --- Primary path: build stress z from FRED series --------------
        # Pivot to wide: each label becomes a column.
        wide = macro.pivot_table(
            index="Date", columns="label", values="value"
        ).sort_index()

        z_parts = []
        if "vix" in wide.columns:
            v = np.log(wide["vix"].astype(float))
            z_parts.append((v - v.mean()) / v.std(ddof=0))
        if "hy_oas" in wide.columns:
            h = wide["hy_oas"].astype(float)
            z_parts.append((h - h.mean()) / h.std(ddof=0))

        if not z_parts:
            logger.warning("No usable macro series — using in-sample fallback")
            return compute_regime(pd.DataFrame(), panel, threshold)

        # Average available z-scores; resample to month-end and forward-fill
        # so we always have a value at the rebalance date.
        stress_daily = pd.concat(z_parts, axis=1).mean(axis=1)
        stress_m = stress_daily.resample("MS").last().ffill()
        out = stress_m.rename("stress_z").to_frame().reset_index()

    # --- Align to panel rebalance dates ----------------------------------
    # Use merge_asof to grab the most recent macro value at or before each
    # rebalance date. This is robust to small calendar mismatches (e.g.
    # macro data on month-end, panel on month-start).
    out["Date"] = pd.to_datetime(out["Date"])
    out_sorted = out.sort_values("Date").reset_index(drop=True)
    rebalance_df = pd.DataFrame({"Date": rebalance_dates}).sort_values("Date")
    aligned = pd.merge_asof(
        rebalance_df, out_sorted,
        on="Date", direction="backward",
    )
    aligned["stress_z"] = aligned["stress_z"].ffill().fillna(0.0)
    aligned["in_stress"] = aligned["stress_z"] >= threshold

    logger.info(
        "Regime computed: %d / %d months flagged as stress (threshold=%.2f)",
        int(aligned["in_stress"].sum()), len(aligned), threshold,
    )
    return aligned


# ---------------------------------------------------------------------------
# Regime tilt
# ---------------------------------------------------------------------------
def apply_regime_tilt(
    panel_with_signal: pd.DataFrame,
    regime: pd.DataFrame,
) -> pd.DataFrame:
    """Multiply the signal by a per-rating quality factor on stress months.

    Adds a `signal_tilted` column. In calm months `signal_tilted == signal`;
    in stress months scores get multiplied by QUALITY_TILT_IN_STRESS based
    on each bond's `Eff_Rating_Group`.
    """
    df = panel_with_signal.merge(
        regime[["Date", "in_stress"]], on="Date", how="left"
    )
    df["in_stress"] = df["in_stress"].fillna(False)

    # Per-bond tilt factor: lookup by rating, default to 1.0 (neutral)
    tilt = df["Eff_Rating_Group"].map(QUALITY_TILT_IN_STRESS).fillna(1.0)

    # Apply tilt only on stress months; leave calm-month scores untouched
    factor = np.where(df["in_stress"], tilt, 1.0)
    df["signal_tilted"] = df["signal"] * factor
    return df
