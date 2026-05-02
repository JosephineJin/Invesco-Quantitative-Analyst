"""Data ingestion module.

Two responsibilities:
1. Read the raw US High Yield bond panel from CSV(s), validate, clean,
   and cache to Parquet for fast re-reads.
2. Download macro series from FRED for the regime overlay.

"""

from __future__ import annotations

import logging
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)


# Columns we expect in the bond panel. If any are missing the loader
# fails loudly rather than silently producing wrong results downstream.
REQUIRED_BOND_COLS = [
    "Date", "Cusip", "Ticker",
    "Class1", "Class2", "Class3",
    "Eff_Rating_Group", "Index_Rating_Number",
    "Years_To_Maturity", "DTS", "OAS", "OAD", "OASD",
    "Yield_To_Worst", "Total_Return_MTD", "Excess_Return_MTD",
]


def _validate_columns(df: pd.DataFrame, required: list[str], source: str) -> None:
    """Raise a clear error if required columns are missing."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{source}: missing required columns {missing}")


def load_bond_panel(
    raw_files: list[str | Path],
    processed_path: str | Path,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Load the bond panel.

    Reads from cached Parquet if it exists; otherwise reads the raw CSVs,
    cleans them, writes the Parquet cache, and returns the cleaned frame.

    Parameters
    ----------
    raw_files : list of paths to the raw CSV files
    processed_path : where to write/read the Parquet cache
    force_rebuild : if True, ignore the cache and re-read from CSV

    Cleaning rules:
      - Coerce Date to datetime; drop rows with unparseable Date.
      - Drop rows with missing Cusip.
      - Coerce numeric columns; rows with NaN in critical numerics
        (OAS, OAD, DTS) are kept here — the universe filter in the
        backtest applies the final drop.
      - Convert return columns from PERCENT to DECIMAL (e.g. 6.41 -> 0.0641)
        so all downstream math is unit-clean. THIS IS THE MOST IMPORTANT
        LINE IN THE FILE — get this wrong and your whole backtest is off
        by 100x.
    """
    processed_path = Path(processed_path)

    # --- Cache path: load and return early if Parquet exists -------------
    if processed_path.exists() and not force_rebuild:
        logger.info("Loading cached bond panel from %s", processed_path)
        df = pd.read_parquet(processed_path)
        logger.info("Loaded %d rows, %d unique dates",
                    len(df), df["Date"].nunique())
        return df

    # --- Slow path: read raw CSVs, clean, cache --------------------------
    logger.info("Building bond panel from %d raw file(s)", len(raw_files))
    frames = []
    for fp in raw_files:
        fp = Path(fp)
        if not fp.exists():
            raise FileNotFoundError(f"Raw data file not found: {fp}")
        logger.info("  reading %s", fp.name)
        # Read everything as string first, then coerce explicitly. This
        # avoids silent dtype surprises when a numeric column has stray
        # text (which happens often in real-world CSVs).
        chunk = pd.read_csv(fp, dtype=str)
        frames.append(chunk)
    df = pd.concat(frames, ignore_index=True)
    logger.info("Concatenated raw shape: %s", df.shape)

    _validate_columns(df, REQUIRED_BOND_COLS, source="bond panel")

    # --- Cleaning --------------------------------------------------------
    initial_rows = len(df)

    # Parse dates; drop unparseable rows
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    bad_dates = df["Date"].isna().sum()
    if bad_dates:
        logger.warning("Dropping %d rows with unparseable Date", bad_dates)
        df = df.dropna(subset=["Date"])

    # Drop rows with missing Cusip — they can't be tracked over time
    df = df.dropna(subset=["Cusip"])

    # Coerce numeric columns. errors="coerce" turns bad values into NaN
    # rather than raising — appropriate here because we want a clean DF
    # with NaNs marking bad data, not a crash.
    numeric_cols = [
        "Index_Rating_Number", "Years_To_Maturity",
        "DTS", "OAS", "OAD", "OASD",
        "Yield_To_Worst", "Total_Return_MTD", "Excess_Return_MTD",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Sort canonically
    df = df.sort_values(["Date", "Cusip"]).reset_index(drop=True)

    # Deduplicate: a Cusip should appear at most once per Date
    dupes = df.duplicated(subset=["Date", "Cusip"]).sum()
    if dupes:
        logger.warning("Dropping %d duplicate (Date, Cusip) rows", dupes)
        df = df.drop_duplicates(subset=["Date", "Cusip"], keep="last")

    # CRITICAL: returns are reported in PERCENT in the source CSV
    # (e.g. 6.41 means +6.41%). Convert to decimals once, here, so all
    # downstream math is clean. If you ever see a Sharpe ratio of 80
    # or a max drawdown of -1300%, this line is the first place to check.
    df["Total_Return_MTD"] = df["Total_Return_MTD"] / 100.0
    df["Excess_Return_MTD"] = df["Excess_Return_MTD"] / 100.0

    logger.info(
        "Cleaned bond panel: %d rows (started %d), %d unique dates",
        len(df), initial_rows, df["Date"].nunique(),
    )

    # --- Cache to Parquet ------------------------------------------------
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed_path, index=False)
    logger.info("Cached panel to %s", processed_path)
    return df

def _fetch_fred_csv(series_id: str, timeout: int = 30) -> pd.DataFrame:
    """Fetch a single FRED series via the public fredgraph.csv endpoint.

    Returns a long-format frame: Date | series | value.

    No API key required — FRED publishes CSVs of every series at a
    well-known URL pattern: 
    https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES_ID>
    
    Raises requests.HTTPError on 4xx/5xx responses; the caller is
    expected to wrap this in a try/except.
    """
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    logger.info("Fetching FRED series %s", series_id)
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()  # Raise on 4xx / 5xx HTTP errors

    df = pd.read_csv(StringIO(resp.text))
    # FRED uses different column names depending on the endpoint version.
    # Newer responses use 'observation_date'; older use 'DATE'. Handle both.
    date_col = "observation_date" if "observation_date" in df.columns else "DATE"
    df = df.rename(columns={date_col: "Date", series_id: "value"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    df["series"] = series_id
    return df[["Date", "series", "value"]]


def load_macro(
    series: dict[str, str],
    macro_path: str | Path,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Load (or fetch + cache) macro series from FRED.

    Parameters
    ----------
    series : dict mapping a friendly label to a FRED series ID, e.g.
        {"vix": "VIXCLS", "hy_oas": "BAMLH0A0HYM2"}
    macro_path : where to write/read the Parquet cache
    force_rebuild : if True, ignore the cache and re-download

    Returns a long-format frame: Date | series | value | label.
    Empty frame (zero rows) if all downloads fail.

    Graceful failure: if FRED is unreachable or returns errors, this
    function logs the failure and returns an empty DataFrame. The caller
    in features.py detects the empty case and falls back to an in-sample
    regime indicator built from the bond panel itself.
    """
    macro_path = Path(macro_path)

    # --- Cache path ------------------------------------------------------
    if macro_path.exists() and not force_rebuild:
        logger.info("Loading cached macro panel from %s", macro_path)
        return pd.read_parquet(macro_path)

    # --- Slow path: download each series ---------------------------------
    frames: list[pd.DataFrame] = []
    for label, series_id in series.items():
        try:
            df = _fetch_fred_csv(series_id)
            df["label"] = label
            frames.append(df)
            logger.info(
                "  %s (%s): %d obs from %s to %s",
                label, series_id, len(df),
                df["Date"].min().date(), df["Date"].max().date(),
            )
        except Exception as e:  # noqa: BLE001 — we genuinely want any failure
            logger.warning(
                "Failed to fetch FRED series %s (%s): %s",
                label, series_id, e,
            )

    # --- Handle total failure --------------------------------------------
    if not frames:
        logger.warning(
            "No macro data fetched; downstream will use fallback regime"
        )
        return pd.DataFrame(columns=["Date", "series", "value", "label"])

    # --- Concatenate, cache, return --------------------------------------
    macro = pd.concat(frames, ignore_index=True)
    macro_path.parent.mkdir(parents=True, exist_ok=True)
    macro.to_parquet(macro_path, index=False)
    logger.info("Cached macro panel to %s (%d rows)", macro_path, len(macro))
    return macro
