"""Tests for data.py.

These tests use a tiny synthetic CSV (written to a temp directory) so
they run in milliseconds, don't depend on the real data files, and
don't break if the real CSVs change. This is the standard pattern for
testing data-ingestion code.

Run with:
    pytest -q
"""

from __future__ import annotations

import pandas as pd
import pytest

from credit_pipeline.data import load_bond_panel, _validate_columns


# ---------------------------------------------------------------------------
# Helper: build a tiny synthetic CSV for tests
# ---------------------------------------------------------------------------
def _write_synthetic_csv(path) -> None:
    """Create a 4-row CSV that mimics the real format."""
    rows = [
        "1/1/2020,AAA001,US0001,Corporate,Industrial,Energy,ABC,BB,13,1/1/2025,5.0,5.0,500.0,4.0,4.0,8.5,1.5,0.5",
        "1/1/2020,AAA002,US0002,Corporate,Industrial,Tech,XYZ,B,16,1/1/2026,6.0,8.0,700.0,4.5,4.4,10.2,2.0,1.0",
        "2/1/2020,AAA001,US0001,Corporate,Industrial,Energy,ABC,BB,13,1/1/2025,4.9,4.8,490.0,3.9,3.9,8.4,-0.5,-0.2",
        "2/1/2020,AAA002,US0002,Corporate,Industrial,Tech,XYZ,B,16,1/1/2026,5.9,7.5,680.0,4.4,4.3,10.0,1.2,0.6",
    ]
    header = "Date,Cusip,ISIN,Class1,Class2,Class3,Ticker,Eff_Rating_Group,Index_Rating_Number,Maturity_Date,Years_To_Maturity,DTS,OAS,OAD,OASD,Yield_To_Worst,Total_Return_MTD,Excess_Return_MTD"
    path.write_text(header + "\n" + "\n".join(rows) + "\n")


# ---------------------------------------------------------------------------
# Test 1: column validation rejects bad input
# ---------------------------------------------------------------------------
def test_validate_columns_raises_on_missing():
    """_validate_columns must fail loudly when required columns are absent."""
    df = pd.DataFrame({"foo": [1, 2], "bar": [3, 4]})
    with pytest.raises(ValueError, match="missing required columns"):
        _validate_columns(df, required=["foo", "baz"], source="test")


# ---------------------------------------------------------------------------
# Test 2: returns get converted from percent to decimal
# ---------------------------------------------------------------------------
def test_returns_converted_to_decimals(tmp_path):
    """The single most important test: returns must be in decimal after load.

    Source CSV has Total_Return_MTD = 1.5 (meaning +1.5%). After loading,
    we should see 0.015 (the decimal form). Getting this wrong means
    every Sharpe ratio downstream is off by 100x.
    """
    csv_path = tmp_path / "synth.csv"
    parquet_path = tmp_path / "synth.parquet"
    _write_synthetic_csv(csv_path)

    df = load_bond_panel(
        raw_files=[csv_path],
        processed_path=parquet_path,
        force_rebuild=True,
    )

    # Source had 1.5 -> after /100 should be 0.015
    first_return = df.loc[df["Cusip"] == "AAA001"].iloc[0]["Total_Return_MTD"]
    assert abs(first_return - 0.015) < 1e-9, (
        f"Expected 0.015 (decimal), got {first_return}. "
        "Are you forgetting to divide returns by 100?"
    )

    # Sanity check: returns should all be << 1.0 in absolute value.
    # If any |return| > 1.0 (i.e. > 100%), conversion didn't happen.
    assert df["Total_Return_MTD"].abs().max() < 1.0, (
        "Some returns exceed 100% in absolute value -- conversion likely failed."
    )


# ---------------------------------------------------------------------------
# Test 3: Parquet cache works (second load skips CSV parsing)
# ---------------------------------------------------------------------------
def test_parquet_cache_used_on_second_call(tmp_path):
    """First call writes Parquet; second call should read from Parquet
    instead of re-parsing the CSV.

    We test this indirectly: delete the source CSV after the first call,
    then call again with the same processed_path. If the cache works,
    the second call succeeds; if not, it fails because the CSV is gone.
    """
    csv_path = tmp_path / "synth.csv"
    parquet_path = tmp_path / "synth.parquet"
    _write_synthetic_csv(csv_path)

    # First call: builds the cache from CSV
    df1 = load_bond_panel(
        raw_files=[csv_path],
        processed_path=parquet_path,
        force_rebuild=False,
    )
    assert parquet_path.exists(), "Parquet cache should have been written"

    # Delete the CSV — now only Parquet exists
    csv_path.unlink()

    # Second call: should succeed by reading the Parquet
    df2 = load_bond_panel(
        raw_files=[csv_path],  # path no longer exists, but cache is hit first
        processed_path=parquet_path,
        force_rebuild=False,
    )

    # Both calls should produce identical data
    pd.testing.assert_frame_equal(df1, df2)


# ---------------------------------------------------------------------------
# Test 4: dates are properly parsed
# ---------------------------------------------------------------------------
def test_dates_parsed_to_datetime(tmp_path):
    """After loading, Date should be a real datetime, not a string."""
    csv_path = tmp_path / "synth.csv"
    parquet_path = tmp_path / "synth.parquet"
    _write_synthetic_csv(csv_path)

    df = load_bond_panel(
        raw_files=[csv_path],
        processed_path=parquet_path,
        force_rebuild=True,
    )

    # Date column should have datetime dtype
    assert pd.api.types.is_datetime64_any_dtype(df["Date"]), (
        f"Expected datetime, got {df['Date'].dtype}"
    )
    # Should have exactly 2 unique months from our synthetic data
    assert df["Date"].nunique() == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# ---------------------------------------------------------------------------
# Test 4: handling of failed downloads
# ---------------------------------------------------------------------------
def test_load_macro_returns_empty_on_failure(monkeypatch, tmp_path):
    """If every FRED fetch fails, load_macro must return an empty
    DataFrame (with the expected columns), NOT raise an exception.
    """
    from credit_pipeline import data as data_module

    # Replace _fetch_fred_csv with one that always raises
    def _always_fail(series_id, timeout=30):
        raise ConnectionError("Simulated network failure")

    monkeypatch.setattr(data_module, "_fetch_fred_csv", _always_fail)

    result = data_module.load_macro(
        series={"vix": "VIXCLS", "hy_oas": "BAMLH0A0HYM2"},
        macro_path=tmp_path / "macro.parquet",
        force_rebuild=True,
    )

    assert result.empty, "Expected empty DataFrame on total fetch failure"
    assert set(result.columns) == {"Date", "series", "value", "label"}, (
        f"Expected columns Date/series/value/label, got {list(result.columns)}"
    )
