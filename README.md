# Systematic Credit Mini-Pipeline

A small, end-to-end research pipeline for a long-only US High Yield credit
strategy. Built for the Invesco second-round take-home project.

## What it does

One command rebuilds everything from raw inputs. Parameters live in
`configs/default.yaml`; no code edits required to sweep over them.

## Quick start

```bash
# 1) Clone
git clone https://github.com/JosephineJin/Invesco-Quantitative-Analyst.git
cd Invesco-Quantitative-Analyst

# 2) Create a venv and install
python3 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

# 3) Place the data files (see "Data setup" below)

# 4) Run tests to confirm setup
python -m pytest -q                    # expect: 18 passed

# 5) Run the full pipeline
python -m credit_pipeline.run
```

The pipeline:
- Reads `data/raw/USHY_INDEX_*.csv`
- Caches a cleaned panel to `data/processed/ushy_panel.parquet`
- Tries to fetch FRED VIX + HY-OAS (caches to `data/processed/macro.parquet`);
  falls back to an in-sample regime indicator if FRED is unreachable
- Writes results CSVs and 4 PNG charts to `outputs/`

Headline result on 2010–Mar 2026: Sharpe 0.88 vs benchmark 0.87, max
drawdown −13% vs −16%. See `MEMO.md` for the full write-up.

## Data setup

The raw bond panel CSVs are not committed (large input data, not
code). To run the pipeline locally:

1. Obtain the two source CSVs: `USHY_INDEX_20260301_part_1.csv` and
   `USHY_INDEX_20260301_part_2.csv`.
2. Place them in `data/raw/`.

The first run reads the CSVs and caches a cleaned Parquet file.
Subsequent runs read from the cache (~10× faster).

## Project layout

```text
Invesco-Quantitative-Analyst/
├── configs/
│   └── default.yaml              # all tunable parameters
├── data/
│   ├── raw/                      # input CSVs (gitignored)
│   └── processed/                # cached Parquet (gitignored)
├── outputs/                      # generated charts and CSVs
├── src/credit_pipeline/
│   ├── __init__.py
│   ├── config.py                 # YAML loader
│   ├── logging_utils.py          # logger setup
│   ├── data.py                   # ingest CSVs + FRED, cache to Parquet
│   ├── features.py               # signal + regime overlay
│   ├── portfolio.py              # selection, weighting, caps, duration
│   ├── evaluation.py             # backtest, diagnostics, plots
│   └── run.py                    # CLI entry point
├── tests/
│   ├── test_data.py              # 5 tests
│   ├── test_features.py          # 4 tests
│   ├── test_portfolio.py         # 5 tests
│   └── test_evaluation.py        # 4 tests
├── MEMO.md                       # 2-page PM-facing write-up
├── pyproject.toml
├── requirements.txt
└── README.md
```

## The signal in one paragraph

For each bond we compute carry per unit of mark-to-market risk:
**OAS / DTS**. We z-score that within (rating bucket × sector) cohorts each
month and pick the top quintile. Intuition: a bond paying disproportionate
spread for the spread-duration risk it carries tends to outperform its
true peers. A regime overlay (FRED VIX + HY-OAS, with an in-sample
fallback) tilts toward higher-quality buckets during stress months. See
`MEMO.md` for the full economic logic and failure modes.

## Configuration

Every parameter (date window, rebalance frequency, signal grouping,
quantile, issuer/sector caps, duration target) lives in
`configs/default.yaml`. A new variant — say, a tighter top-decile selection
— needs no code change:

```yaml
portfolio:
  top_quantile: 0.10        # was 0.20
```

Then: `python -m credit_pipeline.run --config configs/my_variant.yaml`.

## Design decisions worth flagging

- **Parquet caching of the cleaned panel.** First run takes ~12 s to read
  and clean the CSVs; subsequent runs read from Parquet in ~1 s. The
  cache is invalidated with `--force-rebuild` or by deleting
  `data/processed/`.
- **Forward-return alignment is explicit.** The CSV's `Total_Return_MTD`
  is the return *during* the snapshot month. To avoid lookahead bias we
  shift returns by -1 within each Cusip's series in
  `evaluation._attach_forward_returns`. Bonds that drop out of the index
  next month contribute zero — a conservative treatment.
- **Cap-then-renormalise via water-filling, not full optimisation.**
  A QP solver would be more elegant but adds a dependency and a stack of
  failure modes for marginal accuracy gain on a long-only top-quintile
  portfolio.
- **In-sample regime fallback when FRED fails.** Network calls fail; the
  pipeline shouldn't. If the FRED endpoint returns an error, the macro
  loader logs it, returns an empty frame, and `compute_regime` falls back
  to using cross-sectional median OAS from the bond panel itself.
- **Synthetic, hermetic tests.** The test suite builds tiny synthetic
  panels in code so it runs in <3 seconds with no data dependency. Tests
  cover the highest-risk pieces: signal math, portfolio cap enforcement,
  return alignment, and the regime fallback path.
- **Vectorised signal computation.** A naive `groupby(...).transform(lambda)`
  was the first bottleneck (~2 minutes on the full panel). Replaced with
  native groupby reductions for ~30× speedup.

## Failure modes (more in MEMO.md)

1. **Survivorship bias from index reconstitution.** Bonds fall out of the
   index when downgraded to D or matured. The forward-return alignment
   silently zeros them. Detection: monitor `invested_frac` in the
   returns CSV.
2. **Crowded-trade collapse.** OAS / DTS is a well-known retail-friendly
   credit value signal. Detection: track signal IC (rank correlation
   between `signal_tilted` and forward returns) over rolling 6-month
   windows.

## CLI flags

```bash
python -m credit_pipeline.run --help
```

Options:
- `--config CONFIG`     Path to YAML config (default: `configs/default.yaml`).
- `--force-rebuild`     Ignore Parquet caches and re-read raw inputs.

## Reproducibility checklist

- [x] Single command rebuild end-to-end (`python -m credit_pipeline.run`)
- [x] All parameters in YAML
- [x] Logging at INFO with timestamps
- [x] Graceful handling of missing data and failed downloads
- [x] 18 unit tests covering signal, portfolio caps, return alignment,
  regime fallback
- [x] Parquet caching for performance
- [x] Vectorised signal calc (~30× faster than naive)
- [x] Editable install via `pip install -e .`
