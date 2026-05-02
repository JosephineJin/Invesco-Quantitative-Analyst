# Memo: Systematic Credit Mini-Pipeline

**To:** Portfolio managers and analysts, systematic fixed income
**From:** Quant Analyst candidate
**Re:** End-to-end research pipeline for a long-only US HY credit signal
**Date:** May 2026

---

## What I built

A modular, configurable pipeline that turns the firm's monthly US High
Yield bond panel (Jan 2010 – Mar 2026) into a long-only portfolio
prototype with diagnostic outputs. One command rebuilds everything from
raw CSVs; every parameter lives in YAML.

The whole pipeline is ~700 lines of Python across six focused modules
(`data`, `features`, `portfolio`, `evaluation`, `run`, `config`), with a
unit-test suite (18 tests) that runs on synthetic data in under three
seconds.

## The signal: carry per unit of MTM risk, neutralised by peer group

For each bond on each month-end I compute:

> **score = z-score of (OAS / DTS) within (rating bucket × sector)**

The numerator is spread compensation. The denominator (DTS = OAS ×
duration) is a forward-looking measure of how much P&L moves on a
proportional spread shift. Their ratio is value relative to spread risk
taken — conceptually adjacent to a Sharpe ratio for a single name.

Z-scoring within rating × sector cohorts strips out the structural
differences that would otherwise dominate a cross-section: a CCC Energy
name is compared against other CCC Energy names, not against BB Tech.

A **regime overlay** modulates the raw score: when a macro stress index
(FRED VIX + ICE BofA HY OAS, z-scored) is elevated (z ≥ 1.0), scores
are tilted toward higher-quality buckets — BB ×1.10, B unchanged,
CCC ×0.85. This is a tilt, not a regime switch — the strategy stays
invested in HY throughout. If the FRED endpoint fails, the regime
degrades to a self-contained fallback: cross-sectional median OAS,
z-scored over the sample.

**Why this signal:** carry-per-risk has decades of empirical support
across rates, credit, and equities; cross-sectional within-cohort
z-scoring is the cleanest way to extract relative value while remaining
agnostic to absolute spread levels. It is intentionally simple — a
deliberate choice given the engineering-first scope.

## Portfolio construction

Long-only, monthly rebalanced top-quintile selection within an eligible
universe (1–15y maturity, valid OAS/OAD). Equal weights are then
constrained via a water-filling routine that:

  1. Caps any issuer above 2% (`max_issuer_weight`).
  2. Caps any sector above 20% (`max_sector_weight`).
  3. Soft-targets a 2-year portfolio OAD (±1y).

The water-filling preserves the sum-to-one invariant throughout (this
is unit-tested). Constraints are deliberately loose — the goal is a
sane prototype, not an optimised production book.

A subtle finding worth flagging: the OAS/DTS signal has a **structural
short-duration bias** because DTS = OAS × OAD, so the ratio simplifies
toward 1/OAD modulo cohort z-scoring. The portfolio's natural duration
is ~1.4 years rather than the typical HY benchmark of ~4 years. Rather
than fight the universe with an aggressive duration nudge, I set a
realistic target. Two consequences: (a) the strategy is structurally
less rate-sensitive than the index; (b) duration-neutralizing the
signal at construction time is a clear next step.

## Results, 2010 – Mar 2026 (16 years, 195 months)

|                       | Strategy | Equal-weight universe |
|-----------------------|----------|------------------------|
| Annualised return     | 4.7%     | 6.8%                   |
| Annualised volatility | 5.3%     | 7.5%                   |
| Sharpe (excess of 0)  | 0.88     | 0.87                   |
| Max drawdown          | -13.0%   | -16.4%                 |
| Avg monthly turnover  | 24%      | n/a                    |
| Hit rate vs benchmark | 37%      | n/a                    |

The strategy underperforms in total return but achieves a marginally
higher Sharpe with **two-thirds the drawdown of the benchmark**. The
hit rate of 37% combined with a higher Sharpe is a tell-tale signature:
the strategy gives up upside in normal markets to dodge the worst
left-tail months. **Defensive value worked when defensiveness was
needed; it cost in the recoveries.** That trade-off is exactly what the
signal is buying.

The visible drawdowns hit at the right places: late-2015 / early-2016
(Energy blowup), March 2020 (COVID), and 2022 (rates shock). Charts in
`outputs/`: cumulative return, drawdown, sector exposure through time,
and one-way turnover.

## What can break (failure modes and how to detect them)

1. **Survivorship via index reconstitution.** Bonds dropped from the
   index next month (defaulted, downgraded out of HY, redeemed, called)
   silently get a zero forward return in our alignment. This understates
   default losses in realised P&L.
   *Detection:* the returns CSV reports `invested_frac` per month.
   Sustained values <1.0 mean we're not getting full exposure to our
   bets — typically a signal we're systematically missing realised
   losses. Cross-check by tracking the count of positions where
   `FwdTotalReturn` is NaN.

2. **Crowded-trade decay.** Carry-per-DTS is well-known. When this kind
   of signal is widely traded, cross-sectional dispersion in the tilted
   score collapses, and the marginal name we add no longer carries the
   prior expected return.
   *Detection:* compute a rolling 6-month rank-IC between
   `signal_tilted` and forward returns. A sustained drop to zero or
   below is a kill switch — there is no point holding the trade if the
   signal isn't predicting.

3. **Regime mis-classification.** The macro stress index is built from
   two FRED series. If FRED is down we silently switch to an in-sample
   fallback that uses panel OAS — much noisier and *contemporaneous*
   (peeking forward through the cross-section's median).
   *Detection:* the regime path is logged on each run; alert if the
   fallback is active for more than one consecutive month.

4. **Liquidity / executability.** This is a paper portfolio. CCC names
   in particular have wide bid-ask, partial fills, and minimum lot
   sizes that the backtest ignores. A 2% per-issuer cap is permissive
   for a real book.
   *Detection:* layer a transaction-cost model proportional to
   `OASD × bid_offer_bps` and re-run; observed Sharpe should not
   collapse.

## What I would do next

With another day:

  - **Walk-forward signal IC analysis.** Verify the signal isn't
    over-fit to any single decade by reporting IC by year and by
    rating bucket.
  - **Simple TC model.** Add a configurable per-trade cost
    (e.g. half × OASD × 25 bps) so the reported Sharpe is investable.
  - **Duration-neutralize the signal.** Regress OAS/DTS on duration
    within each cohort and use the residual as the score. Removes the
    structural short-duration bias and lets the duration target be
    set freely.

With another week:

  - **Multi-signal blend.** Add a second, lightly-correlated signal
    (e.g. excess-return momentum, or change-in-OAS reversal) and sum
    the z-scores. Diversification of signals usually beats any single
    one in this domain.
  - **Issuer-level downgrade-probability overlay.** Use rating
    transitions visible in the panel to construct a within-month
    downgrade-risk score; subtract a small multiple of it from the
    value signal so we aren't buying value in names heading to D.

## How to run it

pip install -r requirements.txt
pip install -e .
python -m credit_pipeline.run --config configs/default.yaml
python -m pytest -q

All knobs in `configs/default.yaml`. See `README.md` for the design
walkthrough.
