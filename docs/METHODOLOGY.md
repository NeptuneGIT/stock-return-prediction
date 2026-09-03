# Methodology

This document is the durable reference for how this project's data pipeline
and modelling are built to avoid look-ahead bias, the data-quality
conventions the code follows, known issues in the data, and how the ticker
universe is maintained. It exists so that reasoning that would otherwise only
live in code comments and commit history is collected in one place.

## No look-ahead bias

A predictor on a row dated quarter-end `Q` must have been knowable to a
trader **on** `Q`. This is the single rule that governs the panel
construction and validation design:

- **Fundamentals are joined on `filed` (SEC filing date), never `end` (the
  fiscal period the numbers describe).** A company's return on equity or
  debt-to-equity ratio only becomes public knowledge once it is filed with
  the SEC, not as of the date it describes. Joining on the fiscal period
  instead of the filing date would let a model see numbers before they were
  actually available to a real investor, producing backtested performance
  that looks artificially strong. This is enforced with an assertion in
  `scripts/01a_ratios.R` --
  `stopifnot(all(is.na(panel$filed) | panel$filed <= panel$quarter_end))` --
  that fails loudly if any filing date is later than the quarter it was
  joined to.
- **All technical features use trailing windows only.** Momentum, volatility,
  beta, idiosyncratic volatility, and distance-from-52-week-high are all
  computed with `align = "right"` in `rollapply`, never a centered window,
  so each value only looks backward from its quarter-end.
- **The only forward-looking data anywhere in the pipeline is the target
  itself** (`lead(qtr_ret)` when building `exret_next`/`exret_next4`) --
  that is the thing being predicted, not a feature.
- **Walk-forward validation, not k-fold cross-validation, is the only
  out-of-sample validation mechanism.** Standard random k-fold CV would let
  future quarters appear in a training fold used to predict an earlier
  quarter, leaking information a real trader would not have had. Walk-forward
  validation instead builds expanding-window, time-respecting folds where
  every fold trains only on quarters strictly before its own validation
  block. This is what actually prevents look-ahead bias -- not a fixed
  calendar cutoff -- and it holds regardless of where in the calendar a fold
  sits. GBRT's `n.trees` and PLS's `ncomp` are chosen this way; Random
  Forest's `mtry` is fixed at the package regression default (`floor(p/3)`)
  rather than grid-searched, since a multi-value mtry grid repeated across
  every fold proved too slow once the fold set was extended across the
  entire panel. All three models are still evaluated by the same pooled
  out-of-fold walk-forward metric, and the pooled out-of-fold predictions
  this produces are the only historical-quarter predictions ever used for a
  performance metric or the backtest.
- **A second, entire-panel fit exists per model, but may only score the
  current, not-yet-realized quarter.** Each model is refit once on the
  entire panel (all quarters) at its walk-forward-selected hyperparameter.
  This fit is used for variable importance and to forward-score the
  quarter that has not yet realized its target -- what powers
  `scripts/05_top30_forecast.R`. This fit must never be used to generate a
  prediction for a historical quarter: since it has seen every quarter, doing
  so would leak future information into a "prediction" of the past. Any
  future change that blurs this separation should be treated as a
  look-ahead-bias regression, not a refactor.

## Data-quality conventions

- **Missing (`NaN`) is not zero.** A missing fundamental is never filled with
  0, since that would assert a false signal ("this company earns exactly
  $0"). Model packages refuse to run on `NA` by design; that is treated as a
  feature, not something to work around by imputing zeros.
- **Outliers are flagged, not dropped.** Automated dropping biases the panel.
  See `flag_outliers()` in `scripts/02_clean_fundamentals.py` for the
  pattern.
- **Forward-fill is for short gaps, not structural ones.** A tag going quiet
  for 20+ consecutive quarters for one ticker is a data problem to diagnose,
  not something to silently paper over with more forward-fill.
- **CapEx tags are resolved by priority, never summed**, since SEC filers use
  several mutually exclusive tags for the same underlying capital
  expenditure and a filer can even split reporting across two of them within
  its own history (the broadest available tag wins). Revenue tags, by
  contrast, are genuinely mutually exclusive aliases and are correctly
  collapsed via `REVENUE_ALIASES`; these two patterns should not be
  conflated.
- **Growth/ratio features computed off a negative or non-positive base are
  set to `NA`, not computed** (e.g. `ni_growth`, `asset_growth`), since a
  negative-base "growth" percentage is sign-flipped nonsense, not a real
  signal.
- **A predictor only has value in a cross-sectional backtest if it varies
  across tickers within a quarter.** A feature that is constant across
  tickers in a given quarter (most raw national macro series, e.g.
  `DGS10`/`FEDFUNDS`) cannot change the ranking used to pick a "buy the top
  N" portfolio, no matter how predictive it looks in isolation --
  `scripts/03_models.R` deliberately excludes those from its model-facing
  `PREDICTORS` even though `panel.csv` keeps them.

## Known data issues

- **Filer unit-scale and tagging errors.** Individual SEC filers have been
  observed to tag a field at the wrong scale (e.g. thousands of shares
  instead of raw shares, or dollars off by a factor of 10^6) or to tag one
  variant of a field correctly while another variant of the same field is
  wrong in the same filing. `scripts/02_clean_fundamentals.py` detects these
  by comparing a value against a centered rolling median of that same
  ticker's own chronologically nearby quarters (`flag_scale_outliers()` /
  `local_median_scale()`) rather than a single all-time median, since a
  global median over-flags genuine long-run growth (a small-cap's early
  history, or a pre-merger shell company). Values off by more than 50x from
  their local neighborhood are nulled, not guessed at, and forward-filled
  downstream like any other gap. `shares_outstanding` additionally has an
  absolute floor (any value under 1,000,000 is nulled unconditionally),
  since this dataset is S&P 500 constituents only and a sustained multi-year
  tagging error in one filer's primary share-count tag can pollute even a
  ticker's own recent-history baseline.
- **Four tickers (AZO, COST, PEP, YUM) contribute zero complete-case rows**
  under the 15 model-facing predictors, traced to fiscal-calendar-alignment
  gaps in `ni_growth`/`capex_intensity` specific to those four. This is a
  deliberate tradeoff, not an oversight: fixing it would require loosening
  either the TTM-completeness requirement or the negative-base-growth rule
  above, both of which would cost more data-quality guarantees than the four
  tickers are worth.
- **A handful of duplicate `(ticker, quarter_end)` rows exist in
  `panel.csv`.** These rows have identical price/return/technical columns
  but differing fundamentals-derived columns, consistent with the
  point-in-time fundamentals join occasionally fanning out to more than one
  qualifying filing instead of collapsing to exactly one. The root cause has
  not been fixed at the source; `scripts/03_models.R` and
  `scripts/04_results.R` each apply a deterministic first-occurrence dedup
  (`distinct(ticker, quarter_end, .keep_all = TRUE)`) immediately after
  reading `panel.csv`. `scripts/05_top30_forecast.R`'s ticker selection uses
  the *modal* (most common) pooled validation row count rather than a literal
  maximum for the same reason -- the duplicate rows can otherwise inflate one
  ticker's count above the genuine ceiling shared by most tickers.
- **`quantmod::getDividends` (and the JSON parsing beneath it) can crash the
  R process outright**, not just raise a catchable error, when fetching
  dividend-event data from Yahoo Finance in some environments. This is not a
  rate-limiting, network, or malformed-data issue -- it has been isolated to
  the R/jsonlite parsing path itself. `scripts/get_dividends_fallback.py`
  fetches the same data from the same source in Python as a fallback with an
  identical output schema; `main.R` runs the dividends step as its own
  subprocess so a crash there does not take down the rest of the pipeline. A
  separate, transient Yahoo rate-limit has also been observed (plain price
  fetches occasionally failing after a burst of requests, recovering after a
  cooldown) -- this is a different issue from the jsonlite crash and should
  not be assumed to have the same cause.

## Ticker universe

The universe is 222 tickers: the original 28-ticker water/infrastructure
sector universe (regulated water utilities and adjacent industrials) plus 194
additional S&P 500 tickers added to give full GICS sector coverage. Every one
of the 11 GICS sectors is represented, though Financials, Communication
Services, and Real Estate are the thinnest, reflecting a higher rate of
structural rejects in those sectors (see below).

Sector breakdown (original-28 count + added count = total): Industrials
22+14=36, Information Technology 1+31=32, Health Care 0+26=26, Financials
0+24=24, Consumer Discretionary 0+24=24, Utilities 4+13=17, Consumer Staples
0+15=15, Real Estate 0+13=13, Materials 1+12=13, Energy 0+12=12,
Communication Services 0+10=10.

`TICKERS` in `scripts/02_fundamentals.py` is the source of truth; `STOCKS` in
`scripts/01_get_data.R` and `scripts/get_dividends_fallback.py` are kept in
sync with it by hand. This is a deliberate convention in this codebase --
ticker/benchmark lists are hardcoded per-script rather than imported from a
shared config module -- and should be preserved rather than centralized.

**Before adding any ticker to the universe, run
`scripts/verifications.py`.** A candidate can look fine on the surface (long
filing history, a regulated industry) and still be unusable if it never tags
an aggregate CapEx figure or another field the model needs -- this script
scores a candidate on every tag the pipeline actually relies on
(`roe`, `debt_to_equity`, `shares`, `free_cash_flow`, etc.) before it is
added, rather than discovering the gap after a full scrape and clean. Past
screening has surfaced several recurring structural gaps worth checking for
in any future addition: Financials/REITs with an unclassified balance sheet
(no `current_ratio`), companies that redomiciled and changed CIK mid-history,
delisted or renamed tickers, and filers whose bulk company-tickers listing
omits them, requiring a manual CIK override.
