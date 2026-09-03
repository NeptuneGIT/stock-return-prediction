# Quarterly Stock Return Prediction

A quarterly panel of 222 S&P 500 stocks spanning all 11 GICS sectors and
2010–2026, used to train Partial Least Squares (PLS), Random Forest, and
Gradient Boosted Regression Trees (GBRT) models that predict each ticker's
next-quarter return relative to SPY. The trained models are backtested as a
"buy the top N" strategy against SPY and used to produce a forward-looking
top-30 forecast. Built for an undergraduate Economics + Computer Science
ML/Data Analytics course. The full write-up, including the project's
narrative history, is in
[`docs/Stock_Return_Prediction_Report.pdf`](docs/Stock_Return_Prediction_Report.pdf).

**Target variable:** `exret_next` = a ticker's next-quarter return minus
SPY's next-quarter return over the same period.

## Background

The project started with a much narrower goal: an 8-stock universe of
regulated water utilities and water-adjacent industrials (AWK, AWR, BMI, CWT,
JCI, VRT, WTRG, XYL), on the theory that a sector-specific universe would
surface drivers unique to water infrastructure. Three models suited to a
small, correlated predictor set (Lasso, PLS, Bagged Regression Splines) were
trained on 16 fundamental/technical predictors. Every model produced a
negative out-of-sample adjusted R² (Lasso −0.13, PLS −1.26, Bagged Splines
−0.13) — the water universe was expanded to 28 tickers to test whether more
data alone would fix it, and while all three models' test RMSE and adjusted
R² improved, none reached a positive out-of-sample R².

The diagnosis: 8–28 water-sector stocks were both too few observations
relative to the predictor count, and too cross-sectionally correlated — the
tickers shared the same underlying drivers and macro exposures, so most of
the "extra" data was redundant rather than independent signal. *Empirical
Asset Pricing via Machine Learning* (Gu, Kelly & Xiu, 2018 NBER working
paper) corroborates this: even with a sample of nearly 30,000 stocks over 60
years, that literature's models need enough usable data relative to their
predictor set to produce a stable, positive out-of-sample R² — the
water-only universe fell far short of that bar by construction, not because
of a model or tuning deficiency.

In response, three changes were made: the universe was expanded from 8 to
222 tickers pulled from the full S&P 500 across all 11 GICS sectors (directly
addressing the cross-sectional correlation problem); the predictor set
shifted from mostly-fundamental to a mix weighted more heavily toward
technical indicators, after significance testing showed many fundamental
predictors were not significant at the 95% level while most technical ones
held up; and the model shortlist moved from Lasso/PLS/Bagged Splines to
PLS/Random Forest/GBRT, retaining PLS (its relative performance had been
decent) and replacing the other two with methods better suited to a larger,
more varied dataset.

## Data

- **Fundamentals** are scraped from SEC EDGAR's company-facts XBRL API
  (`scripts/02_fundamentals.py`), chosen over alternatives like Macrotrends
  specifically because EDGAR reports each fact's *filing* date, not just the
  fiscal period it describes — without a filing date there is no way to
  align a number with when it actually became public, which risks look-ahead
  bias (see [Methodology](#methodology)).
- **Prices and dividends** come from Yahoo Finance via the `tidyquant` R
  package (`scripts/01_get_data.R`), with a Python fallback for dividends
  (`scripts/get_dividends_fallback.py`) — see
  [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for why the fallback exists.
- **Macro series** (10-year Treasury yield, effective fed funds rate, CPI)
  come from FRED, also via `tidyquant`.
- **Coverage:** 222 tickers across all 11 GICS sectors, quarterly, spanning
  2010–2026, with an 84.1% complete-case rate (12,052 of 14,328
  target-bearing rows) on the 15 predictors used for modelling.

## Predictors

| Category | Predictors |
|---|---|
| Technical / Risk | `mom_1m`, `mom_3m`, `mom_12m` (1/3/12-month momentum), `vol_60d` (60-day volatility), `dollar_volume`, `beta_252d` (252-day beta), `idio_vol_60d` (60-day idiosyncratic volatility), `pct_from_52w_high` |
| Fundamentals (SEC) | `roe`, `debt_to_equity`, `asset_growth`, `ni_growth`, `capex_intensity` |
| Price × Fundamentals | `pb_ratio`, `fcf_yield` |

These 15 names are exactly the `PREDICTORS` list in `scripts/03_models.R`.
`panel.csv` itself carries a few additional columns (e.g. `current_ratio`,
`div_yield`) that are excluded from modelling for missingness reasons — see
`docs/METHODOLOGY.md`.

## Methodology

- **No look-ahead bias.** Fundamentals are joined to a quarter by SEC filing
  date, never the fiscal period the numbers describe, and every technical
  feature uses a trailing (never centered) window. This is enforced with an
  assertion in `scripts/01a_ratios.R` that fails loudly if any filing date is
  later than the quarter it was joined to.
- **Walk-forward validation**, not k-fold cross-validation, is the only
  source of out-of-sample performance numbers in this project: expanding-
  window, time-respecting folds where every fold trains only on quarters
  strictly before its own validation block. Random k-fold CV would let a
  training fold include quarters *after* the one being validated, leaking
  future information into the past. GBRT's `n.trees` and PLS's `ncomp` are
  chosen this way; Random Forest's `mtry` is fixed at the package default
  rather than grid-searched (see `docs/METHODOLOGY.md` for why). The pooled
  out-of-fold predictions this produces are the only historical-quarter
  predictions ever used for a performance metric or the backtest.
- A separate model, refit once on the *entire* panel at each model's
  walk-forward-selected hyperparameter, is used only for variable importance
  and to forward-score the current not-yet-realized quarter for the top-30
  forecast — never to generate a historical-quarter prediction, since that
  fit has seen every quarter and doing so would leak the future into the
  past.

See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for the full reasoning,
data-quality conventions, known data issues, and the ticker-screening
workflow.

## Results

Pooled walk-forward out-of-fold performance across the entire panel (n =
8,951 pooled rows across 11 folds, 2010-03-31 to 2026-03-31), as produced by
`scripts/03_models.R` and written to
`data/processed/model_metrics_summary.csv`:

| Model | Hyperparameter | Walk-forward RMSE | Walk-forward Adjusted R² |
|---|---|---|---|
| GBRT | n.trees = 116 | 0.1312 | +0.0084 |
| PLS | ncomp = 1 | 0.1310 | +0.0113 |
| Random Forest | mtry = 5 (fixed) | 0.1329 | −0.0175 |

GBRT is used throughout this project as the primary model — PLS's adjusted
R² numerically edges it out on this run, and has on other runs too, but the
model choice is not re-litigated run to run. Random Forest is negative. All
three effect sizes are small; see [Limitations](#limitations).

GBRT's top predictors by relative influence (`data/processed/gbrt_importance.csv`):
`mom_12m` (25.8%), `idio_vol_60d` (16.6%), `pct_from_52w_high` (16.6%),
`beta_252d` (10.9%), `mom_1m` (7.4%).

**Backtest** (`scripts/04_results.R`), N=20 "buy the top N" portfolio ranked
on GBRT's pooled out-of-fold walk-forward predictions, held quarterly,
equal-weighted, no transaction costs, over the full ~44-quarter pooled
walk-forward window: see `data/processed/backtest_summary.csv` for the
current run's numbers across the full N sensitivity grid (10/20/30/50).
Every number here is deterministic given the fixed seed and reproduces
exactly on re-run unless the code or `panel.csv` changes.

**Top-30 forward forecast** (`scripts/05_top30_forecast.R`): ranking the 30
tickers with the highest per-ticker pooled walk-forward validation R²,
forecasting Q3 2026 (1-quarter-ahead) and Q2 2027 (4-quarter-ahead, via a
supplementary GBRT model fit against a re-derived 4-quarter-ahead target) —
average predicted outperformance of +1.12% vs. SPY for Q3 2026 and +3.32%
for Q2 2027. See `output/tables/top30_forecast_<date>.csv` for the full
per-ticker report and its printed caveats (read before using this for
anything).

## Repository layout

```
scripts/                     Pipeline stages (see below)
docs/
  METHODOLOGY.md               No-look-ahead rules, data-quality conventions,
                                known issues, ticker-screening workflow
  Stock_Return_Prediction_Report.pdf  Full project write-up
data/raw/                    Scraped, unmodified source data (not committed)
  prices_daily.csv              Daily OHLCV, all tickers + SPY
  dividends_raw.csv             Dividend events
  macro_fred.csv                 FRED macro series
  fundamentals_raw.csv          Raw SEC EDGAR XBRL facts
data/processed/               Derived data and model outputs (not committed)
  fundamentals_clean.csv        Cleaned/derived fundamentals
  panel.csv                     The full quarterly modelling panel
  *_predictions.csv, *_walkforward_*.csv, *_importance.csv
                                 Per-model outputs from scripts/03_models.R
  model_metrics_summary.csv     Combined metrics table, all models
  backtest_*.csv                 Outputs from scripts/04_results.R
output/tables/                 Human-facing report outputs (not committed)
  top30_forecast_<date>.csv     Latest top-30 forecast report
main.R                         Master runner (see below)
```

`data/` and `output/` are gitignored and regenerated by running the
pipeline — see [Running it](#running-it).

## Pipeline stages

| Script | Reads | Writes |
|---|---|---|
| `scripts/01_get_data.R` | — (Yahoo Finance, FRED) | `data/raw/prices_daily.csv`, `dividends_raw.csv`, `macro_fred.csv` |
| `scripts/get_dividends_fallback.py` | — (Yahoo Finance) | `data/raw/dividends_raw.csv` (fallback for the step above — see `docs/METHODOLOGY.md`) |
| `scripts/02_fundamentals.py` | — (SEC EDGAR) | `data/raw/fundamentals_raw.csv` |
| `scripts/02_clean_fundamentals.py` | `fundamentals_raw.csv` | `data/processed/fundamentals_clean.csv` |
| `scripts/01a_ratios.R` | raw prices/dividends/macro + clean fundamentals | `data/processed/panel.csv` |
| `scripts/03_models.R` | `panel.csv` | PLS/RF/GBRT predictions, importances, walk-forward metrics, `model_metrics_summary.csv` |
| `scripts/04_results.R` | GBRT walk-forward predictions + `panel.csv` | `backtest_holdings.csv`, `backtest_quarterly_returns.csv`, `backtest_summary.csv` |
| `scripts/05_top30_forecast.R` | model outputs + `panel.csv` | `output/tables/top30_forecast_<date>.csv` |
| `scripts/verifications.py` | — (SEC EDGAR) | `output/tables/candidate_verification_<date>.csv` — a standalone ticker-screening utility for vetting *future* candidates before adding them to the universe; not part of the run-the-model chain above |

## Installation

**R** (packages: `dplyr`, `readr`, `tidyr`, `lubridate`, `zoo`,
`data.table`, `tidyquant`, `pls`, `randomForest`, `gbm`) — install with:

```
Rscript install_packages.R
```

**Python 3** (packages: `pandas`, `numpy`, `requests`):

```
pip install -r requirements.txt
```

**SEC EDGAR requires a real contact string** in the User-Agent header on
every request. Set it before running any script that hits SEC EDGAR
(`scripts/02_fundamentals.py`, `scripts/verifications.py`):

```
export SEC_USER_AGENT="Your Name your.email@example.com"
```

## Running it

```
Rscript main.R
```

`main.R` runs every stage above in order (data scrape → fundamentals
scrape/clean → panel construction → model training → backtest → top-30
forecast). By default `SKIP_SCRAPE` is `TRUE`, reusing data already on disk
from a prior scrape, since the scrape stages hit rate-limited external APIs
(SEC EDGAR, Yahoo Finance, FRED) and can take a long time — the SEC
fundamentals scrape alone makes one request per ticker with a throttling
delay, and the price/dividend pulls cover 222+ tickers back to 2003. On a
fresh clone with no data on disk, re-scrape everything with:

```
SKIP_SCRAPE=false Rscript main.R
```

## Limitations

- **Effect size is small.** Every model's walk-forward R² is close to zero
  (Random Forest's is negative). Whatever cross-sectional signal exists here
  is faint, not a strong predictive edge.
- **Survivorship bias.** The ticker universe is built from *current* S&P 500
  membership, not point-in-time historical membership, so the panel
  implicitly excludes companies that were delisted, acquired, or dropped
  from the index over the sample period.
- **A backtest "beating SPY" should be read with real skepticism** given the
  two points above — it is at least as consistent with noise as with a
  genuine, if faint, statistical edge. A negative backtest result likewise
  shouldn't be read as proof the model is worthless. Treat the model's
  cross-sectional ranking ability and its point-prediction accuracy as two
  different questions, not confirmation of each other.

## License

MIT — see [`LICENSE`](LICENSE).

## Authors

Abhinnav Abbu, Rohan Bhambri, Sarvesh Soundararajan
