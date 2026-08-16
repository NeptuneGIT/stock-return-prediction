# Quarterly Stock Return Prediction

Final submitted version of a quarterly panel spanning 222 S&P 500 stocks
across all 11 GICS sectors, used to train PLS, Random Forest, and Gradient
Boosted Regression Trees (GBRT) models that predict each ticker's
next-quarter return relative to SPY. Models are validated with walk-forward
(expanding-window) cross-validation to avoid look-ahead bias, backtested as
a "buy the top N" strategy against SPY, and used to produce a forward-looking
top-30 forecast. Built for an undergraduate Economics + Computer Science
ML/Data Analytics course.

**Target variable:** `exret_next` = a ticker's next-quarter return minus
SPY's next-quarter return over the same period.

## Running it

```
Rscript main.R
```

`main.R` runs the full pipeline in order: data scrape (prices/dividends/FRED
+ SEC EDGAR fundamentals), panel construction, model training, backtest, and
top-30 forecast. `SKIP_SCRAPE` at the top of `main.R` defaults to `TRUE` to
reuse data already on disk rather than re-hitting rate-limited external
APIs.

SEC EDGAR requires a real contact string in the User-Agent header on every
request. Set it before running anything that scrapes SEC EDGAR
(`scripts/02_fundamentals.py`, `scripts/verifications.py`):

```
export SEC_USER_AGENT="Your Name your.email@example.com"
```

**R** packages: `dplyr`, `readr`, `tidyr`, `lubridate`, `zoo`, `data.table`,
`tidyquant`, `pls`, `randomForest`, `gbm`.

**Python 3** packages: `pandas`, `numpy`, `requests`.
