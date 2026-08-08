# Quarterly Stock Return Prediction (Water Sector)

Undergraduate Economics + Computer Science ML/Data Analytics course
project. Predicts a stock's next-quarter return in excess of PHO
(Invesco Water Resources ETF) from a mix of technical and fundamental
predictors, using Lasso, Partial Least Squares (PLS), and Bagged
Regression Splines.

**Universe:** 28 tickers -- the original 8 regulated water utilities
(AWK, WTRG, CWT, AWR, XYL, VRT, JCI, BMI) plus 20 water-adjacent
equipment/infrastructure/industrial names, each screened via
`scripts/verifications.py` before being added.

**Target variable:** `exret_next` = a ticker's next-quarter return minus
PHO's next-quarter return.

**No look-ahead bias:** fundamentals are joined to a quarter by SEC
filing date, not the fiscal period they describe, and all technical
features use trailing windows only. Validation is a strict time-based
split (train on quarters before 2023-01-01, test from 2023-01-01
onward) rather than random cross-validation.

## Requirements

SEC EDGAR requires a real contact string in the User-Agent header.
Set it before running anything that hits SEC EDGAR:

```
export SEC_USER_AGENT="Your Name your.email@example.com"
```

## Running it

There is no single runner script yet -- run each stage in order:

```
Rscript scripts/01_get_data.R
python scripts/02_fundamentals.py
python scripts/02_clean_fundamentals.py
Rscript scripts/01a_ratios.R
Rscript scripts/03_models.R
```

If the dividend-download step in `01_get_data.R` fails or returns no
rows, run `python scripts/get_dividends_fallback.py` instead -- it
hits the same Yahoo Finance source and writes the same output schema.
