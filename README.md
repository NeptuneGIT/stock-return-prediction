# Water-Sector Stock Return Prediction

An Economics + Computer Science Machine Learning / Data Analytics course
project. Predicts each stock's next-quarter return in excess of a
sector benchmark, using an 8-stock universe of regulated water utilities
and water-adjacent industrials: AWK, AWR, BMI, CWT, JCI, VRT, WTRG, XYL,
benchmarked against PHO (the water/infrastructure sector ETF).

Three models are fit and compared: Lasso (L1-penalized linear
regression), Partial Least Squares (PLS), and Bagged Regression Splines
(bagged MARS). Predictors combine SEC fundamentals (return on equity,
debt-to-equity, asset growth, net income growth, capex intensity,
current ratio) with market technicals (momentum, volatility, dividend
yield).

Fundamentals are scraped from SEC EDGAR's company-facts API rather than
a source like Macrotrends specifically because EDGAR reports each fact's
filing date, which is what lets every predictor be aligned to what a
trader could actually have known at the time.

## Running it

```
Rscript scripts/01_get_data.R          # prices, dividends, FRED macro data
python scripts/02_fundamentals.py      # SEC EDGAR fundamentals scrape
python scripts/02_clean_fundamentals.py
Rscript scripts/01a_ratios.R           # builds data/processed/panel.csv
Rscript scripts/03_models.R            # fits Lasso / PLS / Bagged Splines
```

Run `python scripts/verifications.py` before adding any ticker to the
universe -- it checks that SEC EDGAR actually has usable coverage for the
tags the model depends on before a candidate is added.
