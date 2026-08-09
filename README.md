# Quarterly Stock Return Prediction

An Economics + Computer Science ML/Data Analytics course project. The
universe just pivoted from a small water-sector set to the full S&P 500:
222 tickers spanning all 11 GICS sectors, screened via
`scripts/verifications.py` before being added. Models are Partial Least
Squares (PLS), Random Forest, and Gradient Boosted Regression Trees
(GBRT), predicting each ticker's next-quarter return relative to SPY
(`exret_next`).

## Pipeline

1. `scripts/01_get_data.R` -- daily prices, dividends, FRED macro series
2. `scripts/02_fundamentals.py` -- SEC EDGAR fundamentals scrape
3. `scripts/02_clean_fundamentals.py` -- cleans and derives ratios
4. `scripts/01a_ratios.R` -- builds the quarterly modelling panel
5. `scripts/03_models.R` -- fits and evaluates PLS / Random Forest / GBRT

Run the R scripts with `Rscript <path>` and the Python scripts with
`python3 <path>`, in the order above. Fundamentals scraping requires the
`SEC_USER_AGENT` environment variable to be set to a real contact string
(SEC EDGAR rejects unidentified traffic).

## Requirements

R: `tidyquant`, `dplyr`, `readr`, `pls`, `randomForest`, `gbm`.
Python 3: `requests`, `pandas`, `numpy`.
