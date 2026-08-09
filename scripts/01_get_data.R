# =============================================================================
# 01_get_data.R
#
# Downloads everything that comes from market/macro sources:
#   - Daily adjusted prices for the stock universe + SPY (the sole benchmark)
#   - Dividend history for the stock universe, needed for the div_yield feature
#   - FRED macroeconomic series
#
# This script ONLY downloads and saves raw data. All feature engineering
# happens in 02_features.R, so you can re-derive features without
# re-hitting the APIs.
#
# The universe has grown from an initial 8-ticker water-utility set to the
# 222-ticker S&P 500 universe below, with every added ticker verified via
# verify_candidate_tickers.py (scripts/verifications.py) first -- see that
# file and 02_fundamentals.py's TICKERS list (byte-identical, kept in sync
# by hand) for the full per-sector breakdown and screening rationale. PHO
# (the original water-sector benchmark) has been removed entirely; SPY is
# now the sole benchmark for exret_next, so BENCHMARKS is a single-element
# vector. START_DATE is 2003-01-01 to give the ~20-year study window
# enough runway for trailing features like 12-month momentum.
#
# Dividends come from Yahoo (via tidyquant) rather than SEC XBRL
# deliberately: Yahoo gives actual ex-dividend CASH events, which is what a
# shareholder receives, whereas SEC's CommonStockDividendsPerShareDeclared
# is a *declared* figure that many filers tag only annually -- blending
# declared-vs-paid across tickers would give div_yield an inconsistent
# meaning per stock. No look-ahead risk: a dividend with ex-date D is
# public on D.
#
# VRT (Vertiv) traded as a SPAC (GS Acquisition Holdings Corp) before its
# 2020-02-07 merger; pre-merger rows are trust-NAV noise pinned near
# $10/share, not real VRT prices, and are dropped before writing
# prices_daily.csv so no feature window spanning the merger gets corrupted
# by non-Vertiv prices.
#
# Output: data/raw/prices_daily.csv
#         data/raw/dividends_raw.csv
#         data/raw/macro_fred.csv
# =============================================================================

library(tidyquant)
library(dplyr)
library(readr)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# NOTE: MSEX, SJW, WTS, and CR were considered and excluded after
# auditing their SEC filings (thin/absent XBRL tagging for a feature
# the model needs -- see EXCLUDED_TICKERS in 02_clean_fundamentals.py
# for the specific reason for each). BMI (Badger Meter) was added as
# SJW's replacement; the 20 tickers below PNR..MTZ were added in the
# original 8 -> 28 water-sector expansion, and the 194 tickers below MTZ
# were added in the subsequent pivot to the full S&P 500 universe. Every
# addition was verified via verify_candidate_tickers.py to report all
# needed tags cleanly -- must stay in sync with TICKERS in
# 02_fundamentals.py (byte-identical list, kept in sync by hand per this
# repo's convention; see that file for the full per-sector breakdown).
STOCKS <- c(
  # Original 8-ticker universe
  "AWK", "WTRG", "CWT", "AWR", "XYL", "VRT", "JCI", "BMI",
  # Added in the 8 -> 28 expansion
  "PNR", "MWA", "AOS", "ITT", "IEX", "FELE", "FLS", "DOV", "ECL", "GRC",
  "ITRI", "MAS", "ROP", "EMR", "PH", "HON", "WMS", "GVA", "PWR", "MTZ",
  # --- Information Technology (32 total: 1 original + 31 new) ---
  "MSFT", "AAPL", "NVDA", "ORCL", "CRM", "ADBE", "CSCO", "AMD", "QCOM",
  "TXN", "IBM", "INTU", "AMAT", "ADI", "LRCX", "KLAC", "MU", "SNPS",
  "CDNS", "ADSK", "MCHP", "ON", "TER", "WDC", "STX", "NTAP", "TYL",
  "PTC", "SWKS", "GRMN", "ZBRA",
  # --- Health Care (26 total, all new) ---
  "JNJ", "UNH", "LLY", "MRK", "PFE", "TMO", "ABT", "DHR", "BMY", "AMGN",
  "GILD", "CVS", "ELV", "HUM", "SYK", "BSX", "ISRG", "ZBH", "BDX", "BAX",
  "VRTX", "BIIB", "MCK", "COR", "HCA", "DVA",
  # --- Financials (24 total, all new) ---
  "C", "GS", "MS", "BNY", "SCHW", "SPGI", "MCO", "ICE", "AON", "MRSH",
  "AJG", "ALL", "PGR", "HIG", "STT", "FITB", "AXP", "COF", "NDAQ",
  "MSCI", "FDS", "CBOE", "NTRS", "WTW",
  # --- Consumer Discretionary (24 total, all new) ---
  "AMZN", "TSLA", "HD", "MCD", "LOW", "SBUX", "TJX", "ORLY", "AZO",
  "ROST", "YUM", "MAR", "HLT", "GM", "APTV", "BBY", "DHI", "PHM", "NVR",
  "WHR", "TSCO", "ULTA", "GPC", "CMG",
  # --- Communication Services (10 total, all new) ---
  "NFLX", "T", "VZ", "TMUS", "EA", "TTWO", "OMC", "LYV", "MTCH", "SIRI",
  # --- Industrials (36 total: 22 original + 14 new) ---
  "CAT", "DE", "UNP", "FDX", "BA", "LMT", "RTX", "GD", "NOC", "CSX",
  "NSC", "WM", "RSG", "GE",
  # --- Consumer Staples (15 total, all new) ---
  "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "MDLZ", "CL", "KMB",
  "GIS", "ADM", "CAG", "CLX", "CHD",
  # --- Energy (12 total, all new) ---
  "CVX", "EOG", "SLB", "MPC", "VLO", "OXY", "WMB", "KMI", "OKE", "DVN",
  "HAL", "TRGP",
  # --- Utilities (17 total: 4 original water utilities + 13 new) ---
  "AEP", "DUK", "SO", "D", "EXC", "XEL", "ED", "PEG", "ES", "FE", "ETR",
  "EIX", "PPL",
  # --- Real Estate (13 total, all new) ---
  "AMT", "EQIX", "CCI", "PSA", "O", "WELL", "AVB", "EQR", "VTR", "IRM",
  "UDR", "HST", "BXP",
  # --- Materials (13 total: 1 original + 12 new) ---
  "APD", "SHW", "FCX", "NEM", "PPG", "NUE", "ALB", "CE", "IFF", "MLM",
  "VMC", "IP"
)

# SPY is the sole benchmark: exret_next is a ticker's next-quarter return
# minus SPY's next-quarter return. PHO (the original water-sector
# benchmark) has been removed entirely, not kept even as a passive
# column -- this project no longer tests a single-sector hypothesis.
BENCHMARKS <- c("SPY")

# Start early enough that a 12-month momentum feature is computable for
# the first quarter of the actual study window (2006, per the ~20-year
# S&P-500-universe history target) with runway to spare. 252 trading
# days of lookback needs roughly 1.5 years of runway to be safe; 2003
# gives the 2006 study start 3 years of it.
START_DATE <- "2003-01-01"
END_DATE   <- Sys.Date()

FRED_SERIES <- c(
  "DGS10",     # 10-Year Treasury constant maturity yield
  "FEDFUNDS",  # Effective federal funds rate
  "CPIAUCSL"   # CPI, all urban consumers (inflation)
)

RAW_DIR <- "data/raw"
dir.create(RAW_DIR, recursive = TRUE, showWarnings = FALSE)

# -----------------------------------------------------------------------------
# Prices
# -----------------------------------------------------------------------------

all_symbols <- c(STOCKS, BENCHMARKS)

message("Downloading daily prices for ", length(all_symbols), " symbols...")

prices_daily <- tq_get(
  all_symbols,
  get  = "stock.prices",
  from = START_DATE,
  to   = END_DATE
)

# tq_get returns a `symbol` column. BOTH `close` and `adjusted` are kept
# and they are NOT interchangeable:
#   - `adjusted` back-corrects for splits AND dividends. Use it for
#     RETURNS (momentum, volatility, the target). Raw close would make
#     a 2-for-1 split look like a -50% return.
#   - `close` is the actual traded price. Use it for VALUATION RATIOS
#     (pb_ratio, fcf_yield, div_yield). Using adjusted there would
#     understate the historical price -- because dividend adjustments
#     pull old prices down -- and therefore overstate every yield.
prices_daily <- prices_daily %>%
  select(symbol, date, open, high, low, close, volume, adjusted) %>%
  arrange(symbol, date)

# VRT (Vertiv) traded as GS Acquisition Holdings Corp, a SPAC, before
# merging with Vertiv on 2020-02-07. Pre-merger rows are trust-NAV
# noise pinned near $10/share -- NOT real Vertiv prices -- and would
# corrupt every VRT return/momentum/volatility feature computed over a
# window that spans the merger. Truncate at 2020-02-10 (first full
# trading day of "real" VRT) rather than 2020-02-07 itself, since the
# merger-day print can still carry pre-merger noise.
VRT_MERGER_CUTOFF <- as.Date("2020-02-10")
n_vrt_dropped <- sum(prices_daily$symbol == "VRT" & prices_daily$date < VRT_MERGER_CUTOFF)
if (n_vrt_dropped > 0) {
  message("Dropping ", n_vrt_dropped,
          " pre-merger VRT rows (SPAC-era trust-NAV noise, before ",
          VRT_MERGER_CUTOFF, ").")
}
prices_daily <- prices_daily %>%
  filter(symbol != "VRT" | date >= VRT_MERGER_CUTOFF)

# Fail loudly if a symbol came back empty -- a silent NULL here would
# quietly drop a stock from the whole study.
missing_syms <- setdiff(all_symbols, unique(prices_daily$symbol))
if (length(missing_syms) > 0) {
  stop("No price data returned for: ", paste(missing_syms, collapse = ", "))
}

# Report coverage so a short history (e.g. a recent IPO) is visible now
# rather than as a confusing gap in the panel later. VRT in particular
# only has data from its 2020 SPAC listing onward.
coverage <- prices_daily %>%
  group_by(symbol) %>%
  summarise(
    n_days = n(),
    first  = min(date),
    last   = max(date),
    .groups = "drop"
  )
print(coverage, n = Inf)

write_csv(prices_daily, file.path(RAW_DIR, "prices_daily.csv"))
message("Saved ", nrow(prices_daily), " price rows.")

# -----------------------------------------------------------------------------
# Dividends                                                            [NEW]
# -----------------------------------------------------------------------------
# One row per ex-dividend event: symbol, date, cash amount per share.
# 02_features.R sums the trailing 365 days of these to get a TTM
# dividend, then divides by the raw close price.
#
# KNOWN ENVIRONMENT ISSUE: this step has been observed to crash the R
# session outright (segfault, not a catchable error) when parsing Yahoo's
# dividend response via quantmod::getDividends -- root-caused to an
# R/jsonlite bug unrelated to rate-limiting, the ticker list, or this
# code. If this step fails or returns no data, run
# `python scripts/get_dividends_fallback.py` instead, which hits the
# identical Yahoo source and writes the identical output schema.

message("Downloading dividend history...")

divs_raw <- tryCatch(
  tq_get(STOCKS, get = "dividends", from = START_DATE, to = END_DATE),
  error = function(e) {
    message("  Dividend download FAILED: ", conditionMessage(e))
    NULL
  }
)

if (is.null(divs_raw) || nrow(divs_raw) == 0) {
  # Deliberately a warning, not a stop(): everything except div_yield
  # is still usable, and 02_features.R handles a missing file. But do
  # NOT quietly proceed to modelling with div_yield silently all-NA.
  warning("No dividend data returned -- div_yield will be unavailable.")
} else {
  # tidyquant has used both `value` and `dividends` as the amount
  # column name across versions, so resolve it rather than hardcode.
  # Guessing wrong here would produce a column of NAs that looks like
  # a genuine "this company pays nothing" signal.
  amount_col <- setdiff(names(divs_raw)[sapply(divs_raw, is.numeric)], "symbol")
  if (length(amount_col) != 1) {
    stop("Could not identify the dividend amount column. Got: ",
         paste(names(divs_raw), collapse = ", "))
  }
  names(divs_raw)[names(divs_raw) == amount_col] <- "dividend"
  
  dividends_raw <- divs_raw %>%
    select(symbol, date, dividend) %>%
    filter(!is.na(dividend), dividend > 0) %>%
    arrange(symbol, date)
  
  write_csv(dividends_raw, file.path(RAW_DIR, "dividends_raw.csv"))
  message("Saved ", nrow(dividends_raw), " dividend events.")
  
  div_coverage <- dividends_raw %>%
    group_by(symbol) %>%
    summarise(
      n_payments = n(),
      first      = min(date),
      last       = max(date),
      last_amt   = dplyr::last(dividend),
      .groups = "drop"
    )
  print(div_coverage, n = Inf)
  
  # All eight of these tickers pay a dividend. A ticker absent here
  # means the fetch failed for it, NOT that it pays nothing -- and
  # those two cases must never be conflated, since one is a real 0%
  # yield and the other is missing data.
  no_divs <- setdiff(STOCKS, unique(dividends_raw$symbol))
  if (length(no_divs) > 0) {
    warning("No dividends returned for: ", paste(no_divs, collapse = ", "),
            " -- verify manually before treating their yield as zero.")
  }
}

# -----------------------------------------------------------------------------
# FRED macro data
# -----------------------------------------------------------------------------

message("Downloading FRED series...")

macro_fred <- tq_get(
  FRED_SERIES,
  get  = "economic.data",
  from = START_DATE,
  to   = END_DATE
)

# tq_get names the value column `price` for economic.data -- rename so
# it isn't mistaken for a stock price downstream.
macro_fred <- macro_fred %>%
  rename(series_id = symbol, value = price) %>%
  arrange(series_id, date)

missing_series <- setdiff(FRED_SERIES, unique(macro_fred$series_id))
if (length(missing_series) > 0) {
  stop("No FRED data returned for: ", paste(missing_series, collapse = ", "))
}

# These series have different native frequencies (DGS10 is daily,
# FEDFUNDS and CPIAUCSL are monthly). 02_features.R handles the
# alignment; this just records what arrived.
macro_coverage <- macro_fred %>%
  group_by(series_id) %>%
  summarise(n_obs = n(), first = min(date), last = max(date), .groups = "drop")
print(macro_coverage)

write_csv(macro_fred, file.path(RAW_DIR, "macro_fred.csv"))
message("Saved ", nrow(macro_fred), " macro rows.")

message("\n01_get_data.R complete.")