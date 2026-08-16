# 01_get_data.R -- pulls daily prices (stocks + SPY), dividend history, and
# FRED macro series. Just downloads raw data, so features can be redone without re-hitting the APIs.
#
# Output: data/raw/prices_daily.csv, dividends_raw.csv, macro_fred.csv.

library(tidyquant)
library(dplyr)
library(readr)

# Full 222-ticker S&P 500 universe, all 11 GICS sectors, grouped by sector
# below. Every ticker screened via verifications.py first. 
STOCKS <- c(
  # Original 8-ticker universe
  "AWK", "WTRG", "CWT", "AWR", "XYL", "VRT", "JCI", "BMI",
  # Added in the 8 -> 28 expansion
  "PNR", "MWA", "AOS", "ITT", "IEX", "FELE", "FLS", "DOV", "ECL", "GRC",
  "ITRI", "MAS", "ROP", "EMR", "PH", "HON", "WMS", "GVA", "PWR", "MTZ",
  # --- Information Technology ---
  "MSFT", "AAPL", "NVDA", "ORCL", "CRM", "ADBE", "CSCO", "AMD", "QCOM",
  "TXN", "IBM", "INTU", "AMAT", "ADI", "LRCX", "KLAC", "MU", "SNPS",
  "CDNS", "ADSK", "MCHP", "ON", "TER", "WDC", "STX", "NTAP", "TYL",
  "PTC", "SWKS", "GRMN", "ZBRA",
  # --- Health Care ---
  "JNJ", "UNH", "LLY", "MRK", "PFE", "TMO", "ABT", "DHR", "BMY", "AMGN",
  "GILD", "CVS", "ELV", "HUM", "SYK", "BSX", "ISRG", "ZBH", "BDX", "BAX",
  "VRTX", "BIIB", "MCK", "COR", "HCA", "DVA",
  # --- Financials  ---
  "C", "GS", "MS", "BNY", "SCHW", "SPGI", "MCO", "ICE", "AON", "MRSH",
  "AJG", "ALL", "PGR", "HIG", "STT", "FITB", "AXP", "COF", "NDAQ",
  "MSCI", "FDS", "CBOE", "NTRS", "WTW",
  # --- Consumer Discretionary ---
  "AMZN", "TSLA", "HD", "MCD", "LOW", "SBUX", "TJX", "ORLY", "AZO",
  "ROST", "YUM", "MAR", "HLT", "GM", "APTV", "BBY", "DHI", "PHM", "NVR",
  "WHR", "TSCO", "ULTA", "GPC", "CMG",
  # --- Communication Services---
  "NFLX", "T", "VZ", "TMUS", "EA", "TTWO", "OMC", "LYV", "MTCH", "SIRI",
  # --- Industrials---
  "CAT", "DE", "UNP", "FDX", "BA", "LMT", "RTX", "GD", "NOC", "CSX",
  "NSC", "WM", "RSG", "GE",
  # --- Consumer Staples  ---
  "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "MDLZ", "CL", "KMB",
  "GIS", "ADM", "CAG", "CLX", "CHD",
  # --- Energy ---
  "CVX", "EOG", "SLB", "MPC", "VLO", "OXY", "WMB", "KMI", "OKE", "DVN",
  "HAL", "TRGP",
  # --- Utilities ---
  "AEP", "DUK", "SO", "D", "EXC", "XEL", "ED", "PEG", "ES", "FE", "ETR",
  "EIX", "PPL",
  # --- Real Estate ---
  "AMT", "EQIX", "CCI", "PSA", "O", "WELL", "AVB", "EQR", "VTR", "IRM",
  "UDR", "HST", "BXP",
  # --- Materials ---
  "APD", "SHW", "FCX", "NEM", "PPG", "NUE", "ALB", "CE", "IFF", "MLM",
  "VMC", "IP"
)

# SPY is the sole benchmark now

BENCHMARKS <- c("SPY")

# Starts 3 years before the study window so 12-month momentum has runway.
START_DATE <- "2003-01-01"
END_DATE   <- Sys.Date()

FRED_SERIES <- c(
  "DGS10",     # 10-Year Treasury constant maturity yield
  "FEDFUNDS",  # Effective federal funds rate
  "CPIAUCSL"   # CPI, all urban consumers (inflation)
)

RAW_DIR <- "data/raw"
dir.create(RAW_DIR, recursive = TRUE, showWarnings = FALSE)

all_symbols <- c(STOCKS, BENCHMARKS)

message("Downloading daily prices for ", length(all_symbols), " symbols...")

prices_daily <- tq_get(
  all_symbols,
  get  = "stock.prices",
  from = START_DATE,
  to   = END_DATE
)

# Keep both close and adjusted -- not interchangeable. adjusted (splits +
# dividends) for returns, close (actual traded price) for valuation ratios.
prices_daily <- prices_daily %>%
  select(symbol, date, open, high, low, close, volume, adjusted) %>%
  arrange(symbol, date)

# VRT traded as a SPAC (GS Acquisition Holdings) before merging 2020-02-07.
# Pre-merger rows are trust-NAV noise pinned near $10, not real VRT prices --
# truncate at 2020-02-10 to be safe of merger-day noise.
VRT_MERGER_CUTOFF <- as.Date("2020-02-10")
n_vrt_dropped <- sum(prices_daily$symbol == "VRT" & prices_daily$date < VRT_MERGER_CUTOFF)
if (n_vrt_dropped > 0) {
  message("Dropping ", n_vrt_dropped,
          " pre-merger VRT rows (SPAC-era trust-NAV noise, before ",
          VRT_MERGER_CUTOFF, ").")
}
prices_daily <- prices_daily %>%
  filter(symbol != "VRT" | date >= VRT_MERGER_CUTOFF)

# Fail loudly if a symbol came back empty rather than silently dropping it.
missing_syms <- setdiff(all_symbols, unique(prices_daily$symbol))
if (length(missing_syms) > 0) {
  stop("No price data returned for: ", paste(missing_syms, collapse = ", "))
}

# Coverage report so a short history (recent IPO, etc.) is visible now.
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



# Known issue: this can segfault the R session. If it fails, run get_dividends_fallback.py instead.

message("Downloading dividend history...")

divs_raw <- tryCatch(
  tq_get(STOCKS, get = "dividends", from = START_DATE, to = END_DATE),
  error = function(e) {
    message("  Dividend download FAILED: ", conditionMessage(e))
    NULL
  }
)

if (is.null(divs_raw) || nrow(divs_raw) == 0) {
  warning("No dividend data returned -- div_yield will be unavailable.")
} else {
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
  
  no_divs <- setdiff(STOCKS, unique(dividends_raw$symbol))
  if (length(no_divs) > 0) {
    warning("No dividends returned for: ", paste(no_divs, collapse = ", "),
            " -- verify manually before treating their yield as zero.")
  }
}


message("Downloading FRED series...")

macro_fred <- tq_get(
  FRED_SERIES,
  get  = "economic.data",
  from = START_DATE,
  to   = END_DATE
)

macro_fred <- macro_fred %>%
  rename(series_id = symbol, value = price) %>%
  arrange(series_id, date)

missing_series <- setdiff(FRED_SERIES, unique(macro_fred$series_id))
if (length(missing_series) > 0) {
  stop("No FRED data returned for: ", paste(missing_series, collapse = ", "))
}

macro_coverage <- macro_fred %>%
  group_by(series_id) %>%
  summarise(n_obs = n(), first = min(date), last = max(date), .groups = "drop")
print(macro_coverage)

write_csv(macro_fred, file.path(RAW_DIR, "macro_fred.csv"))
message("Saved ", nrow(macro_fred), " macro rows.")

message("\n01_get_data.R complete.")