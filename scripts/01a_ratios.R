# =============================================================================
# 02_features.R
#
# Builds the quarterly modelling panel from:
#   - data/raw/prices_daily.csv              (01_get_data.R)
#   - data/raw/dividends_raw.csv             (01_get_data.R)   [NEW]
#   - data/raw/macro_fred.csv                (01_get_data.R)
#   - data/processed/fundamentals_clean.csv  (02b_clean_fundamentals.py)
#
# THE CENTRAL RULE OF THIS SCRIPT:
# Every predictor on a row dated quarter-end Q must have been knowable
# to a trader ON that date. The target is the NEXT quarter's excess
# return. If a predictor leaks information from after Q, the backtest
# is invalid no matter how good the model looks.
#
# Two mechanisms enforce this:
#   1. Fundamentals join on `filed` (the SEC filing date), never `end`
#      (the fiscal period the numbers describe). A trader on 2023-03-31
#      does NOT have Q1 2023's earnings -- those file in late April.
#   2. Technical features use only trailing windows.
#
# Notes on a few predictor decisions:
#   - No pe_ratio/eps_ttm here -- EarningsPerShareDiluted is carried
#     through 02b_clean_fundamentals.py as a raw diagnostic column but
#     isn't turned into a valuation ratio in this panel.
#   - div_yield is computed on this side (it needs a price, which only
#     exists here), while asset_growth/capex_intensity/ni_growth/
#     current_ratio arrive READY-MADE from 02b and are simply carried
#     through the join rather than recomputed here -- the join below can
#     repeat the same filing across quarters when no new one has landed,
#     so a year-over-year change computed AFTER the join would sometimes
#     compare a filing to itself and report a false 0% growth.
#   - pb_ratio and fcf_yield use raw `close`, not `adjusted`: adjusted
#     prices are back-corrected downward for past dividends, which would
#     inflate every historical valuation yield. Returns still use
#     `adjusted`, which is correct for returns.
#   - FEDFUNDS/CPIAUCSL are forward-filled with zoo::na.locf() onto the
#     daily grid BEFORE the roll = Inf macro join, since those monthly
#     series are otherwise NA on all but ~12 rows/year at quarter-end
#     (the roll join finds the nearest date, not the nearest non-NA
#     value per column).
#   - The quarterly bucketing step drops any bucket whose quarter_end is
#     later than the most recently FULLY ELAPSED quarter
#     (last_complete_quarter_end), before qtr_ret_next/exret_next are
#     computed -- otherwise a not-yet-finished quarter's partial data
#     would get bucketed and used as a fake "next quarter" return for
#     the row before it. Fixed here at the source rather than deferred
#     downstream -- see the comment at the fix site.
#   - The universe here is 28 tickers (8 -> 28 expansion) -- see STOCKS
#     in 01_get_data.R. Nothing in this file is ticker-count-specific:
#     PREDICTORS/FILL_FLAGS below are already generic over ticker.
#
# Output: data/processed/panel.csv
# =============================================================================

library(dplyr)
library(tidyr)
library(readr)
library(lubridate)
library(zoo)
library(data.table)

RAW_DIR  <- "data/raw"
PROC_DIR <- "data/processed"
dir.create(PROC_DIR, recursive = TRUE, showWarnings = FALSE)

BENCHMARK   <- "PHO"   # excess return is measured against this
STUDY_START <- as.Date("2015-01-01")

MOM_3M_DAYS  <- 63     # ~3 months of trading days
MOM_12M_DAYS <- 252    # ~12 months of trading days
VOL_DAYS     <- 60     # 60-day realised volatility window

# -----------------------------------------------------------------------------
# 1. Load
# -----------------------------------------------------------------------------

prices <- read_csv(file.path(RAW_DIR, "prices_daily.csv"), show_col_types = FALSE)
macro  <- read_csv(file.path(RAW_DIR, "macro_fred.csv"),  show_col_types = FALSE)
fund   <- read_csv(file.path(PROC_DIR, "fundamentals_clean.csv"), show_col_types = FALSE)

fund <- fund %>% mutate(end = as.Date(end), filed = as.Date(filed))

div_path <- file.path(RAW_DIR, "dividends_raw.csv")
has_divs <- file.exists(div_path)
if (has_divs) {
  dividends <- read_csv(div_path, show_col_types = FALSE) %>%
    mutate(date = as.Date(date))
} else {
  warning("dividends_raw.csv not found -- div_yield will be all NA. ",
          "Re-run 01_get_data.R.")
  dividends <- tibble(symbol = character(), date = as.Date(character()),
                      dividend = numeric())
}

# -----------------------------------------------------------------------------
# 2. Daily technical features
# -----------------------------------------------------------------------------
# All windows are TRAILING -- each value uses only prices up to and
# including that day. No centred or forward-looking windows anywhere.

prices <- prices %>%
  arrange(symbol, date) %>%
  group_by(symbol) %>%
  mutate(
    daily_ret = adjusted / lag(adjusted) - 1,
    
    # Momentum = total return over the trailing window.
    mom_3m  = adjusted / lag(adjusted, MOM_3M_DAYS)  - 1,
    mom_12m = adjusted / lag(adjusted, MOM_12M_DAYS) - 1,
    
    # Realised volatility: rolling SD of daily returns, annualised.
    # align = "right" is what makes this trailing -- the default
    # ("center") would peek at future returns and leak look-ahead.
    vol_60d = rollapply(
      daily_ret, width = VOL_DAYS, FUN = sd,
      fill = NA, align = "right", na.rm = TRUE
    ) * sqrt(252)
  ) %>%
  ungroup()

# -----------------------------------------------------------------------------
# 3. Collapse to quarter-ends
# -----------------------------------------------------------------------------
# One row per symbol per calendar quarter, taking the LAST trading day
# in that quarter. Note this uses CALENDAR quarters for everyone --
# including JCI, whose fiscal year starts in October. That's
# deliberate: returns must be measured over the same window for all
# stocks or the cross-sectional ranking in the broker test is
# comparing different time periods. Fiscal-calendar differences are
# handled on the fundamentals side, via the filing-date join below.

# KNOWN ISSUE FIX: without a cutoff, this bucketing step would assign
# whatever trading days have occurred so far in the CURRENT, not-yet-
# finished quarter to that quarter's quarter_end, using today's price as
# a stand-in for a quarter-end close that doesn't exist yet. The row
# immediately before it would then get qtr_ret_next/exret_next computed
# from that fake partial-quarter return instead of a real completed one
# -- silently wrong, not just missing. Dropping the not-yet-elapsed
# quarter's bucket here, before qtr_ret/qtr_ret_next/exret_next are
# computed below, means every downstream step (the lead()-based target
# construction, the fundamentals roll-join) only ever sees real,
# completed quarters. This is deliberately separate from the
# STUDY_START trim below -- that trims the WARM-UP period, this trims
# the IN-PROGRESS period, and conflating the two would muddy both.
last_complete_quarter_end <- floor_date(Sys.Date(), "quarter") - days(1)

quarterly <- prices %>%
  mutate(quarter_end = ceiling_date(date, "quarter") - days(1)) %>%
  group_by(symbol, quarter_end) %>%
  slice_max(date, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  filter(quarter_end <= last_complete_quarter_end) %>%
  select(symbol, quarter_end, date, close, adjusted, mom_3m, mom_12m, vol_60d)

# Quarterly total return, and the NEXT quarter's return (the target's
# raw ingredient). lead() here is intentional and is the ONLY place
# forward-looking data is allowed -- it's the thing being predicted.
quarterly <- quarterly %>%
  arrange(symbol, quarter_end) %>%
  group_by(symbol) %>%
  mutate(
    qtr_ret      = adjusted / lag(adjusted) - 1,
    qtr_ret_next = lead(qtr_ret)
  ) %>%
  ungroup()

# -----------------------------------------------------------------------------
# 4. Target: next-quarter excess return vs benchmark
# -----------------------------------------------------------------------------

bench <- quarterly %>%
  filter(symbol == BENCHMARK) %>%
  select(quarter_end, bench_ret_next = qtr_ret_next)

panel <- quarterly %>%
  filter(!symbol %in% c("PHO", "SPY")) %>%
  left_join(bench, by = "quarter_end") %>%
  mutate(exret_next = qtr_ret_next - bench_ret_next) %>%
  rename(ticker = symbol)

# -----------------------------------------------------------------------------
# 5. Macro features
# -----------------------------------------------------------------------------
# FRED series arrive at mixed frequencies (DGS10 daily, FEDFUNDS and
# CPIAUCSL monthly) and are released with a lag. Taking the last
# observation ON OR BEFORE the quarter end -- rather than the value
# stamped with that quarter -- keeps this consistent with the
# point-in-time rule.

macro_wide <- macro %>%
  pivot_wider(names_from = series_id, values_from = value) %>%
  arrange(date)

setDT(macro_wide)

# FEDFUNDS and CPIAUCSL are monthly series sitting on DGS10's daily grid,
# so they are NA on all but ~12 rows/year. The roll = Inf join below finds
# the nearest DATE, not the nearest non-NA value per column -- so without
# this fill, a quarter-end that lands on a day DGS10 has data for but
# FEDFUNDS/CPIAUCSL don't would join to an NA instead of carrying forward
# the last published reading. na.rm = FALSE deliberately leaves leading
# NAs (before the series' first release) as NA rather than fabricating a
# value that didn't exist yet. Must happen BEFORE the roll join below.
macro_wide[, FEDFUNDS := na.locf(FEDFUNDS, na.rm = FALSE)]
macro_wide[, CPIAUCSL := na.locf(CPIAUCSL, na.rm = FALSE)]

macro_wide[, join_date := date]

quarter_ends <- sort(unique(panel$quarter_end))
macro_at_qend <- data.table(quarter_end = quarter_ends, join_date = quarter_ends)

setkey(macro_wide, join_date)
setkey(macro_at_qend, join_date)

# roll = Inf carries the last known value forward: at each quarter end,
# use the most recent macro reading that existed by then.
macro_q <- macro_wide[macro_at_qend, roll = Inf]
macro_q <- as_tibble(macro_q) %>%
  select(quarter_end = join_date, DGS10, FEDFUNDS, CPIAUCSL)

# CPI level isn't a useful predictor on its own (it only ever rises).
# Year-over-year inflation rate is the economically meaningful signal.
macro_q <- macro_q %>%
  arrange(quarter_end) %>%
  mutate(cpi_yoy = CPIAUCSL / lag(CPIAUCSL, 4) - 1) %>%
  select(-CPIAUCSL)

panel <- panel %>% left_join(macro_q, by = "quarter_end")

# -----------------------------------------------------------------------------
# 6. Fundamentals: point-in-time as-of join
# -----------------------------------------------------------------------------
# THE most important join in the project. For each ticker-quarter, take
# the most recent filing whose `filed` date is on or before the quarter
# end. Joining on `end` instead would hand the model figures that
# weren't public yet -- the classic look-ahead bias failure.
#
# This is also exactly why asset_growth / ni_growth / capex_intensity /
# current_ratio are computed upstream in 02b rather
# than here: when a ticker files nothing new during a quarter, this
# join hands the SAME filing to two consecutive quarter-ends. A
# year-over-year change computed on the joined frame would compare that
# filing against itself.

fund_dt <- as.data.table(fund)[order(ticker, filed)]
fund_dt[, join_date := filed]

panel_dt <- as.data.table(panel)
# RENAME in place rather than adding a join_date COPY alongside
# quarter_end. data.table's X[Y] join carries Y's non-key columns
# through unchanged, so if quarter_end and join_date both existed here,
# the merge below would come out with BOTH, and rename(quarter_end =
# join_date) afterward would collide into two identically-named
# columns -- the "Names must be unique" error. Renaming instead of
# copying means there's only ever one column to rename back.
setnames(panel_dt, "quarter_end", "join_date")

setkey(fund_dt, ticker, join_date)
setkey(panel_dt, ticker, join_date)

# roll = Inf => most recent filing at or before quarter_end.
merged <- fund_dt[panel_dt, roll = Inf]
panel <- as_tibble(merged) %>%
  rename(quarter_end = join_date) %>%
  mutate(days_since_filing = as.integer(quarter_end - filed))

# Sanity check: filings should be recent-ish. A very large value means
# a stock went several quarters without a usable filing.
stopifnot(all(is.na(panel$filed) | panel$filed <= panel$quarter_end))

# -----------------------------------------------------------------------------
# 7. Trailing-twelve-month dividends per share
# -----------------------------------------------------------------------------
# Summed over the 365 days ENDING at the quarter end, using ex-dividend
# dates. TTM rather than annualising the latest payment, because that
# would embed a guess about future policy; TTM only uses cash that has
# actually been paid.
#
# A ticker with no dividend rows in the window gets 0, not NA -- but
# only if that ticker appears in the dividend file at all. A ticker
# missing entirely from the file keeps NA, because "pays nothing" and
# "we failed to download it" are different facts and must not collapse
# into the same number.

paying_tickers <- unique(dividends$symbol)

div_ttm <- panel %>%
  distinct(ticker, quarter_end) %>%
  left_join(dividends, by = c("ticker" = "symbol"),
            relationship = "many-to-many") %>%
  filter(!is.na(date), date > quarter_end - days(365), date <= quarter_end) %>%
  group_by(ticker, quarter_end) %>%
  summarise(div_ttm = sum(dividend), .groups = "drop")

panel <- panel %>%
  left_join(div_ttm, by = c("ticker", "quarter_end")) %>%
  mutate(div_ttm = if_else(is.na(div_ttm) & ticker %in% paying_tickers,
                           0, div_ttm))

# -----------------------------------------------------------------------------
# 8. Price-dependent ratios
# -----------------------------------------------------------------------------
# These need BOTH a price and a fundamental, which is why they live
# here rather than in the Python cleaner.
#
# All three use raw `close`, NOT `adjusted`. Adjusted prices are
# retroactively lowered to account for dividends already paid, so a
# 2015 adjusted price is well below what the stock actually traded at
# in 2015. Dividing a 2015 fundamental by it would overstate every
# yield, and the distortion grows the further back you go -- which
# would put a fake time trend into the training set.

panel <- panel %>%
  arrange(ticker, quarter_end) %>%
  group_by(ticker) %>%
  mutate(
    # Trailing twelve months = sum of the last 4 quarterly figures.
    # Using a single quarter would make FCF yield 4x too extreme and
    # wildly seasonal (water utilities earn most of their income in
    # summer quarters).
    fcf_ttm = rollapply(free_cash_flow, 4, sum,
                        fill = NA, align = "right", na.rm = FALSE)
  ) %>%
  ungroup() %>%
  mutate(
    market_cap = close * shares_outstanding,
    book_value_per_share = StockholdersEquity / shares_outstanding,
    
    pb_ratio  = if_else(book_value_per_share > 0,
                        close / book_value_per_share, NA_real_),
    fcf_yield = if_else(market_cap > 0, fcf_ttm / market_cap, NA_real_),
    div_yield = if_else(close > 0, div_ttm / close, NA_real_)
  )

# NOTE ON pe_ratio: deliberately gone. It was left NA whenever EPS <= 0,
# which is the right handling but left the column patchy; it has been
# replaced by the six features added in this revision. ni_growth
# inherits the same guard for the same reason -- growth measured off a
# negative base is sign-flipped nonsense, so it is NA rather than
# misleading.

# -----------------------------------------------------------------------------
# 9. Assemble, trim, report
# -----------------------------------------------------------------------------

PREDICTORS <- c(
  "mom_3m", "mom_12m", "vol_60d",           # technical
  "DGS10", "FEDFUNDS", "cpi_yoy",           # macro
  "roe", "debt_to_equity",                  # fundamentals (SEC)
  "current_ratio", "asset_growth",          # fundamentals (SEC) [NEW]
  "ni_growth", "capex_intensity",           # fundamentals (SEC) [NEW]
  "pb_ratio", "fcf_yield", "div_yield"      # price x fundamentals
)

FILL_FLAGS <- c(
  "roe_is_filled", "debt_to_equity_is_filled",
  "current_ratio_is_filled", "asset_growth_is_filled",
  "ni_growth_is_filled", "capex_intensity_is_filled"
)

panel_final <- panel %>%
  filter(quarter_end >= STUDY_START) %>%
  select(
    ticker, quarter_end, filed, days_since_filing,
    close, adjusted, qtr_ret, qtr_ret_next, bench_ret_next,
    exret_next,
    all_of(PREDICTORS),
    any_of(FILL_FLAGS),
    flag_negative_equity, flag_extreme_de, flag_extreme_roe,
    any_of(c("flag_extreme_asset_growth", "flag_extreme_ni_growth"))
  ) %>%
  arrange(ticker, quarter_end)

write_csv(panel_final, file.path(PROC_DIR, "panel.csv"))

message("\nSaved ", nrow(panel_final), " rows -> ", file.path(PROC_DIR, "panel.csv"))

message("\nMissingness by predictor:")
panel_final %>%
  summarise(across(all_of(PREDICTORS), ~ round(mean(is.na(.)), 3))) %>%
  as.data.frame() %>%
  t() %>%
  print()

# The number that decides your actual sample size. glmnet and pls both
# refuse to run on NA, and lm() silently drops those rows -- so this is
# how many stock-quarters the models will really see.
complete_rows <- panel_final %>%
  filter(!is.na(exret_next)) %>%
  filter(if_all(all_of(PREDICTORS), ~ !is.na(.))) %>%
  nrow()
message("\nComplete rows usable for modelling: ", complete_rows,
        " of ", sum(!is.na(panel_final$exret_next)),
        " rows with a target.")
message("If this is much smaller, find the culprit column above before ",
        "building 03_models.R -- one sparse predictor can silently halve ",
        "the training set.")

message("\nHow much of each fundamental feature is forward-filled ",
        "(stale, not freshly reported):")
panel_final %>%
  summarise(across(any_of(FILL_FLAGS), ~ round(mean(., na.rm = TRUE), 2))) %>%
  as.data.frame() %>%
  t() %>%
  print()

message("\nCoverage by ticker:")
panel_final %>%
  group_by(ticker) %>%
  summarise(
    quarters = n(),
    first    = min(quarter_end),
    last     = max(quarter_end),
    target_missing = round(mean(is.na(exret_next)), 2),
    div_missing    = round(mean(is.na(div_yield)), 2),
    .groups = "drop"
  ) %>%
  as.data.frame() %>%
  print()

message("\nMedian days between filing and quarter-end (should be < ~95):")
print(summary(panel_final$days_since_filing))

message("\n02_features.R complete.")