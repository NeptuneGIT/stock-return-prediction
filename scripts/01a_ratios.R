# =============================================================================
# 02_features.R
#
# Builds the quarterly modelling panel from:
#   - data/raw/prices_daily.csv              (01_get_data.R)
#   - data/raw/dividends_raw.csv             (01_get_data.R)
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
# The quarterly bucketing step drops any bucket whose quarter_end is
# later than the most recently fully elapsed quarter
# (last_complete_quarter_end) before qtr_ret_next/exret_next are
# computed -- otherwise a not-yet-finished quarter's partial data could
# get bucketed and used as a fake "next quarter" return for the row
# before it. asset_growth, capex_intensity, ni_growth, and
# current_ratio arrive ready-made from 02b and are simply carried
# through the join rather than recomputed here, since the join repeats
# the same filing across quarters when no new one landed, and a
# year-over-year change computed after that repetition would sometimes
# compare a filing to itself and report 0% growth. pb_ratio and
# fcf_yield use raw `close`, not `adjusted` (adjusted prices are
# back-corrected downward for past dividends, which would inflate every
# historical valuation yield); returns still use `adjusted`, which is
# correct for returns. FEDFUNDS/CPIAUCSL are forward-filled with
# zoo::na.locf() onto the daily grid before the roll = Inf macro join,
# since those monthly series are otherwise NA on all but ~12 rows/year
# at quarter-end (the roll join finds the nearest date, not the nearest
# non-NA value per column).
#
# This project pivoted from a 28-ticker water-sector universe benchmarked
# against PHO to the full 222-ticker S&P 500 universe benchmarked against
# SPY (BENCHMARK below); PHO has been removed entirely, not kept even as
# a passive column. The predictor set grew alongside the pivot: mom_1m
# and dollar_volume were added early as trailing-window technical
# features, and beta_252d (rolling 252-trading-day market beta vs SPY,
# via a rolling-sums Cov/Var identity computed with
# data.table::frollsum), idio_vol_60d (60-day SD of the beta-adjusted
# residual return -- stock-specific risk with the market component
# removed, distinct from vol_60d's total volatility), and
# pct_from_52w_high (proximity to the trailing 52-week high, the George
# & Hwang 2004 anchoring anomaly) were added afterward to give the model
# cross-sectional structure different from what the existing
# momentum/level features already capture. All three inherit the same
# 252-trading-day warm-up requirement mom_12m already has, fully
# absorbed by the runway before STUDY_START.
#
# An alternate sector-relative target also exists alongside the primary
# SPY-relative exret_next: `sector` (GICS sector per ticker, see
# SECTOR_MAP below), `sector_ret_next` (leave-one-out mean of
# qtr_ret_next across the ticker's own GICS sector that quarter,
# excluding the ticker itself so its own return never contaminates its
# own benchmark), and `exret_next_sector` (qtr_ret_next minus
# sector_ret_next). A SPY-relative target asks the model to explain a
# stock's return relative to the whole market, which conflates
# sector-wide moves (energy up, tech down, etc. -- not stock-picking
# skill) with genuine idiosyncratic mispricing; sector-relative isolates
# the latter. exret_next remains this project's primary target;
# exret_next_sector is an additional column for experimentation, built
# entirely from qtr_ret_next (already the one place forward-looking data
# is allowed, since it's the target ingredient, not a feature).
#
# STUDY_START is 2010-01-01: a missingness review of panel.csv found
# 2006-2009 contributed exactly zero complete-case rows across all 222
# tickers, consistent with SEC's XBRL mandate rollout leaving little
# structured fundamentals data before ~2009-2011 regardless of scraper
# quality. See the comment at STUDY_START's definition for the full
# year-by-year evidence.
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

BENCHMARK   <- "SPY"   # excess return is measured against this
# STUDY_START is 2010-01-01, not 2003 or 2006: complete-case rate by year
# is a step function, not a gradual curve -- 2006-2009 contributed
# EXACTLY ZERO complete-case rows across all 222 tickers (3,323 rows,
# 0.0%), consistent with SEC's XBRL mandate rollout (large filers
# required to tag starting ~2009, phased in through ~2011) leaving
# little-to-no structured fundamentals data before then regardless of
# scraper quality. 2010 onward is already 84.1% complete overall and
# climbs to 90-94% by 2014+, so trimming to 2010 loses ZERO real
# training rows (the dropped years already contributed none) while
# still leaving 16 years of history. 02_clean_fundamentals.py's
# STUDY_START and 01_get_data.R's START_DATE deliberately stay at 2003
# -- that's runway for TTM/lag features on the 2010+ panel rows, not
# itself part of the reported study window, so it does not need to
# match this one.
STUDY_START <- as.Date("2010-01-01")

# GICS sector per ticker, used only to build the alternate sector-relative
# target (exret_next_sector, section 4b) -- not used anywhere in the
# point-in-time join or PREDICTORS. Grouped exactly as in 02_fundamentals.py's
# TICKERS list (the 194 tickers added in the S&P-500 pivot carry explicit
# per-ticker sector labels there); the original 28 water-sector tickers'
# assignments were reconstructed from general GICS knowledge and
# cross-checked against known aggregate counts for that group
# (Industrials 22, Utilities 4, Information Technology 1, Materials 1).
# Two tickers were genuinely ambiguous between two plausible sectors:
# VRT (Vertiv) is assigned Industrials/Electrical Equipment and ITRI
# (Itron) is assigned Information Technology/Electronic Equipment --
# this pairing is the only assignment of the two that reconciles exactly
# with the "1 original IT" count; swapping them would still sum
# correctly by sector-size alone but would misclassify one ticker. If a
# future data-quality pass finds authoritative per-ticker GICS data
# (e.g. from a financial data API), re-verify SECTOR_MAP against it
# rather than trusting this reconstruction indefinitely.
SECTOR_MAP <- c(
  # --- Utilities (17: 4 original water utilities + 13 new) ---
  AWK="Utilities", WTRG="Utilities", CWT="Utilities", AWR="Utilities",
  AEP="Utilities", DUK="Utilities", SO="Utilities", D="Utilities",
  EXC="Utilities", XEL="Utilities", ED="Utilities", PEG="Utilities",
  ES="Utilities", FE="Utilities", ETR="Utilities", EIX="Utilities",
  PPL="Utilities",
  # --- Materials (13: 1 original [ECL] + 12 new) ---
  ECL="Materials", APD="Materials", SHW="Materials", FCX="Materials",
  NEM="Materials", PPG="Materials", NUE="Materials", ALB="Materials",
  CE="Materials", IFF="Materials", MLM="Materials", VMC="Materials",
  IP="Materials",
  # --- Information Technology (32: 1 original [ITRI] + 31 new) ---
  ITRI="Information Technology",
  MSFT="Information Technology", AAPL="Information Technology",
  NVDA="Information Technology", ORCL="Information Technology",
  CRM="Information Technology", ADBE="Information Technology",
  CSCO="Information Technology", AMD="Information Technology",
  QCOM="Information Technology", TXN="Information Technology",
  IBM="Information Technology", INTU="Information Technology",
  AMAT="Information Technology", ADI="Information Technology",
  LRCX="Information Technology", KLAC="Information Technology",
  MU="Information Technology", SNPS="Information Technology",
  CDNS="Information Technology", ADSK="Information Technology",
  MCHP="Information Technology", ON="Information Technology",
  TER="Information Technology", WDC="Information Technology",
  STX="Information Technology", NTAP="Information Technology",
  TYL="Information Technology", PTC="Information Technology",
  SWKS="Information Technology", GRMN="Information Technology",
  ZBRA="Information Technology",
  # --- Industrials (36: 22 original + 14 new) ---
  XYL="Industrials", VRT="Industrials", JCI="Industrials", BMI="Industrials",
  PNR="Industrials", MWA="Industrials", AOS="Industrials", ITT="Industrials",
  IEX="Industrials", FELE="Industrials", FLS="Industrials", DOV="Industrials",
  GRC="Industrials", MAS="Industrials", ROP="Industrials", EMR="Industrials",
  PH="Industrials", HON="Industrials", WMS="Industrials", GVA="Industrials",
  PWR="Industrials", MTZ="Industrials",
  CAT="Industrials", DE="Industrials", UNP="Industrials", FDX="Industrials",
  BA="Industrials", LMT="Industrials", RTX="Industrials", GD="Industrials",
  NOC="Industrials", CSX="Industrials", NSC="Industrials", WM="Industrials",
  RSG="Industrials", GE="Industrials",
  # --- Health Care (26, all new) ---
  JNJ="Health Care", UNH="Health Care", LLY="Health Care", MRK="Health Care",
  PFE="Health Care", TMO="Health Care", ABT="Health Care", DHR="Health Care",
  BMY="Health Care", AMGN="Health Care", GILD="Health Care", CVS="Health Care",
  ELV="Health Care", HUM="Health Care", SYK="Health Care", BSX="Health Care",
  ISRG="Health Care", ZBH="Health Care", BDX="Health Care", BAX="Health Care",
  VRTX="Health Care", BIIB="Health Care", MCK="Health Care", COR="Health Care",
  HCA="Health Care", DVA="Health Care",
  # --- Financials (24, all new) ---
  C="Financials", GS="Financials", MS="Financials", BNY="Financials",
  SCHW="Financials", SPGI="Financials", MCO="Financials", ICE="Financials",
  AON="Financials", MRSH="Financials", AJG="Financials", ALL="Financials",
  PGR="Financials", HIG="Financials", STT="Financials", FITB="Financials",
  AXP="Financials", COF="Financials", NDAQ="Financials", MSCI="Financials",
  FDS="Financials", CBOE="Financials", NTRS="Financials", WTW="Financials",
  # --- Consumer Discretionary (24, all new) ---
  AMZN="Consumer Discretionary", TSLA="Consumer Discretionary",
  HD="Consumer Discretionary", MCD="Consumer Discretionary",
  LOW="Consumer Discretionary", SBUX="Consumer Discretionary",
  TJX="Consumer Discretionary", ORLY="Consumer Discretionary",
  AZO="Consumer Discretionary", ROST="Consumer Discretionary",
  YUM="Consumer Discretionary", MAR="Consumer Discretionary",
  HLT="Consumer Discretionary", GM="Consumer Discretionary",
  APTV="Consumer Discretionary", BBY="Consumer Discretionary",
  DHI="Consumer Discretionary", PHM="Consumer Discretionary",
  NVR="Consumer Discretionary", WHR="Consumer Discretionary",
  TSCO="Consumer Discretionary", ULTA="Consumer Discretionary",
  GPC="Consumer Discretionary", CMG="Consumer Discretionary",
  # --- Communication Services (10, all new) ---
  NFLX="Communication Services", T="Communication Services",
  VZ="Communication Services", TMUS="Communication Services",
  EA="Communication Services", TTWO="Communication Services",
  OMC="Communication Services", LYV="Communication Services",
  MTCH="Communication Services", SIRI="Communication Services",
  # --- Consumer Staples (15, all new) ---
  PG="Consumer Staples", KO="Consumer Staples", PEP="Consumer Staples",
  COST="Consumer Staples", WMT="Consumer Staples", PM="Consumer Staples",
  MO="Consumer Staples", MDLZ="Consumer Staples", CL="Consumer Staples",
  KMB="Consumer Staples", GIS="Consumer Staples", ADM="Consumer Staples",
  CAG="Consumer Staples", CLX="Consumer Staples", CHD="Consumer Staples",
  # --- Energy (12, all new) ---
  CVX="Energy", EOG="Energy", SLB="Energy", MPC="Energy", VLO="Energy",
  OXY="Energy", WMB="Energy", KMI="Energy", OKE="Energy", DVN="Energy",
  HAL="Energy", TRGP="Energy",
  # --- Real Estate (13, all new) ---
  AMT="Real Estate", EQIX="Real Estate", CCI="Real Estate", PSA="Real Estate",
  O="Real Estate", WELL="Real Estate", AVB="Real Estate", EQR="Real Estate",
  VTR="Real Estate", IRM="Real Estate", UDR="Real Estate", HST="Real Estate",
  BXP="Real Estate"
)

MOM_1M_DAYS  <- 21     # ~1 month of trading days
MOM_3M_DAYS  <- 63     # ~3 months of trading days
MOM_12M_DAYS <- 252    # ~12 months of trading days
VOL_DAYS     <- 60     # 60-day realised volatility window / dollar-volume window

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
    mom_1m  = adjusted / lag(adjusted, MOM_1M_DAYS)  - 1,
    mom_3m  = adjusted / lag(adjusted, MOM_3M_DAYS)  - 1,
    mom_12m = adjusted / lag(adjusted, MOM_12M_DAYS) - 1,

    # Realised volatility: rolling SD of daily returns, annualised.
    # align = "right" is what makes this trailing -- the default
    # ("center") would peek at future returns and leak look-ahead.
    vol_60d = rollapply(
      daily_ret, width = VOL_DAYS, FUN = sd,
      fill = NA, align = "right", na.rm = TRUE
    ) * sqrt(252),

    # Trailing 60-day average dollar volume (price x shares traded), a
    # liquidity/size proxy that varies meaningfully across the S&P 500's
    # mega-cap-to-small-cap range -- unlike raw national macro series, it
    # actually moves the cross-sectional ranking. Uses `close`, not
    # `adjusted`, matching this file's established close-vs-adjusted
    # convention (close for valuation/liquidity measures, adjusted for
    # returns) -- adjusted prices are retroactively lowered for past
    # dividends and would understate historical dollar volume.
    dollar_volume = rollapply(
      close * volume, width = VOL_DAYS, FUN = mean,
      fill = NA, align = "right", na.rm = TRUE
    )
  ) %>%
  ungroup()

# -----------------------------------------------------------------------------
# 2b. Market beta, idiosyncratic volatility, 52-week-high proximity
# -----------------------------------------------------------------------------
# Added after the sector-relative-target and GBRT-hyperparameter-search
# experiments both left test R^2 near zero -- a search for predictors
# with genuinely DIFFERENT cross-sectional structure, not just more
# momentum/level variants of
# what mom_1m/mom_3m/mom_12m/vol_60d/dollar_volume already capture.
# beta_252d (systematic risk exposure), idio_vol_60d (residual,
# stock-specific risk) and pct_from_52w_high (a well-documented
# behavioral anchoring effect -- George & Hwang 2004) all vary
# meaningfully across a mega-cap-to-small-cap, utility-to-tech universe
# (a regulated water utility's beta is structurally very different from
# a semiconductor's) -- unlike the constant national macro series
# (DGS10/FEDFUNDS/cpi_yoy) already excluded from 03_models.R's
# PREDICTORS for failing exactly this test. All three are TRAILING,
# align = "right", same no-look-ahead discipline as section 2 above.

mkt_ret <- prices %>%
  filter(symbol == BENCHMARK) %>%
  select(date, mkt_ret = daily_ret)

prices <- prices %>% left_join(mkt_ret, by = "date")

# Rolling beta via the algebraic identity beta = Cov(r_i, r_m) / Var(r_m),
# computed from rolling SUMS with data.table::frollsum (a C-level O(n)
# rolling sum) rather than a per-row rolling lm()/cov() call. At ~1.28M
# daily rows across 223 symbols, a naive rolling-regression loop
# (O(n x window)) would be far slower for an identical result.
# frollsum's default na.rm = FALSE means any single NA inside the window
# (e.g. each symbol's very first daily_ret, which is NA because lag() has
# nothing to look back to) makes that whole window NA -- the same
# warm-up behaviour mom_12m already has via lag(), not a new gap, and
# fully absorbed by the 2003-2010 runway before STUDY_START.
prices_dt <- as.data.table(prices)
setorder(prices_dt, symbol, date)

BETA_DAYS <- MOM_12M_DAYS  # 252 trading days -- same window as mom_12m

prices_dt[, `:=`(
  roll_sum_ri   = frollsum(daily_ret, BETA_DAYS),
  roll_sum_rm   = frollsum(mkt_ret, BETA_DAYS),
  roll_sum_rirm = frollsum(daily_ret * mkt_ret, BETA_DAYS),
  roll_sum_rm2  = frollsum(mkt_ret^2, BETA_DAYS)
), by = symbol]

prices_dt[, beta_252d := (roll_sum_rirm - roll_sum_ri * roll_sum_rm / BETA_DAYS) /
                          (roll_sum_rm2 - roll_sum_rm^2 / BETA_DAYS)]
prices_dt[, c("roll_sum_ri", "roll_sum_rm", "roll_sum_rirm", "roll_sum_rm2") := NULL]

prices <- as_tibble(prices_dt)

prices <- prices %>%
  arrange(symbol, date) %>%
  group_by(symbol) %>%
  mutate(
    # Idiosyncratic volatility: trailing 60-day SD of the return left
    # over after removing the beta-implied market component -- distinct
    # from vol_60d (TOTAL volatility, which mixes market-wide and
    # stock-specific risk together and is highly correlated across
    # tickers during broad market turbulence).
    resid_ret = daily_ret - beta_252d * mkt_ret,
    idio_vol_60d = rollapply(
      resid_ret, width = VOL_DAYS, FUN = sd,
      fill = NA, align = "right", na.rm = TRUE
    ) * sqrt(252),

    # Proximity to the trailing 52-week high, using `adjusted` (like the
    # other momentum features, not `close`) since this is a
    # return/momentum-family signal, not a valuation/liquidity one.
    # Always <= 0; 0 means today's adjusted close IS the 52-week high.
    # zoo::rollmax is a dedicated C-level rolling-max (like frollsum
    # above), not a generic rollapply(FUN = max) call, for the same
    # performance reason.
    pct_from_52w_high = adjusted / zoo::rollmax(adjusted, k = MOM_12M_DAYS, fill = NA, align = "right") - 1
  ) %>%
  ungroup() %>%
  select(-resid_ret, -mkt_ret)

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
  select(symbol, quarter_end, date, close, adjusted,
         mom_1m, mom_3m, mom_12m, vol_60d, dollar_volume,
         beta_252d, idio_vol_60d, pct_from_52w_high)

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
  filter(symbol != BENCHMARK) %>%
  left_join(bench, by = "quarter_end") %>%
  mutate(exret_next = qtr_ret_next - bench_ret_next) %>%
  rename(ticker = symbol)

stopifnot(all(unique(panel$ticker) %in% names(SECTOR_MAP)))

# -----------------------------------------------------------------------------
# 4b. Alternate target: next-quarter excess return vs GICS sector peers
# -----------------------------------------------------------------------------
# exret_next (above) benchmarks against SPY, the whole market -- which
# conflates sector-wide moves (e.g. all of Energy rallying on oil prices)
# with genuine stock-specific mispricing, since a sector-wide move affects
# every stock in that sector roughly alike and shows up as "signal" in
# exret_next even though no model could have picked the WINNER within
# that sector from it. exret_next_sector instead benchmarks each ticker
# against the mean next-quarter return of its own GICS sector peers in
# this universe, isolating the within-sector, idiosyncratic component.
#
# LEAVE-ONE-OUT mean, not a simple sector average: including a ticker's
# own return in its own benchmark mechanically shrinks its resulting
# excess return toward zero (self-cancellation), and that effect is
# largest for the smallest sector here, Communication Services, at only
# 10 constituents (1/10 = 10% self-weight). Excluding each ticker from
# its own sector average removes that bias for every sector, not just
# the small ones.
#
# Built entirely from qtr_ret_next, the same forward-looking target
# ingredient exret_next already uses -- this is target construction, not
# a feature, so it does not reintroduce look-ahead bias.
panel <- panel %>%
  mutate(sector = SECTOR_MAP[ticker]) %>%
  group_by(sector, quarter_end) %>%
  mutate(
    sector_n_nonna   = sum(!is.na(qtr_ret_next)),
    sector_sum_nonna = sum(qtr_ret_next, na.rm = TRUE),
    sector_loo_n     = sector_n_nonna - if_else(is.na(qtr_ret_next), 0L, 1L),
    sector_loo_sum   = sector_sum_nonna - if_else(is.na(qtr_ret_next), 0, qtr_ret_next),
    sector_ret_next  = if_else(sector_loo_n > 0, sector_loo_sum / sector_loo_n, NA_real_)
  ) %>%
  ungroup() %>%
  select(-sector_n_nonna, -sector_sum_nonna, -sector_loo_n, -sector_loo_sum) %>%
  mutate(exret_next_sector = qtr_ret_next - sector_ret_next)

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
# 8b. Accruals (quality-of-earnings / Sloan anomaly)
# -----------------------------------------------------------------------------
# Added alongside beta_252d/idio_vol_60d/pct_from_52w_high in section
# 2b, as part of the same "find genuinely cross-sectional
# predictors" push -- but this one is fundamentals-based rather than
# price-based, deliberately chosen to AVOID re-litigating an already-
# rejected path: 02_clean_fundamentals.py's docstring explicitly notes
# operating_margin and asset_turnover were both dropped because "Revenue
# coverage is too sparse and uneven across this universe" (see that
# file's NOTE ON operating_margin). Every standard margin/turnover ratio
# needs Revenue as a denominator, so they inherit that same coverage
# problem. Accruals sidesteps it entirely: NetIncomeLoss, OperatingCashFlow
# and Assets are all resolved by 02_clean_fundamentals.py's existing
# PRIORITY logic already and used elsewhere in this file (ni_ttm feeds
# ni_growth; Assets feeds asset_growth) -- no new SEC tag, no new scrape,
# and no dependency on Revenue's coverage gaps.
#
# accruals = (ni_ttm - ocf_ttm) / Assets -- the Sloan (1996) accrual
# anomaly: net income built more from non-cash accounting accruals than
# from actual operating cash flow is a well-documented signal of lower
# earnings quality, historically associated with weaker future returns.
# This is a genuinely different economic dimension from every predictor
# already in PREDICTORS (growth, leverage, valuation, momentum, risk) --
# earnings QUALITY, not earnings level or its trend.
#
# ocf_ttm follows the exact same TTM-sum-of-4-quarters pattern as
# capex_ttm/fcf_ttm above (na.rm = FALSE -- a TTM figure built from an
# incomplete year is not a real TTM figure, same reasoning as fcf_ttm).
# Guarded on Assets > 0 (should be true for every going concern in this
# universe; NA rather than a division by a non-positive/missing value).
panel <- panel %>%
  arrange(ticker, quarter_end) %>%
  group_by(ticker) %>%
  mutate(
    ocf_ttm = rollapply(OperatingCashFlow, 4, sum,
                        fill = NA, align = "right", na.rm = FALSE)
  ) %>%
  ungroup() %>%
  mutate(
    accruals = if_else(Assets > 0, (ni_ttm - ocf_ttm) / Assets, NA_real_)
  )

# -----------------------------------------------------------------------------
# 9. Assemble, trim, report
# -----------------------------------------------------------------------------

PREDICTORS <- c(
  "mom_1m", "mom_3m", "mom_12m", "vol_60d", "dollar_volume",  # technical
  "beta_252d", "idio_vol_60d", "pct_from_52w_high",  # technical/risk
  "DGS10", "FEDFUNDS", "cpi_yoy",           # macro
  "roe", "debt_to_equity",                  # fundamentals (SEC)
  "current_ratio", "asset_growth",          # fundamentals (SEC) [NEW]
  "ni_growth", "capex_intensity",           # fundamentals (SEC) [NEW]
  "accruals",                               # fundamentals (SEC)
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
    sector, sector_ret_next, exret_next_sector,
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