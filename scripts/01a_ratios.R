# 01a_ratios.R -- builds the panel. Reads prices/dividends/macro (01_get_data.R)
# and fundamentals_clean.csv (02_clean_fundamentals.py), outputs
# data/processed/panel.csv: quarterly technical/risk features, macro series,
# point-in-time fundamentals, and the targets (exret_next, sector-relative,
# 4-quarter-ahead).

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
# 2006-2009 has zero complete-case rows across all 222 tickers -- SEC's
# XBRL mandate didn't roll out till ~2009-2011. 2010+ is already ~84%
# complete-case, climbing to 90-94% by 2014+, so starting here costs
# nothing real and still leaves 16 years.
STUDY_START <- as.Date("2010-01-01")

SECTOR_MAP <- c(
  # --- Utilities---
  AWK="Utilities", WTRG="Utilities", CWT="Utilities", AWR="Utilities",
  AEP="Utilities", DUK="Utilities", SO="Utilities", D="Utilities",
  EXC="Utilities", XEL="Utilities", ED="Utilities", PEG="Utilities",
  ES="Utilities", FE="Utilities", ETR="Utilities", EIX="Utilities",
  PPL="Utilities",
  # --- Materials ---
  ECL="Materials", APD="Materials", SHW="Materials", FCX="Materials",
  NEM="Materials", PPG="Materials", NUE="Materials", ALB="Materials",
  CE="Materials", IFF="Materials", MLM="Materials", VMC="Materials",
  IP="Materials",
  # --- Information Technology  ---
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
  # --- Industrials  ---
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
  # --- Health Care  ---
  JNJ="Health Care", UNH="Health Care", LLY="Health Care", MRK="Health Care",
  PFE="Health Care", TMO="Health Care", ABT="Health Care", DHR="Health Care",
  BMY="Health Care", AMGN="Health Care", GILD="Health Care", CVS="Health Care",
  ELV="Health Care", HUM="Health Care", SYK="Health Care", BSX="Health Care",
  ISRG="Health Care", ZBH="Health Care", BDX="Health Care", BAX="Health Care",
  VRTX="Health Care", BIIB="Health Care", MCK="Health Care", COR="Health Care",
  HCA="Health Care", DVA="Health Care",
  # --- Financials  ---
  C="Financials", GS="Financials", MS="Financials", BNY="Financials",
  SCHW="Financials", SPGI="Financials", MCO="Financials", ICE="Financials",
  AON="Financials", MRSH="Financials", AJG="Financials", ALL="Financials",
  PGR="Financials", HIG="Financials", STT="Financials", FITB="Financials",
  AXP="Financials", COF="Financials", NDAQ="Financials", MSCI="Financials",
  FDS="Financials", CBOE="Financials", NTRS="Financials", WTW="Financials",
  # --- Consumer Discretionary  ---
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
  # --- Communication Services  ---
  NFLX="Communication Services", T="Communication Services",
  VZ="Communication Services", TMUS="Communication Services",
  EA="Communication Services", TTWO="Communication Services",
  OMC="Communication Services", LYV="Communication Services",
  MTCH="Communication Services", SIRI="Communication Services",
  # --- Consumer Staples  ---
  PG="Consumer Staples", KO="Consumer Staples", PEP="Consumer Staples",
  COST="Consumer Staples", WMT="Consumer Staples", PM="Consumer Staples",
  MO="Consumer Staples", MDLZ="Consumer Staples", CL="Consumer Staples",
  KMB="Consumer Staples", GIS="Consumer Staples", ADM="Consumer Staples",
  CAG="Consumer Staples", CLX="Consumer Staples", CHD="Consumer Staples",
  # --- Energy  ---
  CVX="Energy", EOG="Energy", SLB="Energy", MPC="Energy", VLO="Energy",
  OXY="Energy", WMB="Energy", KMI="Energy", OKE="Energy", DVN="Energy",
  HAL="Energy", TRGP="Energy",
  # --- Real Estate ---
  AMT="Real Estate", EQIX="Real Estate", CCI="Real Estate", PSA="Real Estate",
  O="Real Estate", WELL="Real Estate", AVB="Real Estate", EQR="Real Estate",
  VTR="Real Estate", IRM="Real Estate", UDR="Real Estate", HST="Real Estate",
  BXP="Real Estate"
)

MOM_1M_DAYS  <- 21     # ~1 month of trading days
MOM_3M_DAYS  <- 63     # ~3 months of trading days
MOM_12M_DAYS <- 252    # ~12 months of trading days
VOL_DAYS     <- 60     # 60-day realised volatility window / dollar-volume window

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


prices <- prices %>%
  arrange(symbol, date) %>%
  group_by(symbol) %>%
  mutate(
    daily_ret = adjusted / lag(adjusted) - 1,

    # Momentum = total return over the trailing window.
    mom_1m  = adjusted / lag(adjusted, MOM_1M_DAYS)  - 1,
    mom_3m  = adjusted / lag(adjusted, MOM_3M_DAYS)  - 1,
    mom_12m = adjusted / lag(adjusted, MOM_12M_DAYS) - 1,

    # Rolling SD of daily returns, annualised. align = "right" is the
    vol_60d = rollapply(
      daily_ret, width = VOL_DAYS, FUN = sd,
      fill = NA, align = "right", na.rm = TRUE
    ) * sqrt(252),
    dollar_volume = rollapply(
      close * volume, width = VOL_DAYS, FUN = mean,
      fill = NA, align = "right", na.rm = TRUE
    )
  ) %>%
  ungroup()


mkt_ret <- prices %>%
  filter(symbol == BENCHMARK) %>%
  select(date, mkt_ret = daily_ret)

prices <- prices %>% left_join(mkt_ret, by = "date")

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

    resid_ret = daily_ret - beta_252d * mkt_ret,
    idio_vol_60d = rollapply(
      resid_ret, width = VOL_DAYS, FUN = sd,
      fill = NA, align = "right", na.rm = TRUE
    ) * sqrt(252),

    pct_from_52w_high = adjusted / zoo::rollmax(adjusted, k = MOM_12M_DAYS, fill = NA, align = "right") - 1
  ) %>%
  ungroup() %>%
  select(-resid_ret, -mkt_ret)


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


quarterly <- quarterly %>%
  arrange(symbol, quarter_end) %>%
  group_by(symbol) %>%
  mutate(
    qtr_ret       = adjusted / lag(adjusted) - 1,
    qtr_ret_next  = lead(qtr_ret),
    qtr_ret_next4 = lead(adjusted, 4) / adjusted - 1
  ) %>%
  ungroup()



bench <- quarterly %>%
  filter(symbol == BENCHMARK) %>%
  select(quarter_end, bench_ret_next = qtr_ret_next, bench_ret_next4 = qtr_ret_next4)

panel <- quarterly %>%
  filter(symbol != BENCHMARK) %>%
  left_join(bench, by = "quarter_end") %>%
  mutate(
    exret_next  = qtr_ret_next  - bench_ret_next,
    exret_next4 = qtr_ret_next4 - bench_ret_next4
  ) %>%
  rename(ticker = symbol)

stopifnot(all(unique(panel$ticker) %in% names(SECTOR_MAP)))


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



macro_wide <- macro %>%
  pivot_wider(names_from = series_id, values_from = value) %>%
  arrange(date)

setDT(macro_wide)


macro_wide[, FEDFUNDS := na.locf(FEDFUNDS, na.rm = FALSE)]
macro_wide[, CPIAUCSL := na.locf(CPIAUCSL, na.rm = FALSE)]

macro_wide[, join_date := date]

quarter_ends <- sort(unique(panel$quarter_end))
macro_at_qend <- data.table(quarter_end = quarter_ends, join_date = quarter_ends)

setkey(macro_wide, join_date)
setkey(macro_at_qend, join_date)


macro_q <- macro_wide[macro_at_qend, roll = Inf]
macro_q <- as_tibble(macro_q) %>%
  select(quarter_end = join_date, DGS10, FEDFUNDS, CPIAUCSL)

macro_q <- macro_q %>%
  arrange(quarter_end) %>%
  mutate(cpi_yoy = CPIAUCSL / lag(CPIAUCSL, 4) - 1) %>%
  select(-CPIAUCSL)

panel <- panel %>% left_join(macro_q, by = "quarter_end")


fund_dt <- as.data.table(fund)[order(ticker, filed)]
fund_dt[, join_date := filed]

panel_dt <- as.data.table(panel)

setnames(panel_dt, "quarter_end", "join_date")

setkey(fund_dt, ticker, join_date)
setkey(panel_dt, ticker, join_date)


merged <- fund_dt[panel_dt, roll = Inf]
panel <- as_tibble(merged) %>%
  rename(quarter_end = join_date) %>%
  mutate(days_since_filing = as.integer(quarter_end - filed))

stopifnot(all(is.na(panel$filed) | panel$filed <= panel$quarter_end))


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


panel <- panel %>%
  arrange(ticker, quarter_end) %>%
  group_by(ticker) %>%
  mutate(

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


# 9. Assemble, trim, report


PREDICTORS <- c(
  "mom_1m", "mom_3m", "mom_12m", "vol_60d", "dollar_volume",  # technical
  "beta_252d", "idio_vol_60d", "pct_from_52w_high",  # technical/risk 
  "DGS10", "FEDFUNDS", "cpi_yoy",           # macro
  "roe", "debt_to_equity",                  # fundamentals (SEC)
  "current_ratio", "asset_growth",          # fundamentals (SEC)
  "ni_growth", "capex_intensity",           # fundamentals (SEC) 
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
    qtr_ret_next4, bench_ret_next4, exret_next4,
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

message("\n01a_ratios.R complete.")