# 04_results.R -- backtests "buy the top N": at each out-of-sample
# walk-forward quarter, rank tickers by GBRT's predicted exret_next, go
# long an equal-weighted basket of the top N, hold for that quarter.
# Compares against SPY over the full pooled walk-forward window.
#
# GBRT is used here as the primary model (see 03_models.R for the full
# comparison against PLS and Random Forest) -- its walk-forward R^2
# doesn't always beat PLS's, all three are small and arguably within
# noise at this sample size.
#
# N is a sensitivity grid (10/20/30/50), not one hardcoded number -- the
# course spec doesn't hand down a single "correct" N. N=20 is the
# primary portfolio for the detailed breakdown.
#
# No look-ahead: every prediction row comes from a fold-specific model
# trained only on quarters strictly before its own quarter_end, never
# the entire-panel production model. Ranking by `pred` at quarter_end Q
# and holding through Q's realized return is a legitimate backtest.
#
# Input:  data/processed/gbrt_walkforward_predictions.csv (03_models.R)
#         data/processed/panel.csv              (01a_ratios.R)
# Output: data/processed/backtest_holdings.csv         (primary N=20)
#         data/processed/backtest_quarterly_returns.csv (primary N=20)
#         data/processed/backtest_summary.csv          (full N grid)

library(dplyr)
library(readr)
library(tidyr)

PROC_DIR <- "data/processed"
N_GRID <- c(10, 20, 30, 50)
N_PRIMARY <- 20

# -----------------------------------------------------------------------------
# 1. Load GBRT pooled out-of-fold walk-forward predictions + realized outcomes
# -----------------------------------------------------------------------------

gbm_pred <- read_csv(file.path(PROC_DIR, "gbrt_walkforward_predictions.csv"), show_col_types = FALSE)

# panel.csv carries a handful of duplicate (ticker, quarter_end) rows --
# the point-in-time fundamentals join occasionally matches more than one
# qualifying filing instead of collapsing to exactly one; root cause not
# fixed at the source. Deduplicated here the same deterministic way
# (first occurrence in CSV row order) as in 03_models.R, since this
# script reads panel.csv directly. Skip this and the join below turns
# many-to-many, failing the sanity check below.
panel <- read_csv(file.path(PROC_DIR, "panel.csv"), show_col_types = FALSE) %>%
  select(ticker, quarter_end, sector, qtr_ret_next, bench_ret_next) %>%
  distinct(ticker, quarter_end, .keep_all = TRUE)

bt <- gbm_pred %>%
  inner_join(panel, by = c("ticker", "quarter_end")) %>%
  rename(pred_exret_next = pred, actual_exret_next = actual)

n_join_lost <- nrow(gbm_pred) - nrow(bt)
if (n_join_lost != 0) {
  stop("Join between gbrt_walkforward_predictions.csv and panel.csv lost ",
       n_join_lost, " row(s) -- ticker/quarter_end keys should match ",
       "exactly since both come from the same panel.csv build. Investigate ",
       "before trusting any backtest number below.")
}

# Sanity check: bench_ret_next has to be identical for every ticker in a
# quarter -- one market-wide number, not ticker-specific.
bench_check <- bt %>% group_by(quarter_end) %>% summarise(n_distinct_bench = n_distinct(bench_ret_next), .groups = "drop")
if (any(bench_check$n_distinct_bench > 1)) {
  stop("bench_ret_next is not constant within at least one quarter -- ",
       "SPY's realized return should be a single market-wide number per ",
       "quarter. Check panel.csv's benchmark join before trusting the ",
       "equity curve below.")
}

n_quarters <- n_distinct(bt$quarter_end)
n_tickers  <- n_distinct(bt$ticker)
message("Loaded ", nrow(bt), " pooled out-of-fold (ticker x quarter) ",
        "walk-forward predictions: ", n_tickers, " tickers across ",
        n_quarters, " quarters (", min(bt$quarter_end), " to ",
        max(bt$quarter_end), ").")

# -----------------------------------------------------------------------------
# 2. "Broker Test" cross-sectional check: does GBRT's ranking actually
#    differentiate tickers within each quarter, or are predictions
#    degenerate (e.g. all identical, which would make "top N" arbitrary)?
# -----------------------------------------------------------------------------
# Same principle applied to predictor selection elsewhere in this project
# -- a signal only matters if it varies across tickers within a quarter --
# applied here to the model's output. Reported per quarter, not assumed.

broker_test <- bt %>%
  group_by(quarter_end) %>%
  summarise(
    n_tickers_available = n(),
    pred_sd             = sd(pred_exret_next),
    pred_min            = min(pred_exret_next),
    pred_max            = max(pred_exret_next),
    .groups = "drop"
  )

message("\n--- Broker Test: cross-sectional spread of GBRT predictions, by quarter ---")
print(broker_test, n = 100)

if (any(broker_test$pred_sd < 1e-8)) {
  stop("At least one quarter has ~zero cross-sectional spread in GBRT ",
       "predictions -- ranking into a top-N portfolio would be arbitrary ",
       "that quarter. Investigate before trusting the backtest.")
}
if (any(broker_test$n_tickers_available < max(N_GRID))) {
  warning("At least one quarter has fewer tickers available than the ",
          "largest N in N_GRID (", max(N_GRID), ") -- that quarter's ",
          "largest-N portfolio will simply hold every available ticker ",
          "instead of a true top-N subset. See the table above for which ",
          "quarter(s).")
}

# -----------------------------------------------------------------------------
# 3. Build top-N equal-weighted portfolios for the full N sensitivity grid
# -----------------------------------------------------------------------------
# Equal-weighted, held one quarter, no transaction costs, no
# re-optimization within the quarter -- simplest implementation of "buy
# the top N", matching the course-spec framing.

build_portfolio <- function(n_top) {
  bt %>%
    group_by(quarter_end) %>%
    arrange(desc(pred_exret_next), .by_group = TRUE) %>%
    mutate(rank = row_number()) %>%
    filter(rank <= n_top) %>%
    summarise(
      n_held        = n(),
      portfolio_ret = mean(qtr_ret_next),
      spy_ret       = first(bench_ret_next),
      .groups = "drop"
    ) %>%
    mutate(excess_ret = portfolio_ret - spy_ret)
}

max_drawdown <- function(cum_growth) {
  running_peak <- cummax(cum_growth)
  min((cum_growth - running_peak) / running_peak)
}

summarize_portfolio <- function(qret, n_top) {
  n_q <- nrow(qret)
  port_cum <- cumprod(1 + qret$portfolio_ret)
  spy_cum  <- cumprod(1 + qret$spy_ret)
  data.frame(
    n_top                  = n_top,
    n_quarters             = n_q,
    portfolio_total_return = tail(port_cum, 1) - 1,
    spy_total_return       = tail(spy_cum, 1) - 1,
    portfolio_ann_return   = tail(port_cum, 1)^(4 / n_q) - 1,
    spy_ann_return         = tail(spy_cum, 1)^(4 / n_q) - 1,
    portfolio_ann_vol      = sd(qret$portfolio_ret) * sqrt(4),
    spy_ann_vol            = sd(qret$spy_ret) * sqrt(4),
    portfolio_max_drawdown = max_drawdown(port_cum),
    spy_max_drawdown       = max_drawdown(spy_cum),
    hit_rate               = mean(qret$excess_ret > 0),
    mean_excess_ret        = mean(qret$excess_ret),
    information_ratio      = mean(qret$excess_ret) / sd(qret$excess_ret) * sqrt(4)
  )
}

message("\n--- Top-N backtest, sensitivity across N (equal-weighted, quarterly rebalance) ---")
backtest_summary <- bind_rows(lapply(N_GRID, function(n) {
  qret <- build_portfolio(n)
  summarize_portfolio(qret, n)
}))
print(backtest_summary)

backtest_summary_path <- file.path(PROC_DIR, "backtest_summary.csv")
write_csv(backtest_summary, backtest_summary_path)
message("Saved N-sensitivity summary -> ", backtest_summary_path)

# -----------------------------------------------------------------------------
# 4. Detailed quarterly breakdown + holdings for the primary N (N=20)
# -----------------------------------------------------------------------------

primary_holdings <- bt %>%
  group_by(quarter_end) %>%
  arrange(desc(pred_exret_next), .by_group = TRUE) %>%
  mutate(rank = row_number()) %>%
  filter(rank <= N_PRIMARY) %>%
  select(quarter_end, rank, ticker, sector, pred_exret_next, actual_exret_next, qtr_ret_next) %>%
  ungroup()

holdings_path <- file.path(PROC_DIR, "backtest_holdings.csv")
write_csv(primary_holdings, holdings_path)
message("\nSaved ", nrow(primary_holdings), " holdings rows (top ", N_PRIMARY,
        " per quarter) -> ", holdings_path)

primary_qret <- build_portfolio(N_PRIMARY) %>%
  mutate(
    portfolio_cum_growth = cumprod(1 + portfolio_ret),
    spy_cum_growth       = cumprod(1 + spy_ret)
  )

message("\n--- Primary backtest (N = ", N_PRIMARY, ") quarterly returns ---")
print(primary_qret)

qret_path <- file.path(PROC_DIR, "backtest_quarterly_returns.csv")
write_csv(primary_qret, qret_path)
message("Saved quarterly returns + equity curve -> ", qret_path)

primary_summary <- backtest_summary %>% filter(n_top == N_PRIMARY)
message("\n--- Primary backtest (N = ", N_PRIMARY, ") summary ---")
message("Portfolio total return: ", signif(100 * primary_summary$portfolio_total_return, 4),
        "%  |  SPY total return: ", signif(100 * primary_summary$spy_total_return, 4), "%")
message("Portfolio annualized return: ", signif(100 * primary_summary$portfolio_ann_return, 4),
        "%  |  SPY annualized return: ", signif(100 * primary_summary$spy_ann_return, 4), "%")
message("Portfolio annualized vol: ", signif(100 * primary_summary$portfolio_ann_vol, 4),
        "%  |  SPY annualized vol: ", signif(100 * primary_summary$spy_ann_vol, 4), "%")
message("Portfolio max drawdown: ", signif(100 * primary_summary$portfolio_max_drawdown, 4),
        "%  |  SPY max drawdown: ", signif(100 * primary_summary$spy_max_drawdown, 4), "%")
message("Hit rate (quarters beating SPY): ", signif(100 * primary_summary$hit_rate, 4), "%")
message("Mean quarterly excess return: ", signif(100 * primary_summary$mean_excess_ret, 4), "%")
message("Information ratio (annualized): ", signif(primary_summary$information_ratio, 4))

# -----------------------------------------------------------------------------
# 5. Honest interpretation
# -----------------------------------------------------------------------------
# Rides on GBRT's own walk-forward R^2, small and sometimes negative.
# Uses today's S&P 500 membership, not point-in-time, so survivorship
# bias too. Any "beats SPY" result is at least as consistent with noise
# as a genuine edge. Report the numbers plainly, don't overstate either way.

message("\n04_results.R complete.")
