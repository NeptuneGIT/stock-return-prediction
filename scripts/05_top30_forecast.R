# 05_top30_forecast.R -- standalone report, not part of the main pipeline:
# ranks tickers by GBRT per-ticker pooled walk-forward R^2, takes the top
# 30, projects a $10 investment's value at Q3 2026 and Q2 2027.
#
# Output: output/tables/top30_forecast_<date>.csv

library(dplyr)
library(readr)

PROC_DIR <- "data/processed"
RAW_DIR  <- "data/raw"
OUT_DIR  <- "output/tables"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

N_TOP <- 30


ticker_r2 <- read_csv(file.path(PROC_DIR, "gbrt_ticker_r2.csv"), show_col_types = FALSE)


row_counts <- table(ticker_r2$n_val_rows)
full_window <- as.integer(names(row_counts)[which.max(row_counts)])
message("Requiring the modal (most common) ", full_window, " pooled ",
        "walk-forward validation row(s): ", sum(ticker_r2$n_val_rows == full_window),
        " of ", nrow(ticker_r2), " tickers qualify.")

top30 <- ticker_r2 %>%
  filter(n_val_rows == full_window) %>%
  arrange(desc(r2)) %>%
  slice_head(n = N_TOP)

if (nrow(top30) < N_TOP) {
  warning("Only ", nrow(top30), " tickers had the full pooled walk-forward ",
          "validation window -- fewer than the requested top ", N_TOP, ".")
}


forward <- read_csv(file.path(PROC_DIR, "gbrt_forward_predictions.csv"), show_col_types = FALSE)
spy_assumption <- read_csv(file.path(PROC_DIR, "spy_forward_assumption.csv"), show_col_types = FALSE)

spy_1q <- spy_assumption$realized_mean_ret[spy_assumption$horizon == "1q"]
spy_4q <- spy_assumption$realized_mean_ret[spy_assumption$horizon == "4q"]

missing_forward <- setdiff(top30$ticker, forward$ticker)
if (length(missing_forward) > 0) {
  warning("Top-", N_TOP, " ticker(s) with no forward prediction (excluded ",
          "by 03_models.R's forward-scoring step for a predictor gap at ",
          "the current quarter): ", paste(missing_forward, collapse = ", "),
          " -- dropped from this report.")
}

top30 <- top30 %>% inner_join(forward, by = "ticker")


prices <- read_csv(file.path(RAW_DIR, "prices_daily.csv"), show_col_types = FALSE)

latest_close <- prices %>%
  filter(symbol %in% top30$ticker) %>%
  group_by(symbol) %>%
  slice_max(date, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  transmute(ticker = symbol, close_today = close, close_date = date)

top30 <- top30 %>% inner_join(latest_close, by = "ticker")

panel_sector <- read_csv(file.path(PROC_DIR, "panel.csv"), show_col_types = FALSE) %>%
  distinct(ticker, sector)

gbrt_pred_1q <- read_csv(file.path(PROC_DIR, "gbrt_walkforward_predictions.csv"), show_col_types = FALSE)
gbrt_pred_4q <- read_csv(file.path(PROC_DIR, "gbrt_walkforward_predictions_4q.csv"), show_col_types = FALSE)

resid_sd_1q <- sd(gbrt_pred_1q$actual - gbrt_pred_1q$pred)
resid_sd_4q <- sd(gbrt_pred_4q$actual - gbrt_pred_4q$pred)

message("1Q model pooled walk-forward out-of-fold residual SD: ", signif(resid_sd_1q, 4),
        " | 4Q model: ", signif(resid_sd_4q, 4),
        " (used below as a simple +/-1 SD band, NOT a statistical ",
        "confidence interval).")

report <- top30 %>%
  left_join(panel_sector, by = "ticker") %>%
  mutate(
    pred_stock_ret_q3_2026 = pred_exret_next_1q + spy_1q,
    pred_stock_ret_q2_2027 = pred_exret_next_4q + spy_4q,

    price_q3_2026    = close_today * (1 + pred_stock_ret_q3_2026),
    price_q3_2026_lo = close_today * (1 + pred_stock_ret_q3_2026 - resid_sd_1q),
    price_q3_2026_hi = close_today * (1 + pred_stock_ret_q3_2026 + resid_sd_1q),
    value10_q3_2026  = 10 * (1 + pred_stock_ret_q3_2026),

    price_q2_2027    = close_today * (1 + pred_stock_ret_q2_2027),
    price_q2_2027_lo = close_today * (1 + pred_stock_ret_q2_2027 - resid_sd_4q),
    price_q2_2027_hi = close_today * (1 + pred_stock_ret_q2_2027 + resid_sd_4q),
    value10_q2_2027  = 10 * (1 + pred_stock_ret_q2_2027),

    spy_1q_assumed_ret = spy_1q,
    spy_4q_assumed_ret = spy_4q
  ) %>%
  select(
    ticker, sector, r2, adj_r2, n_val_rows, close_today, close_date,
    pred_exret_next_1q, pred_stock_ret_q3_2026,
    price_q3_2026, price_q3_2026_lo, price_q3_2026_hi, value10_q3_2026,
    pred_exret_next_4q, pred_stock_ret_q2_2027,
    price_q2_2027, price_q2_2027_lo, price_q2_2027_hi, value10_q2_2027,
    spy_1q_assumed_ret, spy_4q_assumed_ret
  ) %>%
  arrange(desc(r2))


out_path <- file.path(OUT_DIR, paste0("top30_forecast_", format(Sys.Date(), "%Y%m%d"), ".csv"))
write_csv(report, out_path)
message("\nSaved ", nrow(report), " rows -> ", out_path)

message("\n--- Top 10 by 1Q per-ticker pooled walk-forward R^2 ---")
print(head(report, 10) %>% select(ticker, sector, r2, value10_q3_2026, value10_q2_2027), n = 10)

message("\n--- Caveats (read before using this report for anything) ---")
message("- Ticker selection ranks by 1Q GBRT per-ticker POOLED WALK-FORWARD ",
        "VALIDATION R^2 (out-of-fold predictions pooled across every walk-",
        "forward fold, not a single fixed test period), requiring the full ",
        full_window, " pooled validation row(s); GBRT's pooled walk-forward ",
        "R^2 (see 03_models.R output for this run's exact figure) is small ",
        "and has sometimes NOT been the highest of the three models tested ",
        "(PLS has numerically edged it out on some runs) -- so even the ",
        "BEST individually-ranked tickers here are selected from a model ",
        "with weak aggregate out-of-sample skill. GBRT is used throughout ",
        "this project as the primary model; read every ranking and figure ",
        "below with that weak aggregate skill in mind, not as evidence GBRT ",
        "is the strongest model on every run.")
message("- Q2 2027 uses a genuinely separate 4-quarter-ahead GBRT model with a ",
        "SHORTER, noisier usable quarter range than the 1Q model -- its ",
        "walk-forward R^2 should be read with even more caution.")
message("- Both GBRT models used for the forward predictions below (1Q and ",
        "4Q) are refit on 03_models.R's ENTIRE panel (not a pre-2023 ",
        "subset) before forward-scoring the current not-yet-realized ",
        "quarter -- this is deliberate (a forward forecast should use all ",
        "available history, not blind itself to 2023-2026 data) and is ",
        "look-ahead-safe because it only ever scores a quarter that comes ",
        "AFTER every training quarter; see 03_models.R's top-of-file note ",
        "for the full reasoning and the hard rule that this same fit must ",
        "NEVER be used to score a historical quarter.")
message("- SPY's own future return is NOT modelled -- it is assumed to equal ",
        "the mean of SPY's REALIZED return over the pooled walk-forward ",
        "validation window at the matching horizon (1Q assumption = ",
        signif(spy_1q, 4), ", 4Q assumption = ", signif(spy_4q, 4),
        "). A materially different real SPY return would shift every ",
        "price/value figure here.")
message("- close_today is the latest daily close (data/raw/prices_daily.csv), ",
        "which is more recent than the predictor snapshot the forward ",
        "predictions are anchored to -- see close_date per ticker.")
message("- price_*_lo/hi bands are +/-1 SD of POOLED walk-forward ",
        "out-of-fold residuals, NOT a statistical confidence interval.")

message("\n05_top30_forecast.R complete.")
