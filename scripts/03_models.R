# 03_models.R -- fits and compares the three models required by the course spec
# (PLS, Random Forest, GBRT) against the quarterly modelling panel, and produces
# the forward-looking scores that scripts/05_top30_forecast.R consumes.
#
# Target: exret_next, a ticker's next-quarter return minus SPY's next-quarter
# return over the same period (built in scripts/01a_ratios.R). GBRT is this
# project's headline model; PLS and Random Forest are fit identically and
# reported alongside it as the comparison baseline -- see docs/METHODOLOGY.md
# for the full model-selection narrative and why GBRT is reported as the
# primary result despite PLS's walk-forward R^2 numerically edging it out on
# some runs.
#
# Validation: walk-forward only, no fixed calendar train/test split. Folds are
# expanding-window and time-respecting -- every fold trains only on quarters
# strictly before its own validation block -- which is what actually prevents
# look-ahead bias here, not any particular calendar cutoff. GBRT's n.trees and
# PLS's ncomp are chosen this way; Random Forest's mtry is fixed at the package
# regression default (floor(p/3)) rather than grid-searched (see
# fit_rf_walkforward()'s comment below for why). The pooled out-of-fold
# predictions this produces are the ONLY historical-quarter out-of-sample
# numbers this script or any downstream script may report.
#
# Each model is also refit once on the ENTIRE panel at its walk-forward-selected
# hyperparameter. This entire-panel fit is used only for variable importance and
# to forward-score the current not-yet-realized quarter (for
# scripts/05_top30_forecast.R) -- it must NEVER be used to generate a
# historical-quarter prediction, since it has seen every quarter and doing so
# would leak future information into the past. See docs/METHODOLOGY.md for the
# full reasoning; treat any change that blurs this separation as a
# look-ahead-bias regression, not a refactor.
#
# A second, supplementary GBRT model is also fit here against a 4-quarter-ahead
# target (exret_next4, built in 01a_ratios.R) purely to power
# scripts/05_top30_forecast.R's "next year" forecast with a genuinely
# re-estimated model instead of naively compounding the 1-quarter result. This
# does not change which model is primary -- GBRT on the 1-quarter exret_next
# target remains the result reported in the README and the project report.
#
# Input:  data/processed/panel.csv          (01a_ratios.R)
# Output: data/processed/ols_diagnostics[_sector].csv
#         data/processed/random_forest_predictions[_sector].csv
#         data/processed/random_forest_importance[_sector].csv
#         data/processed/random_forest_walkforward_cv.csv
#         data/processed/pls_predictions[_sector].csv
#         data/processed/pls_coefficients[_sector].csv
#         data/processed/pls_walkforward_cv.csv
#         data/processed/gbrt_predictions[_sector].csv      (in-sample, entire panel)
#         data/processed/gbrt_importance[_sector].csv
#         data/processed/gbrt_walkforward_cv.csv
#         data/processed/gbrt_walkforward_predictions.csv   (pooled out-of-fold, 1Q)
#         data/processed/gbrt_ticker_r2.csv                 (pooled out-of-fold, per ticker)
#         data/processed/gbrt_predictions_4q.csv            (in-sample, entire panel)
#         data/processed/gbrt_walkforward_cv_4q.csv
#         data/processed/gbrt_walkforward_predictions_4q.csv (pooled out-of-fold, 4Q)
#         data/processed/gbrt_forward_predictions.csv
#         data/processed/spy_forward_assumption.csv
# =============================================================================

library(dplyr)
library(readr)
library(pls)
library(randomForest)
library(gbm)

PROC_DIR <- "data/processed"
SEED <- 42


TARGET_VAR <- "exret_next"
OUTPUT_SUFFIX <- if (TARGET_VAR == "exret_next") "" else "_sector"

PREDICTORS <- c(
  "mom_1m", "mom_3m", "mom_12m", "vol_60d", "dollar_volume",  # technical
  "beta_252d", "idio_vol_60d", "pct_from_52w_high",  # technical/risk
  "roe", "debt_to_equity",                  # fundamentals (SEC)
  "asset_growth",                           # fundamentals (SEC)
  "ni_growth", "capex_intensity",           # fundamentals (SEC)
  "pb_ratio", "fcf_yield"                   # price x fundamentals
)


# 1. Load the panel, deduplicate, and drop rows with no realized target yet

panel <- read_csv(file.path(PROC_DIR, "panel.csv"), show_col_types = FALSE)

# panel.csv can contain a handful of duplicate (ticker, quarter_end) rows --
# identical price/return/technical columns but differing fundamentals-derived
# columns, consistent with the point-in-time fundamentals join in 01a_ratios.R
# occasionally fanning out to more than one qualifying filing instead of
# collapsing to exactly one. This dedup is a deterministic downstream
# workaround (first occurrence wins), not a fix at the source -- see
# docs/METHODOLOGY.md for the known-issue writeup.
n_before_dedup <- nrow(panel)
panel <- panel %>% distinct(ticker, quarter_end, .keep_all = TRUE)
n_dup_dropped <- n_before_dedup - nrow(panel)
if (n_dup_dropped > 0) {
  warning(n_dup_dropped, " duplicate (ticker, quarter_end) row(s) dropped ",
          "from panel.csv. This is a pre-existing panel.csv data-integrity ",
          "issue (see docs/METHODOLOGY.md), not something introduced by ",
          "this script.")
}


panel_raw <- panel

forward_raw <- panel_raw %>% filter(quarter_end == max(quarter_end))

n_no_target <- sum(is.na(panel[[TARGET_VAR]]))
message("Dropping ", n_no_target, " row(s) with no known ", TARGET_VAR, " ",
        "(most recent quarter per ticker, not yet realised).")

panel <- panel %>% filter(!is.na(.data[[TARGET_VAR]]))

# 2. Complete-case filter on predictors


is_complete <- panel %>%
  select(all_of(PREDICTORS)) %>%
  complete.cases()

message("\nRows dropped for missing predictor(s), by ticker:")
panel %>%
  mutate(complete = is_complete) %>%
  group_by(ticker) %>%
  summarise(
    total    = n(),
    dropped  = sum(!complete),
    kept     = sum(complete),
    .groups  = "drop"
  ) %>%
  as.data.frame() %>%
  print()

panel_model <- panel[is_complete, ]


if (nrow(panel_model) == 0) {
  stop("Complete-case filtering left zero rows. Check the missingness ",
       "table above for the culprit predictor.")
}
ticker_counts <- table(panel_model$ticker)
if (any(ticker_counts == 0)) {
  stop("Complete-case filtering left zero rows for at least one ticker: ",
       paste(names(ticker_counts)[ticker_counts == 0], collapse = ", "))
}

message("\nTotal rows after filtering: ", nrow(panel_model), " of ",
        nrow(panel), " rows with a known target.")


full_df <- panel_model %>% arrange(ticker, quarter_end)

x_full <- as.matrix(full_df[, PREDICTORS])
y_full <- full_df[[TARGET_VAR]]



# 3. Diagnostic: correlations + unpenalized OLS over the entire panel
#    (a descriptive baseline only, not used to select or tune any model below)

message("\n--- Diagnostic: correlations with ", TARGET_VAR, " (entire panel) ---")
full_cor <- sapply(PREDICTORS, function(p) cor(x_full[, p], y_full))
print(sort(full_cor, decreasing = TRUE))

message("\n--- Diagnostic: unpenalized OLS on the entire panel ---")

ols_df <- as.data.frame(x_full)
ols_df$target <- y_full
ols_fit <- lm(target ~ ., data = ols_df)
ols_summary <- summary(ols_fit)
print(ols_summary)

ols_coefs <- as.data.frame(ols_summary$coefficients)
colnames(ols_coefs) <- c("ols_estimate", "ols_std_error", "ols_t_value", "ols_p_value")
ols_coefs$predictor <- rownames(ols_coefs)
ols_coefs$predictor[ols_coefs$predictor == "(Intercept)"] <- "(Intercept)"

diagnostics <- data.frame(predictor = c("(Intercept)", PREDICTORS)) %>%
  left_join(
    data.frame(predictor = names(full_cor), correlation = as.numeric(full_cor)),
    by = "predictor"
  ) %>%
  left_join(ols_coefs, by = "predictor") %>%
  arrange(desc(abs(correlation)))

ols_diagnostics_path <- file.path(PROC_DIR, paste0("ols_diagnostics", OUTPUT_SUFFIX, ".csv"))
write_csv(diagnostics, ols_diagnostics_path)
message("\nSaved diagnostics (correlation + OLS coefficients/p-values) -> ",
        ols_diagnostics_path)



# 4. Shared metric helpers and walk-forward fold construction

rmse <- function(actual, pred) sqrt(mean((actual - pred)^2))
r_squared <- function(actual, pred) {
  1 - sum((actual - pred)^2) / sum((actual - mean(actual))^2)
}

adj_r_squared <- function(r2, n, p) {
  1 - (1 - r2) * (n - 1) / (n - p - 1)
}

WF_WARMUP_QUARTERS <- 20
WF_FOLD_QUARTERS <- 4

build_walkforward_folds <- function(quarters, warmup_quarters, fold_quarters) {
  quarters <- sort(unique(quarters))
  n_q <- length(quarters)
  if (n_q <= warmup_quarters) {
    stop("Not enough quarters (", n_q, ") for a ", warmup_quarters,
         "-quarter walk-forward warmup.")
  }
  fold_starts <- seq(warmup_quarters + 1, n_q, by = fold_quarters)

  fold_starts <- fold_starts[fold_starts + fold_quarters - 1 <= n_q]
  lapply(fold_starts, function(start_idx) {
    end_idx <- start_idx + fold_quarters - 1
    list(
      train_quarters = quarters[1:(start_idx - 1)],
      val_quarters   = quarters[start_idx:end_idx]
    )
  })
}

fit_gbrt_walkforward <- function(df, target_col, predictors, folds,
                                  n_trees_ceiling, shrinkage, depth,
                                  seed, p_adj) {
  fold_fits <- lapply(seq_along(folds), function(i) {
    fold <- folds[[i]]
    fold_train <- df[df$quarter_end %in% fold$train_quarters, ]
    fold_val   <- df[df$quarter_end %in% fold$val_quarters, ]

    fold_train_df <- as.data.frame(fold_train[, predictors])
    fold_train_df$target <- fold_train[[target_col]]

    set.seed(seed)
    fold_fit <- gbm(
      target ~ ., data = fold_train_df, distribution = "gaussian",
      n.trees = n_trees_ceiling, shrinkage = shrinkage,
      interaction.depth = depth, cv.folds = 0, n.cores = 1
    )

    fold_val_x <- as.data.frame(fold_val[, predictors])
    staged_pred <- predict(fold_fit, newdata = fold_val_x,
                            n.trees = 1:n_trees_ceiling)

    list(fold = i, val_quarters = fold$val_quarters,
         ticker = fold_val$ticker, quarter_end = fold_val$quarter_end,
         actual = fold_val[[target_col]], staged_pred = staged_pred)
  })

  # matrix: n_trees_ceiling rows x n_folds cols
  rmse_by_fold <- sapply(fold_fits, function(fd) {
    apply(fd$staged_pred, 2, function(p) rmse(fd$actual, p))
  })
  avg_rmse_curve <- rowMeans(rmse_by_fold)
  best_iter <- which.min(avg_rmse_curve)


  wf_actual <- unlist(lapply(fold_fits, function(fd) fd$actual))
  wf_pred   <- unlist(lapply(fold_fits, function(fd) fd$staged_pred[, best_iter]))
  wf_ticker <- unlist(lapply(fold_fits, function(fd) fd$ticker))
  wf_quarter_end <- do.call(c, lapply(fold_fits, function(fd) fd$quarter_end))
  wf_rmse_val <- rmse(wf_actual, wf_pred)
  wf_r2_val   <- r_squared(wf_actual, wf_pred)
  wf_adj_r2_val <- adj_r_squared(wf_r2_val, length(wf_actual), p_adj)

  fold_summary <- do.call(rbind, lapply(fold_fits, function(fd) {
    fold_curve <- apply(fd$staged_pred, 2, function(p) rmse(fd$actual, p))
    data.frame(
      fold = fd$fold,
      val_start = min(fd$val_quarters), val_end = max(fd$val_quarters),
      n_val_rows = length(fd$actual),
      fold_best_n_trees = which.min(fold_curve),
      rmse_at_global_best_n_trees = fold_curve[best_iter]
    )
  }))

  list(best_iter = best_iter, wf_rmse = wf_rmse_val, wf_r2 = wf_r2_val,
       wf_adj_r2 = wf_adj_r2_val, wf_n = length(wf_actual),
       wf_actual = wf_actual, wf_pred = wf_pred,
       wf_ticker = wf_ticker, wf_quarter_end = wf_quarter_end,
       fold_summary = fold_summary, n_folds = length(folds))
}

fit_pls_walkforward <- function(df, target_col, predictors, folds,
                                 max_ncomp, seed, p_adj) {
  fold_fits <- lapply(seq_along(folds), function(i) {
    fold <- folds[[i]]
    fold_train <- df[df$quarter_end %in% fold$train_quarters, ]
    fold_val   <- df[df$quarter_end %in% fold$val_quarters, ]

    fold_train_df <- as.data.frame(fold_train[, predictors])
    fold_train_df$target <- fold_train[[target_col]]

    set.seed(seed)
    fold_fit <- plsr(target ~ ., data = fold_train_df, scale = TRUE,
                      validation = "none", ncomp = max_ncomp)

    fold_val_x <- as.data.frame(fold_val[, predictors])
    staged_pred <- drop(predict(fold_fit, newdata = fold_val_x, ncomp = 1:max_ncomp))
    if (is.null(dim(staged_pred))) staged_pred <- matrix(staged_pred, ncol = max_ncomp)


    intercept_only <- rep(mean(fold_train_df$target), nrow(fold_val_x))
    staged_pred_full <- cbind(intercept_only, staged_pred)

    list(fold = i, val_quarters = fold$val_quarters,
         actual = fold_val[[target_col]], staged_pred_full = staged_pred_full)
  })


  rmse_by_fold <- sapply(fold_fits, function(fd) {
    apply(fd$staged_pred_full, 2, function(p) rmse(fd$actual, p))
  })
  avg_rmse_curve <- rowMeans(rmse_by_fold)

  ncomp_min <- which.min(avg_rmse_curve[-1])
  min_row <- ncomp_min + 1

  se_at_min <- sd(rmse_by_fold[min_row, ]) / sqrt(ncol(rmse_by_fold))
  threshold <- avg_rmse_curve[min_row] + se_at_min
  within_1se <- which(avg_rmse_curve <= threshold) - 1  # back to ncomp values
  ncomp_1se <- min(within_1se)

  pool_at <- function(ncomp) {
    actual <- unlist(lapply(fold_fits, function(fd) fd$actual))
    pred   <- unlist(lapply(fold_fits, function(fd) fd$staged_pred_full[, ncomp + 1]))
    r2 <- r_squared(actual, pred)
    list(rmse = rmse(actual, pred), r2 = r2,
         adj_r2 = adj_r_squared(r2, length(actual), p_adj), n = length(actual))
  }
  wf_min <- pool_at(ncomp_min)
  wf_1se <- pool_at(ncomp_1se)

  fold_summary <- do.call(rbind, lapply(fold_fits, function(fd) {
    fold_curve <- apply(fd$staged_pred_full, 2, function(p) rmse(fd$actual, p))
    data.frame(
      fold = fd$fold,
      val_start = min(fd$val_quarters), val_end = max(fd$val_quarters),
      n_val_rows = length(fd$actual),
      fold_best_ncomp = which.min(fold_curve[-1]),
      rmse_at_global_ncomp_min = fold_curve[min_row]
    )
  }))

  list(ncomp_min = ncomp_min, ncomp_1se = ncomp_1se,
       wf_rmse_min = wf_min$rmse, wf_r2_min = wf_min$r2, wf_adj_r2_min = wf_min$adj_r2,
       wf_rmse_1se = wf_1se$rmse, wf_r2_1se = wf_1se$r2, wf_adj_r2_1se = wf_1se$adj_r2,
       wf_n = wf_min$n,
       fold_summary = fold_summary, n_folds = length(folds))
}

# Random Forest walk-forward validation. Unlike GBRT's n.trees and PLS's
# ncomp, mtry is passed in fixed rather than grid-searched over candidate
# values within this function: once the fold set spans the entire panel
# (rather than a small pre-2023 subset), a multi-value mtry grid -- each
# candidate a full untuned randomForest() fit, repeated across every fold --
# became too slow to run in practice. mtry is fixed at the package
# regression default (floor(p/3)) instead; Random Forest is still evaluated
# by the same pooled out-of-fold walk-forward metric as the other two
# models at that fixed value.
fit_rf_walkforward <- function(df, target_col, predictors, folds, mtry,
                                ntree, seed, p_adj) {
  fold_fits <- lapply(seq_along(folds), function(i) {
    fold <- folds[[i]]
    fold_train <- df[df$quarter_end %in% fold$train_quarters, ]
    fold_val   <- df[df$quarter_end %in% fold$val_quarters, ]

    x_fold_train <- as.matrix(fold_train[, predictors])
    y_fold_train <- fold_train[[target_col]]
    x_fold_val   <- as.matrix(fold_val[, predictors])
    y_fold_val   <- fold_val[[target_col]]

    set.seed(seed)
    fold_fit <- randomForest(x = x_fold_train, y = y_fold_train,
                              mtry = mtry, ntree = ntree)
    pred_val <- as.numeric(predict(fold_fit, newdata = x_fold_val))

    list(fold = i, val_quarters = fold$val_quarters,
         actual = y_fold_val, pred = pred_val)
  })

  wf_actual <- unlist(lapply(fold_fits, function(fd) fd$actual))
  wf_pred   <- unlist(lapply(fold_fits, function(fd) fd$pred))
  wf_rmse_val <- rmse(wf_actual, wf_pred)
  wf_r2_val   <- r_squared(wf_actual, wf_pred)

  fold_summary <- do.call(rbind, lapply(fold_fits, function(fd) {
    data.frame(
      fold = fd$fold,
      val_start = min(fd$val_quarters), val_end = max(fd$val_quarters),
      n_val_rows = length(fd$actual),
      val_rmse = rmse(fd$actual, fd$pred)
    )
  }))

  list(mtry = mtry, wf_rmse = wf_rmse_val, wf_r2 = wf_r2_val,
       wf_adj_r2 = adj_r_squared(wf_r2_val, length(wf_actual), p_adj),
       wf_n = length(wf_actual),
       fold_summary = fold_summary,
       n_folds = length(folds))
}


per_ticker_r2 <- function(df, actual_col, pred_col) {
  df %>%
    group_by(ticker) %>%
    summarise(
      n_val_rows = n(),
      r2 = 1 - sum((.data[[actual_col]] - .data[[pred_col]])^2) /
               sum((.data[[actual_col]] - mean(.data[[actual_col]]))^2),
      rmse = sqrt(mean((.data[[actual_col]] - .data[[pred_col]])^2)),
      .groups = "drop"
    ) %>%
    arrange(desc(r2))
}

wf_folds <- build_walkforward_folds(full_df$quarter_end, WF_WARMUP_QUARTERS, WF_FOLD_QUARTERS)
message("\nWalk-forward validation: ", length(wf_folds), " expanding-window fold(s) over ",
        length(unique(full_df$quarter_end)), " quarters (",
        min(full_df$quarter_end), " to ", max(full_df$quarter_end), "), ",
        WF_WARMUP_QUARTERS, "-quarter warmup, ", WF_FOLD_QUARTERS,
        "-quarter validation blocks -- spans the ENTIRE panel (no fixed ",
        "train/test split). Shared by Random Forest, PLS, and the 1Q GBRT ",
        "model below. Every fold still trains only on quarters strictly ",
        "before its own validation block.")


# 5. Random Forest: walk-forward validation, then a single entire-panel fit
#    (the entire-panel fit is used for variable importance only -- see the
#    top-of-file note on why it must never score a historical quarter)

RF_MTRY <- max(floor(length(PREDICTORS) / 3), 1)
RF_NTREE <- 500

rf_wf <- fit_rf_walkforward(full_df, TARGET_VAR, PREDICTORS, wf_folds,
                             RF_MTRY, RF_NTREE, SEED, length(PREDICTORS))

rf_walkforward_path <- file.path(PROC_DIR, "random_forest_walkforward_cv.csv")
write_csv(rf_wf$fold_summary, rf_walkforward_path)
message("\nRandom Forest walk-forward validation (mtry = fixed at ", RF_MTRY,
        " -- package regression default floor(p/3), ntree = ", RF_NTREE, "):")
message("  Walk-forward RMSE = ", signif(rf_wf$wf_rmse, 4),
        "  R^2 = ", signif(rf_wf$wf_r2, 4),
        "  Adjusted R^2 = ", signif(rf_wf$wf_adj_r2, 4),
        "  (n = ", rf_wf$wf_n, " pooled out-of-fold rows across ", rf_wf$n_folds, " folds)")
message("Saved ", nrow(rf_wf$fold_summary), " walk-forward fold rows -> ", rf_walkforward_path)

set.seed(SEED)
rf_fit <- randomForest(
  x = x_full, y = y_full,
  mtry = RF_MTRY, ntree = RF_NTREE, importance = TRUE
)

message("\nRandom Forest OOB performance (supplementary diagnostic over the ",
        "entire panel -- mtry is fixed, not chosen by OOB error):")
message("  OOB MSE = ", signif(rf_fit$mse[rf_fit$ntree], 4),
        "  OOB R^2 = ", signif(rf_fit$rsq[rf_fit$ntree], 4))

message("\nVariable importance (%IncMSE / IncNodePurity):")
print(importance(rf_fit))

# Predict + evaluate: in-sample (entire panel) vs. walk-forward

rf_pred_full <- as.numeric(predict(rf_fit, newdata = x_full))

message("\nRandom Forest performance:")
rf_metrics <- data.frame(
  fit       = c("in_sample", "walkforward"),
  n         = c(nrow(full_df), rf_wf$wf_n),
  rmse      = c(rmse(y_full, rf_pred_full), rf_wf$wf_rmse),
  r_squared = c(r_squared(y_full, rf_pred_full), rf_wf$wf_r2)
)
rf_metrics$adjusted_r_squared <- adj_r_squared(rf_metrics$r_squared, rf_metrics$n, length(PREDICTORS))
print(rf_metrics)

# Write Random Forest outputs

rf_predictions <- full_df %>%
  transmute(ticker, quarter_end, actual = .data[[TARGET_VAR]], pred = rf_pred_full)

rf_predictions_path <- file.path(PROC_DIR, paste0("random_forest_predictions", OUTPUT_SUFFIX, ".csv"))
write_csv(rf_predictions, rf_predictions_path)

rf_importance <- as.data.frame(importance(rf_fit))
rf_importance$predictor <- rownames(rf_importance)
rf_importance <- rf_importance %>%
  select(predictor, everything()) %>%
  arrange(desc(`%IncMSE`))
rf_importance_path <- file.path(PROC_DIR, paste0("random_forest_importance", OUTPUT_SUFFIX, ".csv"))
write_csv(rf_importance, rf_importance_path)

message("\nSaved ", nrow(rf_predictions), " in-sample predictions -> ", rf_predictions_path)
message("Saved ", nrow(rf_importance), " importance rows -> ", rf_importance_path)


# 6. PLS (Partial Least Squares): walk-forward validation, then a single
#    entire-panel fit. ncomp is chosen two ways -- the minimum-average-RMSE
#    value (ncomp_min) and a more conservative one-standard-error rule
#    (ncomp_1se) -- and both are reported throughout.


MAX_NCOMP <- length(PREDICTORS)

pls_wf <- fit_pls_walkforward(full_df, TARGET_VAR, PREDICTORS, wf_folds,
                               MAX_NCOMP, SEED, length(PREDICTORS))

pls_walkforward_path <- file.path(PROC_DIR, "pls_walkforward_cv.csv")
write_csv(pls_wf$fold_summary, pls_walkforward_path)
message("\nPLS walk-forward validation (max ncomp = ", MAX_NCOMP, "):")
message("  ncomp (min average validation RMSE across ", pls_wf$n_folds,
        " folds) = ", pls_wf$ncomp_min)
message("  ncomp (one-standard-error rule) = ", pls_wf$ncomp_1se,
        if (pls_wf$ncomp_1se == 0) " (intercept-only -- one-SE rule found no component reliably beats the mean)" else "")
message("  Walk-forward RMSE/R^2/Adjusted R^2 @ ncomp_min: ", signif(pls_wf$wf_rmse_min, 4),
        " / ", signif(pls_wf$wf_r2_min, 4), " / ", signif(pls_wf$wf_adj_r2_min, 4),
        "  (n = ", pls_wf$wf_n, " pooled out-of-fold rows)")
message("  Walk-forward RMSE/R^2/Adjusted R^2 @ ncomp_1se: ", signif(pls_wf$wf_rmse_1se, 4),
        " / ", signif(pls_wf$wf_r2_1se, 4), " / ", signif(pls_wf$wf_adj_r2_1se, 4))
message("Saved ", nrow(pls_wf$fold_summary), " walk-forward fold rows -> ", pls_walkforward_path)

ncomp_min <- pls_wf$ncomp_min
ncomp_1se <- pls_wf$ncomp_1se

set.seed(SEED)
pls_fit <- plsr(target ~ ., data = ols_df, scale = TRUE, validation = "none", ncomp = MAX_NCOMP)

message("\nPLS coefficients @ ncomp_min:")
pls_coef_min <- drop(coef(pls_fit, ncomp = ncomp_min, intercept = TRUE))
print(pls_coef_min)
if (ncomp_1se == 0) {
  message("\nPLS coefficients @ ncomp_1se: intercept-only (all predictor coefficients 0)")
  pls_coef_1se <- setNames(c(mean(y_full), rep(0, length(PREDICTORS))),
                            names(pls_coef_min))
  print(pls_coef_1se)
  pls_pred_full_1se <- rep(mean(y_full), nrow(full_df))
} else {
  message("\nPLS coefficients @ ncomp_1se:")
  pls_coef_1se <- drop(coef(pls_fit, ncomp = ncomp_1se, intercept = TRUE))
  print(pls_coef_1se)
  pls_pred_full_1se <- as.numeric(predict(pls_fit, newdata = as.data.frame(x_full), ncomp = ncomp_1se))
}

pls_pred_full_min <- as.numeric(predict(pls_fit, newdata = as.data.frame(x_full), ncomp = ncomp_min))

message("\nPLS performance:")
pls_metrics <- data.frame(
  fit       = c("in_sample", "in_sample", "walkforward", "walkforward"),
  ncomp     = c("ncomp_min", "ncomp_1se", "ncomp_min", "ncomp_1se"),
  n         = c(nrow(full_df), nrow(full_df), pls_wf$wf_n, pls_wf$wf_n),
  rmse      = c(rmse(y_full, pls_pred_full_min), rmse(y_full, pls_pred_full_1se),
                 pls_wf$wf_rmse_min, pls_wf$wf_rmse_1se),
  r_squared = c(r_squared(y_full, pls_pred_full_min), r_squared(y_full, pls_pred_full_1se),
                 pls_wf$wf_r2_min, pls_wf$wf_r2_1se)
)
pls_metrics$adjusted_r_squared <- adj_r_squared(pls_metrics$r_squared, pls_metrics$n, length(PREDICTORS))
print(pls_metrics)

pls_predictions <- full_df %>%
  transmute(ticker, quarter_end,
            actual = .data[[TARGET_VAR]],
            pred_ncomp_min = pls_pred_full_min,
            pred_ncomp_1se = pls_pred_full_1se)

pls_predictions_path <- file.path(PROC_DIR, paste0("pls_predictions", OUTPUT_SUFFIX, ".csv"))
write_csv(pls_predictions, pls_predictions_path)

pls_coefficients <- data.frame(
  predictor        = names(pls_coef_min),
  coef_ncomp_min = as.numeric(pls_coef_min),
  coef_ncomp_1se = as.numeric(pls_coef_1se)
)
pls_coefficients_path <- file.path(PROC_DIR, paste0("pls_coefficients", OUTPUT_SUFFIX, ".csv"))
write_csv(pls_coefficients, pls_coefficients_path)

message("\nSaved ", nrow(pls_predictions), " PLS in-sample predictions -> ", pls_predictions_path)
message("Saved ", nrow(pls_coefficients), " PLS coefficients -> ", pls_coefficients_path)


# 7. GBRT (Gradient Boosted Regression Trees, gbm package), 1-quarter-ahead
#    target -- this project's primary model. Walk-forward validation, then a
#    single entire-panel fit used for variable importance and forward
#    scoring only (see the top-of-file note).

GBM_N_TREES <- 2000
GBM_SHRINKAGE <- 0.01
GBM_INTERACTION_DEPTH <- 3

gbrt_wf <- fit_gbrt_walkforward(full_df, TARGET_VAR, PREDICTORS, wf_folds,
                                 GBM_N_TREES, GBM_SHRINKAGE, GBM_INTERACTION_DEPTH,
                                 SEED, length(PREDICTORS))

gbrt_walkforward_path <- file.path(PROC_DIR, "gbrt_walkforward_cv.csv")
write_csv(gbrt_wf$fold_summary, gbrt_walkforward_path)
message("\nGBRT walk-forward validation (shrinkage = ", GBM_SHRINKAGE,
        ", interaction.depth = ", GBM_INTERACTION_DEPTH, ", n.trees ceiling = ",
        GBM_N_TREES, "):")
message("  n.trees (min average validation RMSE across ", gbrt_wf$n_folds,
        " folds -- early stopping) = ", gbrt_wf$best_iter, " of ", GBM_N_TREES)
message("  Walk-forward RMSE = ", signif(gbrt_wf$wf_rmse, 4),
        "  R^2 = ", signif(gbrt_wf$wf_r2, 4),
        "  Adjusted R^2 = ", signif(gbrt_wf$wf_adj_r2, 4),
        "  (n = ", gbrt_wf$wf_n, " pooled out-of-fold rows)")
message("Saved ", nrow(gbrt_wf$fold_summary), " walk-forward fold rows -> ", gbrt_walkforward_path)

best_iter <- gbrt_wf$best_iter

set.seed(SEED)
gbm_fit <- gbm(
  target ~ .,
  data = ols_df,
  distribution = "gaussian",
  n.trees = GBM_N_TREES,
  shrinkage = GBM_SHRINKAGE,
  interaction.depth = GBM_INTERACTION_DEPTH,
  cv.folds = 0,
  n.cores = 1
)

message("\nVariable importance (relative influence):")
gbm_importance <- summary(gbm_fit, n.trees = best_iter, plotit = FALSE)
print(gbm_importance)



gbm_pred_full <- as.numeric(predict(gbm_fit, newdata = full_df, n.trees = best_iter))

message("\nGBRT performance:")
gbm_metrics <- data.frame(
  fit       = c("in_sample", "walkforward"),
  n         = c(nrow(full_df), gbrt_wf$wf_n),
  rmse      = c(rmse(y_full, gbm_pred_full), gbrt_wf$wf_rmse),
  r_squared = c(r_squared(y_full, gbm_pred_full), gbrt_wf$wf_r2)
)
gbm_metrics$adjusted_r_squared <- adj_r_squared(gbm_metrics$r_squared, gbm_metrics$n, length(PREDICTORS))
print(gbm_metrics)


gbm_predictions <- full_df %>%
  transmute(ticker, quarter_end, actual = .data[[TARGET_VAR]], pred = gbm_pred_full)

gbm_predictions_path <- file.path(PROC_DIR, paste0("gbrt_predictions", OUTPUT_SUFFIX, ".csv"))
write_csv(gbm_predictions, gbm_predictions_path)

gbm_importance_out <- gbm_importance %>%
  rename(predictor = var, importance = rel.inf) %>%
  arrange(desc(importance))
gbm_importance_path <- file.path(PROC_DIR, paste0("gbrt_importance", OUTPUT_SUFFIX, ".csv"))
write_csv(gbm_importance_out, gbm_importance_path)

gbrt_walkforward_predictions <- data.frame(
  ticker = gbrt_wf$wf_ticker,
  quarter_end = gbrt_wf$wf_quarter_end,
  actual = gbrt_wf$wf_actual,
  pred = gbrt_wf$wf_pred
) %>% arrange(ticker, quarter_end)

gbrt_wf_predictions_path <- file.path(PROC_DIR, "gbrt_walkforward_predictions.csv")
write_csv(gbrt_walkforward_predictions, gbrt_wf_predictions_path)

message("\nSaved ", nrow(gbm_predictions), " in-sample predictions -> ", gbm_predictions_path)
message("Saved ", nrow(gbm_importance_out), " importance rows -> ", gbm_importance_path)
message("Saved ", nrow(gbrt_walkforward_predictions), " pooled out-of-fold predictions -> ",
        gbrt_wf_predictions_path)



gbrt_ticker_r2 <- per_ticker_r2(
  data.frame(ticker = gbrt_wf$wf_ticker, actual = gbrt_wf$wf_actual, pred = gbrt_wf$wf_pred),
  "actual", "pred"
) %>%
  mutate(adj_r2 = ifelse(n_val_rows - length(PREDICTORS) - 1 > 0,
                          adj_r_squared(r2, n_val_rows, length(PREDICTORS)),
                          NA_real_))

message("\nPer-ticker GBRT walk-forward R^2 -- top 10:")
print(head(gbrt_ticker_r2, 10))
message(sum(gbrt_ticker_r2$n_val_rows == max(gbrt_ticker_r2$n_val_rows)), " of ",
        nrow(gbrt_ticker_r2), " tickers have the full ",
        max(gbrt_ticker_r2$n_val_rows), " pooled walk-forward validation rows; ",
        "the rest have fewer (shorter history) and their R^2/Adjusted R^2 ",
        "should be read with more caution (computed on fewer points).")

gbrt_ticker_r2_path <- file.path(PROC_DIR, "gbrt_ticker_r2.csv")
write_csv(gbrt_ticker_r2, gbrt_ticker_r2_path)
message("Saved ", nrow(gbrt_ticker_r2), " per-ticker walk-forward R^2 rows -> ", gbrt_ticker_r2_path)


# 8. Supplementary 4-quarter-ahead GBRT model (exret_next4). Fit purely to
#    power scripts/05_top30_forecast.R's "next year" forecast with a
#    genuinely re-estimated model rather than compounding the 1Q result;
#    does not change GBRT-on-exret_next's status as the primary model.

TARGET_VAR_4Q <- "exret_next4"

message("\n--- Supplementary 4Q GBRT model (", TARGET_VAR_4Q, ") ---")
n_no_target_4q <- sum(is.na(panel_raw[[TARGET_VAR_4Q]]))
message("Dropping ", n_no_target_4q, " row(s) with no known ", TARGET_VAR_4Q,
        " (need 4 fully-elapsed future quarters).")

panel_4q <- panel_raw %>% filter(!is.na(.data[[TARGET_VAR_4Q]]))

is_complete_4q <- panel_4q %>% select(all_of(PREDICTORS)) %>% complete.cases()
panel_model_4q <- panel_4q[is_complete_4q, ]

if (nrow(panel_model_4q) == 0) {
  stop("4Q complete-case filtering left zero rows.")
}

full_df_4q <- panel_model_4q %>% arrange(ticker, quarter_end)

message("4Q model rows: ", nrow(full_df_4q), " (date range ",
        min(full_df_4q$quarter_end), " to ", max(full_df_4q$quarter_end),
        " -- shorter than the 1Q model's range because the most recent ~4 ",
        "quarters have no realized 4-quarter-ahead target yet).")

x_full_4q <- as.matrix(full_df_4q[, PREDICTORS])
y_full_4q <- full_df_4q[[TARGET_VAR_4Q]]

ols_df_4q <- as.data.frame(x_full_4q)
ols_df_4q$target <- y_full_4q

wf_folds_4q <- build_walkforward_folds(full_df_4q$quarter_end, WF_WARMUP_QUARTERS, WF_FOLD_QUARTERS)

gbrt_wf_4q <- fit_gbrt_walkforward(full_df_4q, TARGET_VAR_4Q, PREDICTORS, wf_folds_4q,
                                    GBM_N_TREES, GBM_SHRINKAGE, GBM_INTERACTION_DEPTH,
                                    SEED, length(PREDICTORS))

gbrt_walkforward_4q_path <- file.path(PROC_DIR, "gbrt_walkforward_cv_4q.csv")
write_csv(gbrt_wf_4q$fold_summary, gbrt_walkforward_4q_path)
message("4Q GBRT walk-forward: n.trees (early stopping, ", gbrt_wf_4q$n_folds,
        " folds) = ", gbrt_wf_4q$best_iter, " of ", GBM_N_TREES)

best_iter_4q <- gbrt_wf_4q$best_iter

set.seed(SEED)
gbm_fit_4q <- gbm(
  target ~ ., data = ols_df_4q, distribution = "gaussian",
  n.trees = GBM_N_TREES, shrinkage = GBM_SHRINKAGE,
  interaction.depth = GBM_INTERACTION_DEPTH, cv.folds = 0, n.cores = 1
)

gbm_pred_full_4q <- as.numeric(predict(gbm_fit_4q, newdata = full_df_4q, n.trees = best_iter_4q))

message("4Q GBRT performance:")
gbm_metrics_4q <- data.frame(
  fit       = c("in_sample", "walkforward"),
  n         = c(nrow(full_df_4q), gbrt_wf_4q$wf_n),
  rmse      = c(rmse(y_full_4q, gbm_pred_full_4q), gbrt_wf_4q$wf_rmse),
  r_squared = c(r_squared(y_full_4q, gbm_pred_full_4q), gbrt_wf_4q$wf_r2)
)
gbm_metrics_4q$adjusted_r_squared <- adj_r_squared(gbm_metrics_4q$r_squared, gbm_metrics_4q$n, length(PREDICTORS))
print(gbm_metrics_4q)
message("Read this walk-forward R^2 with EXTRA caution vs. the 1Q model's -- ",
        "the usable quarter range here is shorter, so it is noisier by ",
        "construction, independent of whether the underlying 4-quarter ",
        "relationship is genuinely any stronger or weaker than the ",
        "1-quarter one.")

gbm_predictions_4q <- full_df_4q %>%
  transmute(ticker, quarter_end, actual = .data[[TARGET_VAR_4Q]], pred = gbm_pred_full_4q)

gbm_predictions_4q_path <- file.path(PROC_DIR, "gbrt_predictions_4q.csv")
write_csv(gbm_predictions_4q, gbm_predictions_4q_path)
message("Saved ", nrow(gbm_predictions_4q), " 4Q GBRT in-sample predictions -> ", gbm_predictions_4q_path)

gbrt_walkforward_predictions_4q <- data.frame(
  ticker = gbrt_wf_4q$wf_ticker,
  quarter_end = gbrt_wf_4q$wf_quarter_end,
  actual = gbrt_wf_4q$wf_actual,
  pred = gbrt_wf_4q$wf_pred
) %>% arrange(ticker, quarter_end)

gbrt_wf_predictions_4q_path <- file.path(PROC_DIR, "gbrt_walkforward_predictions_4q.csv")
write_csv(gbrt_walkforward_predictions_4q, gbrt_wf_predictions_4q_path)
message("Saved ", nrow(gbrt_walkforward_predictions_4q), " 4Q pooled out-of-fold predictions -> ",
        gbrt_wf_predictions_4q_path)


# 9. Combined model-metrics summary -- every in-sample/walk-forward metric
#    table above, in one file, for the README/report to cite directly

model_metrics_summary <- bind_rows(
  rf_metrics %>% mutate(model = "random_forest", target = "exret_next (1Q)",
                         variant = NA_character_, hyperparam = paste0("mtry=", RF_MTRY)),
  pls_metrics %>% mutate(model = "pls", target = "exret_next (1Q)", variant = ncomp,
                          hyperparam = ifelse(ncomp == "ncomp_min", paste0("ncomp=", ncomp_min), paste0("ncomp=", ncomp_1se))) %>%
    select(-ncomp),
  gbm_metrics %>% mutate(model = "gbrt", target = "exret_next (1Q)",
                          variant = NA_character_, hyperparam = paste0("n.trees=", best_iter)),
  gbm_metrics_4q %>% mutate(model = "gbrt", target = "exret_next4 (4Q, supplementary)",
                             variant = NA_character_, hyperparam = paste0("n.trees=", best_iter_4q))
) %>%
  select(model, target, fit, variant, hyperparam, n, rmse, r_squared, adjusted_r_squared)

model_metrics_summary_path <- file.path(PROC_DIR, "model_metrics_summary.csv")
write_csv(model_metrics_summary, model_metrics_summary_path)
message("\nSaved combined model-comparison summary (RF/PLS/GBRT, in-sample + walk-forward) -> ",
        model_metrics_summary_path)


# 10. Forward-score the current not-yet-realized quarter (both horizons) using
#     the entire-panel-trained GBRT fits, and record the SPY forward-return
#     assumption used to turn those excess-return predictions into absolute
#     price/dollar projections. Powers scripts/05_top30_forecast.R only --
#     this is never a historical-quarter prediction.

is_complete_forward <- forward_raw %>% select(all_of(PREDICTORS)) %>% complete.cases()
excluded_forward <- forward_raw$ticker[!is_complete_forward]
if (length(excluded_forward) > 0) {
  message("\nExcluding ", length(excluded_forward), " ticker(s) from forward ",
          "scoring for missing predictor(s) at the current (",
          unique(forward_raw$quarter_end), ") row: ",
          paste(excluded_forward, collapse = ", "))
}
forward_scorable <- forward_raw[is_complete_forward, ]

forward_pred_1q <- as.numeric(predict(gbm_fit,    newdata = forward_scorable, n.trees = best_iter))
forward_pred_4q <- as.numeric(predict(gbm_fit_4q, newdata = forward_scorable, n.trees = best_iter_4q))

gbrt_forward_predictions <- forward_scorable %>%
  transmute(ticker, quarter_end, close, adjusted,
            pred_exret_next_1q = forward_pred_1q,
            pred_exret_next_4q = forward_pred_4q)

gbrt_forward_path <- file.path(PROC_DIR, "gbrt_forward_predictions.csv")
write_csv(gbrt_forward_predictions, gbrt_forward_path)
message("\nSaved ", nrow(gbrt_forward_predictions), " forward (Q3 2026 / Q2 ",
        "2027) GBRT predictions -> ", gbrt_forward_path)


wf_quarters_1q <- unique(gbrt_wf$wf_quarter_end)
wf_quarters_4q <- unique(gbrt_wf_4q$wf_quarter_end)

spy_1q_assumed <- full_df %>%
  filter(quarter_end %in% wf_quarters_1q) %>%
  distinct(quarter_end, bench_ret_next) %>%
  pull(bench_ret_next) %>%
  mean(na.rm = TRUE)
spy_4q_assumed <- full_df_4q %>%
  filter(quarter_end %in% wf_quarters_4q) %>%
  distinct(quarter_end, bench_ret_next4) %>%
  pull(bench_ret_next4) %>%
  mean(na.rm = TRUE)

spy_forward_assumption <- data.frame(
  horizon = c("1q", "4q"),
  realized_mean_ret = c(spy_1q_assumed, spy_4q_assumed),
  n_quarters_basis = c(length(wf_quarters_1q), length(wf_quarters_4q))
)
spy_forward_path <- file.path(PROC_DIR, "spy_forward_assumption.csv")
write_csv(spy_forward_assumption, spy_forward_path)
message("Saved SPY forward-return assumption (1Q = ", signif(spy_1q_assumed, 4),
        ", 4Q = ", signif(spy_4q_assumed, 4), ") -> ", spy_forward_path)

message("\n03_models.R complete.")
