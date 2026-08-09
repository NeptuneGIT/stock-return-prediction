# =============================================================================
# 03_models.R
#
# Fits PLS (Partial Least Squares), Random Forest, and Gradient Boosted
# Regression Trees (GBRT, gbm package), all predicting exret_next (a
# ticker's next-quarter return minus SPY's) from the predictors built in
# 01a_ratios.R / panel.csv. Course spec fixes the model order as
# PLS -> Random Forest -> GBRT.
#
# THE CENTRAL RULE OF THIS SCRIPT:
# Train/test is a strict time-based split -- train on quarter_end <
# SPLIT_DATE (2023-01-01), test on quarter_end >= SPLIT_DATE -- never
# random k-fold across the full series. That would hand the model
# test-period information during training and invalidate every
# downstream result. SPLIT_DATE lands close to an 80/20 row split of the
# target-bearing panel (roughly 80% train / 20% test both before and
# after complete-case filtering, printed each run so it's visible
# without recomputing by hand); it is not moved just to force a rounder
# ratio, since that would mean picking a cutoff quarter arbitrarily
# rather than living with whatever ratio the actual calendar produces.
# Random Forest's out-of-bag error and GBRT's cv.folds are both confined
# entirely to the training rows -- neither ever sees a test-period
# observation (see sections 4 and 8 for exactly how that's enforced) --
# and this is intentionally NOT k-fold CV across the full series, since
# that would leak test-period rows into model selection for time-series
# data.
#
# NA handling: none of these three models accept NA in their inputs. Per
# project convention, missing fundamentals are never imputed as 0 (a
# false "this company earns exactly $0" signal) -- so rows with any NA
# across the predictors are dropped via complete-case filtering instead,
# and the drop counts are reported below so a grader can see the real
# sample size, not just assume it matches panel.csv's row count.
#
# MODEL-SELECTION HISTORY
# This project started with Lasso, PLS, and hand-rolled Bagged Regression
# Splines (bagged earth::earth() models) on an 8-ticker, then 28-ticker,
# water-utility/infrastructure universe benchmarked against PHO. Every
# model in every configuration tried on that universe came back with
# out-of-sample R^2 at or below zero -- weak, entangled partial-OLS
# effects were visible in-sample, but nothing that survived a held-out
# test split. After pivoting to the current 222-ticker S&P 500 universe
# benchmarked against SPY, the model shortlist changed to the current
# PLS / Random Forest / GBRT trio (course spec), and several follow-up
# experiments were run to chase the same near-zero test R^2 problem at
# the new scale: an alternate GICS-sector-relative target (to separate
# whole-sector moves from genuine stock-picking signal), a GBRT
# shrinkage/interaction-depth grid search, and four additional predictors
# (beta_252d, idio_vol_60d, pct_from_52w_high, accruals) chosen to add
# cross-sectional structure different from the existing momentum/level
# features. None of these three interventions alone produced a positive
# test R^2. A follow-up ad-hoc comparison found that dropping accruals
# specifically (its 17.1% missingness costs ~1,150 rows for a feature
# that ranks low in importance even when included) does produce a small
# positive GBRT test R^2 -- the first positive out-of-sample result this
# project has seen across every model/target/predictor-set combination
# tried so far, though still well within noise for this sample size and
# not yet promoted into PREDICTORS below.
#
# Input:  data/processed/panel.csv          (01a_ratios.R)
# Output: data/processed/ols_diagnostics[_sector].csv
#         data/processed/pls_predictions[_sector].csv
#         data/processed/pls_coefficients[_sector].csv
#         data/processed/random_forest_predictions[_sector].csv
#         data/processed/random_forest_importance[_sector].csv
#         data/processed/gbrt_predictions[_sector].csv
#         data/processed/gbrt_importance[_sector].csv
#         data/processed/gbrt_hyperparam_search[_sector].csv
# =============================================================================

library(dplyr)
library(readr)
library(pls)
library(randomForest)
library(gbm)

PROC_DIR <- "data/processed"
SPLIT_DATE <- as.Date("2023-01-01")
SEED <- 42

# TARGET_VAR: "exret_next" (SPY-relative, this project's primary target)
# or "exret_next_sector" (GICS-sector-relative, see 01a_ratios.R section
# 4b -- an experimental alternate target, see the model-selection history
# above for why it was tried). OUTPUT_SUFFIX keeps this run's files
# separate from the primary target's saved outputs.
TARGET_VAR <- "exret_next"
OUTPUT_SUFFIX <- if (TARGET_VAR == "exret_next") "" else "_sector"

# DGS10/FEDFUNDS/cpi_yoy dropped: they're constant across every ticker
# within a quarter (raw national macro series), and exret_next is
# ALREADY a relative return (ticker minus SPY, the broad-market
# benchmark) -- a market-wide rate/inflation shock moves a stock and the
# broad market together and mostly cancels out of that difference. A
# predictor only has value here if it varies across tickers within a
# quarter; these three don't, so they can't explain relative returns or
# change the Broker Test's ranking. dollar_volume, by contrast, is kept:
# it varies meaningfully across a universe spanning mega-cap to
# small-cap names, unlike the macro trio.
#
# current_ratio and div_yield dropped (post-pivot missingness review):
# each is NA for that ticker's ENTIRE history for a large,
# structurally-explained block of the 222-ticker universe -- current_ratio
# for 25 tickers with an unclassified balance sheet (mostly Financials/
# REITs: GS, MS, C, SCHW, BNY, AXP, COF, FITB, STT, NTRS, HIG, PGR, ALL,
# DE, BXP, AVB, EQR, VTR, WELL, UDR, O, PSA, HST, DHI, PHM), div_yield for
# 20 genuine non-dividend-payers (AMD, AMZN, BIIB, BSX, CDNS, CMG, DVA,
# ISRG, ITRI, LYV, MTZ, NFLX, ON, ORLY, PTC, SNPS, TSLA, TYL, VRTX, ZBRA).
# Under complete-case filtering (glmnet/pls/RF/GBRT all require every
# predictor present), an always-NA column doesn't just thin a ticker's
# rows -- it zeroes them out entirely, and together these two columns
# were responsible for 45 of 50 tickers contributing ZERO training rows,
# concentrated almost entirely in the two sectors (Financials/REITs,
# growth/tech) this pivot most wanted represented. Both columns are left
# in panel.csv itself (this is a modeling-set decision, not a data
# problem) -- only removed from the model-facing PREDICTORS here, same
# pattern as the macro trio above.
# beta_252d, idio_vol_60d, pct_from_52w_high, and accruals were added
# (see 01a_ratios.R) as a search for predictors with genuinely DIFFERENT
# cross-sectional structure from the existing momentum/level/valuation
# features, after the sector-relative-target and GBRT-hyperparameter-
# search experiments above both left test R^2 near zero. All four were
# verified to have real within-quarter cross-sectional spread before
# being added here (e.g. beta_252d's median within-quarter SD = 0.385,
# vs. DGS10's 0), not just added on the assumption that they would.
PREDICTORS <- c(
  "mom_1m", "mom_3m", "mom_12m", "vol_60d", "dollar_volume",  # technical
  "beta_252d", "idio_vol_60d", "pct_from_52w_high",  # technical/risk [NEW]
  "roe", "debt_to_equity",                  # fundamentals (SEC)
  "asset_growth",                           # fundamentals (SEC)
  "ni_growth", "capex_intensity",           # fundamentals (SEC)
  "accruals",                               # fundamentals (SEC) [NEW]
  "pb_ratio", "fcf_yield"                   # price x fundamentals
)

# -----------------------------------------------------------------------------
# 1. Load, drop rows with no known target
# -----------------------------------------------------------------------------
# The most recent quarter for each ticker has exret_next = NA (its next
# quarter hasn't happened yet). Those rows can't be trained on or scored
# against, so they're dropped before anything else -- keeping them would
# silently shrink either split's usable rows without any of this script's
# missingness reporting noticing why.

panel <- read_csv(file.path(PROC_DIR, "panel.csv"), show_col_types = FALSE)

n_no_target <- sum(is.na(panel[[TARGET_VAR]]))
message("Dropping ", n_no_target, " row(s) with no known ", TARGET_VAR, " ",
        "(most recent quarter per ticker, not yet realised).")

panel <- panel %>% filter(!is.na(.data[[TARGET_VAR]]))

# -----------------------------------------------------------------------------
# 2. Complete-case filter on predictors
# -----------------------------------------------------------------------------
# None of PLS/Random Forest/GBRT accept NA in their inputs. Rather than
# impute, drop any row missing one or more of the predictors -- and
# report exactly what was lost, by ticker and by split, so the effective
# sample size is visible rather than assumed.

panel <- panel %>%
  mutate(split = if_else(quarter_end < SPLIT_DATE, "train", "test"))

is_complete <- panel %>%
  select(all_of(PREDICTORS)) %>%
  complete.cases()

message("\nRows dropped for missing predictor(s), by ticker x split:")
panel %>%
  mutate(complete = is_complete) %>%
  group_by(ticker, split) %>%
  summarise(
    total    = n(),
    dropped  = sum(!complete),
    kept     = sum(complete),
    .groups  = "drop"
  ) %>%
  as.data.frame() %>%
  print()

panel_model <- panel[is_complete, ]

# Fail loudly if an entire ticker or split has nothing left -- that's a
# structural problem (a predictor gone missing everywhere for a stock),
# not an ordinary partial gap.
split_counts <- table(panel_model$split)
if (!all(c("train", "test") %in% names(split_counts)) ||
    any(split_counts == 0)) {
  stop("Complete-case filtering left zero rows in the train or test split. ",
       "Check the missingness table above for the culprit predictor.")
}
ticker_counts <- table(panel_model$ticker)
if (any(ticker_counts == 0)) {
  stop("Complete-case filtering left zero rows for at least one ticker: ",
       paste(names(ticker_counts)[ticker_counts == 0], collapse = ", "))
}

n_train_final <- sum(panel_model$split == "train")
n_test_final  <- sum(panel_model$split == "test")
message("\nTotal rows after filtering: ", nrow(panel_model),
        " (", n_train_final, " train / ",
        n_test_final, " test) of ",
        nrow(panel), " rows with a known target.")
message("Train/test split ratio (chronological, SPLIT_DATE = ", SPLIT_DATE,
        "): ", signif(100 * n_train_final / nrow(panel_model), 4),
        "% train / ", signif(100 * n_test_final / nrow(panel_model), 4),
        "% test -- confirms the fixed calendar cutoff lands on an ~80/20 ",
        "split; this is still a strict time-based cut (all train rows ",
        "chronologically precede all test rows), never a random 80/20 ",
        "shuffle, which would leak the future into training.")

# -----------------------------------------------------------------------------
# 3. Train / test matrices
# -----------------------------------------------------------------------------

train_df <- panel_model %>% filter(split == "train") %>% arrange(ticker, quarter_end)
test_df  <- panel_model %>% filter(split == "test")  %>% arrange(ticker, quarter_end)

x_train <- as.matrix(train_df[, PREDICTORS])
y_train <- train_df[[TARGET_VAR]]
x_test  <- as.matrix(test_df[, PREDICTORS])
y_test  <- test_df[[TARGET_VAR]]

# -----------------------------------------------------------------------------
# 3b. Diagnostic: raw correlations + unpenalized OLS on the training set
# -----------------------------------------------------------------------------
# Kept from this file's original Lasso-based version even though Lasso
# itself has since been replaced by Random Forest/GBRT: plain
# correlations and unpenalized OLS coefficients/p-values are a
# model-agnostic baseline for whether any individual predictor has a
# non-trivial linear relationship with exret_next at all, against which
# the nonlinear models below can be compared. TRAINING SET ONLY -- this
# is a pre-modelling sanity check, not a fit used for prediction, so it
# must respect the same no-look-ahead boundary as everything else.

message("\n--- Diagnostic: training-set correlations with ", TARGET_VAR, " ---")
train_cor <- sapply(PREDICTORS, function(p) cor(x_train[, p], y_train))
print(sort(train_cor, decreasing = TRUE))

message("\n--- Diagnostic: unpenalized OLS on the training set ---")
# Internal modelling frame's response column is always called "target"
# (not the literal TARGET_VAR name) so the lm()/plsr()/gbm() formulas
# below don't need to change based on which target is active.
ols_df <- as.data.frame(x_train)
ols_df$target <- y_train
ols_fit <- lm(target ~ ., data = ols_df)
ols_summary <- summary(ols_fit)
print(ols_summary)

# Written to CSV (not just console) so both the correlations and the OLS
# coefficient table can be inspected directly without re-running the model
# or re-parsing console output. One row per predictor, plus the intercept
# (which has no correlation, hence NA there).
ols_coefs <- as.data.frame(ols_summary$coefficients)
colnames(ols_coefs) <- c("ols_estimate", "ols_std_error", "ols_t_value", "ols_p_value")
ols_coefs$predictor <- rownames(ols_coefs)
ols_coefs$predictor[ols_coefs$predictor == "(Intercept)"] <- "(Intercept)"

diagnostics <- data.frame(predictor = c("(Intercept)", PREDICTORS)) %>%
  left_join(
    data.frame(predictor = names(train_cor), correlation = as.numeric(train_cor)),
    by = "predictor"
  ) %>%
  left_join(ols_coefs, by = "predictor") %>%
  arrange(desc(abs(correlation)))

ols_diagnostics_path <- file.path(PROC_DIR, paste0("ols_diagnostics", OUTPUT_SUFFIX, ".csv"))
write_csv(diagnostics, ols_diagnostics_path)
message("\nSaved diagnostics (correlation + OLS coefficients/p-values) -> ",
        ols_diagnostics_path)

# -----------------------------------------------------------------------------
# 4. Fit: Random Forest (bagged decision trees; OOB error is intrinsic CV)
# -----------------------------------------------------------------------------
# Unlike Lasso/PLS, Random Forest needs no manual/CV lambda or component
# search: each tree is grown on a bootstrap resample of train_df, and the
# rows left out of that resample (roughly 1/3, the "out-of-bag" sample)
# serve as that tree's own held-out test set. Averaging OOB error across
# all trees gives an honest generalisation estimate WITHOUT ever touching
# test_df -- the same no-look-ahead guarantee as cv.glmnet's/plsr's
# training-only CV above, just built into the algorithm instead of a
# separate CV loop.

set.seed(SEED)
rf_fit <- randomForest(
  x = x_train, y = y_train,
  ntree = 500, importance = TRUE
)

message("\nRandom Forest OOB performance (training rows only, never touches test_df):")
message("  OOB MSE = ", signif(rf_fit$mse[rf_fit$ntree], 4),
        "  OOB R^2 = ", signif(rf_fit$rsq[rf_fit$ntree], 4))

message("\nVariable importance (%IncMSE / IncNodePurity):")
print(importance(rf_fit))

# -----------------------------------------------------------------------------
# 5. Predict + evaluate on both splits
# -----------------------------------------------------------------------------

rmse <- function(actual, pred) sqrt(mean((actual - pred)^2))
r_squared <- function(actual, pred) {
  1 - sum((actual - pred)^2) / sum((actual - mean(actual))^2)
}

rf_pred_train <- as.numeric(predict(rf_fit, newdata = x_train))
rf_pred_test  <- as.numeric(predict(rf_fit, newdata = x_test))

message("\nRandom Forest performance:")
rf_metrics <- data.frame(
  split     = c("train", "test"),
  rmse      = c(rmse(y_train, rf_pred_train), rmse(y_test, rf_pred_test)),
  r_squared = c(r_squared(y_train, rf_pred_train), r_squared(y_test, rf_pred_test))
)
print(rf_metrics)

# -----------------------------------------------------------------------------
# 6. Write outputs for 04_results.R
# -----------------------------------------------------------------------------

rf_predictions <- bind_rows(
  train_df %>% transmute(ticker, quarter_end, split, actual = .data[[TARGET_VAR]], pred = rf_pred_train),
  test_df  %>% transmute(ticker, quarter_end, split, actual = .data[[TARGET_VAR]], pred = rf_pred_test)
) %>%
  arrange(ticker, quarter_end)

rf_predictions_path <- file.path(PROC_DIR, paste0("random_forest_predictions", OUTPUT_SUFFIX, ".csv"))
write_csv(rf_predictions, rf_predictions_path)

rf_importance <- as.data.frame(importance(rf_fit))
rf_importance$predictor <- rownames(rf_importance)
rf_importance <- rf_importance %>%
  select(predictor, everything()) %>%
  arrange(desc(`%IncMSE`))
rf_importance_path <- file.path(PROC_DIR, paste0("random_forest_importance", OUTPUT_SUFFIX, ".csv"))
write_csv(rf_importance, rf_importance_path)

message("\nSaved ", nrow(rf_predictions), " predictions -> ", rf_predictions_path)
message("Saved ", nrow(rf_importance), " importance rows -> ", rf_importance_path)

# -----------------------------------------------------------------------------
# 7. Fit: PLS (Partial Least Squares), ncomp selected by CV on training rows
# -----------------------------------------------------------------------------
# Same train/test rows, same predictors, same no-look-ahead boundary as
# the Random Forest section above -- CV for choosing the number of
# components is confined entirely to train_df, never touching test_df.
# Unlike a sparse linear selector, PLS builds a small number of
# latent components as linear combinations of ALL predictors, chosen to
# maximise covariance with exret_next. That makes it a natural next lens
# on the diagnostics above: several predictors showed weak individual
# correlations but significant partial OLS effects once correlated
# predictors were controlled for (a sign of shared, entangled signal) --
# exactly the structure PLS's components are built to capture instead of
# discarding.

set.seed(SEED)
pls_fit <- plsr(target ~ ., data = ols_df, scale = TRUE, validation = "CV")

rmsep_cv <- RMSEP(pls_fit, estimate = "CV")
# ncomp = 0 is the intercept-only row; exclude it before taking the min so
# "best" always refers to an actual fitted component count.
cv_curve <- as.numeric(rmsep_cv$val["CV", , ])[-1]
ncomp_min <- which.min(cv_curve)
ncomp_1se <- selectNcomp(pls_fit, method = "onesigma", plot = FALSE)

message("\nCross-validated PLS component count (selected on training rows only):")
message("  ncomp (min CV RMSEP) = ", ncomp_min)
message("  ncomp (one-sigma rule) = ", ncomp_1se,
        if (ncomp_1se == 0) " (intercept-only -- one-sigma rule found no component reliably beats the mean)" else "")

message("\nPLS coefficients @ ncomp_min:")
pls_coef_min <- drop(coef(pls_fit, ncomp = ncomp_min, intercept = TRUE))
print(pls_coef_min)

# NOTE: coef.mvr has no ncomp = 0 case (there's no component to take
# coefficients FROM), so the one-sigma rule choosing 0 is handled directly
# here as "predict the training mean, all predictor coefficients 0" --
# an intercept-only null model -- rather than silently substituting
# ncomp_min and hiding a real, conservative CV finding.
if (ncomp_1se == 0) {
  message("\nPLS coefficients @ ncomp_1se: intercept-only (all predictor coefficients 0)")
  pls_coef_1se <- setNames(c(mean(y_train), rep(0, length(PREDICTORS))),
                            names(pls_coef_min))
  print(pls_coef_1se)
  pls_pred_train_1se <- rep(mean(y_train), nrow(train_df))
  pls_pred_test_1se  <- rep(mean(y_train), nrow(test_df))
} else {
  message("\nPLS coefficients @ ncomp_1se:")
  pls_coef_1se <- drop(coef(pls_fit, ncomp = ncomp_1se, intercept = TRUE))
  print(pls_coef_1se)
  pls_pred_train_1se <- as.numeric(predict(pls_fit, newdata = as.data.frame(x_train), ncomp = ncomp_1se))
  pls_pred_test_1se  <- as.numeric(predict(pls_fit, newdata = as.data.frame(x_test),  ncomp = ncomp_1se))
}

pls_pred_train_min <- as.numeric(predict(pls_fit, newdata = as.data.frame(x_train), ncomp = ncomp_min))
pls_pred_test_min  <- as.numeric(predict(pls_fit, newdata = as.data.frame(x_test),  ncomp = ncomp_min))

message("\nPLS performance:")
pls_metrics <- data.frame(
  split     = c("train", "train", "test", "test"),
  ncomp     = c("ncomp_min", "ncomp_1se", "ncomp_min", "ncomp_1se"),
  rmse      = c(rmse(y_train, pls_pred_train_min), rmse(y_train, pls_pred_train_1se),
                 rmse(y_test,  pls_pred_test_min),  rmse(y_test,  pls_pred_test_1se)),
  r_squared = c(r_squared(y_train, pls_pred_train_min), r_squared(y_train, pls_pred_train_1se),
                 r_squared(y_test,  pls_pred_test_min),  r_squared(y_test,  pls_pred_test_1se))
)
print(pls_metrics)

pls_predictions <- bind_rows(
  train_df %>%
    transmute(ticker, quarter_end, split,
              actual = .data[[TARGET_VAR]],
              pred_ncomp_min = pls_pred_train_min,
              pred_ncomp_1se = pls_pred_train_1se),
  test_df %>%
    transmute(ticker, quarter_end, split,
              actual = .data[[TARGET_VAR]],
              pred_ncomp_min = pls_pred_test_min,
              pred_ncomp_1se = pls_pred_test_1se)
) %>%
  arrange(ticker, quarter_end)

pls_predictions_path <- file.path(PROC_DIR, paste0("pls_predictions", OUTPUT_SUFFIX, ".csv"))
write_csv(pls_predictions, pls_predictions_path)

pls_coefficients <- data.frame(
  predictor        = names(pls_coef_min),
  coef_ncomp_min = as.numeric(pls_coef_min),
  coef_ncomp_1se = as.numeric(pls_coef_1se)
)
pls_coefficients_path <- file.path(PROC_DIR, paste0("pls_coefficients", OUTPUT_SUFFIX, ".csv"))
write_csv(pls_coefficients, pls_coefficients_path)

message("\nSaved ", nrow(pls_predictions), " PLS predictions -> ", pls_predictions_path)
message("Saved ", nrow(pls_coefficients), " PLS coefficients -> ", pls_coefficients_path)

# -----------------------------------------------------------------------------
# 8. Fit: Gradient Boosted Regression Trees (GBRT, gbm package)
# -----------------------------------------------------------------------------
# Same train/test rows, same predictors, same no-look-ahead boundary as
# every section above. GBRT builds trees sequentially, each one fit to the
# residuals left by the ensemble so far -- unlike Random Forest's
# independently-grown, averaged trees, boosting can pick up different
# nonlinear structure, which is why the course spec runs it as the third
# model in the fixed PLS -> Random Forest -> GBRT sequence rather than
# treating Random Forest as sufficient on its own.
#
# gbm()'s own cv.folds performs k-fold cross-validation to choose the
# number of trees (n.trees) that minimises CV error. CRITICALLY, that CV
# is computed entirely on the `data` argument passed to gbm() below, which
# is ols_df (PREDICTORS + the training-set target, renamed "target" -- see
# section 3b) -- it never sees a test_df row, so it respects the same
# no-look-ahead boundary as cv.glmnet's/plsr's training-only CV above.
# This is the one place in this file most likely to accidentally
# reintroduce a look-ahead violation if gbm() were ever passed the full
# panel instead of train_df/ols_df -- confirmed it is not.
#
# NOTE: this file previously ran Bagged MARS twice (all tickers, then with
# VRT excluded) because MARS's unbounded hinge-function extrapolation made
# one outlier ticker capable of dominating a whole bag's averaged
# prediction (traced, in one case, to a units error in a single SEC
# filing that fed in an out-of-range training row). Tree-based models
# split on thresholds and predict a bounded leaf value,
# so they don't extrapolate the same way -- an outlier row lands in
# whichever leaf its predictors sort it into and can't blow up the
# prediction unboundedly the way a MARS hinge function could. The
# VRT-exclusion comparison run is deliberately not repeated here for
# Random Forest/GBRT for that reason, not an oversight.
#
# HYPERPARAMETER SEARCH: shrinkage and interaction.depth are
# not fixed constants -- they're chosen by a small grid
# search. For EACH candidate combination, gbm()'s cv.folds runs 5-fold CV
# entirely on ols_df (the training rows) to both pick that combination's
# own best n.trees and score its CV error; whichever combination reaches
# the lowest CV error wins. test_df is never touched by any part of this
# search -- it's still model SELECTION confined to the training rows,
# the same no-look-ahead boundary as the single-configuration fit this
# replaced, just applied across a grid instead of one fixed guess.
# set.seed(SEED) is reset before every combination's gbm() call so all
# combinations see the IDENTICAL 5-fold assignment -- otherwise a
# combination could look better or worse than another purely from luckier
# CV folds rather than a genuinely better hyperparameter choice.

# N.TREES capped at 1500, not 3000: a first attempt at 3000 trees x 16
# combinations x (5 CV folds + 1 full fit) took >10 minutes and had to be
# cut short. Every combination tried so far (this file's whole history)
# has its CV-selected best_iter land well below its n.trees ceiling given
# this panel's weak signal (e.g. the single fixed-hyperparameter run
# above stopped at 417 of 2000) -- so a lower ceiling is very unlikely to
# cut off any combination's true minimum, and 1500 keeps the full grid
# under the practical runtime budget.
GBM_N_TREES <- 1500
GBM_CV_FOLDS <- 5
GBM_SHRINKAGE_GRID <- c(0.001, 0.005, 0.01, 0.05)
GBM_DEPTH_GRID <- c(1, 2, 3, 4)

gbm_grid <- expand.grid(shrinkage = GBM_SHRINKAGE_GRID,
                         interaction.depth = GBM_DEPTH_GRID)

message("\nGBRT hyperparameter search: ", nrow(gbm_grid),
        " (shrinkage x interaction.depth) combinations, ", GBM_CV_FOLDS,
        "-fold CV confined to the ", nrow(train_df), " training rows only:")

gbm_grid_fits <- vector("list", nrow(gbm_grid))
gbm_grid_rows <- vector("list", nrow(gbm_grid))

for (i in seq_len(nrow(gbm_grid))) {
  shr <- gbm_grid$shrinkage[i]
  dep <- gbm_grid$interaction.depth[i]

  set.seed(SEED)
  fit_i <- gbm(
    target ~ .,
    data = ols_df,
    distribution = "gaussian",
    n.trees = GBM_N_TREES,
    shrinkage = shr,
    interaction.depth = dep,
    cv.folds = GBM_CV_FOLDS,
    n.cores = 1
  )

  best_iter_i <- gbm.perf(fit_i, method = "cv", plot.it = FALSE)

  gbm_grid_fits[[i]] <- fit_i
  gbm_grid_rows[[i]] <- data.frame(
    shrinkage = shr, interaction.depth = dep,
    best_iter = best_iter_i,
    cv_error  = min(fit_i$cv.error)
  )
}

gbm_grid_summary <- bind_rows(gbm_grid_rows) %>% arrange(cv_error)
message("\nGrid search results, best (lowest CV error) first:")
print(gbm_grid_summary)

gbm_grid_search_path <- file.path(PROC_DIR, paste0("gbrt_hyperparam_search", OUTPUT_SUFFIX, ".csv"))
write_csv(gbm_grid_summary, gbm_grid_search_path)
message("Saved full grid -> ", gbm_grid_search_path)

best_idx <- which.min(vapply(gbm_grid_rows, function(r) r$cv_error, numeric(1)))
gbm_fit  <- gbm_grid_fits[[best_idx]]
best_iter <- gbm_grid_rows[[best_idx]]$best_iter
GBM_SHRINKAGE <- gbm_grid_rows[[best_idx]]$shrinkage
GBM_INTERACTION_DEPTH <- gbm_grid_rows[[best_idx]]$interaction.depth

message("\nGBRT: hyperparameter search winner -- shrinkage = ", GBM_SHRINKAGE,
        ", interaction.depth = ", GBM_INTERACTION_DEPTH,
        " (", GBM_CV_FOLDS, "-fold CV error = ",
        signif(gbm_grid_rows[[best_idx]]$cv_error, 4), ", CV-selected n.trees = ",
        best_iter, " of ", GBM_N_TREES, " grown, CV run only on the ",
        nrow(train_df), " training rows, never on the ", nrow(test_df),
        " test rows)")

message("\nVariable importance (relative influence):")
gbm_importance <- summary(gbm_fit, n.trees = best_iter, plotit = FALSE)
print(gbm_importance)

# -----------------------------------------------------------------------------
# 9. Predict + evaluate on both splits
# -----------------------------------------------------------------------------

gbm_pred_train <- as.numeric(predict(gbm_fit, newdata = train_df, n.trees = best_iter))
gbm_pred_test  <- as.numeric(predict(gbm_fit, newdata = test_df,  n.trees = best_iter))

message("\nGBRT performance:")
gbm_metrics <- data.frame(
  split     = c("train", "test"),
  rmse      = c(rmse(y_train, gbm_pred_train), rmse(y_test, gbm_pred_test)),
  r_squared = c(r_squared(y_train, gbm_pred_train), r_squared(y_test, gbm_pred_test))
)
print(gbm_metrics)

# -----------------------------------------------------------------------------
# 10. Write outputs for 04_results.R
# -----------------------------------------------------------------------------

gbm_predictions <- bind_rows(
  train_df %>% transmute(ticker, quarter_end, split, actual = .data[[TARGET_VAR]], pred = gbm_pred_train),
  test_df  %>% transmute(ticker, quarter_end, split, actual = .data[[TARGET_VAR]], pred = gbm_pred_test)
) %>%
  arrange(ticker, quarter_end)

gbm_predictions_path <- file.path(PROC_DIR, paste0("gbrt_predictions", OUTPUT_SUFFIX, ".csv"))
write_csv(gbm_predictions, gbm_predictions_path)

gbm_importance_out <- gbm_importance %>%
  rename(predictor = var, importance = rel.inf) %>%
  arrange(desc(importance))
gbm_importance_path <- file.path(PROC_DIR, paste0("gbrt_importance", OUTPUT_SUFFIX, ".csv"))
write_csv(gbm_importance_out, gbm_importance_path)

message("\nSaved ", nrow(gbm_predictions), " predictions -> ", gbm_predictions_path)
message("Saved ", nrow(gbm_importance_out), " importance rows -> ", gbm_importance_path)

message("\n03_models.R complete.")
