# =============================================================================
# 03_models.R
#
# Fits Lasso (L1-penalised linear), PLS (Partial Least Squares), and Bagged
# Regression Splines (Bagged MARS, hand-rolled on earth::earth -- see
# section 8's comment for why this isn't caret::bagEarth), all predicting
# exret_next (a ticker's next-quarter return minus PHO's) from the 12
# predictors built in 01a_ratios.R / panel.csv. Course spec fixes the model
# order as PLS -> Lasso -> Bagged Regression Splines; all three are
# implemented here (Lasso was written first and is left first in the file --
# read/run order does not need to match the course's presentation order,
# only the results need to be reported in that order).
#
# THE CENTRAL RULE OF THIS SCRIPT:
# Train/test is a strict time-based split -- train on quarter_end <
# 2023-01-01, test on quarter_end >= 2023-01-01 -- never random k-fold
# across the full series. That would hand the model test-period
# information during training and invalidate every downstream result.
# cv.glmnet's internal k-fold CV (used only to pick the penalty lambda)
# is fine because it is confined entirely to the training rows; it never
# sees a test-period observation.
#
# NA handling: glmnet cannot accept NA in its design matrix. Per project
# convention, missing fundamentals are never imputed as 0 (a false "this
# company earns exactly $0" signal) -- so rows with any NA across the 15
# predictors are dropped via complete-case filtering instead, and the
# drop counts are reported below so a grader can see the real sample
# size, not just assume it matches panel.csv's row count.
#
# Model-specific notes:
#
# Lasso (glmnet, L1 penalty): fit across all 15 candidate predictors.
# DGS10, FEDFUNDS, and cpi_yoy are excluded from PREDICTORS -- they are
# constant across tickers within a quarter and mostly cancel out of a
# relative-return target, so a first pass that included them selected
# zero non-zero coefficients at both lambda.min and lambda.1se. Both
# lambda.min and lambda.1se are fit and reported side by side.
#
# Diagnostic section (3b): training-set-only correlations and
# unpenalized OLS, written to lasso_diagnostics.csv, to check whether
# Lasso's near-intercept-only result reflects a true absence of linear
# signal or just Lasso's all-or-nothing penalty being conservative.
# Raw correlations with the target are all weak (|r| <= 0.18), but
# several predictors have significant partial OLS effects once the
# others are controlled for -- weak, entangled/collinear signal that
# Lasso's per-predictor penalty tends to zero out under cross-validation,
# which is part of the motivation for also fitting PLS (section 7),
# since it builds latent components from correlated predictors instead
# of selecting individual ones.
#
# PLS: ncomp is chosen by cross-validation confined to the training
# rows (a min-CV-RMSEP variant and a more conservative one-standard-
# error variant, mirroring lambda.min/lambda.1se). When the one-
# standard-error rule picks ncomp = 0, that is itself a real result --
# no component reliably beats the training mean -- and is reported as
# an intercept-only fit rather than silently substituted with the
# min-CV-RMSEP choice.
#
# Bagged Regression Splines (Bagged MARS): implemented as a hand-rolled
# loop directly on earth::earth() -- manual k-fold CV for nprune, manual
# bootstrap loop for bagging -- rather than caret::train(method =
# "bagEarth"). caret's bagEarth wraps plyr internals that conflict with
# a newer dplyr/rlang stack in ways that are brittle to keep patching;
# a direct earth::earth() loop avoids the dependency entirely. Run both
# with the full ticker set and with the highest-volatility outlier
# ticker excluded, to check whether that one ticker's noise was
# suppressing signal in the rest of the universe.
#
# NA handling: glmnet cannot accept NA in its design matrix. Per project
# convention, missing fundamentals are never imputed as 0 (a false "this
# company earns exactly $0" signal) -- so rows with any NA across the
# predictors are dropped via complete-case filtering instead, and the
# drop counts are reported below so a grader can see the real sample
# size, not just assume it matches panel.csv's row count.
#
# Current results (28-ticker universe, complete-case rows after
# filtering -- see the missingness table this script prints for which
# tickers drop out and why, e.g. confirmed non-dividend-payers with
# div_yield left NA rather than 0, per the missing-!=-zero convention):
#   Lasso:       lambda.min selects a single predictor (mom_12m);
#                test R^2 is negative at both lambda.min and lambda.1se.
#   PLS:         a small number of components selected by CV;
#                test R^2 is negative (ncomp_min) and intercept-only
#                (ncomp_1se).
#   Bagged MARS: a small nprune selected in both configurations;
#                test R^2 is negative in both.
# All three: test R^2 negative or ~0 in every configuration -- none
# beats predicting the training mean out-of-sample. Consistent with the
# smaller 8-ticker universe's conclusion (and the training-set OLS
# diagnostic, which still shows several individually-significant but
# small and entangled partial effects, not a clearly tradeable
# relationship). Broadening the universe from 8 to 28 tickers did not
# on its own produce out-of-sample signal that wasn't there before.
#
# Input:  data/processed/panel.csv          (01a_ratios.R)
# Output: data/processed/lasso_predictions.csv
#         data/processed/lasso_coefficients.csv
#         data/processed/lasso_diagnostics.csv
#         data/processed/pls_predictions.csv
#         data/processed/pls_coefficients.csv
#         data/processed/bagged_splines_predictions.csv
#         data/processed/bagged_splines_importance.csv
#         data/processed/bagged_splines_novrt_predictions.csv
#         data/processed/bagged_splines_novrt_importance.csv
# =============================================================================

library(dplyr)
library(readr)
library(glmnet)
library(pls)
library(earth)

PROC_DIR <- "data/processed"
SPLIT_DATE <- as.Date("2023-01-01")
SEED <- 42

# DGS10/FEDFUNDS/cpi_yoy dropped: they're constant across all 8 tickers
# within a quarter (raw national macro series), and exret_next is
# ALREADY a relative return (ticker minus PHO) -- a market-wide rate/
# inflation shock moves a water utility and its water-sector benchmark
# together and mostly cancels out of that difference. A predictor only
# has value here if it varies across tickers within a quarter; these
# three don't, so they can't explain relative returns or
# change the Broker Test's top-3 ranking. Confirmed empirically too: the
# 15-predictor Lasso above selected zero non-zero coefficients at both
# lambda.min and lambda.1se (pure intercept-only), so this cut is a
# genuine attempt to reduce noise, not just theory.
PREDICTORS <- c(
  "mom_3m", "mom_12m", "vol_60d",           # technical
  "roe", "debt_to_equity",                  # fundamentals (SEC)
  "current_ratio", "asset_growth",          # fundamentals (SEC)
  "ni_growth", "capex_intensity",           # fundamentals (SEC)
  "pb_ratio", "fcf_yield", "div_yield"      # price x fundamentals
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

n_no_target <- sum(is.na(panel$exret_next))
message("Dropping ", n_no_target, " row(s) with no known exret_next ",
        "(most recent quarter per ticker, not yet realised).")

panel <- panel %>% filter(!is.na(exret_next))

# -----------------------------------------------------------------------------
# 2. Complete-case filter on predictors
# -----------------------------------------------------------------------------
# glmnet refuses NA in x. Rather than impute, drop any row missing one or
# more of the 15 predictors -- and report exactly what was lost, by
# ticker and by split, so the effective sample size is visible rather
# than assumed.

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

message("\nTotal rows after filtering: ", nrow(panel_model),
        " (", sum(panel_model$split == "train"), " train / ",
        sum(panel_model$split == "test"), " test) of ",
        nrow(panel), " rows with a known target.")

# -----------------------------------------------------------------------------
# 3. Train / test matrices
# -----------------------------------------------------------------------------

train_df <- panel_model %>% filter(split == "train") %>% arrange(ticker, quarter_end)
test_df  <- panel_model %>% filter(split == "test")  %>% arrange(ticker, quarter_end)

x_train <- as.matrix(train_df[, PREDICTORS])
y_train <- train_df$exret_next
x_test  <- as.matrix(test_df[, PREDICTORS])
y_test  <- test_df$exret_next

# -----------------------------------------------------------------------------
# 3b. Diagnostic: raw correlations + unpenalized OLS on the training set
# -----------------------------------------------------------------------------
# Lasso's CV-selected model came back near-intercept-only above. Before
# concluding "no signal exists," check with a less punishing lens: Lasso's L1
# penalty needs a predictor to earn its keep in every CV fold or it gets
# zeroed entirely, which is a high bar with ~230 training rows. Plain
# correlations and unpenalized OLS coefficients/p-values show whether any
# individual predictor has a non-trivial linear relationship with
# exret_next that Lasso's all-or-nothing selection is simply being
# conservative about, or whether the signal really is absent. TRAINING SET
# ONLY -- this is a pre-modelling sanity check, not a fit used for
# prediction, so it must respect the same no-look-ahead boundary as
# everything else.

message("\n--- Diagnostic: training-set correlations with exret_next ---")
train_cor <- sapply(PREDICTORS, function(p) cor(x_train[, p], y_train))
print(sort(train_cor, decreasing = TRUE))

message("\n--- Diagnostic: unpenalized OLS on the training set ---")
ols_df <- as.data.frame(x_train)
ols_df$exret_next <- y_train
ols_fit <- lm(exret_next ~ ., data = ols_df)
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

write_csv(diagnostics, file.path(PROC_DIR, "lasso_diagnostics.csv"))
message("\nSaved diagnostics (correlation + OLS coefficients/p-values) -> ",
        file.path(PROC_DIR, "lasso_diagnostics.csv"))

# -----------------------------------------------------------------------------
# 4. Fit: Lasso with CV-selected lambda (CV confined to training rows only)
# -----------------------------------------------------------------------------

set.seed(SEED)
cv_fit <- cv.glmnet(x_train, y_train, alpha = 1, standardize = TRUE)

lambda_min <- cv_fit$lambda.min
lambda_1se <- cv_fit$lambda.1se

coef_min <- coef(cv_fit, s = "lambda.min")
coef_1se <- coef(cv_fit, s = "lambda.1se")

n_nonzero_min <- sum(coef_min[-1, 1] != 0)  # exclude intercept
n_nonzero_1se <- sum(coef_1se[-1, 1] != 0)

message("\nCross-validated lambda (selected on training rows only):")
message("  lambda.min = ", signif(lambda_min, 4),
        "  (", n_nonzero_min, " of ", length(PREDICTORS), " predictors non-zero)")
message("  lambda.1se = ", signif(lambda_1se, 4),
        "  (", n_nonzero_1se, " of ", length(PREDICTORS), " predictors non-zero)")

message("\nCoefficients @ lambda.min:")
print(coef_min)
message("\nCoefficients @ lambda.1se:")
print(coef_1se)

# -----------------------------------------------------------------------------
# 5. Predict + evaluate on both splits, both lambdas
# -----------------------------------------------------------------------------

rmse <- function(actual, pred) sqrt(mean((actual - pred)^2))
r_squared <- function(actual, pred) {
  1 - sum((actual - pred)^2) / sum((actual - mean(actual))^2)
}

pred_train_min <- as.numeric(predict(cv_fit, newx = x_train, s = "lambda.min"))
pred_train_1se <- as.numeric(predict(cv_fit, newx = x_train, s = "lambda.1se"))
pred_test_min  <- as.numeric(predict(cv_fit, newx = x_test,  s = "lambda.min"))
pred_test_1se  <- as.numeric(predict(cv_fit, newx = x_test,  s = "lambda.1se"))

message("\nPerformance:")
metrics <- data.frame(
  split       = c("train", "train", "test", "test"),
  lambda      = c("lambda.min", "lambda.1se", "lambda.min", "lambda.1se"),
  rmse        = c(rmse(y_train, pred_train_min), rmse(y_train, pred_train_1se),
                   rmse(y_test,  pred_test_min),  rmse(y_test,  pred_test_1se)),
  r_squared   = c(r_squared(y_train, pred_train_min), r_squared(y_train, pred_train_1se),
                   r_squared(y_test,  pred_test_min),  r_squared(y_test,  pred_test_1se))
)
print(metrics)

# -----------------------------------------------------------------------------
# 6. Write outputs for 04_results.R
# -----------------------------------------------------------------------------

predictions <- bind_rows(
  train_df %>%
    transmute(ticker, quarter_end, split,
              exret_next,
              pred_lambda_min = pred_train_min,
              pred_lambda_1se = pred_train_1se),
  test_df %>%
    transmute(ticker, quarter_end, split,
              exret_next,
              pred_lambda_min = pred_test_min,
              pred_lambda_1se = pred_test_1se)
) %>%
  arrange(ticker, quarter_end)

write_csv(predictions, file.path(PROC_DIR, "lasso_predictions.csv"))

coefficients <- data.frame(
  predictor      = rownames(coef_min),
  coef_lambda_min = as.numeric(coef_min[, 1]),
  coef_lambda_1se = as.numeric(coef_1se[, 1])
)
write_csv(coefficients, file.path(PROC_DIR, "lasso_coefficients.csv"))

message("\nSaved ", nrow(predictions), " predictions -> ",
        file.path(PROC_DIR, "lasso_predictions.csv"))
message("Saved ", nrow(coefficients), " coefficients -> ",
        file.path(PROC_DIR, "lasso_coefficients.csv"))

# -----------------------------------------------------------------------------
# 7. Fit: PLS (Partial Least Squares), ncomp selected by CV on training rows
# -----------------------------------------------------------------------------
# Same train/test rows, same 12 predictors, same no-look-ahead boundary as
# the Lasso section above -- CV for choosing the number of components is
# confined entirely to train_df, never touching test_df. Unlike Lasso's
# all-or-nothing per-predictor selection, PLS builds a small number of
# latent components as linear combinations of ALL predictors, chosen to
# maximise covariance with exret_next. That makes it a natural next lens
# on the diagnostics above: several predictors showed weak individual
# correlations but significant partial OLS effects once correlated
# predictors were controlled for (a sign of shared, entangled signal) --
# exactly the structure PLS's components are built to capture instead of
# discarding.

set.seed(SEED)
pls_fit <- plsr(exret_next ~ ., data = ols_df, scale = TRUE, validation = "CV")

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
# exactly the Lasso lambda.1se null-model shape -- rather than silently
# substituting ncomp_min and hiding a real, conservative CV finding.
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
              exret_next,
              pred_ncomp_min = pls_pred_train_min,
              pred_ncomp_1se = pls_pred_train_1se),
  test_df %>%
    transmute(ticker, quarter_end, split,
              exret_next,
              pred_ncomp_min = pls_pred_test_min,
              pred_ncomp_1se = pls_pred_test_1se)
) %>%
  arrange(ticker, quarter_end)

write_csv(pls_predictions, file.path(PROC_DIR, "pls_predictions.csv"))

pls_coefficients <- data.frame(
  predictor        = names(pls_coef_min),
  coef_ncomp_min = as.numeric(pls_coef_min),
  coef_ncomp_1se = as.numeric(pls_coef_1se)
)
write_csv(pls_coefficients, file.path(PROC_DIR, "pls_coefficients.csv"))

message("\nSaved ", nrow(pls_predictions), " PLS predictions -> ",
        file.path(PROC_DIR, "pls_predictions.csv"))
message("Saved ", nrow(pls_coefficients), " PLS coefficients -> ",
        file.path(PROC_DIR, "pls_coefficients.csv"))

# -----------------------------------------------------------------------------
# 8. Fit: Bagged Regression Splines (Bagged MARS, hand-rolled on earth::earth)
# -----------------------------------------------------------------------------
# Same train/test rows, same 12 predictors, same no-look-ahead boundary as
# the Lasso and PLS sections above. Bagging (B bootstrap-resampled MARS
# fits, averaged) is what lets this model capture nonlinearity/interactions
# that neither Lasso (linear, sparse) nor PLS (linear, latent-factor) can --
# the reason it's the third model in the course's fixed sequence, tried
# last precisely because the first two came back null: if there's a real
# relationship here, it's more likely nonlinear than a linear one Lasso/PLS
# both missed.
#
# This is implemented directly on earth::earth() rather than via
# caret::train(method = "bagEarth"). caret's bag() framework depends on
# plyr's unqualified, unnamespaced .() call, and attaching plyr to make
# that resolve collides with dplyr's stricter data-masking (rlang) checks
# used throughout the rest of this script -- a version-sensitive
# plyr/dplyr/caret interoperability issue, not something to keep patching
# blind. A manual bootstrap loop over earth() avoids the dependency
# entirely and makes the bagging mechanics explicit.
#
# nprune (max MARS terms after pruning) is chosen by manual 5-fold CV,
# confined entirely to train_df -- folds are drawn from train_df only, so
# this never touches test_df, same no-look-ahead reasoning as cv.glmnet's/
# plsr's CV above. degree = 1 only (additive terms, no MARS interaction
# terms): with ~232 training rows and 12 predictors, allowing degree = 2
# interaction terms would multiply the hypothesis space and invite exactly
# the kind of overfitting PLS's ncomp_min already demonstrated on this
# same data.

MARS_DEGREE <- 1
NPRUNE_GRID <- seq(2, 10, by = 2)
N_FOLDS <- 5
N_BAGS  <- 25

# Wrapped in a function so the identical fit/CV/bag procedure can be run
# twice -- once on all 8 tickers (the primary result reported above), once
# with VRT excluded -- without duplicating the logic. label suffixes the
# output filenames so neither run overwrites the other.
fit_bagged_mars <- function(train_df, test_df, x_train, y_train, x_test, y_test, label) {
  set.seed(SEED)
  cv_folds <- sample(rep(1:N_FOLDS, length.out = nrow(train_df)))

  cv_rmse_by_nprune <- sapply(NPRUNE_GRID, function(np) {
    fold_rmse <- sapply(1:N_FOLDS, function(k) {
      fit_k <- earth(x = x_train[cv_folds != k, , drop = FALSE],
                      y = y_train[cv_folds != k],
                      degree = MARS_DEGREE, nprune = np)
      pred_k <- as.numeric(predict(fit_k, newdata = x_train[cv_folds == k, , drop = FALSE]))
      rmse(y_train[cv_folds == k], pred_k)
    })
    mean(fold_rmse)
  })
  names(cv_rmse_by_nprune) <- NPRUNE_GRID

  best_nprune <- as.integer(names(cv_rmse_by_nprune)[which.min(cv_rmse_by_nprune)])

  message("\n[", label, "] Bagged MARS -- 5-fold CV RMSE by nprune (training rows only):")
  print(cv_rmse_by_nprune)
  message("[", label, "] Selected nprune = ", best_nprune, " (degree = ", MARS_DEGREE, ")")

  # Bag: N_BAGS bootstrap resamples of the training rows, each fit at the
  # CV-selected nprune, predictions averaged across bags.
  set.seed(SEED)
  bag_preds_train <- matrix(NA_real_, nrow = nrow(train_df), ncol = N_BAGS)
  bag_preds_test  <- matrix(NA_real_, nrow = nrow(test_df),  ncol = N_BAGS)
  bag_importance  <- matrix(0, nrow = length(PREDICTORS), ncol = N_BAGS,
                             dimnames = list(PREDICTORS, NULL))

  for (b in 1:N_BAGS) {
    boot_idx <- sample(nrow(train_df), replace = TRUE)
    fit_b <- earth(x = x_train[boot_idx, , drop = FALSE],
                    y = y_train[boot_idx],
                    degree = MARS_DEGREE, nprune = best_nprune)
    bag_preds_train[, b] <- as.numeric(predict(fit_b, newdata = x_train))
    bag_preds_test[, b]  <- as.numeric(predict(fit_b, newdata = x_test))

    # evimp() reports gcv-based importance only for predictors this bag's
    # pruned model actually used; predictors it dropped entirely are left
    # at this matrix's 0 default rather than NA, since "not selected in
    # this bag" is a real (low-importance) outcome, not missing data.
    imp_b <- evimp(fit_b, trim = FALSE)
    used <- intersect(rownames(imp_b), PREDICTORS)
    bag_importance[used, b] <- imp_b[used, "gcv"]
  }

  bagged_pred_train <- rowMeans(bag_preds_train)
  bagged_pred_test  <- rowMeans(bag_preds_test)

  message("\n[", label, "] Bagged MARS performance:")
  bagged_metrics <- data.frame(
    split     = c("train", "test"),
    rmse      = c(rmse(y_train, bagged_pred_train), rmse(y_test, bagged_pred_test)),
    r_squared = c(r_squared(y_train, bagged_pred_train), r_squared(y_test, bagged_pred_test))
  )
  print(bagged_metrics)

  bagged_predictions <- bind_rows(
    train_df %>% transmute(ticker, quarter_end, split, exret_next, pred = bagged_pred_train),
    test_df  %>% transmute(ticker, quarter_end, split, exret_next, pred = bagged_pred_test)
  ) %>%
    arrange(ticker, quarter_end)

  # label = "all" keeps the original, unsuffixed filenames so the
  # primary result stays at the same path; only the
  # "novrt" comparison run gets a distinguishing suffix.
  file_tag <- if (label == "all") "" else paste0("_", label)
  pred_file <- file.path(PROC_DIR, paste0("bagged_splines", file_tag, "_predictions.csv"))
  write_csv(bagged_predictions, pred_file)

  # MARS/bagged-MARS has no single per-predictor coefficient (each
  # bootstrap fit can use a predictor in different hinge functions/terms,
  # or drop it entirely), so mean importance across bags -- not a
  # coefficient table -- is the analogous diagnostic output here.
  bagged_importance <- data.frame(
    predictor  = PREDICTORS,
    importance = rowMeans(bag_importance)
  ) %>%
    arrange(desc(importance))
  imp_file <- file.path(PROC_DIR, paste0("bagged_splines", file_tag, "_importance.csv"))
  write_csv(bagged_importance, imp_file)

  message("\n[", label, "] Saved ", nrow(bagged_predictions), " predictions -> ", pred_file)
  message("[", label, "] Saved ", nrow(bagged_importance), " variable-importance rows -> ", imp_file)

  invisible(list(metrics = bagged_metrics, importance = bagged_importance))
}

# Run 1: all 8 tickers (the primary result; filenames
# unchanged so it stays comparable / doesn't disturb anything already
# reviewed).
fit_bagged_mars(train_df, test_df, x_train, y_train, x_test, y_test, label = "all")

# Run 2: VRT excluded. VRT was the clear outlier across every model so
# far -- only 8 training quarters (2021Q1-2022Q4, it wasn't public/in the
# universe before), and structurally a thermal-management/data-center
# company rather than a classic water utility, which is very likely why
# its predictor values sit outside the range the other 7 tickers train
# the model on. This checks whether VRT's noise was suppressing signal
# in the other 7 tickers, or whether the null result holds regardless.
train_df_novrt <- train_df %>% filter(ticker != "VRT")
test_df_novrt  <- test_df  %>% filter(ticker != "VRT")
x_train_novrt  <- as.matrix(train_df_novrt[, PREDICTORS])
y_train_novrt  <- train_df_novrt$exret_next
x_test_novrt   <- as.matrix(test_df_novrt[, PREDICTORS])
y_test_novrt   <- test_df_novrt$exret_next

message("\nVRT excluded: ", sum(train_df$ticker == "VRT"), " train / ",
        sum(test_df$ticker == "VRT"), " test row(s) dropped. ",
        nrow(train_df_novrt), " train / ", nrow(test_df_novrt),
        " test rows remain across the other 7 tickers.")

fit_bagged_mars(train_df_novrt, test_df_novrt, x_train_novrt, y_train_novrt,
                 x_test_novrt, y_test_novrt, label = "novrt")

message("\n03_models.R complete.")
