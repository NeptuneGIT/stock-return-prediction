# main.R -- runs the full pipeline: scrape, clean, merge, train, backtest, forecast.
# Each stage is its own subprocess, not source()/import -- the dividends step
# (scripts/01_get_data.R's quantmod::getDividends call) can segfault the R
# process outright in some environments, and running it in-process could take
# the whole pipeline down with it.

SKIP_SCRAPE <- TRUE

run_rscript <- function(path) {
  message("\n=== Running ", path, " ===")
  status <- system2("Rscript", shQuote(path))
  if (status != 0) {
    stop(path, " failed (exit status ", status, "). See output above.")
  }
}

run_python <- function(path) {
  message("\n=== Running ", path, " ===")
  python_bin <- if (nzchar(Sys.which("python"))) "python" else "python3"
  status <- system2(python_bin, shQuote(path))
  if (status != 0) {
    stop(path, " failed (exit status ", status, "). See output above.")
  }
}

# Stage 1-3: scrape (prices/dividends/macro, SEC fundamentals) and clean it

if (!SKIP_SCRAPE) {
  run_rscript("scripts/01_get_data.R")
  div_path <- "data/raw/dividends_raw.csv"
  needs_fallback <- !file.exists(div_path) || nrow(read.csv(div_path, nrows = 2)) == 0
  if (needs_fallback) {
    message("\ndividends_raw.csv missing or empty -- falling back to the ",
            "Python dividend fetcher.")
    run_python("scripts/get_dividends_fallback.py")
  }

  run_python("scripts/02_fundamentals.py")
  run_python("scripts/02_clean_fundamentals.py")
} else {
  message("SKIP_SCRAPE is TRUE -- reusing the committed data/raw and ",
          "data/processed/fundamentals_clean.csv instead of re-scraping. ",
          "Set SKIP_SCRAPE <- FALSE at the top of this file to re-run the ",
          "full scrape.")
}

# Stage 4: merge everything into the modelling panel
run_rscript("scripts/01a_ratios.R")

# Stage 5: train and walk-forward validate PLS / Random Forest / GBRT
run_rscript("scripts/03_models.R")

# Stage 6: backtest the buy-the-top-N strategy against SPY
run_rscript("scripts/04_results.R")

# Stage 7: rank tickers and forecast the top 30
run_rscript("scripts/05_top30_forecast.R")

message("\n=== Pipeline complete. See output/tables/ for the top-30 forecast. ===")
