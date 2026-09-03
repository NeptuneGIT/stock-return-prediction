# install_packages.R -- installs the R packages the pipeline needs.
# Run once: Rscript install_packages.R

packages <- c(
  "dplyr", "readr", "tidyr", "lubridate", "zoo", "data.table",
  "tidyquant", "pls", "randomForest", "gbm"
)

missing <- packages[!packages %in% rownames(installed.packages())]
if (length(missing) > 0) {
  install.packages(missing, repos = "https://cloud.r-project.org")
} else {
  message("All required packages are already installed.")
}
