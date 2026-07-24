# Databricks notebook source
# MAGIC %md
# MAGIC # Regression for Large Matrices with Joint Column and Row Reduction - Visualizations
# MAGIC Loss, total time, and screening plots for all CUR reduction pipelines.

# COMMAND ----------

# DBTITLE 1,Setup & Imports
import sys
import os
import pickle
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath("__file__")))

from evaluation import evaluate_experiment
from visualizations import (
    plot_loss_boxplots,
    plot_screening_boxplots,
    plot_score_distributions,
    plot_n_selected,
)

# COMMAND ----------

# DBTITLE 1,Configuration
# === Paths ===
BASE = "/Users/timschmale/Documents/GIT/Masterthesis/"
DATA_FOLDER = "Simulation Data/p1000_n1000"
RESULTS_FOLDER = "Results/p1000"
SAVE_PATH = os.path.join(BASE, "Plots/p1000")  # PDF output

# === Parameters ===
k_vector = [10, 20, 25, 50, 100, 200, 300, 400, 500]
REPS = 10
OUTER_REPS = 10
DATASET = "p1000"

# COMMAND ----------

# DBTITLE 1,Load Results from Pickles
def load_all_seeds(base, results_folder, save_name, seed_from=1, seed_to=11):
    """Load per-seed pickles and merge into one results dict."""
    all_results = {}
    path = os.path.join(base, results_folder)
    for seed in range(seed_from, seed_to):
        fpath = os.path.join(path, f"{save_name}_seed_{seed}.pkl")
        if os.path.exists(fpath):
            with open(fpath, "rb") as f:
                all_results[seed] = pickle.load(f)
        else:
            print(f"  [WARN] Missing: {fpath}")
    print(f"Loaded {len(all_results)} seeds for '{save_name}'.")
    return all_results

# Load all pipeline results
results_ols_col_row = load_all_seeds(BASE, RESULTS_FOLDER, "results_ols_col_row")
results_ols_col_only = load_all_seeds(BASE, RESULTS_FOLDER, "results_ols_col_only")
results_ols_row_col = load_all_seeds(BASE, RESULTS_FOLDER, "results_ols_row_col")
results_logistic = load_all_seeds(BASE, RESULTS_FOLDER, "results_logistic")
results_lasso = load_all_seeds(BASE, RESULTS_FOLDER, "results_lasso")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loss & Total Time

# COMMAND ----------

# DBTITLE 1,OLS Column→Row: Loss Plots
plot_loss_boxplots(results_ols_col_row, k_vector, metric="rmse_test",
                   pipeline="col_row", dataset=DATASET, save_path=SAVE_PATH)

plot_loss_boxplots(results_ols_col_row, k_vector, metric="rmse_train",
                   pipeline="col_row", dataset=DATASET, save_path=SAVE_PATH)

plot_loss_boxplots(results_ols_col_row, k_vector, metric="time_total",
                   pipeline="col_row", dataset=DATASET, save_path=SAVE_PATH)

# COMMAND ----------

# DBTITLE 1,OLS Column Only: Loss Plots
plot_loss_boxplots(results_ols_col_only, k_vector, metric="rmse_test",
                   pipeline="col_only", dataset=DATASET, save_path=SAVE_PATH)

plot_loss_boxplots(results_ols_col_only, k_vector, metric="rmse_train",
                   pipeline="col_only", dataset=DATASET, save_path=SAVE_PATH)

plot_loss_boxplots(results_ols_col_only, k_vector, metric="time_total",
                   pipeline="col_only", dataset=DATASET, save_path=SAVE_PATH)

# COMMAND ----------

# DBTITLE 1,OLS Row→Column (Sketching): Loss Plots
plot_loss_boxplots(results_ols_row_col, k_vector, metric="rmse_test",
                   pipeline="row_col", dataset=DATASET, save_path=SAVE_PATH)

plot_loss_boxplots(results_ols_row_col, k_vector, metric="rmse_train",
                   pipeline="row_col", dataset=DATASET, save_path=SAVE_PATH)

plot_loss_boxplots(results_ols_row_col, k_vector, metric="time_total",
                   pipeline="row_col", dataset=DATASET, save_path=SAVE_PATH)

# COMMAND ----------

# DBTITLE 1,Logistic Regression: Loss Plots
plot_loss_boxplots(results_logistic, k_vector, metric="brier_test",
                   pipeline="logistic", dataset=DATASET, save_path=SAVE_PATH)

plot_loss_boxplots(results_logistic, k_vector, metric="ce_test",
                   pipeline="logistic", dataset=DATASET, save_path=SAVE_PATH)

plot_loss_boxplots(results_logistic, k_vector, metric="time_total",
                   pipeline="logistic", dataset=DATASET, save_path=SAVE_PATH)

# COMMAND ----------

# DBTITLE 1,Lasso: Loss Plots
plot_loss_boxplots(results_lasso, k_vector, metric="rmse_test",
                   pipeline="lasso", dataset=DATASET, save_path=SAVE_PATH)

plot_loss_boxplots(results_lasso, k_vector, metric="rmse_train",
                   pipeline="lasso", dataset=DATASET, save_path=SAVE_PATH)

plot_loss_boxplots(results_lasso, k_vector, metric="time_total",
                   pipeline="lasso", dataset=DATASET, save_path=SAVE_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Screening (TPR, FDR, MCC)

# COMMAND ----------

# DBTITLE 1,Screening: Evaluate All Pipelines
# OLS Pipelines
screening_col_row = evaluate_experiment(results_ols_col_row, BASE, DATA_FOLDER, REPS)
screening_col_only = evaluate_experiment(results_ols_col_only, BASE, DATA_FOLDER, REPS)
screening_row_col = evaluate_experiment(results_ols_row_col, BASE, DATA_FOLDER, REPS)

# Logistic
screening_logistic = evaluate_experiment(results_logistic, BASE, DATA_FOLDER, REPS)

# Lasso
screening_lasso = evaluate_experiment(results_lasso, BASE, DATA_FOLDER, REPS)

print("Screening evaluation done.")
print(f"  col_row:  {len(screening_col_row)} rows")
print(f"  col_only: {len(screening_col_only)} rows")
print(f"  row_col:  {len(screening_row_col)} rows")
print(f"  logistic: {len(screening_logistic)} rows")
print(f"  lasso:    {len(screening_lasso)} rows")

# COMMAND ----------

# DBTITLE 1,Screening: OLS Column→Row
plot_screening_boxplots(screening_col_row, pipeline="col_row", dataset=DATASET,
                        save_path=SAVE_PATH)

# COMMAND ----------

# DBTITLE 1,Screening: OLS Column Only
plot_screening_boxplots(screening_col_only, pipeline="col_only", dataset=DATASET,
                        save_path=SAVE_PATH)

# COMMAND ----------

# DBTITLE 1,Screening: OLS Row→Column
plot_screening_boxplots(screening_row_col, pipeline="row_col", dataset=DATASET,
                        save_path=SAVE_PATH)

# COMMAND ----------

# DBTITLE 1,Screening: Logistic Regression
plot_screening_boxplots(screening_logistic, pipeline="logistic", dataset=DATASET,
                        save_path=SAVE_PATH)

# COMMAND ----------

# DBTITLE 1,Screening: Lasso (alle 3 Varianten)
plot_screening_boxplots(screening_lasso, pipeline="lasso", dataset=DATASET,
                        save_path=SAVE_PATH)
