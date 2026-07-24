# Databricks notebook source
# MAGIC %md
# MAGIC # Regression for Large Matrices with Joint Column and Row Reduction - Reduction
# MAGIC
# MAGIC Runs all pipelines and saves per-seed pickles.

# COMMAND ----------

# DBTITLE 1,Setup & Imports
import sys
import os

# Notebook und Module liegen im selben Verzeichnis (refactored/)
sys.path.append(os.path.dirname(os.path.abspath("__file__")))

from pipeline import (
    run_ols_experiment,
    run_logistic_experiment,
    run_lasso_experiment,
)

# COMMAND ----------

# DBTITLE 1,Configuration
# === Paths ===
BASE = "/Users/timschmale/Documents/GIT/Masterthesis/"
DATA_FOLDER = "Simulation Data/p1000_n1000"  # or p5000_n5000, p10000_n10000
RESULTS_FOLDER = "Results/p1000"

# === Experiment parameters ===
k_vector = [10, 20, 25, 50, 100, 200, 300, 400, 500]
REPS = 10
OUTER_REPS = 10
TEST_SIZE = 0.2
SEED_FROM = 1
SEED_TO = None  # defaults to SEED_FROM + OUTER_REPS

# COMMAND ----------

# MAGIC %md
# MAGIC ## OLS Pipelines

# COMMAND ----------

# DBTITLE 1,Pipeline 1: OLS Column → Row
results_ols_col_row = run_ols_experiment(
    k_vector=k_vector,
    base=BASE,
    data_folder=DATA_FOLDER,
    results_folder=RESULTS_FOLDER,
    reps=REPS,
    outer_reps=OUTER_REPS,
    test_size=TEST_SIZE,
    order="column_first",
    save_name="results_ols_col_row",
    seed_from=SEED_FROM,
    seed_to=SEED_TO,
)

# COMMAND ----------

# DBTITLE 1,Pipeline 2: OLS Column Only
results_ols_col_only = run_ols_experiment(
    k_vector=k_vector,
    base=BASE,
    data_folder=DATA_FOLDER,
    results_folder=RESULTS_FOLDER,
    reps=REPS,
    outer_reps=OUTER_REPS,
    test_size=TEST_SIZE,
    order="column_only",
    save_name="results_ols_col_only",
    seed_from=SEED_FROM,
    seed_to=SEED_TO,
)

# COMMAND ----------

# DBTITLE 1,Pipeline 3: OLS Row → Column (Sketching)
results_ols_row_col = run_ols_experiment(
    k_vector=k_vector,
    base=BASE,
    data_folder=DATA_FOLDER,
    results_folder=RESULTS_FOLDER,
    reps=REPS,
    outer_reps=OUTER_REPS,
    test_size=TEST_SIZE,
    order="row_first",
    sketch_method="rademacher",
    save_name="results_ols_row_col",
    seed_from=SEED_FROM,
    seed_to=SEED_TO,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Logistic Regression

# COMMAND ----------

# DBTITLE 1,Pipeline 4: Logistic Regression (Column → Row + Coreset)
results_logistic = run_logistic_experiment(
    k_vector=k_vector,
    base=BASE,
    data_folder=DATA_FOLDER,
    results_folder=RESULTS_FOLDER,
    reps=REPS,
    outer_reps=OUTER_REPS,
    test_size=TEST_SIZE,
    save_name="results_logistic",
    seed_from=SEED_FROM,
    seed_to=SEED_TO,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lasso (3 Variants)

# COMMAND ----------

# DBTITLE 1,Pipeline 5: Lasso (3 Varianten: Theoretical, Binary Search, CLS+Lasso)
results_lasso = run_lasso_experiment(
    k_vector=k_vector,
    base=BASE,
    data_folder=DATA_FOLDER,
    results_folder=RESULTS_FOLDER,
    reps=REPS,
    outer_reps=OUTER_REPS,
    test_size=TEST_SIZE,
    sketch_method="rademacher",
    save_name="results_lasso",
    seed_from=SEED_FROM,
    seed_to=SEED_TO,
)
