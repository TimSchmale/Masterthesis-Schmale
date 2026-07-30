"""Experiment pipelines for CUR reduction studies.

One runner function per model type, with shared functions
for data loading, score caching, and result persistence.

Timing Conventions (consistent across all pipelines)
----------------------------------------------------
All timings use time.perf_counter() (monotonic, high-resolution).

- time_scores:     Time to compute the column score vector (SVD-based).
                   Measured internally in compute_all_scores().
- time_reduction:  Time for ALL steps to achieve dimensionality reduction:
                   column_first: column_reduction + row_reduction_leverage
                   column_only:  column_reduction
                   row_first:    row_reduction_sketch + column_reduction
                   logistic:     column_reduction + estimate_mu + row_reduction_coreset
                   lasso:        row_reduction_sketch + alpha_search [+ column_reduction for CLS]
                   Note: Lasso's alpha search IS its sparsity/reduction mechanism
                   (analogous to CUR's column selection), hence included here.
- time_model:      Time for model.fit() ONLY (no predict, no metric eval).
                   Measured internally in fit_ols/fit_logistic/fit_lasso.
                   Comparable across all pipelines.
- time_fit:        (Lasso only) Same as time_model, kept for backward compat.
- time_alpha_search: (Lasso only) Time for alpha selection (binary search
                   or theoretical computation). Included in time_reduction,
                   stored separately for detailed analysis.

Total computational cost for fair comparison:
    time_total = time_scores + time_reduction + time_model
"""

import numpy as np
import pandas as pd
import pickle
import time
import os
from pathlib import Path
def numpy_train_test_split(X, y, test_size, seed):
    """Deterministic NumPy-based train/test split.

    Uses a separate RNG instance (default_rng) seeded with `seed` so that
    the split is reproducible and independent of the global numpy state.
    Matches the original implementation used in all old pipeline files.

    Parameters
    ----------
    X : ndarray of shape (n, p)
    y : ndarray of shape (n,)
    test_size : float
        Fraction of samples in the test set.
    seed : int
        Random seed for shuffling.

    Returns
    -------
    X_train, X_test, y_train, y_test : ndarray
    """
    n = X.shape[0]
    n_test = int(np.floor(n * test_size))
    n_train = n - n_test

    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

from scoring import compute_all_scores, get_cross_leverage_scores
from reduction import (
    column_reduction,
    row_reduction_leverage,
    row_reduction_sketch,
    row_reduction_coreset,
    estimate_mu,
)
from modeling import fit_ols, fit_logistic, fit_lasso


# =============================================================================
# Shared Functions
# =============================================================================

def load_data(base, folder, reps, response="continuous"):
    """Load simulation data from CSV files.

    Parameters
    ----------
    base : str or Path
        Base directory.
    folder : str
        Subfolder containing X*.csv, y*.csv files.
    reps : int
        Number of replications to load.
    response : {"continuous", "binary"}
        "continuous" loads y*.csv, "binary" loads y_binary*.csv.

    Returns
    -------
    X_list : list of ndarray
    y_list : list of ndarray
    """
    X_list = []
    y_list = []

    for i in range(reps):
        X = pd.read_csv(f"{base}/{folder}/X{i + 1}.csv").to_numpy()
        X_list.append(X)

        if response == "binary":
            y = pd.read_csv(f"{base}/{folder}/y_binary{i + 1}.csv").to_numpy().ravel()
        else:
            y = pd.read_csv(f"{base}/{folder}/y{i + 1}.csv").to_numpy().ravel()
        y_list.append(y)

    return X_list, y_list


def compute_scores_cached(X_list, y_list, k_vector):
    """Precompute all scores for all k and replications.

    Scores are independent of the random seed and can be reused
    across all outer repetitions.

    Parameters
    ----------
    X_list : list of ndarray
    y_list : list of ndarray
    k_vector : list of int

    Returns
    -------
    dict : {k: {method: [score_per_rep]}}
    dict : {k: {method: [time_per_rep]}}
    """
    cached_scores = {}
    cached_timings = {}

    for k in k_vector:
        scores_k = {"LS": [], "CLS": [], "RS": [], "CS": []}
        timings_k = {"LS": [], "CLS": [], "RS": [], "CS": []}

        for i in range(len(X_list)):
            s, t = compute_all_scores(X_list[i], y_list[i], k)
            for method in s:
                scores_k[method].append(s[method])
                timings_k[method].append(t.get(method, 0))

        cached_scores[k] = scores_k
        cached_timings[k] = timings_k
        print(f"  Scores for k={k} done.")

    return cached_scores, cached_timings


def save_results(path, results):
    """Save results dict as pickle."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(results, f)


# =============================================================================
# OLS Pipeline
# =============================================================================

def run_ols_experiment(
    k_vector,
    base,
    data_folder,
    results_folder,
    reps=10,
    outer_reps=10,
    test_size=0.2,
    order="column_first",
    save_name="results_ols",
    seed_from=1,
    seed_to=None
):
    """Run OLS sampling-variance experiment.

    Parameters
    ----------
    k_vector : list of int
        Target ranks to evaluate.
    base : str or Path
        Base directory.
    data_folder : str
        Subfolder with X*.csv, y*.csv.
    results_folder : str
        Subfolder for output.
    reps : int
        Number of data replications.
    outer_reps : int
        Number of random seeds.
    test_size : float
        Train/test split proportion.
    order : {"column_first", "row_first", "column_only"}
        Reduction order.
    sketch_method : {"gaussian", "rademacher"}
        Only used when order="row_first".
    save_name : str
        Filename prefix for pickles.
    seed_from : int
        Starting seed (for resumability).
    seed_to : int or None
        Ending seed (exclusive). Defaults to seed_from + outer_reps.

    Returns
    -------
    dict : {seed: {k: {method: {metric: [values]}}}}
    """
    base = Path(base)
    if seed_to is None:
        seed_to = seed_from + outer_reps

    # Load data
    print(f"Loading data from {data_folder}...")
    X_list, y_list = load_data(base, data_folder, reps, response="continuous")
    print(f"  Shape: {X_list[0].shape}, {reps} replications.")

    # Run per-seed experiments
    all_results = {}
    out_path = base / results_folder
    os.makedirs(out_path, exist_ok=True)

    for seed in range(seed_from, seed_to):
        print(f"\nSeed {seed}...")
        np.random.seed(seed + 42)

        # Split all replications for this seed
        X_train_list, X_test_list = [], []
        y_train_list, y_test_list = [], []
        for i in range(reps):
            X_tr, X_te, y_tr, y_te = numpy_train_test_split(
                X_list[i], y_list[i],
                test_size=test_size, seed=seed
            )
            X_train_list.append(X_tr)
            X_test_list.append(X_te)
            y_train_list.append(y_tr)
            y_test_list.append(y_te)

        # Compute scores on TRAINING data for this seed
        print(f"  Computing scores on training data...")
        cached_scores, cached_timings = compute_scores_cached(
            X_train_list, y_train_list, k_vector
        )

        seed_results = {}

        for k in k_vector:
            method_results = {}

            if order == "row_first":
                # =============================================================
                # Row-First: 1 sketch per rep, shared across all methods.
                # Scores computed once on the sketch, then column reduction
                # per method on the SAME sketch.
                # =============================================================

                # Step 1: Sketch all reps (timed)
                R_list = []
                y_sk_list = []
                time_sketch_list = []
                for i in range(reps):
                    t0 = time.perf_counter()
                    R, y_sk = row_reduction_sketch(
                        X_train_list[i], y_train_list[i], k
                    )
                    time_sketch_list.append(time.perf_counter() - t0)
                    R_list.append(R)
                    y_sk_list.append(y_sk)

                # Step 2: Compute scores on all sketches (shared across methods)
                all_sketch_scores = []  # list of dicts per rep
                all_sketch_timings = []  # list of dicts per rep
                for i in range(reps):
                    s, t = compute_all_scores(
                        R_list[i], y_sk_list[i], k, rank_reduce=False
                    )
                    all_sketch_scores.append(s)
                    all_sketch_timings.append(t)

                # Step 3: Column reduction (all methods) + abort check
                reductions_rf = {}
                abort_reps_rf = set()

                for method in ("LS", "CLS", "RS", "CS"):
                    reductions_rf[method] = {}
                    for i in range(reps):
                        scores_i = all_sketch_scores[i][method]

                        t0 = time.perf_counter()
                        col_res = column_reduction(R_list[i], scores_i, k)
                        X_red = col_res["C"]
                        y_red = y_sk_list[i]
                        sel_cols = col_res["selected_columns"]
                        time_col = time.perf_counter() - t0
                        # time_reduction = sketch + column reduction
                        time_red = time_sketch_list[i] + time_col

                        reductions_rf[method][i] = {
                            "X_red": X_red, "y_red": y_red,
                            "sel_cols": sel_cols, "time_red": time_red
                        }

                        if X_red.shape[1] > X_red.shape[0]:
                            abort_reps_rf.add(i)

                if abort_reps_rf:
                    print(f"  [INFO] k={k}: Reps {sorted(abort_reps_rf)} aborted "
                          f"(underdetermined for >=1 method). Skipping ALL methods.")

                # Step 4: Modeling (skip abort_reps for ALL methods)
                for method in ("LS", "CLS", "RS", "CS"):
                    metrics = {"rmse_test": [], "rmse_train": [],
                               "time_model": [], "time_reduction": [],
                               "time_scores": [], "selected_columns": []}

                    for i in range(reps):
                        red = reductions_rf[method][i]
                        X_red = red["X_red"]
                        y_red = red["y_red"]
                        sel_cols = red["sel_cols"]
                        time_red = red["time_red"]

                        if i in abort_reps_rf:
                            metrics["rmse_test"].append(np.nan)
                            metrics["rmse_train"].append(np.nan)
                            metrics["time_model"].append(np.nan)
                            metrics["time_reduction"].append(time_red)
                            metrics["time_scores"].append(all_sketch_timings[i][method])
                            metrics["selected_columns"].append(sel_cols)
                            continue

                        # Modeling (time_model = only model.fit via internal timing)
                        result = fit_ols(X_red, y_red, X_test_list[i], y_test_list[i],
                                         selected_columns=sel_cols)

                        metrics["rmse_test"].append(result["rmse_test"])
                        metrics["rmse_train"].append(result["rmse_train"])
                        metrics["time_model"].append(result["time_fit"])
                        metrics["time_reduction"].append(time_red)
                        metrics["time_scores"].append(all_sketch_timings[i][method])
                        metrics["selected_columns"].append(sel_cols)

                    method_results[method] = metrics

            else:
                # =============================================================
                # Column-First / Column-Only: two-phase (reduce, then model)
                # =============================================================

                # Phase 1: Compute all reductions and identify abort reps
                reductions_cf = {}
                abort_reps_cf = set()

                for method in ("LS", "CLS", "RS", "CS"):
                    reductions_cf[method] = {}
                    for i in range(reps):
                        scores_i = cached_scores[k][method][i]

                        t0 = time.perf_counter()

                        if order == "column_first":
                            col_res = column_reduction(X_train_list[i], scores_i, k)
                            C = col_res["C"]
                            sel_cols = col_res["selected_columns"]
                            row_res = row_reduction_leverage(C, y_train_list[i], k)
                            X_red = row_res["R"]
                            y_red = row_res["y"]

                        elif order == "column_only":
                            col_res = column_reduction(X_train_list[i], scores_i, k)
                            X_red = col_res["C"]
                            y_red = y_train_list[i]
                            sel_cols = col_res["selected_columns"]

                        time_red = time.perf_counter() - t0

                        reductions_cf[method][i] = {
                            "X_red": X_red, "y_red": y_red,
                            "sel_cols": sel_cols, "time_red": time_red
                        }

                        if X_red.shape[1] > X_red.shape[0]:
                            abort_reps_cf.add(i)

                if abort_reps_cf:
                    print(f"  [INFO] k={k}: Reps {sorted(abort_reps_cf)} aborted "
                          f"(underdetermined for >=1 method). Skipping ALL methods.")

                # Phase 2: Modeling (skip abort_reps for ALL methods)
                for method in ("LS", "CLS", "RS", "CS"):
                    metrics = {"rmse_test": [], "rmse_train": [],
                               "time_model": [], "time_reduction": [],
                               "time_scores": [], "selected_columns": []}

                    for i in range(reps):
                        red = reductions_cf[method][i]
                        X_red = red["X_red"]
                        y_red = red["y_red"]
                        sel_cols = red["sel_cols"]
                        time_red = red["time_red"]

                        if i in abort_reps_cf:
                            metrics["rmse_test"].append(np.nan)
                            metrics["rmse_train"].append(np.nan)
                            metrics["time_model"].append(np.nan)
                            metrics["time_reduction"].append(time_red)
                            metrics["time_scores"].append(cached_timings[k][method][i])
                            metrics["selected_columns"].append(sel_cols)
                            continue

                        # Modeling (time_model = only model.fit via internal timing)
                        result = fit_ols(X_red, y_red, X_test_list[i], y_test_list[i],
                                         selected_columns=sel_cols)

                        metrics["rmse_test"].append(result["rmse_test"])
                        metrics["rmse_train"].append(result["rmse_train"])
                        metrics["time_model"].append(result["time_fit"])
                        metrics["time_reduction"].append(time_red)
                        metrics["time_scores"].append(cached_timings[k][method][i])
                        metrics["selected_columns"].append(sel_cols)

                    method_results[method] = metrics

            # Oracle baseline: OLS on true non-zero beta columns
            method_results["Full"] = _run_full_ols(
                X_train_list, X_test_list, y_train_list, y_test_list,
                base, data_folder, reps
            )

            seed_results[k] = method_results

        all_results[seed] = seed_results
        save_results(
            str(out_path / f"{save_name}_seed_{seed}.pkl"),
            seed_results
        )
        print(f"  Saved seed {seed}.")

    return all_results


def _run_full_ols(X_train_list, X_test_list, y_train_list, y_test_list,
                  base, data_folder, reps):
    """Oracle baseline: OLS on true non-zero beta columns."""
    metrics = {"rmse_test": [], "rmse_train": [],
               "time_model": [], "time_reduction": [], "time_scores": []}

    for i in range(reps):
        # Load true beta and get support
        beta = pd.read_csv(
            f"{base}/{data_folder}/beta{i + 1}.csv"
        ).to_numpy().ravel()
        support = np.where(beta != 0)[0]

        # Fit on true support (time_model = internal time_fit)
        result = fit_ols(
            X_train_list[i][:, support], y_train_list[i],
            X_test_list[i], y_test_list[i],
            selected_columns=support
        )
        metrics["rmse_test"].append(result["rmse_test"])
        metrics["rmse_train"].append(result["rmse_train"])
        metrics["time_model"].append(result["time_fit"])
        metrics["time_reduction"].append(0.0)
        metrics["time_scores"].append(0.0)

    return metrics


# =============================================================================
# Logistic Regression Pipeline
# =============================================================================

def run_logistic_experiment(
    k_vector,
    base,
    data_folder,
    results_folder,
    reps=10,
    outer_reps=10,
    test_size=0.2,
    save_name="results_logistic",
    seed_from=1,
    seed_to=None
):
    """Run logistic regression sampling-variance experiment.

    Uses column-first reduction with coreset-based row reduction.

    Parameters
    ----------
    k_vector : list of int
    base : str or Path
    data_folder : str
    results_folder : str
    reps : int
    outer_reps : int
    test_size : float
    save_name : str
    seed_from : int
    seed_to : int or None

    Returns
    -------
    dict : {seed: {k: {method: {metric: [values]}}}}
    """
    base = Path(base)
    if seed_to is None:
        seed_to = seed_from + outer_reps

    # Load data (binary response)
    print(f"Loading data from {data_folder}...")
    X_list, y_list = load_data(base, data_folder, reps, response="binary")
    print(f"  Shape: {X_list[0].shape}, {reps} replications.")

    # Run per-seed experiments
    all_results = {}
    out_path = base / results_folder
    os.makedirs(out_path, exist_ok=True)

    for seed in range(seed_from, seed_to):
        print(f"\nSeed {seed}...")
        np.random.seed(seed + 42)

        # Split all replications for this seed
        X_train_list, X_test_list = [], []
        y_train_list, y_test_list = [], []
        for i in range(reps):
            X_tr, X_te, y_tr, y_te = numpy_train_test_split(
                X_list[i], y_list[i],
                test_size=test_size, seed=seed
            )
            X_train_list.append(X_tr)
            X_test_list.append(X_te)
            y_train_list.append(y_tr)
            y_test_list.append(y_te)

        # Compute scores on TRAINING data for this seed
        print(f"  Computing scores on training data...")
        cached_scores, cached_timings = compute_scores_cached(
            X_train_list, y_train_list, k_vector
        )

        seed_results = {}

        for k in k_vector:
            method_results = {}

            # Phase 1: Compute all reductions and identify abort reps
            reductions_log = {}
            abort_reps_log = set()

            for method in ("LS", "CLS", "RS", "CS"):
                reductions_log[method] = {}
                for i in range(reps):
                    scores_i = cached_scores[k][method][i]

                    # --- Reduction (column-first + coreset) ---
                    t0 = time.perf_counter()
                    col_res = column_reduction(X_train_list[i], scores_i, k)
                    C = col_res["C"]
                    sel_cols = col_res["selected_columns"]

                    mu = estimate_mu(C, y_train_list[i])
                    row_res = row_reduction_coreset(C, y_train_list[i], mu, k)
                    X_red = row_res["R"]
                    y_red = row_res["y"]
                    time_red = time.perf_counter() - t0

                    reductions_log[method][i] = {
                        "X_red": X_red, "y_red": y_red,
                        "sel_cols": sel_cols, "time_red": time_red
                    }

                    if X_red.shape[1] > X_red.shape[0]:
                        abort_reps_log.add(i)

            if abort_reps_log:
                print(f"  [INFO] k={k}: Reps {sorted(abort_reps_log)} aborted "
                      f"(underdetermined for >=1 method). Skipping ALL methods.")

            # Phase 2: Modeling (skip abort_reps for ALL methods)
            for method in ("LS", "CLS", "RS", "CS"):
                metrics = {"brier_test": [], "ce_test": [],
                           "time_model": [], "time_reduction": [],
                           "time_scores": [], "selected_columns": []}

                for i in range(reps):
                    red = reductions_log[method][i]
                    X_red = red["X_red"]
                    y_red = red["y_red"]
                    sel_cols = red["sel_cols"]
                    time_red = red["time_red"]

                    if i in abort_reps_log:
                        metrics["brier_test"].append(np.nan)
                        metrics["ce_test"].append(np.nan)
                        metrics["time_model"].append(np.nan)
                        metrics["time_reduction"].append(time_red)
                        metrics["time_scores"].append(cached_timings[k][method][i])
                        metrics["selected_columns"].append(sel_cols)
                        continue

                    # Modeling (time_model = only model.fit via internal timing)
                    result = fit_logistic(X_red, y_red, X_test_list[i], y_test_list[i],
                                          selected_columns=sel_cols)

                    metrics["brier_test"].append(result["brier_test"])
                    metrics["ce_test"].append(result["ce_test"])
                    metrics["time_model"].append(result["time_fit"])
                    metrics["time_reduction"].append(time_red)
                    metrics["time_scores"].append(cached_timings[k][method][i])
                    metrics["selected_columns"].append(sel_cols)

                method_results[method] = metrics

            # Oracle baseline: Logistic on true non-zero beta columns
            method_results["Full"] = _run_full_logistic(
                X_train_list, X_test_list, y_train_list, y_test_list,
                base, data_folder, reps
            )

            seed_results[k] = method_results

        all_results[seed] = seed_results
        save_results(
            str(out_path / f"{save_name}_seed_{seed}.pkl"),
            seed_results
        )
        print(f"  Saved seed {seed}.")

    return all_results


def _run_full_logistic(X_train_list, X_test_list, y_train_list, y_test_list,
                       base, data_folder, reps):
    """Oracle baseline: Logistic on true non-zero beta columns."""
    metrics = {"brier_test": [], "ce_test": [],
               "time_model": [], "time_reduction": [], "time_scores": []}

    for i in range(reps):
        beta = pd.read_csv(
            f"{base}/{data_folder}/beta{i + 1}.csv"
        ).to_numpy().ravel()
        support = np.where(beta != 0)[0]

        # time_model = internal time_fit
        result = fit_logistic(
            X_train_list[i][:, support], y_train_list[i],
            X_test_list[i], y_test_list[i],
            selected_columns=support
        )
        metrics["brier_test"].append(result["brier_test"])
        metrics["ce_test"].append(result["ce_test"])
        metrics["time_model"].append(result["time_fit"])
        metrics["time_reduction"].append(0.0)
        metrics["time_scores"].append(0.0)

    return metrics


# =============================================================================
# Lasso Pipeline
# =============================================================================

def _init_lasso_metrics():
    """Initialize empty metrics dict for a Lasso variant."""
    return {"rmse_test": [], "rmse_train": [],
            "n_features": [], "alpha": [],
            "time_model": [], "time_fit": [], "time_alpha_search": [],
            "time_reduction": [], "time_scores": [],
            "selected_columns": [], "coef": []}


def run_lasso_experiment(
    k_vector,
    base,
    data_folder,
    results_folder,
    reps=10,
    outer_reps=10,
    test_size=0.2,
    save_name="results_lasso",
    seed_from=1,
    seed_to=None
):
    """Run Lasso sampling-variance experiment with all 3 variants.

    Runs three Lasso variants simultaneously per (seed, k, rep):
    1. Lasso_theo: Sketch -> Lasso (alpha=1/sqrt(2k)) on FULL sketched matrix
    2. Lasso_binary: Sketch -> Lasso (binary search alpha) on FULL sketched matrix
    3. CUR+Lasso: Sketch -> Column Reduction (per score method) -> Lasso

    Parameters
    ----------
    k_vector : list of int
    base : str or Path
    data_folder : str
    results_folder : str
    reps : int
    outer_reps : int
    test_size : float
    sketch_method : {"gaussian", "rademacher"}
    save_name : str
    seed_from : int
    seed_to : int or None

    Returns
    -------
    dict : {seed: {k: {method: {metric: [values]}}}}
        Methods include: "Lasso_theo", "Lasso_binary",
        "Lasso_CLS", "Full"
    """
    base = Path(base)
    if seed_to is None:
        seed_to = seed_from + outer_reps

    # Load data
    print(f"Loading data from {data_folder}...")
    X_list, y_list = load_data(base, data_folder, reps, response="continuous")
    print(f"  Shape: {X_list[0].shape}, {reps} replications.")

    # Run per-seed experiments
    all_results = {}
    out_path = base / results_folder
    os.makedirs(out_path, exist_ok=True)

    for seed in range(seed_from, seed_to):
        print(f"\nSeed {seed}...")
        np.random.seed(seed + 42)

        # Split all replications for this seed
        X_train_list, X_test_list = [], []
        y_train_list, y_test_list = [], []
        for i in range(reps):
            X_tr, X_te, y_tr, y_te = numpy_train_test_split(
                X_list[i], y_list[i],
                test_size=test_size, seed=seed
            )
            X_train_list.append(X_tr)
            X_test_list.append(X_te)
            y_train_list.append(y_tr)
            y_test_list.append(y_te)

        # Compute scores on TRAINING data for this seed
        print(f"  Computing scores on training data...")
        cached_scores, cached_timings = compute_scores_cached(
            X_train_list, y_train_list, k_vector
        )

        seed_results = {}

        for k in k_vector:
            method_results = {}

            # Initialize metrics for all 3 variants
            metrics_theo = _init_lasso_metrics()
            metrics_binary = _init_lasso_metrics()
            metrics_cls = _init_lasso_metrics()

            for i in range(reps):
                # --- Sketch (shared across all 3 variants) ---
                t0 = time.perf_counter()
                R, y_sk = row_reduction_sketch(
                    X_train_list[i], y_train_list[i], k
                )
                time_sketch = time.perf_counter() - t0

                # --- Variant 1: Lasso_theo on full sketched matrix ---
                res_theo = fit_lasso(
                    R, y_sk, X_test_list[i], y_test_list[i],
                    k=k, mode="theoretical"
                )

                metrics_theo["rmse_test"].append(res_theo["rmse_test"])
                metrics_theo["rmse_train"].append(res_theo["rmse_train"])
                metrics_theo["n_features"].append(res_theo["n_features"])
                metrics_theo["alpha"].append(res_theo["alpha"])
                metrics_theo["time_model"].append(res_theo["time_fit"])
                metrics_theo["time_fit"].append(res_theo["time_fit"])
                metrics_theo["time_alpha_search"].append(res_theo["time_alpha_search"])
                metrics_theo["time_reduction"].append(time_sketch + res_theo["time_alpha_search"])
                metrics_theo["time_scores"].append(0.0)
                metrics_theo["selected_columns"].append(None)
                metrics_theo["coef"].append(res_theo["coef"])

                # --- Variant 2: Lasso_binary on full sketched matrix ---
                res_binary = fit_lasso(
                    R, y_sk, X_test_list[i], y_test_list[i],
                    k=k, mode="binary_search"
                )

                metrics_binary["rmse_test"].append(res_binary["rmse_test"])
                metrics_binary["rmse_train"].append(res_binary["rmse_train"])
                metrics_binary["n_features"].append(res_binary["n_features"])
                metrics_binary["alpha"].append(res_binary["alpha"])
                metrics_binary["time_model"].append(res_binary["time_fit"])
                metrics_binary["time_fit"].append(res_binary["time_fit"])
                metrics_binary["time_alpha_search"].append(res_binary["time_alpha_search"])
                metrics_binary["time_reduction"].append(time_sketch + res_binary["time_alpha_search"])
                metrics_binary["time_scores"].append(0.0)
                metrics_binary["selected_columns"].append(None)
                metrics_binary["coef"].append(res_binary["coef"])

                # --- Variant 3: CLS Column Reduction + Lasso ---
                # Own CLS scores with rank_reduce=False (full rank)
                t0_scores = time.perf_counter()
                scores_cls = np.abs(get_cross_leverage_scores(
                    X_train_list[i], y_train_list[i], k, rank_reduce=False
                ))
                time_cls_scores = time.perf_counter() - t0_scores

                t0 = time.perf_counter()
                col_res = column_reduction(R, scores_cls, k)
                time_col = time.perf_counter() - t0

                X_red = col_res["C"]
                sel_cols = col_res["selected_columns"]

                # No soft abort needed: Lasso handles p > n natively via L1 regularization
                res_cls = fit_lasso(
                    X_red, y_sk, X_test_list[i], y_test_list[i],
                    k=k, mode="binary_search",
                    selected_columns=sel_cols
                )

                metrics_cls["rmse_test"].append(res_cls["rmse_test"])
                metrics_cls["rmse_train"].append(res_cls["rmse_train"])
                metrics_cls["n_features"].append(res_cls["n_features"])
                metrics_cls["alpha"].append(res_cls["alpha"])
                metrics_cls["time_model"].append(res_cls["time_fit"])
                metrics_cls["time_fit"].append(res_cls["time_fit"])
                metrics_cls["time_alpha_search"].append(res_cls["time_alpha_search"])
                metrics_cls["time_reduction"].append(time_sketch + time_col + res_cls["time_alpha_search"])
                metrics_cls["time_scores"].append(time_cls_scores)
                metrics_cls["selected_columns"].append(sel_cols)
                metrics_cls["coef"].append(res_cls["coef"])

            # Store all variants
            method_results["Lasso_theo"] = metrics_theo
            method_results["Lasso_binary"] = metrics_binary
            method_results["Lasso_CLS"] = metrics_cls

            # Shared benchmark: OLS on true support (same as Pipeline 1/2/3 Full)
            method_results["Full"] = _run_full_ols_for_lasso(
                X_train_list, X_test_list, y_train_list, y_test_list,
                base, data_folder, reps
            )

            seed_results[k] = method_results

        all_results[seed] = seed_results
        save_results(
            str(out_path / f"{save_name}_seed_{seed}.pkl"),
            seed_results
        )
        print(f"  Saved seed {seed}.")

    return all_results


def _run_full_ols_for_lasso(X_train_list, X_test_list, y_train_list, y_test_list,
                            base, data_folder, reps):
    """Shared benchmark: OLS on true non-zero beta columns.

    Same as _run_full_ols but returns metrics in Lasso-compatible format
    so Pipeline 5 results can be directly compared against Pipeline 3.
    """
    metrics = _init_lasso_metrics()

    for i in range(reps):
        beta = pd.read_csv(
            f"{base}/{data_folder}/beta{i + 1}.csv"
        ).to_numpy().ravel()
        support = np.where(beta != 0)[0]

        result = fit_ols(
            X_train_list[i][:, support], y_train_list[i],
            X_test_list[i], y_test_list[i],
            selected_columns=support
        )
        metrics["rmse_test"].append(result["rmse_test"])
        metrics["rmse_train"].append(result["rmse_train"])
        metrics["n_features"].append(len(support))
        metrics["alpha"].append(0.0)
        metrics["time_model"].append(result["time_fit"])
        metrics["time_fit"].append(result["time_fit"])
        metrics["time_alpha_search"].append(0.0)
        metrics["time_reduction"].append(0.0)
        metrics["time_scores"].append(0.0)
        metrics["selected_columns"].append(support.tolist())
        metrics["coef"].append(result["coef"])

    return metrics
