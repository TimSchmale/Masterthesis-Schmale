"""Screening evaluation for CUR reduction experiments.

Computes TPR, FDR, and MCC for feature selection quality.

Works generically with the pipeline results dict structure:
    {seed: {k: {method: {metric: [values_per_rep]}}}}
"""

import numpy as np
import pandas as pd
from pathlib import Path


# =============================================================================
# Data Loading
# =============================================================================

def load_beta(base, folder, reps):
    """Load true beta vectors from simulation data.

    Parameters
    ----------
    base : str or Path
        Base directory.
    folder : str
        Subfolder containing beta*.csv files.
    reps : int
        Number of replications.

    Returns
    -------
    beta_list : list of ndarray
        True coefficient vectors.
    support_list : list of ndarray
        Indices of non-zero coefficients per replication.
    """
    base = Path(base)
    beta_list = []
    support_list = []

    for i in range(reps):
        beta_i = pd.read_csv(base / folder / f"beta{i + 1}.csv").to_numpy().ravel()
        beta_list.append(beta_i)
        support_list.append(np.where(beta_i != 0.0)[0])

    return beta_list, support_list


# =============================================================================
# Feature Selection Extraction
# =============================================================================

def get_selected_features(selected_columns, coef=None):
    """Extract the effective selected feature indices in original column space.

    Handles three cases:
    1. CUR only (OLS/Logistic): selected_columns is the answer.
    2. Lasso without CUR (Lasso_theo/binary): non-zero coef indices.
    3. Lasso with CUR (Lasso_CLS): non-zero coef mapped back to
       original space via selected_columns.

    Parameters
    ----------
    selected_columns : ndarray or None
        Column indices from CUR reduction (None if no CUR was applied).
    coef : ndarray or None
        Lasso coefficients (None for OLS/Logistic pipelines).

    Returns
    -------
    ndarray
        Indices of selected features in original column space.
    """
    if coef is not None:
        nonzero_idx = np.where(coef != 0)[0]
        if selected_columns is not None:
            # Lasso + CUR: map back to original column indices
            return np.asarray(selected_columns)[nonzero_idx]
        else:
            # Lasso without CUR: indices are already in original space
            return nonzero_idx
    else:
        # Pure CUR selection (OLS, Logistic)
        return np.asarray(selected_columns)


# =============================================================================
# Screening Metrics
# =============================================================================

def compute_confusion(selected, support, p):
    """Compute confusion matrix entries for feature selection.

    Parameters
    ----------
    selected : ndarray
        Indices of selected features.
    support : ndarray
        Indices of true non-zero features.
    p : int
        Total number of features.

    Returns
    -------
    dict : {tp, fp, fn, tn}
    """
    selected_set = set(selected)
    support_set = set(support)

    tp = len(selected_set & support_set)
    fp = len(selected_set - support_set)
    fn = len(support_set - selected_set)
    tn = p - len(selected_set | support_set)

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def compute_tpr(conf):
    """True Positive Rate (Sensitivity/Recall): TP / (TP + FN).

    What fraction of true features was correctly selected?
    """
    denom = conf["tp"] + conf["fn"]
    if denom == 0:
        return 0.0
    return conf["tp"] / denom


def compute_fdr(conf):
    """False Discovery Rate: FP / (FP + TP).

    What fraction of selected features is irrelevant?
    """
    denom = conf["fp"] + conf["tp"]
    if denom == 0:
        return 0.0
    return conf["fp"] / denom


def compute_mcc(conf):
    """Matthews Correlation Coefficient.

    Balanced measure accounting for all four confusion entries.
    Range: [-1, +1], 0 = random.
    """
    tp, fp, fn, tn = conf["tp"], conf["fp"], conf["fn"], conf["tn"]
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return 0.0
    return (tp * tn - fp * fn) / denom


# =============================================================================
# Main Evaluation Functions
# =============================================================================

def evaluate_experiment(results, base, data_folder, reps):
    """Full screening evaluation of pipeline results.

    Computes TPR, FDR, MCC for every (seed, k, method, rep).
    Works with OLS, Logistic, and Lasso pipeline outputs.

    Parameters
    ----------
    results : dict
        Pipeline output: {seed: {k: {method: {metric: [values]}}}}.
    base : str or Path
        Base directory for simulation data.
    data_folder : str
        Subfolder containing beta*.csv files.
    reps : int
        Number of replications.

    Returns
    -------
    pd.DataFrame
        Columns: seed, k, method, rep, tpr, fdr, mcc, n_selected.
    """
    # Load true betas
    beta_list, support_list = load_beta(base, data_folder, reps)

    # Total number of features
    p = len(beta_list[0])

    rows = []

    for seed, seed_data in results.items():
        for k, k_data in seed_data.items():
            for method, metrics in k_data.items():
                # Skip Full baseline (no feature selection)
                if method == "Full":
                    continue

                # Check what selection info is available
                has_sel_cols = "selected_columns" in metrics
                has_coef = "coef" in metrics

                if not has_sel_cols and not has_coef:
                    continue

                n_reps = len(metrics.get("selected_columns", metrics.get("coef", [])))

                for i in range(n_reps):
                    sel_cols = metrics["selected_columns"][i] if has_sel_cols else None
                    coef = metrics["coef"][i] if has_coef else None

                    # Skip soft-aborted reps (model not fitted → NaN in loss)
                    aborted = False
                    for m in ("rmse_test", "brier_test"):
                        if m in metrics and i < len(metrics[m]):
                            val = metrics[m][i]
                            if val is not None and np.isnan(val):
                                aborted = True
                                break
                    if aborted:
                        continue

                    # Skip if no selection info for this rep
                    if sel_cols is None and coef is None:
                        continue

                    # Get effective selection
                    selected = get_selected_features(sel_cols, coef)

                    # Compute confusion matrix and metrics
                    conf = compute_confusion(selected, support_list[i], p)

                    rows.append({
                        "seed": seed,
                        "k": k,
                        "method": method,
                        "rep": i,
                        "tpr": compute_tpr(conf),
                        "fdr": compute_fdr(conf),
                        "mcc": compute_mcc(conf),
                        "n_selected": len(selected)
                    })

    return pd.DataFrame(rows)


def evaluate_ols_experiment(results, base, data_folder, reps):
    """Screening evaluation for OLS pipeline (CUR selection only).

    Parameters
    ----------
    results : dict
        Output from run_ols_experiment().
    base, data_folder, reps : as in evaluate_experiment.

    Returns
    -------
    pd.DataFrame
        Columns: seed, k, method, rep, hit_rate, beta_share, n_selected.
    """
    return evaluate_experiment(results, base, data_folder, reps)


def evaluate_lasso_experiment(results, base, data_folder, reps):
    """Screening evaluation for Lasso pipeline.

    For Lasso_theo/binary: uses coef != 0 as implicit selection.
    For Lasso_CLS: maps Lasso's non-zero coef back through CUR columns.

    Parameters
    ----------
    results : dict
        Output from run_lasso_experiment().
    base, data_folder, reps : as in evaluate_experiment.

    Returns
    -------
    pd.DataFrame
        Columns: seed, k, method, rep, hit_rate, beta_share, n_selected.
    """
    return evaluate_experiment(results, base, data_folder, reps)


def evaluate_logistic_experiment(results, base, data_folder, reps):
    """Screening evaluation for Logistic pipeline (CUR selection only).

    Parameters
    ----------
    results : dict
        Output from run_logistic_experiment().
    base, data_folder, reps : as in evaluate_experiment.

    Returns
    -------
    pd.DataFrame
        Columns: seed, k, method, rep, hit_rate, beta_share, n_selected.
    """
    return evaluate_experiment(results, base, data_folder, reps)


# =============================================================================
# Summary Statistics
# =============================================================================

def summarize_screening(df, groupby=None):
    """Compute summary statistics for screening results.

    Parameters
    ----------
    df : pd.DataFrame
        Output from evaluate_experiment().
    groupby : list of str or None
        Columns to group by. Default: ["k", "method"].

    Returns
    -------
    pd.DataFrame
        Mean, std, median for hit_rate and beta_share.
    """
    if groupby is None:
        groupby = ["k", "method"]

    return df.groupby(groupby).agg(
        hit_rate_mean=("hit_rate", "mean"),
        hit_rate_std=("hit_rate", "std"),
        hit_rate_median=("hit_rate", "median"),
        beta_share_mean=("beta_share", "mean"),
        beta_share_std=("beta_share", "std"),
        beta_share_median=("beta_share", "median"),
        n_selected_mean=("n_selected", "mean"),
    ).reset_index()
