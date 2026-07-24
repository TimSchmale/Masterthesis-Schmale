"""Visualizations for CUR reduction experiments.

All plots use plotnine with the standard layout:
- x-axis: "Full" first, then "Method | k" (method alphabetical, k numerical)
- All boxplots side by side (no facets)
- theme_bw, figure_size=(25, 16), legend bottom, x-labels 90° rotated
"""

import os
import numpy as np
import pandas as pd
from IPython.display import display
from plotnine import (
    ggplot, aes,
    geom_boxplot, geom_histogram,
    facet_wrap, theme_bw, theme_minimal, theme, element_text,
    labs
)


# =============================================================================
# Shared Helpers
# =============================================================================

def _build_x_order(df):
    """Build ordered x_label column: Full first, then Method | k."""
    df = df.copy()
    df["k"] = df["k"].astype(int)

    df_full = df[df["Method"] == "Full"].copy()
    df_rest = df[df["Method"] != "Full"].copy()

    if len(df_full) > 0:
        df_full["x_label"] = "Full"

    df_rest["x_label"] = df_rest["Method"] + " | " + df_rest["k"].astype(str)

    # Build category order
    combos = df_rest[["Method", "k"]].drop_duplicates()
    method_sorted = sorted(combos["Method"].unique())
    k_sorted = sorted(combos["k"].unique())

    x_order = []
    if len(df_full) > 0:
        x_order.append("Full")

    for m in method_sorted:
        for k in k_sorted:
            if ((combos["Method"] == m) & (combos["k"] == k)).any():
                x_order.append(f"{m} | {k}")

    df_plot = pd.concat([df_full, df_rest], ignore_index=True)
    df_plot["x_label"] = pd.Categorical(
        df_plot["x_label"], categories=x_order, ordered=True
    )
    df_plot = df_plot.sort_values("x_label")

    return df_plot


def _standard_theme():
    """Standard plot theme matching original format."""
    return (
        theme_minimal()
        + theme(
            figure_size=(25, 16),
            axis_text_x=element_text(rotation=90, ha="center", size=16),
            axis_text_y=element_text(size=16),
            axis_title_x=element_text(size=20),
            axis_title_y=element_text(size=16),
            legend_position="bottom",
            legend_text=element_text(size=18),
            legend_title=element_text(size=20),
        )
    )


def _save_plot(p, save_path, filename):
    """Save plot as PDF if save_path is provided."""
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, filename)
        p.save(file_path, limitsize=False)
        print(f"Saved: {file_path}")


# =============================================================================
# Loss / Performance Plots
# =============================================================================

def plot_loss_boxplots(
    results, k_vector, metric="rmse_test",
    pipeline="col_row", dataset="p1000",
    aggregate="raw", save_path=None
):
    """Boxplots of loss metrics across methods and k values.

    Parameters
    ----------
    results : dict
        Pipeline output: {seed: {k: {method: {metric: [values]}}}}.
    k_vector : list of int
        Target ranks to include.
    metric : str
        Key in the metrics dict: "rmse_test", "rmse_train",
        "brier_test", "ce_test", "time_model", "time_fit",
        "time_reduction", "time_scores".
    pipeline : str
        Pipeline name for filename.
    dataset : str
        Dataset name for filename.
    aggregate : {"raw", "mean", "median"}
        How to aggregate across replications per seed.
    save_path : str or None
        Directory for PDF output.
    """
    rows = []

    # Filter valid k values (present in all seeds)
    valid_k = [k for k in k_vector
               if all(k in results[s] for s in results)]

    for seed in results:
        for k in valid_k:
            for method, metrics in results[seed][k].items():

                # Compute time_total on the fly
                if metric == "time_total":
                    t_scores = np.asarray(metrics.get("time_scores", [0.0] * len(next(iter(metrics.values())))))
                    t_reduction = np.asarray(metrics.get("time_reduction", [0.0] * len(t_scores)))
                    # Lasso uses time_fit, OLS/Logistic uses time_model
                    if "time_fit" in metrics:
                        t_model = np.asarray(metrics["time_fit"])
                    elif "time_model" in metrics:
                        t_model = np.asarray(metrics["time_model"])
                    else:
                        continue
                    values = t_scores + t_reduction + t_model
                elif metric not in metrics:
                    continue
                else:
                    values = np.asarray(metrics[metric])

                # Skip all-NaN
                if np.all(np.isnan(values)):
                    continue

                if aggregate == "raw":
                    for rep_idx, val in enumerate(values):
                        if np.isnan(val):
                            continue
                        rows.append({
                            "Seed": seed, "k": k, "Method": method,
                            "Replication": rep_idx + 1, "Loss": val
                        })
                elif aggregate == "mean":
                    rows.append({
                        "Seed": seed, "k": k, "Method": method,
                        "Replication": None, "Loss": float(np.nanmean(values))
                    })
                elif aggregate == "median":
                    rows.append({
                        "Seed": seed, "k": k, "Method": method,
                        "Replication": None, "Loss": float(np.nanmedian(values))
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        print("No data to plot.")
        return

    df_plot = _build_x_order(df)

    # Metric display name
    metric_names = {
        "rmse_test": "RMSE (Test)",
        "rmse_train": "RMSE (Train)",
        "brier_test": "Brier Score (Test)",
        "ce_test": "Cross-Entropy (Test)",
        "time_model": "Modeling Time (s)",
        "time_fit": "Fit Time (s)",
        "time_reduction": "Reduction Time (s)",
        "time_scores": "Score Computation Time (s)",
        "time_alpha_search": "Alpha Search Time (s)",
    }
    y_label = metric_names.get(metric, metric)

    p = (
        ggplot(df_plot, aes(x="x_label", y="Loss", fill="Method"))
        + geom_boxplot(width=0.4)
        + _standard_theme()
        + labs(x="Method | k", y=y_label)
    )

    _save_plot(p, save_path, f"results_{pipeline}_{metric}_{dataset}.pdf")
    display(p)


# =============================================================================
# Screening Plots
# =============================================================================

def plot_screening_boxplots(
    screening_df, pipeline="col_row", dataset="p1000",
    save_path=None
):
    """Combined boxplots for all screening metrics (TPR, FDR, MCC).

    Three rows of boxplots in one plot, faceted by metric.
    x-axis: Method | k (same format as loss plots).

    Parameters
    ----------
    screening_df : pd.DataFrame
        Output from evaluate_experiment(): columns seed, k, method, rep,
        tpr, fdr, mcc, n_selected.
    pipeline : str
        Pipeline name for filename.
    dataset : str
        Dataset name for filename.
    save_path : str or None
        Directory for PDF output.
    """
    df = screening_df.rename(columns={"method": "Method"}).copy()

    # Melt to long format: one row per (seed, k, Method, rep, Metric)
    metric_names = {
        "tpr": "True Positive Rate",
        "fdr": "False Discovery Rate",
        "mcc": "Matthews Correlation Coefficient",
    }

    df_long = df.melt(
        id_vars=["seed", "k", "Method", "rep"],
        value_vars=["tpr", "fdr", "mcc"],
        var_name="Metric_key",
        value_name="Value"
    )
    df_long["Metric"] = df_long["Metric_key"].map(metric_names)

    # Enforce facet order: TPR, FDR, MCC
    facet_order = ["True Positive Rate", "False Discovery Rate", "Matthews Correlation Coefficient"]
    df_long["Metric"] = pd.Categorical(
        df_long["Metric"], categories=facet_order, ordered=True
    )

    # Build x_label (reuse logic but on the melted df)
    df_long["Loss"] = df_long["Value"]
    df_plot = _build_x_order(df_long)

    p = (
        ggplot(df_plot, aes(x="x_label", y="Loss", fill="Method"))
        + geom_boxplot(width=0.4)
        + facet_wrap("~Metric", ncol=1, scales="free_y")
        + _standard_theme()
        + labs(x="Method | k", y="")
    )

    _save_plot(p, save_path, f"results_{pipeline}_screening_{dataset}.pdf")
    display(p)


# =============================================================================
# Score Distribution Plots
# =============================================================================

def plot_score_distributions(scores_single, pipeline=None, dataset=None,
                             save_path=None):
    """Plot score distributions for a single replication.

    Parameters
    ----------
    scores_single : dict
        {method_name: score_vector} for one replication.
    pipeline, dataset : str
        Used for filename.
    save_path : str or None
    """
    rows = []
    for method, arr in scores_single.items():
        arr = np.asarray(arr).reshape(-1)
        for v in arr:
            rows.append({"Method": method, "value": v})

    df = pd.DataFrame(rows)

    p = (
        ggplot(df, aes(x="value"))
        + geom_histogram(bins=80, fill="#4C72B0", alpha=0.7)
        + facet_wrap("~Method", ncol=2, scales="free")
        + theme_minimal()
        + labs(x="Score", y="Count")
    )

    if save_path is not None:
        _save_plot(p, save_path, f"results_{pipeline}_scores_{dataset}.pdf")

    display(p)
    return p


# =============================================================================
# Dimension Plots
# =============================================================================

def plot_n_selected(screening_df, pipeline="col_row", dataset="p1000",
                    save_path=None):
    """Boxplot of number of selected features across methods and k.

    Parameters
    ----------
    screening_df : pd.DataFrame
        Output from evaluate_experiment().
    """
    df = screening_df.rename(columns={"method": "Method"}).copy()
    df["Loss"] = df["n_selected"]

    df_plot = _build_x_order(df)

    p = (
        ggplot(df_plot, aes(x="x_label", y="Loss", fill="Method"))
        + geom_boxplot(width=0.4)
        + _standard_theme()
        + labs(x="Method | k", y="Number of Selected Features")
    )

    _save_plot(p, save_path, f"results_{pipeline}_n_selected_{dataset}.pdf")
    display(p)
