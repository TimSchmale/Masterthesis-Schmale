import pandas as pd
import numpy as np
from IPython.display import display
from plotnine import (
    ggplot, aes, geom_boxplot, geom_line, geom_point, facet_wrap,
    labs, theme_minimal, theme, scale_x_discrete, element_text, scale_y_continuous, geom_histogram, theme_bw, scale_y_log10
)
import os
def visualize_distributions(scores):
    """
    Visualize score distributions across replications for each method.

    For each method, the score arrays are converted into long format and
    displayed as faceted histograms, one facet per replication.
    """

    # iterate over methods and their score lists
    for method, score_list in scores.items():

        # collect long-format rows for this method
        rows = []
        for rep_idx, arr in enumerate(score_list, start=1):
            arr = np.asarray(arr).reshape(-1)
            for v in arr:
                rows.append({
                    "value": v,
                    "Replication": rep_idx
                })

        # build DataFrame for plotting
        df_method = pd.DataFrame(rows)

        # create histogram faceted by replication
        p = (
            ggplot(df_method, aes(x="value"))
            + geom_histogram(bins=80, fill="#4C72B0", alpha=0.7)
            + facet_wrap("~Replication", ncol=5)
            + theme_bw()
            + labs(
                title=f"{method} Score Distribution by Replication",
                x="Score",
                y="Count"
            )
        )

        # display plot
        display(p)

def plot_facets_all(results_dict, k_vector, dataset, metric="brier", save_path=None):
    """
    Create facet-based boxplots for loss metrics across k and methods.

    The function aggregates loss values across all k, reshapes the data
    into long format, and produces two facet plots:
    (1) faceted by method with k on the x-axis,
    (2) faceted by k with method on the x-axis.
    """

    # collect loss data across all k values
    rows = []
    for k in k_vector:

        # select appropriate metric dictionary
        if metric == "brier":
            loss_dict = results_dict[k]["brierloss"]
        elif metric == "rmse":
            loss_dict = results_dict[k]["rmse"]
        elif metric == "time":
            loss_dict = results_dict[k]["time_scores"]
        else:
            raise ValueError("metric must be 'brier', 'rmse' or 'time'")

        # determine number of replications
        n_reps = len(next(iter(loss_dict.values())))

        # build DataFrame for this k
        df = pd.DataFrame(loss_dict)
        df["Replication"] = np.arange(1, n_reps + 1)
        df["k"] = k
        rows.append(df)

    # concatenate all k-level DataFrames
    loss_all = pd.concat(rows, ignore_index=True)

    # reshape to long format
    loss_long = loss_all.melt(
        id_vars=["Replication", "k"],
        var_name="Method",
        value_name="Loss"
    )

    # convert k to categorical for discrete x-axis
    loss_long["k"] = loss_long["k"].astype("category")

    # map metric to readable name
    metric_name = {
        "brier": "Brier Score",
        "rmse": "RMSE",
        "time": "Score Calculation Time"
    }[metric]

    # remove Full model for method-facet plot
    loss_long_no_full = loss_long[loss_long["Method"] != "Full"]

    # create facet plot by method
    p_method = (
        ggplot(loss_long_no_full, aes(x="k", y="Loss", fill="Method"))
        + geom_boxplot()
        + facet_wrap("~ Method", scales="fixed", ncol=2)
        + theme_bw()
        + theme(figure_size=(18, 20))
        + labs(
            title=f"{metric_name} per Method across k",
            x="k",
            y=metric_name
        )
    )

    # create facet plot by k
    p_k = (
        ggplot(loss_long, aes(x="Method", y="Loss", fill="Method"))
        + geom_boxplot()
        + facet_wrap("~ k", scales="fixed", ncol=3)
        + theme_bw()
        + theme(figure_size=(18, 20))
        + labs(
            title=f"{metric_name} per k across Methods",
            x="Method",
            y=metric_name
        )
    )

    # optionally save plots as PDF
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

        file_method = os.path.join(save_path, f"facet_method_{metric}_{dataset}.pdf")
        file_k = os.path.join(save_path, f"facet_k_{metric}_{dataset}.pdf")

        p_method.save(file_method)
        p_k.save(file_k)

        print(f"Saved:\n - {file_method}\n - {file_k}")

    # display both plots
    display(p_method)
    display(p_k)

def _derive_counts_from_entry(entry):
    """
    Extract selected column and row counts from a results entry.

    The function supports both explicit count fields and raw lists of
    selected indices, falling back to the latter if counts are missing.
    """

    # extract column counts if available
    sel_cols = entry.get("selected_columns_counts", None)

    # extract row counts if available
    sel_rows = entry.get("selected_rows_counts", None)

    # fallback: use raw selected column indices
    if sel_cols is None and "selected_columns" in entry:
        sel_cols = entry["selected_columns"]

    # fallback: use raw selected row indices
    if sel_rows is None and "selected_rows" in entry:
        sel_rows = entry["selected_rows"]

    return sel_cols, sel_rows

def plot_dimension_comparison(results_dict, k_vector, show_median_lines=True, ncol=2):
    """
    Visualize selected column and row counts across k and methods.

    The function extracts selection counts, reshapes them into long format,
    and produces boxplots faceted by method. Optional median lines can be
    added for clearer trend visualization.
    """

    # collect long-format rows for all k values
    rows = []
    for k in k_vector:

        # ensure k exists in results
        if k not in results_dict:
            raise KeyError(f"k={k} nicht in results_dict vorhanden.")

        # extract counts for this k
        entry = results_dict[k]
        sel_cols, sel_rows = _derive_counts_from_entry(entry)

        if sel_cols is None:
            raise ValueError(f"Keine selected_columns_counts oder C für k={k} gefunden.")

        # iterate over methods and replications
        for method, col_counts in sel_cols.items():

            # extract row counts if available
            row_counts = sel_rows.get(method, [np.nan] * len(col_counts)) if sel_rows is not None else [np.nan] * len(col_counts)

            # build long-format rows
            for rep_idx, (cc, rc) in enumerate(zip(col_counts, row_counts), start=1):
                rows.append({
                    "k": int(k),
                    "k_str": str(k),
                    "Method": method,
                    "Replication": rep_idx,
                    "SelectedColumns": int(cc) if not np.isnan(cc) else np.nan,
                    "SelectedRows": int(rc) if not np.isnan(rc) else np.nan
                })

    # build DataFrame
    df_counts = pd.DataFrame(rows)
    if df_counts.empty:
        raise ValueError("Keine Counts extrahiert. Prüfe results_dict Struktur.")

    # enforce categorical ordering for k
    k_order = sorted(df_counts["k"].unique())
    k_labels = [str(x) for x in k_order]
    df_counts["k_str"] = pd.Categorical(df_counts["k_str"], categories=k_labels, ordered=True)

    # compute median lines for columns
    med_cols = (
        df_counts
        .groupby(["Method", "k_str"], observed=True)["SelectedColumns"]
        .median()
        .reset_index()
        .rename(columns={"SelectedColumns": "MedianSelectedColumns"})
    )

    # compute median lines for rows
    med_rows = (
        df_counts
        .groupby(["Method", "k_str"], observed=True)["SelectedRows"]
        .median()
        .reset_index()
        .rename(columns={"SelectedRows": "MedianSelectedRows"})
    )

    # create boxplot for selected columns
    p_cols = (
        ggplot(df_counts, aes(x="k_str", y="SelectedColumns"))
        + geom_boxplot(aes(fill="Method"))
        + facet_wrap("~Method", ncol=ncol, scales="fixed")
        + labs(title="Selected Columns per k and Method", x="k", y="Anzahl ausgewählte Spalten")
        + theme_minimal()
        + theme(axis_text_x=element_text(rotation=45, hjust=1))
    )

    # add median lines if enabled
    if show_median_lines:
        p_cols = p_cols + geom_line(
            data=med_cols, mapping=aes(x="k_str", y="MedianSelectedColumns", group=1),
            color="orange", size=1
        ) + geom_point(
            data=med_cols, mapping=aes(x="k_str", y="MedianSelectedColumns"),
            color="orange", size=2
        )

    # create boxplot for selected rows
    p_rows = (
        ggplot(df_counts, aes(x="k_str", y="SelectedRows"))
        + geom_boxplot(aes(fill="Method"))
        + facet_wrap("~Method", ncol=ncol, scales="fixed")
        + labs(title="Selected Rows per k and Method", x="k", y="Anzahl ausgewählte Zeilen")
        + theme_minimal()
        + theme(axis_text_x=element_text(rotation=45, hjust=1))
    )

    # add median lines if enabled
    if show_median_lines:
        p_rows = p_rows + geom_line(
            data=med_rows, mapping=aes(x="k_str", y="MedianSelectedRows", group=1),
            color="darkgreen", size=1
        ) + geom_point(
            data=med_rows, mapping=aes(x="k_str", y="MedianSelectedRows"),
            color="darkgreen", size=2
        )

    # display both plots
    display(p_cols)
    display(p_rows)