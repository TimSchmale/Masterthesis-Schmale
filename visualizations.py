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
        elif metric == "time_scores":
            loss_dict = results_dict[k]["time_scores"]
        elif metric == "time_model":
            loss_dict = results_dict[k]["time_model"]
        else:
            raise ValueError("metric must be 'brier', 'rmse', 'time_model' or 'time_scores'")

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
        "time_scores": "Score Calculation Time",
        "time_model": "Modeling Time"
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

def plot_facets_all_seeds(results, k_vector, dataset, metric="rmse", aggregate="raw", save_path=None):
    """
    Facet-based boxplots aggregated across seeds, replications, methods, and k.

    Supported metrics:
        - "rmse"  -> results[seed][k]["loss"][method]["raw"]
        - "time"  -> results[seed][k]["time_scores"][method]
        - "brier" -> results[seed][k]["brier"][method]["raw"] (if present)

    aggregate:
        - "raw"   -> plot all replication values
        - "mean"  -> plot mean per (seed, k, method)
        - "median"-> plot median per (seed, k, method)
    """

    # filter k_vector to only those present in results
    valid_k = []
    for k in k_vector:
        ok = True
        for seed in results.keys():
            for method in results[seed][k]["loss"].keys():
                vals = results[seed][k]["loss"][method]["raw"]
                if any(np.isnan(vals)):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            valid_k.append(k)

    k_vector = valid_k

    rows = []

    for seed in results.keys():
        for k in k_vector:

            # select metric dictionary
            if metric == "rmse":
                metric_dict = results[seed][k]["loss"]
            elif metric == "time_scores":
                metric_dict = results[seed][k]["time_scores"]
            elif metric == "time_model":
                metric_dict = results[seed][k]["time_model"]
            elif metric == "brier":
                metric_dict = results[seed][k]["brier"]
            else:
                raise ValueError("metric must be 'rmse', 'time_scores', or 'brier'.")

            for method, entry in metric_dict.items():

                # extract values
                if metric in ["rmse", "brier", "time_scores", "time_model"]:
                    values = entry["raw"]

                # aggregate
                if aggregate == "raw":
                    for rep_idx, val in enumerate(values):
                        rows.append({
                            "Seed": seed,
                            "k": k,
                            "Method": method,
                            "Replication": rep_idx + 1,
                            "Loss": val
                        })

                elif aggregate == "mean":
                    rows.append({
                        "Seed": seed,
                        "k": k,
                        "Method": method,
                        "Replication": None,
                        "Loss": float(np.mean(values))
                    })

                elif aggregate == "median":
                    rows.append({
                        "Seed": seed,
                        "k": k,
                        "Method": method,
                        "Replication": None,
                        "Loss": float(np.median(values))
                    })

                else:
                    raise ValueError("aggregate must be 'raw', 'mean', or 'median'.")

    df = pd.DataFrame(rows)
    df["k"] = df["k"].astype("category")

    # Full model only exists for rmse/brier
    if metric in ["rmse", "brier"]:
        df_no_full = df[df["Method"] != "Full"]
    else:
        df_no_full = df

    metric_name = {
        "rmse": "RMSE",
        "time_scores": "Score Computation Time (s)",
        "brier": "Brier Score",
        "time_model": "Modeling Time (s)",
    }[metric]

    # facet by method
    p_method = (
        ggplot(df_no_full, aes(x="k", y="Loss", fill="Method"))
        + geom_boxplot()
        + facet_wrap("~ Method", scales="fixed")
        + theme_bw()
        + theme(figure_size=(18, 20))
        + labs(
            title=f"{metric_name} per Method across k (aggregate={aggregate})",
            x="k",
            y=metric_name
        )
    )

    # facet by k
    p_k = (
        ggplot(df, aes(x="Method", y="Loss", fill="Method"))
        + geom_boxplot()
        + facet_wrap("~ k", scales="fixed")
        + theme_bw()
        + theme(figure_size=(18, 20))
        + labs(
            title=f"{metric_name} per k across Methods (aggregate={aggregate})",
            x="Method",
            y=metric_name
        )
    )

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        p_method.save(os.path.join(save_path, f"facet_method_{metric}_{dataset}_{aggregate}.pdf"))
        p_k.save(os.path.join(save_path, f"facet_k_{metric}_{dataset}_{aggregate}.pdf"))
        print("Plots saved.")

    display(p_method)
    display(p_k)

def extract_dimensions(entry):
    """
    Extract selected column and row counts from a results entry.
    Works for both Row→Column and Column→Row pipelines.
    """

    sel_cols = {}
    sel_rows = {}

    # --- Columns ---
    C = entry.get("selected_columns", None)
    if isinstance(C, dict):
        for method, rep_lists in C.items():
            if isinstance(rep_lists, list):
                sel_cols[method] = [
                    len(cols) if isinstance(cols, list) else np.nan
                    for cols in rep_lists
                ]

    # --- Rows ---
    R = entry.get("selected_rows", None)
    if isinstance(R, dict):
        for method, rep_lists in R.items():
            if isinstance(rep_lists, list):
                sel_rows[method] = [
                    len(rows) if isinstance(rows, list) else np.nan
                    for rows in rep_lists
                ]

    if len(sel_cols) == 0:
        return None, None

    return sel_cols, sel_rows
def plot_dimension_facets(results, k_vector, ncol=2):
    """
    Faceted visualization of selected column and row counts across k and methods.
    Compatible with both Row→Column and Column→Row pipelines.
    """

    rows = []

    for seed in results.keys():
        for k in k_vector:

            if k not in results[seed]:
                print(f"[WARN] seed={seed}, k={k} fehlt – übersprungen.")
                continue

            entry = results[seed][k]
            sel_cols, sel_rows = extract_dimensions(entry)

            if sel_cols is None:
                print(f"[WARN] seed={seed}, k={k} hat keine gültigen selected_columns – übersprungen.")
                continue

            for method, col_counts in sel_cols.items():

                # row counts fallback
                if sel_rows is not None and method in sel_rows:
                    row_counts = sel_rows[method]
                else:
                    row_counts = [np.nan] * len(col_counts)

                for rep_idx, (cc, rc) in enumerate(zip(col_counts, row_counts), start=1):
                    rows.append({
                        "Seed": seed,
                        "k": k,
                        "k_str": str(k),
                        "Method": method,
                        "Replication": rep_idx,
                        "SelectedColumns": cc,
                        "SelectedRows": rc
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Keine gültigen Dimensionen gefunden.")

    # numerisch sortieren
    k_sorted = sorted(df["k"].unique())  # echte numerische Sortierung
    k_labels = [str(x) for x in k_sorted]  # Labels als Strings

    df["k_str"] = pd.Categorical(df["k_str"], categories=k_labels, ordered=True)

    # Median lines
    med_cols = df.groupby(["Method", "k_str"])["SelectedColumns"].median().reset_index()
    med_rows = df.groupby(["Method", "k_str"])["SelectedRows"].median().reset_index()

    # --- Columns ---
    p_cols = (
        ggplot(df, aes(x="k_str", y="SelectedColumns"))
        + geom_boxplot(aes(fill="Method"))
        + facet_wrap("~Method", ncol=ncol)
        + theme_minimal()
        + labs(title="Selected Columns per k and Method", x="k", y="Columns")
    )

    # --- Rows ---
    p_rows = (
        ggplot(df, aes(x="k_str", y="SelectedRows"))
        + geom_boxplot(aes(fill="Method"))
        + facet_wrap("~Method", ncol=ncol)
        + theme_minimal()
        + labs(title="Selected Rows per k and Method", x="k", y="Rows")
    )

    display(p_cols)
    display(p_rows)
