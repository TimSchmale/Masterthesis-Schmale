import pandas as pd
import numpy as np
from IPython.display import display
from plotnine import (
    ggplot, aes, geom_boxplot, geom_line, geom_point, facet_wrap,
    labs, theme_minimal, theme, scale_x_discrete, element_text, scale_y_continuous, geom_histogram, theme_bw
)
# ==============================================================================================
# visualize_distributions: Function to visualize scoring distributions
# ==============================================================================================

def visualize_distributions(scores):

    for method, score_list in scores.items():
        rows = []

        # long format
        for rep_idx, arr in enumerate(score_list, start=1):
            arr = np.asarray(arr).reshape(-1)
            for v in arr:
                rows.append({
                    "value": v,
                    "Replication": rep_idx
                })
        df_method = pd.DataFrame(rows)

        # plot
        p = (
                ggplot(df_method, aes(x="value"))
                + geom_histogram(bins=80, fill="#4C72B0", alpha=0.7)
                + facet_wrap("~Replication", ncol=5)  # → 10 Subplots
                + theme_bw()
                + labs(
            title=f"{method} Score Distribution by Replication",
            x="Score",
            y="Count"
        )
        )

        # print plot
        display(p)


def plot_facets_all(results_dict, k_vector, metric="brier"):

    # ============================
    # 1) Collect data über alle k
    # ============================
    rows = []

    for k in k_vector:
        if metric == "brier":
            loss_dict = results_dict[k]["brierloss"]
        elif metric == "rmse":
            loss_dict = results_dict[k]["RMSE"]
        elif metric == "time":
            loss_dict = results_dict[k]["Time"]
        else:
            raise ValueError("metric must be 'brier', 'rmse' or 'time'")

        n_reps = len(next(iter(loss_dict.values())))

        df = pd.DataFrame(loss_dict)
        df["Replication"] = np.arange(1, n_reps + 1)
        df["k"] = k

        rows.append(df)

    loss_all = pd.concat(rows, ignore_index=True)

    loss_long = loss_all.melt(
        id_vars=["Replication", "k"],
        var_name="Method",
        value_name="Loss"
    )

    loss_long["k"] = loss_long["k"].astype("category")

    metric_name = {
        "brier": "Brier Score",
        "rmse": "RMSE",
        "time": "Score Calculation Time"
    }[metric]

    # ============================
    # 2) Facet nach Methode (x = k)
    # ============================
    p_method = (
        ggplot(loss_long, aes(x="k", y="Loss", fill="Method"))
        + geom_boxplot()
        + facet_wrap("~ Method", scales="fixed", ncol=2)
        + theme_minimal()
        + theme(figure_size=(18, 20))
        + labs(
            title=f"{metric_name} per Method across k",
            x="k",
            y=metric_name
        )
    )

    # ============================
    # 3) Facet nach k (x = Methode)
    # ============================
    p_k = (
        ggplot(loss_long, aes(x="Method", y="Loss", fill="Method"))
        + geom_boxplot()
        + facet_wrap("~ k", scales="fixed", ncol=2)
        + theme_minimal()
        + theme(figure_size=(18, 20))
        + labs(
            title=f"{metric_name} per k across Methods",
            x="Method",
            y=metric_name
        )
    )

    display(p_method)
    display(p_k)

def _derive_counts_from_entry(entry):

    sel_cols = entry.get("selected_columns_counts", None)
    sel_rows = entry.get("selected_rows_counts", None)

    if sel_cols is None and "C" in entry:
        sel_cols = {}
        for key, lst in entry["C"].items():
            method = key.replace("C_", "").upper()
            sel_cols[method] = [len(d.get("selected_columns", [])) for d in lst]

    if sel_rows is None and "R" in entry:
        sel_rows = {}
        for key, lst in entry["R"].items():
            method = key.replace("R_", "").upper()
            sel_rows[method] = [len(d.get("selected_rows", [])) for d in lst]

    return sel_cols, sel_rows

def plot_dimension_comparison(results_dict, k_vector, show_median_lines=True, ncol=2):

    # 1) Counts extrahieren und in long DataFrame bringen
    rows = []
    for k in k_vector:
        if k not in results_dict:
            raise KeyError(f"k={k} nicht in results_dict vorhanden.")
        entry = results_dict[k]
        sel_cols, sel_rows = _derive_counts_from_entry(entry)
        if sel_cols is None:
            raise ValueError(f"Keine selected_columns_counts oder C für k={k} gefunden.")

        for method, col_counts in sel_cols.items():
            row_counts = sel_rows.get(method, [np.nan] * len(col_counts)) if sel_rows is not None else [np.nan] * len(col_counts)
            for rep_idx, (cc, rc) in enumerate(zip(col_counts, row_counts), start=1):
                rows.append({
                    "k": int(k),
                    "k_str": str(k),
                    "Method": method,
                    "Replication": rep_idx,
                    "SelectedColumns": int(cc) if not np.isnan(cc) else np.nan,
                    "SelectedRows": int(rc) if not np.isnan(rc) else np.nan
                })

    df_counts = pd.DataFrame(rows)
    if df_counts.empty:
        raise ValueError("Keine Counts extrahiert. Prüfe results_dict Struktur.")

    # 2) Ordnung der k-Werte als Strings (für diskrete x-Achse)
    k_order = sorted(df_counts["k"].unique())
    k_labels = [str(x) for x in k_order]
    df_counts["k_str"] = pd.Categorical(df_counts["k_str"], categories=k_labels, ordered=True)

    # 3) Median‑Daten vorbereiten (für Linien)
    med_cols = (
        df_counts
        .groupby(["Method", "k_str"], observed=True)["SelectedColumns"]
        .median()
        .reset_index()
        .rename(columns={"SelectedColumns": "MedianSelectedColumns"})
    )
    med_rows = (
        df_counts
        .groupby(["Method", "k_str"], observed=True)["SelectedRows"]
        .median()
        .reset_index()
        .rename(columns={"SelectedRows": "MedianSelectedRows"})
    )

    # 4) Plot für Selected Columns (Boxplot + optionale Medianlinie)
    p_cols = (
        ggplot(df_counts, aes(x="k_str", y="SelectedColumns"))
        + geom_boxplot(aes(fill="Method"))
        + facet_wrap("~Method", ncol=ncol, scales="fixed")
        + labs(title="Selected Columns per k and Method", x="k", y="Anzahl ausgewählte Spalten")
        + theme_minimal()
        + theme(axis_text_x=element_text(rotation=45, hjust=1))
    )

    if show_median_lines:
        # geom_line/point mit eigenem DataFrame; group=1 sorgt für Linien pro Facet
        p_cols = p_cols + geom_line(
            data=med_cols, mapping=aes(x="k_str", y="MedianSelectedColumns", group=1),
            color="orange", size=1
        ) + geom_point(
            data=med_cols, mapping=aes(x="k_str", y="MedianSelectedColumns"),
            color="orange", size=2
        )

    # 5) Plot für Selected Rows
    p_rows = (
        ggplot(df_counts, aes(x="k_str", y="SelectedRows"))
        + geom_boxplot(aes(fill="Method"))
        + facet_wrap("~Method", ncol=ncol, scales="fixed")
        + labs(title="Selected Rows per k and Method", x="k", y="Anzahl ausgewählte Zeilen")
        + theme_minimal()
        + theme(axis_text_x=element_text(rotation=45, hjust=1))
    )

    if show_median_lines:
        p_rows = p_rows + geom_line(
            data=med_rows, mapping=aes(x="k_str", y="MedianSelectedRows", group=1),
            color="darkgreen", size=1
        ) + geom_point(
            data=med_rows, mapping=aes(x="k_str", y="MedianSelectedRows"),
            color="darkgreen", size=2
        )

    # 6) Display und Rückgabe
    display(p_cols)
    display(p_rows)