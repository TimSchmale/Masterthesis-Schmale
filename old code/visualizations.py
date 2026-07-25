from IPython.display import display
from plotnine import (
    ggplot, aes,
    geom_histogram, geom_point, geom_smooth, geom_boxplot,
    facet_wrap, theme_minimal, theme_bw, theme, element_text,
    labs
)
import pandas as pd
import numpy as np

import os

def visualize_score_facets_single(scores_single, save_path=None,
                                  pipeline=None, metric=None, dataset=None):
    """
    Plot score distributions for a single replication and optionally save as PDF.

    scores_single: dict with keys CLS, CS, LS, RS
                   each containing ONE score vector (not a list).
    save_path: directory where PDF should be saved
    pipeline, metric, dataset: used for naming the output file
    """

    rows = []

    for method, arr in scores_single.items():
        arr = np.asarray(arr).reshape(-1)
        for v in arr:
            rows.append({
                "Method": method,
                "value": v
            })

    df = pd.DataFrame(rows)

    p = (
        ggplot(df, aes(x="value"))
        + geom_histogram(bins=80, fill="#4C72B0", alpha=0.7)
        + facet_wrap("~Method", ncol=2, scales="free")
        + theme_minimal()
        + labs(
            x="Score",
            y="Count"
        )
    )

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

        file_path = os.path.join(
            save_path,
            f"results_{pipeline}_scores_{dataset}.pdf"
        )

        p.save(file_path, limitsize=False, height=3, width=8)

        print(f"Saved: {file_path}")

    return p

def plot_all_boxplots(
    results, k_vector, dataset, metric="rmse", aggregate="raw", pipeline = "col_row", save_path=None
):
    rows = []

    # 1. k filtern: nur k verwenden, die in allen Seeds existieren
    valid_k = []
    for k in k_vector:
        ok = True
        for seed in results.keys():
            if k not in results[seed]:
                ok = False
                break
        if ok:
            valid_k.append(k)

    # 2. Werte einsammeln
    for seed in results.keys():
        for k in valid_k:

            # Metric dictionary auswählen
            if metric == "rmse":
                metric_dict = results[seed][k]["loss"]
            elif metric == "rmse_train":
                metric_dict = results[seed][k]["train_loss"]
            elif metric == "ce":
                metric_dict = results[seed][k]["ce"]
            elif metric == "ce_train":
                metric_dict = results[seed][k]["ce_train"]
            elif metric == "time_scores":
                metric_dict = results[seed][k]["time_scores"]
            elif metric == "time_model":
                metric_dict = results[seed][k]["time_model"]
            else:
                raise ValueError("Unknown metric.")

            for method, entry in metric_dict.items():
                values = np.asarray(entry["raw"])

                # komplett leere / NaN-Kombis überspringen
                if np.all(np.isnan(values)):
                    continue

                if aggregate == "raw":
                    for rep_idx, val in enumerate(values):
                        if np.isnan(val):
                            continue
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
                        "Loss": float(np.nanmean(values))
                    })

                elif aggregate == "median":
                    rows.append({
                        "Seed": seed,
                        "k": k,
                        "Method": method,
                        "Replication": None,
                        "Loss": float(np.nanmedian(values))
                    })

                else:
                    raise ValueError("aggregate must be 'raw', 'mean', or 'median'.")

    df = pd.DataFrame(rows)

    # 3. numerische Sortierung
    df["k"] = df["k"].astype(int)

    # Full extrahieren (einmalig)
    df_full = df[df["Method"] == "Full"]
    df_no_full = df[df["Method"] != "Full"]

    # Full bekommt eigenes Label
    if len(df_full) > 0:
        df_full = df_full.copy()
        df_full["x_label"] = "Full"

    # reguläre Methoden sortieren: Methode → k
    combos = df_no_full[["Method", "k"]].drop_duplicates()
    method_sorted = sorted(combos["Method"].unique())
    k_sorted = sorted(combos["k"].unique())

    df_no_full["x_label"] = df_no_full["Method"] + " | " + df_no_full["k"].astype(str)

    # Kategorien nur für existierende Kombis
    x_order = []

    # zuerst Full
    if len(df_full) > 0:
        x_order.append("Full")

    # dann Methoden → k
    for m in method_sorted:
        for k in k_sorted:
            if ((combos["Method"] == m) & (combos["k"] == k)).any():
                x_order.append(f"{m} | {k}")

    # zusammenführen
    df_plot = pd.concat([df_full, df_no_full], ignore_index=True)

    df_plot["x_label"] = pd.Categorical(df_plot["x_label"], categories=x_order, ordered=True)
    df_plot = df_plot.sort_values("x_label")

    # 4. Metric-Namen
    metric_name = {
        "rmse": "RMSE",
        "rmse_train": "RMSE (Train)",
        "ce": "Cross-Entropy",
        "ce_train": "Cross-Entropy (Train)",
        "time_scores": "Score Computation Time (s)",
        "time_model": "Modeling Time (s)"
    }[metric]

    # 5. Plot
    p = (
        ggplot(df_plot, aes(x="x_label", y="Loss", fill="Method"))
        + geom_boxplot(width=0.4)
        + theme_bw()
        + theme(
            figure_size=(25, 16),
            axis_text_x=element_text(rotation=90, ha="center", size=16),
            axis_text_y=element_text(size=16),
            axis_title_x=element_text(size=20),
            axis_title_y=element_text(size=16),
            legend_position='bottom',
            legend_text=element_text(size=18),
            legend_title=element_text(size=20)
        )

        + labs(
            x="Method | k",
            y=metric_name
        )
    )

    # 6. Speichern
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, f"results_{pipeline}_{metric}_{dataset}.pdf")
        p.save(file_path, limitsize=False)
        print(f"Saved: {file_path}")

    display(p)

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
