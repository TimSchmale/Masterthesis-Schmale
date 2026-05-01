import pandas as pd
import numpy as np
from IPython.display import display
from plotnine import ggplot, aes, geom_histogram, theme_bw, labs, geom_bar, theme_minimal, geom_boxplot, facet_wrap

def evaluate_screening_performance(base, folder, C, reps, beta_lasso = None):
    # determine the beta hit lists
    beta, beta_nonzero, cls_hits, ls_hits, rs_hits, cs_hits, lasso_hits = get_beta_hits(base, folder, C, reps, beta_lasso)

    if beta_lasso is None:
        lasso_hits = None

    # visualize the percentages of betas hit by method
    visualize_hit_percentages(beta_nonzero, cls_hits, ls_hits, rs_hits, cs_hits, reps, lasso_hits)

    cls_share, ls_share, rs_share, cs_share, lasso_share= get_beta_share(beta, C, beta_lasso)

    if beta_lasso is None:
        lasso_share = None

    visualize_beta_share(cls_share, ls_share, rs_share, cs_share, lasso_share)

def get_beta_hits(base, folder, C, reps, beta_lasso=None):

    beta = []
    beta_nonzero = []

    cls_hits, ls_hits, rs_hits, cs_hits = [], [], [], []
    lasso_hits = []

    for i in range(reps):

        beta.append(
            pd.read_csv(f"{base}/{folder}/beta{i + 1}.csv").to_numpy().reshape(-1)
        )

        beta_nonzero.append(np.where(beta[i] != 0.0)[0])

        cls_cols = C['C_cls'][i]['selected_columns']
        ls_cols = C['C_ls'][i]['selected_columns']
        rs_cols = C['C_rs'][i]['selected_columns']
        cs_cols = C['C_cs'][i]['selected_columns']

        cls_hits.append(np.intersect1d(cls_cols, beta_nonzero[i]))
        ls_hits.append(np.intersect1d(ls_cols, beta_nonzero[i]))
        rs_hits.append(np.intersect1d(rs_cols, beta_nonzero[i]))
        cs_hits.append(np.intersect1d(cs_cols, beta_nonzero[i]))

        # --- LASSO ---
        if beta_lasso is not None:
            lasso_cols = np.where(beta_lasso[i] != 0)[0]
            lasso_hits.append(np.intersect1d(lasso_cols, beta_nonzero[i]))

    return beta, beta_nonzero, cls_hits, ls_hits, rs_hits, cs_hits, lasso_hits

def visualize_hit_percentages(beta_nonzero, cls_hits, ls_hits, rs_hits, cs_hits, reps, lasso_hits=None):

    no_cls_hits = []
    no_ls_hits = []
    no_rs_hits = []
    no_cs_hits = []
    no_lasso_hits = []

    for i in range(reps):
        ...
        if lasso_hits is not None:
            no_lasso_hits.append(len(lasso_hits[i]))

    for i in range(reps):
        no_cls_hits.append(len(cls_hits[i]))
        no_ls_hits.append(len(ls_hits[i]))
        no_rs_hits.append(len(rs_hits[i]))
        no_cs_hits.append(len(cs_hits[i]))

    no_hits = {
        "CLS": no_cls_hits,
        "LS": no_ls_hits,
        "RS": no_rs_hits,
        "CS": no_cs_hits
    }

    if lasso_hits is not None:
        no_hits["Lasso"] = no_lasso_hits

    # Create DataFrame
    n_reps = len(next(iter(no_hits.values())))
    hits_df = pd.DataFrame(no_hits)
    hits_df["Replication"] = np.arange(1, n_reps + 1)

    # Melt into long format
    hits_long = hits_df.melt(
        id_vars="Replication",
        var_name="Method",
        value_name="Hits"
    )
    hits_long["Hit Percentage"] = hits_long["Hits"] / np.shape(beta_nonzero[0]) * 100

    # Boxplot
    p_box = (
            ggplot(hits_long, aes(x="Method", y="Hit Percentage", fill="Method"))
            + geom_boxplot()
            + theme_minimal()
            + labs(
        title="Hit Percentage per Method",
        x="Method",
        y="Hit Percentage"
    )
    )

    # print plot
    display(p_box)

def get_beta_share(beta, C, beta_lasso=None):
    reps = np.shape(beta)[0]

    cls_share = []
    ls_share = []
    rs_share = []
    cs_share = []
    lasso_share = []

    for i in range(reps):
        beta_total = np.sum(np.abs(beta[i]))

        cls_cols = C['C_cls'][i]['selected_columns']
        ls_cols = C['C_ls'][i]['selected_columns']
        rs_cols = C['C_rs'][i]['selected_columns']
        cs_cols = C['C_cs'][i]['selected_columns']

        cls = np.sum(np.abs(beta[i][cls_cols]))
        ls = np.sum(np.abs(beta[i][ls_cols]))
        rs = np.sum(np.abs(beta[i][rs_cols]))
        cs = np.sum(np.abs(beta[i][cs_cols]))

        cls_share.append(cls / beta_total * 100)
        ls_share.append(ls / beta_total * 100)
        rs_share.append(rs / beta_total * 100)
        cs_share.append(cs / beta_total * 100)

        if beta_lasso is not None:
            lasso_cols = np.where(beta_lasso[i] != 0)[0]
            lasso_val = np.sum(np.abs(beta[i][lasso_cols]))
            lasso_share.append(lasso_val / beta_total * 100)

    return cls_share, ls_share, rs_share, cs_share, lasso_share

def visualize_beta_share(cls_share, ls_share, rs_share, cs_share, lasso_share = None):

    reps = np.shape(cls_share)[0]

    shares = {
        "CLS": cls_share,
        "LS": ls_share,
        "RS": rs_share,
        "CS": cs_share
    }

    if lasso_share is not None:
        shares["Lasso"] = lasso_share

    # Create DataFrame
    n_reps = len(next(iter(shares.values())))
    shares_df = pd.DataFrame(shares)
    shares_df["Replication"] = np.arange(1, n_reps + 1)

    # Melt into long format
    hits_long = shares_df.melt(
        id_vars="Replication",
        var_name="Method",
        value_name="Share"
    )

    # Boxplot
    p_box = (
            ggplot(hits_long, aes(x="Method", y="Share", fill="Method"))
            + geom_boxplot()
            + theme_minimal()
            + labs(
        title="Beta Share per Method",
        x="Method",
        y="Beta Share"
    )
    )

    # print plot
    display(p_box)
