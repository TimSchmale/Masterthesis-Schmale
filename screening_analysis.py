import pandas as pd
import numpy as np
from IPython.display import display
from plotnine import ggplot, aes, geom_boxplot, theme_minimal, labs, facet_wrap


def get_beta_hits(base, folder, C, reps, beta_lasso=None):
    """
    Compute screening hits for CUR-based column selection and optional Lasso models.

    For each replication, the true coefficient vector β is loaded and its non-zero
    support is extracted. The selected columns from each CUR method (CLS, LS, RS, CS)
    are compared against the true support to determine hit sets. If Lasso coefficient
    vectors are provided, their selected supports are evaluated analogously.

    Parameters
    ----------
    base : str
        Base directory containing simulation data.
    folder : str
        Subfolder containing beta files.
    C : dict
        Dictionary containing selected column indices for each CUR method.
    reps : int
        Number of replications.
    beta_lasso : dict or None
        Dictionary of Lasso coefficient vectors per method.

    Returns
    -------
    tuple
        (beta, beta_nonzero, cls_hits, ls_hits, rs_hits, cs_hits, lasso_hits)
    """

    # prepare containers for true betas and supports
    beta = []
    beta_nonzero = []

    # prepare hit containers for CUR methods
    cls_hits, ls_hits, rs_hits, cs_hits = [], [], [], []

    # prepare Lasso hit container if provided
    lasso_hits = {}
    if beta_lasso is not None:
        for method in beta_lasso.keys():
            lasso_hits[method] = []

    # iterate over replications
    for i in range(reps):

        # load true beta vector
        beta_i = pd.read_csv(f"{base}/{folder}/beta{i + 1}.csv").to_numpy().reshape(-1)
        beta.append(beta_i)

        # extract true non-zero support
        beta_nonzero.append(np.where(beta_i != 0.0)[0])

        # extract selected columns for each CUR method
        cls_cols = C['C_cls'][i]['selected_columns']
        ls_cols  = C['C_ls'][i]['selected_columns']
        rs_cols  = C['C_rs'][i]['selected_columns']
        cs_cols  = C['C_cs'][i]['selected_columns']

        # compute intersections with true support
        cls_hits.append(np.intersect1d(cls_cols, beta_nonzero[i]))
        ls_hits.append(np.intersect1d(ls_cols, beta_nonzero[i]))
        rs_hits.append(np.intersect1d(rs_cols, beta_nonzero[i]))
        cs_hits.append(np.intersect1d(cs_cols, beta_nonzero[i]))

        # compute Lasso hits if provided
        if beta_lasso is not None:
            for method in beta_lasso.keys():
                lasso_cols = np.where(beta_lasso[method][i] != 0)[0]
                lasso_hits[method].append(np.intersect1d(lasso_cols, beta_nonzero[i]))

    return beta, beta_nonzero, cls_hits, ls_hits, rs_hits, cs_hits, lasso_hits

def visualize_hit_percentages(beta_nonzero, cls_hits, ls_hits, rs_hits, cs_hits, reps, lasso_hits=None):
    """
    Visualize hit percentages for CUR-based screening and optional Lasso models.

    Hit counts are normalized by the number of true non-zero coefficients and
    displayed as boxplots across replications.
    """

    # collect hit counts for CUR methods
    no_hits = {
        "CLS": [len(x) for x in cls_hits],
        "LS":  [len(x) for x in ls_hits],
        "RS":  [len(x) for x in rs_hits],
        "CS":  [len(x) for x in cs_hits]
    }

    # add Lasso hit counts if available
    if lasso_hits is not None:
        for method in lasso_hits.keys():
            no_hits[f"Lasso_{method}"] = [len(x) for x in lasso_hits[method]]

    # build DataFrame
    n_reps = len(next(iter(no_hits.values())))
    hits_df = pd.DataFrame(no_hits)
    hits_df["Replication"] = np.arange(1, n_reps + 1)

    # reshape to long format
    hits_long = hits_df.melt(
        id_vars="Replication",
        var_name="Method",
        value_name="Hits"
    )

    # compute hit percentages
    p = len(beta_nonzero[0])
    hits_long["Hit Percentage"] = hits_long["Hits"] / p * 100

    # create boxplot
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

    display(p_box)
def get_beta_share(beta, C, beta_lasso=None):
    """
    Compute the share of total coefficient magnitude captured by selected columns.

    For each method, the absolute values of β restricted to the selected columns
    are summed and normalized by the total ℓ₁-norm of β.
    """

    # number of replications
    reps = len(beta)

    # prepare share containers for CUR methods
    cls_share, ls_share, rs_share, cs_share = [], [], [], []

    # prepare Lasso share container if provided
    lasso_share = {}
    if beta_lasso is not None:
        for method in beta_lasso.keys():
            lasso_share[method] = []

    # iterate over replications
    for i in range(reps):

        # compute total |beta|-mass
        beta_total = np.sum(np.abs(beta[i]))

        # extract selected columns
        cls_cols = C['C_cls'][i]['selected_columns']
        ls_cols  = C['C_ls'][i]['selected_columns']
        rs_cols  = C['C_rs'][i]['selected_columns']
        cs_cols  = C['C_cs'][i]['selected_columns']

        # compute share for each CUR method
        cls_share.append(np.sum(np.abs(beta[i][cls_cols])) / beta_total * 100)
        ls_share.append(np.sum(np.abs(beta[i][ls_cols])) / beta_total * 100)
        rs_share.append(np.sum(np.abs(beta[i][rs_cols])) / beta_total * 100)
        cs_share.append(np.sum(np.abs(beta[i][cs_cols])) / beta_total * 100)

        # compute Lasso share if provided
        if beta_lasso is not None:
            for method in beta_lasso.keys():
                lasso_cols = np.where(beta_lasso[method][i] != 0)[0]
                val = np.sum(np.abs(beta[i][lasso_cols])) / beta_total * 100
                lasso_share[method].append(val)

    return cls_share, ls_share, rs_share, cs_share, lasso_share

def visualize_beta_share(cls_share, ls_share, rs_share, cs_share, lasso_share=None):
    """
    Visualize the percentage of coefficient magnitude captured by each method.

    Beta-share values quantify how much of the total signal mass is recovered
    by the selected columns of each method.
    """

    # collect share values for CUR methods
    shares = {
        "CLS": cls_share,
        "LS":  ls_share,
        "RS":  rs_share,
        "CS":  cs_share
    }

    # add Lasso share values if available
    if lasso_share is not None:
        for method in lasso_share.keys():
            shares[f"Lasso_{method}"] = lasso_share[method]

    # build DataFrame
    n_reps = len(next(iter(shares.values())))
    shares_df = pd.DataFrame(shares)
    shares_df["Replication"] = np.arange(1, n_reps + 1)

    # reshape to long format
    shares_long = shares_df.melt(
        id_vars="Replication",
        var_name="Method",
        value_name="Share"
    )

    # create boxplot
    p_box = (
        ggplot(shares_long, aes(x="Method", y="Share", fill="Method"))
        + geom_boxplot()
        + theme_minimal()
        + labs(
            title="Beta Share per Method",
            x="Method",
            y="Beta Share"
        )
    )

    display(p_box)

def evaluate_screening_performance_with_lasso(base, folder, C, reps, beta_lasso):
    """
    Evaluate screening performance for CUR-based methods and Lasso models.

    Computes hit percentages and beta-share values and visualizes both.
    """

    # compute hits
    beta, beta_nonzero, cls_hits, ls_hits, rs_hits, cs_hits, lasso_hits = get_beta_hits(
        base, folder, C, reps, beta_lasso
    )

    # visualize hit percentages
    visualize_hit_percentages(
        beta_nonzero,
        cls_hits,
        ls_hits,
        rs_hits,
        cs_hits,
        reps,
        lasso_hits
    )

    # compute beta-share
    cls_share, ls_share, rs_share, cs_share, lasso_share = get_beta_share(
        beta, C, beta_lasso
    )

    # visualize beta-share
    visualize_beta_share(
        cls_share,
        ls_share,
        rs_share,
        cs_share,
        lasso_share
    )

def evaluate_screening_performance(base, folder, C, reps):
    """
    Evaluate screening performance for CUR-based methods without Lasso models.

    Computes hit percentages and beta-share values and visualizes both.
    """

    # compute hits
    beta, beta_nonzero, cls_hits, ls_hits, rs_hits, cs_hits, lasso_hits = get_beta_hits(
        base, folder, C, reps
    )

    # visualize hit percentages
    visualize_hit_percentages(
        beta_nonzero,
        cls_hits,
        ls_hits,
        rs_hits,
        cs_hits,
        reps,
        lasso_hits=None
    )

    # compute beta-share
    cls_share, ls_share, rs_share, cs_share, lasso_share = get_beta_share(
        beta, C, beta_lasso=None
    )

    # visualize beta-share
    visualize_beta_share(
        cls_share,
        ls_share,
        rs_share,
        cs_share,
        lasso_share=None
    )
