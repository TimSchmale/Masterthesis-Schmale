import pandas as pd
import numpy as np
from IPython.display import display
from plotnine import ggplot, aes, geom_histogram, theme_bw, labs, geom_line, theme_minimal, geom_boxplot, facet_wrap

def evaluate_screening_performance(base, folder, C, reps):
    # determine the beta hit lists
    beta, beta_nonzero, cls_hits, ls_hits, rs_hits, cs_hits = get_beta_hits(base, folder, C, reps)

    # visualize the percentages of betas hit by method
    visualize_hit_percentages(beta_nonzero, cls_hits, ls_hits, rs_hits, cs_hits, reps)

def get_beta_hits(base, folder, C, reps):

    # iterate over the reps
    beta = []
    beta_nonzero = []

    cls_hits = []
    ls_hits = []
    rs_hits = []
    cs_hits = []
    for i in range(reps):
        # read in the beta vector
        beta.append(
            pd.read_csv(f"{base}/{folder}/beta{i + 1}.csv").to_numpy().reshape(-1)
        )

        # determine the indices of the beta values != 0
        beta_nonzero.append(np.where(beta[i] != 0.0)[0])

        # get the selected columns for each method as well as the intersection = hits with beta
        cls_cols = C['C_cls'][i]['selected_columns']
        ls_cols = C['C_ls'][i]['selected_columns']
        rs_cols = C['C_rs'][i]['selected_columns']
        cs_cols = C['C_cs'][i]['selected_columns']

        cls_hits.append(np.intersect1d(cls_cols, beta_nonzero[i]))
        ls_hits.append(np.intersect1d(ls_cols, beta_nonzero[i]))
        rs_hits.append(np.intersect1d(rs_cols, beta_nonzero[i]))
        cs_hits.append(np.intersect1d(cs_cols, beta_nonzero[i]))

    return beta, beta_nonzero, cls_hits, ls_hits, rs_hits, cs_hits

def visualize_hit_percentages(beta_nonzero, cls_hits, ls_hits, rs_hits, cs_hits, reps):

    no_cls_hits = []
    no_ls_hits = []
    no_rs_hits = []
    no_cs_hits = []
    for i in range(reps):
        no_cls_hits.append(len(cls_hits[i]))
        no_ls_hits.append(len(ls_hits[i]))
        no_rs_hits.append(len(rs_hits[i]))
        no_cs_hits.append(len(cs_hits[i]))

    no_hits = {"CLS": no_cls_hits, "LS": no_ls_hits, "RS": no_rs_hits, "CS": no_cs_hits}

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