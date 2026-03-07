import pandas as pd
import numpy as np
from plotnine import ggplot, aes, geom_histogram, theme_bw, labs, geom_line, theme_minimal, geom_boxplot, facet_wrap
from IPython.display import display

# ==============================================================================================
# visualize_distributions: Function to visualize scoring distributions
# ==============================================================================================

def visualize_distributions(scores):
    """
    Visualize the distributions of different scoring methods across replications.

    This function generates histogram plots for each method in the `scores` dictionary.
    Each method will have a single figure, with faceted subplots per replication
    (e.g., 10 replications arranged in 2 rows of 5 columns).

    Parameters
    ----------
    scores : dict
        Dictionary containing score lists for each method.
        Keys are method names (str), e.g., "LS", "CLS", "RS", "CS".
        Values are lists of arrays, where each array corresponds to one replication.

    Returns
    -------
    None
        Displays one faceted histogram plot per method.

    Notes
    -----
    - Each histogram shows the distribution of scores for the corresponding replication.
    - Uses plotnine/ggplot for plotting and automatically reshapes the scores into long format.
    - Recommended: ensure all replications have the same number of features for consistent plotting.
    """
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


# ==============================================================================================
# Function to plot RMSE comparison (line + boxplot)
# ==============================================================================================
def plot_rmse_comparison(rmse_dict):
    """
    Create line plot and boxplot of RMSE per method across replications.

    Parameters
    ----------
    rmse_dict : dict
        Keys = method names (str), values = list of RMSE per replication

    Returns
    -------
    rmse_df : pd.DataFrame
        Long-format DataFrame used for plotting
    p_line : plotnine.ggplot
        Line plot of RMSE across replications
    p_box : plotnine.ggplot
        Boxplot of RMSE distribution per method
    """
    # Create DataFrame
    n_reps = len(next(iter(rmse_dict.values())))
    rmse_df = pd.DataFrame(rmse_dict)
    rmse_df["Replication"] = np.arange(1, n_reps + 1)

    # Melt into long format
    rmse_long = rmse_df.melt(
        id_vars="Replication",
        var_name="Method",
        value_name="RMSE"
    )

    # Line plot
    p_line = (
            ggplot(rmse_long, aes(x="Replication", y="RMSE", color="Method"))
            + geom_line(size=1.2)
            + theme_minimal()
            + labs(
        title="RMSE Comparison Across Methods",
        x="Replication",
        y="RMSE"
    )
    )

    # Boxplot
    p_box = (
            ggplot(rmse_long, aes(x="Method", y="RMSE", fill="Method"))
            + geom_boxplot()
            + theme_minimal()
            + labs(
        title="RMSE Distribution per Method",
        x="Method",
        y="RMSE"
    )
    )

    # Display plots
    # display(p_line)
    display(p_box)

    return rmse_df, p_line, p_box

# ==============================================================================================
# Function to plot RMSE comparison (line + boxplot)
# ==============================================================================================
def plot_time_comparison(time_dict):
    """
    Create line plot and boxplot of Time per method across replications.

    Parameters
    ----------
    time_dict : dict
        Keys = method names (str), values = list of Time per replication

    Returns
    -------
    time_df : pd.DataFrame
        Long-format DataFrame used for plotting
    p_line : plotnine.ggplot
        Line plot of Time across replications
    p_box : plotnine.ggplot
        Boxplot of Time distribution per method
    """
    # Create DataFrame
    n_reps = len(next(iter(time_dict.values())))
    time_df = pd.DataFrame(time_dict)
    time_df["Replication"] = np.arange(1, n_reps + 1)

    # Melt into long format
    time_long = time_df.melt(
        id_vars="Replication",
        var_name="Method",
        value_name="Score Calculation Time"
    )

    # Line plot
    p_line = (
            ggplot(time_long, aes(x="Replication", y="Time", color="Method"))
            + geom_line(size=1.2)
            + theme_minimal()
            + labs(
        title="Score Calculation Time Comparison Across Methods",
        x="Replication",
        y="Score Calculation Time"
    )
    )

    # Boxplot
    p_box = (
            ggplot(time_long, aes(x="Method", y="Time", fill="Method"))
            + geom_boxplot()
            + theme_minimal()
            + labs(
        title="Score Calculation Time Distribution per Method",
        x="Method",
        y="Score Calculation Time"
    )
    )

    # Display plots
    # display(p_line)
    display(p_box)

    return time_df, p_line, p_box