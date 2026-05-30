import numpy as np
import pandas as pd
import time
from scoring_functions import get_column_leverage_scores, get_row_leverage_scores, get_random_scores, get_combined_scores, get_cross_leverage_scores
from statsmodels.sandbox.distributions.genpareto import shape

from sklearn.preprocessing import StandardScaler

from scoring_functions import *
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
import random
from visualizations import *

def row_reduction(k, X, y, gaussian=False):
    """
    Row reduction using sketching-based sampling.

    This function applies either Gaussian or Rademacher sketching
    to reduce the number of rows in the design matrix. The sketch
    size is determined by k * log(d). The reduced matrix and the
    corresponding reduced response vector are returned.

    Parameters
    ----------
    k : int
        Target rank for determining the sketch size.
    X : array-like or DataFrame
        Design matrix of shape (n, d).
    y : array-like or Series
        Response vector of length n.
    gaussian : bool
        If True, Gaussian sketching is used. Otherwise, Rademacher.

    Returns
    -------
    tuple
        (R, y_reduced) where R is the row-reduced matrix and
        y_reduced is the reduced response vector.
    """

    # convert X to numpy array
    X = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)

    # convert y to numpy column vector
    y = y.to_numpy().reshape(-1, 1) if isinstance(y, (pd.Series, pd.DataFrame)) else y.reshape(-1, 1)

    # extract dimensions
    n, d = X.shape

    # compute sketch size
    r = int(np.ceil(k * np.log(d)))

    # allocate output matrices
    R = np.zeros((r, d))
    y_reduced = np.zeros((r, 1))

    # perform sketching if r < n
    if r < n:

        # iterate over all rows
        for i in range(n):

            # generate sketch vector
            if gaussian:
                sketch_vec = np.random.randn(r, 1)
            else:
                sketch_vec = np.random.choice([-1, 1], size=(r, 1)) / np.sqrt(r)

            # update reduced matrix
            R += sketch_vec @ X[i:i+1, :]

            # update reduced response
            y_reduced += sketch_vec * y[i]

    # if r >= n, no reduction is needed
    else:
        R = X.copy()
        y_reduced = y.copy()

    return R, y_reduced

def column_reduction(R, scores, k):
    """
    Column reduction using EXPECTED(c) sampling.

    This function samples columns independently using scaled
    probabilities derived from the provided score vector. Selected
    columns are rescaled according to the CUR theorem.

    Parameters
    ----------
    R : array-like
        Row-reduced matrix of shape (r, d).
    scores : array-like
        Column score vector of length d.
    k : int
        Target rank for determining the number of sampled columns.

    Returns
    -------
    dict
        Dictionary containing:
            "C" : reduced matrix (r x t)
            "selected_columns" : list of selected column indices
    """

    # convert R to numpy array
    R = np.asarray(R)

    # extract dimensions
    r, d = R.shape

    # compute sampling probabilities
    probs = scores / scores.sum()

    # compute expected number of sampled columns
    c = int(np.ceil(k * np.log(k)))

    # compute scaled probabilities
    scaled_probs = np.minimum(c * probs, 1)

    # draw Bernoulli samples
    z = np.random.rand(d)
    sampled = np.where(z <= scaled_probs)[0]

    # ensure at least one column is selected
    if len(sampled) == 0:
        sampled = np.array([np.argmax(scaled_probs)])

    # compute rescaling factors
    D_inv = 1 / np.sqrt(scaled_probs[sampled])

    # build reduced matrix
    C = R[:, sampled] * D_inv

    return {
        "C": C,
        "selected_columns": sampled.tolist()
    }

def data_reduction(k, df_train, y_train, gaussian=False):
    """
    Perform row-first data reduction followed by column reduction.

    This function first applies sketching-based row reduction to each
    replication. It then computes LS, CLS, RS, and CS column scores on
    the row-reduced matrices and applies EXPECTED(c) column sampling.

    Parameters
    ----------
    k : int
        Target rank for determining sketch and sample sizes.
    df_train : list of DataFrames
        Training matrices for each replication.
    y_train : list of Series
        Response vectors for each replication.
    gaussian : bool
        Whether Gaussian sketching is used for row reduction.

    Returns
    -------
    tuple
        (scores, time_scores, selected_columns, selected_rows, R_list, y_list)
    """

    # determine number of replications
    n_reps = len(df_train)

    # initialize containers
    selected_rows = []
    selected_columns = {"LS": [], "CLS": [], "RS": [], "CS": []}
    scores = {"LS": [], "CLS": [], "RS": [], "CS": []}
    time_scores = {"LS": [], "CLS": [], "RS": [], "CS": []}

    R_list = []
    y_list = []

    # perform row reduction for each replication
    for i in range(n_reps):

        # apply sketching-based row reduction
        R, y_red = row_reduction(k, df_train[i], y_train[i], gaussian)

        # store reduced matrices
        R_list.append(R)
        y_list.append(y_red)

        # store selected row indices
        selected_rows.append(list(range(R.shape[0])))

    # compute column scores for each replication
    for i in range(n_reps):

        # extract reduced matrix and response
        R = R_list[i]
        y = y_list[i]

        # compute LS scores
        start = time.perf_counter()
        scores["LS"].append(get_column_leverage_scores(R, k))
        time_scores["LS"].append(time.perf_counter() - start)

        # compute CLS scores
        start = time.perf_counter()
        scores["CLS"].append(get_cross_leverage_scores(R, y))
        time_scores["CLS"].append(time.perf_counter() - start)

        # compute RS scores
        start = time.perf_counter()
        scores["RS"].append(get_random_scores(R))
        time_scores["RS"].append(time.perf_counter() - start)

        # compute CS scores
        start = time.perf_counter()
        scores["CS"].append(get_combined_scores(R, y, k, p_leverage=0.2))
        time_scores["CS"].append(time.perf_counter() - start)

    # perform column reduction for each replication
    for i in range(n_reps):

        # extract reduced matrix
        R = R_list[i]

        # apply column reduction for each score type
        selected_columns["LS"].append(column_reduction(R, scores["LS"][i], k)["selected_columns"])
        selected_columns["CLS"].append(column_reduction(R, np.abs(scores["CLS"][i]), k)["selected_columns"])
        selected_columns["RS"].append(column_reduction(R, scores["RS"][i], k)["selected_columns"])
        selected_columns["CS"].append(column_reduction(R, scores["CS"][i], k)["selected_columns"])

    return scores, time_scores, selected_columns, selected_rows, R_list, y_list

def linear_modeling(selected_columns, R_list, y_list, df_train, df_test, y_train, y_test):
    """
    Fit linear models on reduced matrices and compute RMSE.

    This function fits linear regression models on the row-reduced
    matrices using the selected columns. It computes both training
    and test RMSE for each method and replication.

    Parameters
    ----------
    selected_columns : dict
        Selected column indices per method and replication.
    R_list : list
        Row-reduced matrices for each replication.
    y_list : list
        Row-reduced response vectors.
    df_train : list of DataFrames
        Original training matrices.
    df_test : list of DataFrames
        Original test matrices.
    y_train : list of Series
        Training responses.
    y_test : list of Series
        Test responses.

    Returns
    -------
    tuple
        (rmse_train, rmse_test)
    """

    # determine number of replications
    n_reps = len(df_test)

    # initialize RMSE containers
    rmse_train = {"LS": [], "CLS": [], "RS": [], "CS": []}
    rmse_test = {"LS": [], "CLS": [], "RS": [], "CS": []}

    # loop over replications
    for i in range(n_reps):

        # loop over methods
        for method in ["LS", "CLS", "RS", "CS"]:

            # extract selected columns
            cols = selected_columns[method][i]

            # fit model on row-reduced matrix
            model = LinearRegression().fit(R_list[i][:, cols], y_list[i].ravel())

            # compute train RMSE
            X_train_red = df_train[i].to_numpy()[:, cols]
            preds_train = model.predict(X_train_red)
            rmse_train[method].append(
                np.sqrt(mean_squared_error(y_train[i], preds_train))
            )

            # compute test RMSE
            X_test_red = df_test[i].to_numpy()[:, cols]
            preds_test = model.predict(X_test_red)
            rmse_test[method].append(
                np.sqrt(mean_squared_error(y_test[i], preds_test))
            )

    return rmse_train, rmse_test

def compute_full_rmse(df_train, df_test, y_train, y_test, base, folder):
    """
    Compute oracle RMSE using the true non-zero coefficients.

    This function loads the true beta vector for each replication,
    identifies the non-zero coefficients, fits a linear model on
    the corresponding columns, and computes test RMSE.

    Parameters
    ----------
    df_train : list of DataFrames
        Training matrices.
    df_test : list of DataFrames
        Test matrices.
    y_train : list of Series
        Training responses.
    y_test : list of Series
        Test responses.
    base : str
        Base directory.
    folder : str
        Subfolder containing beta files.

    Returns
    -------
    list
        Oracle RMSE values for each replication.
    """

    # container for oracle RMSE values
    rmse_full = []

    # loop over replications
    for i in range(len(df_train)):

        # load beta vector
        beta_df = pd.read_csv(f"{base}/{folder}/beta{i+1}.csv")
        beta = beta_df.select_dtypes(include=[np.number]).to_numpy().reshape(-1)

        # ensure beta length matches number of features
        p = df_train[i].shape[1]
        if len(beta) > p:
            beta = beta[:p]
        elif len(beta) < p:
            raise ValueError(f"Beta length {len(beta)} < number of features {p} in replication {i+1}.")

        # identify non-zero coefficients
        selected = np.where(beta != 0)[0]

        # extract relevant columns
        X_train = df_train[i].to_numpy()[:, selected]
        X_test = df_test[i].to_numpy()[:, selected]

        # extract responses
        y_tr = y_train[i].to_numpy().reshape(-1)
        y_te = y_test[i].to_numpy().reshape(-1)

        # fit oracle model
        model = LinearRegression().fit(X_train, y_tr)

        # compute predictions
        preds = model.predict(X_test)

        # compute RMSE
        rmse_full.append(np.sqrt(mean_squared_error(y_te, preds)))

    return rmse_full

def apply_col_after_row_reduction(k, seed, base, folder, reps, gaussian=False):
    """
    Full pipeline for Row → Column reduction and linear modeling.

    This function loads simulation data, performs sketching-based
    row reduction, computes column scores, applies EXPECTED(c)
    column sampling, fits linear models, computes RMSE, and adds
    the oracle RMSE benchmark.

    Parameters
    ----------
    k : int
        Target rank.
    seed : int
        Random seed.
    base : str
        Base directory.
    folder : str
        Subfolder containing simulation data.
    reps : int
        Number of replications.
    gaussian : bool
        Whether Gaussian sketching is used.

    Returns
    -------
    dict
        Dictionary containing:
            scores
            time_scores
            selected_columns
            selected_rows
            rmse_train
            rmse_test
    """

    # load simulation data
    print("Reading simulation data...")
    df_train = [pd.read_csv(f"{base}/{folder}/X_train{i+1}.csv") for i in range(reps)]
    df_test  = [pd.read_csv(f"{base}/{folder}/X_test{i+1}.csv") for i in range(reps)]
    y_train  = [pd.read_csv(f"{base}/{folder}/y_train{i+1}.csv") for i in range(reps)]
    y_test   = [pd.read_csv(f"{base}/{folder}/y_test{i+1}.csv") for i in range(reps)]

    # set random seed
    random.seed(seed)
    np.random.seed(seed)

    # perform row-first reduction
    print("Performing data reduction...")
    scores, time_scores, selected_columns, selected_rows, R_list, y_list = \
        data_reduction(k, df_train, y_train, gaussian)

    # fit linear models
    print("Fitting linear models...")
    rmse_train, rmse_test = linear_modeling(
        selected_columns,
        R_list,
        y_list,
        df_train,
        df_test,
        y_train,
        y_test
    )

    # compute oracle RMSE
    print("Computing full model benchmark...")
    rmse_full = compute_full_rmse(df_train, df_test, y_train, y_test, base, folder)
    rmse_test["Full"] = rmse_full

    print("Completed.")

    return {
        "scores": scores,
        "time_scores": time_scores,
        "selected_columns": selected_columns,
        "selected_rows": selected_rows,
        "rmse_train": rmse_train,
        "rmse_test": rmse_test
    }

def run_sampling_variance_col_after_row(
    k,
    base,
    folder,
    reps=10,
    outer_reps=10,
    gaussian=False
):
    """
    Sampling variance analysis for the Row→Column CUR pipeline.

    This function runs the full row-first reduction pipeline multiple
    times using different random seeds. Each run produces 'reps'
    RMSE values per method. For each outer repetition and each method,
    the function computes:
        - raw   : list of RMSE values (length reps)
        - mean  : average RMSE
        - median: median RMSE

    Additionally, the function stores:
        - train RMSE values
        - selected columns
        - selected rows
        - score vectors
        - score computation times

    The output enables full sampling variance and stability analysis.

    Parameters
    ----------
    k : int
        Target rank for sketch and sampling sizes.
    base : str
        Base directory containing simulation data.
    folder : str
        Subfolder containing the simulation files.
    reps : int
        Number of replications inside each wrapper run.
    outer_reps : int
        Number of wrapper repetitions with different seeds.
    gaussian : bool
        Whether Gaussian sketching is used for row reduction.

    Returns
    -------
    dict
        Nested dictionary containing sampling variance results for each seed.
    """

    # container for all outer seeds
    results = {}

    # loop over outer repetitions (different seeds)
    for outer_seed in range(1, outer_reps + 1):

        # print progress information
        print(f"Running outer repetition with seed = {outer_seed}")

        # run the full row→column pipeline once
        out = apply_col_after_row_reduction(
            k=k,
            seed=outer_seed,
            base=base,
            folder=folder,
            reps=reps,
            gaussian=gaussian
        )

        # extract RMSE values
        rmse_test = out["rmse_test"]
        rmse_train = out["rmse_train"]

        # extract structural information
        selected_columns = out["selected_columns"]
        selected_rows = out["selected_rows"]
        scores = out["scores"]
        time_scores = out["time_scores"]

        # container for aggregated results of this seed
        seed_result = {}

        # container for test loss summaries
        loss_summary = {}

        # container for train loss summaries
        train_loss_summary = {}

        # iterate over all methods (LS, CLS, RS, CS, Full)
        for method in rmse_test.keys():

            # handle "Full" separately because it has no training RMSE
            if method == "Full":

                # extract raw test RMSE values
                raw_vals_test = rmse_test["Full"]

                # compute mean and median for test RMSE
                loss_summary["Full"] = {
                    "raw": raw_vals_test,
                    "mean": float(np.mean(raw_vals_test)),
                    "median": float(np.median(raw_vals_test))
                }

                # skip training RMSE for "Full"
                continue

            # extract raw test RMSE values
            raw_vals_test = rmse_test[method]

            # compute mean and median for test RMSE
            loss_summary[method] = {
                "raw": raw_vals_test,
                "mean": float(np.mean(raw_vals_test)),
                "median": float(np.median(raw_vals_test))
            }

            # extract raw train RMSE values
            raw_vals_train = rmse_train[method]

            # compute mean and median for train RMSE
            train_loss_summary[method] = {
                "raw": raw_vals_train,
                "mean": float(np.mean(raw_vals_train)),
                "median": float(np.median(raw_vals_train))
            }

        # store all results for this seed
        seed_result["loss"] = loss_summary
        seed_result["train_loss"] = train_loss_summary
        seed_result["selected_columns"] = selected_columns
        seed_result["selected_rows"] = selected_rows
        seed_result["scores"] = scores
        seed_result["time_scores"] = time_scores

        # save seed result
        results[outer_seed] = seed_result

    # return full sampling variance structure
    return results