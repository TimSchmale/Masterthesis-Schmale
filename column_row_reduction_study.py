import pandas as pd
import numpy as np
import random
import gc
from scoring_functions import (
    get_column_leverage_scores,
    get_row_leverage_scores,
    get_random_scores,
    get_combined_scores,
    get_cross_leverage_scores
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import time

def column_reduction(X, scores, k):
    """
    Column reduction using the EXPECTED(c) sampling rule.

    Columns are sampled independently using scaled probabilities
    derived from the provided score vector. Selected columns are
    rescaled according to the CUR theorem.

    Parameters
    ----------
    X : array-like
        Design matrix of shape (n, p).
    scores : array-like
        Column score vector of length p.
    k : int
        Target rank for CUR approximation.

    Returns
    -------
    dict
        Dictionary containing:
            "C" : reduced matrix (n x t)
            "selected_columns" : list of selected column indices
    """

    # convert input to numpy array
    X = np.asarray(X)
    n, p = X.shape

    # compute sampling probabilities
    probs = scores / scores.sum()

    # compute expected number of sampled columns
    c = int(np.ceil(k * np.log(k)))

    # compute scaled probabilities for Bernoulli sampling
    scaled_probs = np.minimum(c * probs, 1)

    # draw Bernoulli samples
    z = np.random.rand(p)
    sampled = np.where(z <= scaled_probs)[0]

    # ensure at least one column is selected
    if len(sampled) == 0:
        sampled = np.array([np.argmax(scaled_probs)])

    # compute rescaling factors
    D_inv = 1 / np.sqrt(scaled_probs[sampled])

    # build reduced matrix
    C = X[:, sampled] * D_inv

    return {
        "C": C,
        "selected_columns": sampled.tolist()
    }

def row_reduction(C, y, k):
    """
    Row reduction using the EXPECTED(r) sampling rule.

    Rows are sampled independently using scaled row leverage scores.
    Selected rows are rescaled according to the CUR theorem.

    Parameters
    ----------
    C : array-like
        Column-reduced matrix of shape (n, c).
    y : array-like
        Response vector of length n.
    k : int
        Target rank for CUR approximation.

    Returns
    -------
    dict
        Dictionary containing:
            "R" : reduced matrix (t x c)
            "y" : reduced response vector (t,)
            "selected_rows" : list of selected row indices
    """

    # convert inputs to numpy arrays
    C = np.asarray(C)
    y = np.asarray(y).reshape(-1)

    n, c = C.shape

    # compute row leverage scores
    scores = get_row_leverage_scores(C, k)

    # compute sampling probabilities
    probs = scores / scores.sum()

    # compute expected number of sampled rows
    r = int(np.ceil(c * np.log(c)))

    # compute scaled probabilities
    scaled_probs = np.minimum(r * probs, 1)

    # draw Bernoulli samples
    z = np.random.rand(n)
    sampled = np.where(z <= scaled_probs)[0]

    # ensure at least one row is selected
    if len(sampled) == 0:
        sampled = np.array([np.argmax(scaled_probs)])

    # compute rescaling factors
    D_inv = 1 / np.sqrt(scaled_probs[sampled])

    # build reduced matrix
    R = C[sampled, :] * D_inv[:, None]

    # build reduced response vector
    y_reduced = y[sampled]

    return {
        "R": R,
        "y": y_reduced,
        "selected_rows": sampled.tolist()
    }

def data_reduction(k, df_train, y_train, row_reduce=True):
    """
    Compute score vectors and perform column and optional row reduction.

    This function computes LS, CLS, RS, and CS scores for each replication.
    It then performs EXPECTED(c) column reduction and, if enabled,
    EXPECTED(r) row reduction.

    Parameters
    ----------
    k : int
        Target rank for CUR approximation.
    df_train : list of DataFrames
        Training matrices for each replication.
    y_train : list of Series
        Response vectors for each replication.
    row_reduce : bool
        Whether row reduction should be applied.

    Returns
    -------
    tuple
        (scores, time_scores, C, R)
    """

    # determine number of replications
    n_reps = len(df_train)

    # initialize score containers
    scores = {"LS": [], "CLS": [], "RS": [], "CS": []}
    time_scores = {"LS": [], "CLS": [], "RS": [], "CS": []}

    # compute score vectors
    for i in range(n_reps):

        # extract X and y
        X = df_train[i].to_numpy()
        y = y_train[i].to_numpy().reshape(-1)

        # compute LS scores
        start = time.perf_counter()
        scores["LS"].append(get_column_leverage_scores(X, k))
        time_scores["LS"].append(time.perf_counter() - start)

        # compute CLS scores
        start = time.perf_counter()
        scores["CLS"].append(get_cross_leverage_scores(X, y))
        time_scores["CLS"].append(time.perf_counter() - start)

        # compute RS scores
        start = time.perf_counter()
        scores["RS"].append(get_random_scores(X))
        time_scores["RS"].append(time.perf_counter() - start)

        # compute CS scores
        start = time.perf_counter()
        scores["CS"].append(get_combined_scores(X, y, k, p_leverage=0.2))
        time_scores["CS"].append(time.perf_counter() - start)

        del X, y
        gc.collect()

    # perform column reduction
    C = {"LS": [], "CLS": [], "RS": [], "CS": []}

    for i in range(n_reps):

        # extract X
        X = df_train[i].to_numpy()

        # apply column reduction for each score type
        C["LS"].append(column_reduction(X, scores["LS"][i], k))
        C["CLS"].append(column_reduction(X, np.abs(scores["CLS"][i]), k))
        C["RS"].append(column_reduction(X, scores["RS"][i], k))
        C["CS"].append(column_reduction(X, scores["CS"][i], k))

        del X
        gc.collect()

    # perform row reduction if enabled
    R = None
    if row_reduce:

        # initialize row reduction container
        R = {"LS": [], "CLS": [], "RS": [], "CS": []}

        for i in range(n_reps):

            # extract y
            y = y_train[i].to_numpy().reshape(-1)

            # apply row reduction for each method
            R["LS"].append(row_reduction(C["LS"][i]["C"], y, k))
            R["CLS"].append(row_reduction(C["CLS"][i]["C"], y, k))
            R["RS"].append(row_reduction(C["RS"][i]["C"], y, k))
            R["CS"].append(row_reduction(C["CS"][i]["C"], y, k))

            del y
            gc.collect()

    return scores, time_scores, C, R

def linear_modeling(C, R, df_train, df_test, y_train, y_test):
    """
    Fit linear regression models on reduced matrices and compute RMSE.

    If row reduction is applied, models are fitted on R and predictions
    are made using the corresponding selected columns. Otherwise,
    models are fitted directly on the column-reduced matrices C.

    Parameters
    ----------
    C : dict
        Column-reduced matrices and selected columns.
    R : dict or None
        Row-reduced matrices and selected rows.
    df_train : list of DataFrames
        Training matrices.
    df_test : list of DataFrames
        Test matrices.
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

        # check if row reduction is used
        if R is not None:

            # fit models on row-reduced matrices
            models = {
                "LS": LinearRegression().fit(R["LS"][i]["R"], R["LS"][i]["y"]),
                "CLS": LinearRegression().fit(R["CLS"][i]["R"], R["CLS"][i]["y"]),
                "RS": LinearRegression().fit(R["RS"][i]["R"], R["RS"][i]["y"]),
                "CS": LinearRegression().fit(R["CS"][i]["R"], R["CS"][i]["y"])
            }

            # compute RMSE for each method
            for key in rmse_test.keys():

                # extract selected columns
                cols = C[key][i]["selected_columns"]

                # compute train RMSE
                X_train_red = df_train[i].to_numpy()[:, cols]
                preds_train = models[key].predict(X_train_red)
                rmse_train[key].append(
                    np.sqrt(mean_squared_error(y_train[i], preds_train))
                )

                # compute test RMSE
                X_test_red = df_test[i].to_numpy()[:, cols]
                preds_test = models[key].predict(X_test_red)
                rmse_test[key].append(
                    np.sqrt(mean_squared_error(y_test[i], preds_test))
                )

        else:

            # extract training response
            y_tr = y_train[i].to_numpy().reshape(-1)

            # fit models on column-reduced matrices
            models = {
                "LS": LinearRegression().fit(C["LS"][i]["C"], y_tr),
                "CLS": LinearRegression().fit(C["CLS"][i]["C"], y_tr),
                "RS": LinearRegression().fit(C["RS"][i]["C"], y_tr),
                "CS": LinearRegression().fit(C["CS"][i]["C"], y_tr)
            }

            # compute RMSE for each method
            for key in rmse_test.keys():

                # extract selected columns
                cols = C[key][i]["selected_columns"]

                # compute train RMSE
                X_train_red = df_train[i].to_numpy()[:, cols]
                preds_train = models[key].predict(X_train_red)
                rmse_train[key].append(
                    np.sqrt(mean_squared_error(y_train[i], preds_train))
                )

                # compute test RMSE
                X_test_red = df_test[i].to_numpy()[:, cols]
                preds_test = models[key].predict(X_test_red)
                rmse_test[key].append(
                    np.sqrt(mean_squared_error(y_test[i], preds_test))
                )

    return rmse_train, rmse_test

def compute_full_rmse(df_train, df_test, y_train, y_test, base, folder):
    """
    Compute oracle RMSE using the true non-zero coefficients.

    The function loads the true beta vector for each replication,
    extracts the non-zero positions, fits a linear model on the
    corresponding columns, and computes test RMSE.

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
        List of oracle RMSE values for each replication.
    """

    # container for oracle RMSE values
    rmse_full = []

    # loop over replications
    for i in range(len(df_train)):

        # load true beta vector
        beta = pd.read_csv(
            f"{base}/{folder}/beta{i + 1}.csv",
            header=0
        ).to_numpy().reshape(-1)

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

        del X_train, X_test, y_tr, y_te, preds
        gc.collect()

    return rmse_full

def apply_row_after_col_reduction(k, seed, base, folder, reps, row_reduction=True):
    """
    Full CUR-based reduction and modeling pipeline.

    This function loads simulation data, computes score vectors,
    performs column and optional row reduction, fits linear models,
    computes RMSE, and computes the oracle RMSE benchmark.

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
    row_reduction : bool
        Whether row reduction should be applied.

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

    # print progress information
    print(f"Reading simulation data for {reps} replications...")

    # load training and test data
    df_train = [pd.read_csv(f"{base}/{folder}/X_train{i+1}.csv") for i in range(reps)]
    df_test = [pd.read_csv(f"{base}/{folder}/X_test{i+1}.csv") for i in range(reps)]
    y_train = [pd.read_csv(f"{base}/{folder}/y_train{i+1}.csv") for i in range(reps)]
    y_test = [pd.read_csv(f"{base}/{folder}/y_test{i+1}.csv") for i in range(reps)]

    # set random seed
    print("Setting random seed...")
    random.seed(seed)
    np.random.seed(seed)

    # perform score computation and reduction
    print("Performing data reduction (scores + column reduction)...")
    scores, time_scores, C, R = data_reduction(k, df_train, y_train, row_reduction)

    # fit linear models
    print("Fitting linear models on reduced data...")
    rmse_train, rmse_test = linear_modeling(C, R, df_train, df_test, y_train, y_test)

    # compute oracle RMSE
    print("Computing full-model benchmark (oracle RMSE)...")
    rmse_full = compute_full_rmse(df_train, df_test, y_train, y_test, base, folder)
    rmse_test["Full"] = rmse_full

    print("Data Reduction & Modeling completed.")

    # extract only selected column indices
    selected_columns_clean = {
        method: [C[method][i]["selected_columns"] for i in range(reps)]
        for method in ["LS", "CLS", "RS", "CS"]
    }

    # extract only selected row indices (if row reduction enabled)
    if row_reduction:
        selected_rows_clean = {
            method: [R[method][i]["selected_rows"] for i in range(reps)]
            for method in ["LS", "CLS", "RS", "CS"]
        }
    else:
        selected_rows_clean = None

    return {
        "scores": scores,
        "time_scores": time_scores,
        "selected_columns": selected_columns_clean,
        "selected_rows": selected_rows_clean,
        "rmse_train": rmse_train,
        "rmse_test": rmse_test
    }


def run_sampling_variance_row_after_col(
    k,
    base,
    folder,
    reps=10,
    outer_reps=10,
    row_reduction=True
):
    """
    Extended sampling variance analysis for the Row-after-Column CUR pipeline.

    This function runs the full CUR-based reduction and modeling pipeline
    multiple times using different random seeds. Each run produces 'reps'
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
        Target rank for CUR approximation.
    base : str
        Base directory containing simulation data.
    folder : str
        Subfolder containing the simulation files.
    reps : int
        Number of replications inside each wrapper run.
    outer_reps : int
        Number of wrapper repetitions with different seeds.
    row_reduction : bool
        Whether row reduction should be applied.

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

        # run the full pipeline once for this seed
        out = apply_row_after_col_reduction(
            k=k,
            seed=outer_seed,
            base=base,
            folder=folder,
            reps=reps,
            row_reduction=row_reduction
        )

        # extract test RMSE values
        rmse_test = out["rmse_test"]

        # extract train RMSE values
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