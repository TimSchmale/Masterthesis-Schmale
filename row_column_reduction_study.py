import numpy as np
import pandas as pd
import time
import pickle
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
    Row reduction using sketching-based dimensionality reduction.

    This function applies either Gaussian or Rademacher sketching to reduce
    the number of rows in the design matrix. The sketch size is determined
    by r = k * log(d). The resulting sketched matrix R preserves the
    subspace structure of X in expectation and is used as the basis for
    subsequent column reduction.

    Parameters
    ----------
    k : int
        Target rank controlling the sketch size.
    X : array-like or DataFrame
        Design matrix of shape (n, d).
    y : array-like or Series
        Response vector of length n.
    gaussian : bool
        If True, Gaussian sketching is used. Otherwise, Rademacher sketching.

    Returns
    -------
    R : ndarray of shape (r, d)
        Row-reduced design matrix.
    y_reduced : ndarray of shape (r, 1)
        Corresponding sketched response vector.
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

    This function samples columns independently using scaled probabilities
    derived from the provided score vector. The expected number of sampled
    columns is c = k * log(k). Selected columns are rescaled according to
    the CUR theorem to preserve unbiasedness.

    Parameters
    ----------
    R : ndarray of shape (r, d)
        Row-reduced design matrix.
    scores : array-like of length d
        Column score vector (e.g., LS, CLS, RS, CS).
    k : int
        Target rank controlling the expected number of sampled columns.

    Returns
    -------
    dict
        {
            "C" : ndarray of shape (r, t)
                Column-reduced matrix.
            "selected_columns" : list of int
                Indices of sampled columns.
        }
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
    Row-first data reduction pipeline.

    This function performs:
    1. Row reduction via sketching (Gaussian or Rademacher).
    2. Column score computation on the sketched matrices (LS, CLS, RS, CS).
    3. Column reduction via EXPECTED(c) sampling.

    If any method selects more columns than available sketch rows (t > r),
    the configuration is invalid for linear modeling and the function
    returns None-values (soft abort).

    Parameters
    ----------
    k : int
        Target rank controlling sketch and sampling sizes.
    df_train : list of DataFrames
        Training matrices for each replication.
    y_train : list of Series
        Response vectors for each replication.
    gaussian : bool
        Whether Gaussian sketching is used.

    Returns
    -------
    scores : dict
        Score vectors per method and replication.
    time_scores : dict
        Score computation times per method.
    selected_columns : dict
        Selected column indices per method and replication.
    selected_rows : list
        Selected row indices per replication.
    R_list : list
        Row-reduced matrices.
    y_list : list
        Row-reduced response vectors.

    Or (soft abort)
    ---------------
    None, None, None, None, None, None
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

        R = R_list[i]
        r = R.shape[0]  # number of rows after sketching

        # LS
        cols_ls = column_reduction(R, scores["LS"][i], k)["selected_columns"]
        if len(cols_ls) > r:
            print(f"[WARN] Aborting k={k} for method=LS,: {len(cols_ls)} columns > {r} rows. Skipping.")
            return None, None, None, None, None, None
        selected_columns["LS"].append(cols_ls)

        # CLS
        cols_cls = column_reduction(R, np.abs(scores["CLS"][i]), k)["selected_columns"]
        if len(cols_cls) > r:
            print(f"[WARN] Aborting k={k} for method=CLS,: {len(cols_cls)} columns > {r} rows. Skipping.")
            return None, None, None, None, None, None
        selected_columns["CLS"].append(cols_cls)

        # RS
        cols_rs = column_reduction(R, scores["RS"][i], k)["selected_columns"]
        if len(cols_rs) > r:
            print(f"[WARN] Aborting k={k} for method=RS,: {len(cols_rs)} columns > {r} rows. Skipping.")
            return None, None, None, None, None, None
        selected_columns["RS"].append(cols_rs)

        # CS
        cols_cs = column_reduction(R, scores["CS"][i], k)["selected_columns"]
        if len(cols_cs) > r:
            print(f"[WARN] Aborting k={k} for method=CS,: {len(cols_cs)} columns > {r} rows. Skipping.")
            return None, None, None, None, None, None
        selected_columns["CS"].append(cols_cs)

    return scores, time_scores, selected_columns, selected_rows, R_list, y_list

def linear_modeling(selected_columns, R_list, y_list, df_train, df_test, y_train, y_test):
    """
    Fit linear models on reduced matrices and compute RMSE.

    For each replication and each CUR method (LS, CLS, RS, CS), a linear
    regression model is fitted on the row- and column-reduced matrix.
    Training and test RMSE are computed, along with model fitting times.

    Parameters
    ----------
    selected_columns : dict
        Selected column indices per method and replication.
    R_list : list
        Row-reduced matrices.
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
    rmse_train : dict
        Training RMSE values per method and replication.
    rmse_test : dict
        Test RMSE values per method and replication.
    time_model : dict
        Model fitting times per method.
    """


    # determine number of replications
    n_reps = len(df_test)

    # initialize RMSE containers
    rmse_train = {"LS": [], "CLS": [], "RS": [], "CS": []}
    rmse_test = {"LS": [], "CLS": [], "RS": [], "CS": []}
    time_model = {"LS": [], "CLS": [], "RS": [], "CS": []}

    # initialize time containers
    time_model = {"LS": [], "CLS": [], "RS": [], "CS": []}

    # loop over replications
    for i in range(n_reps):

        # loop over methods
        for method in ["LS", "CLS", "RS", "CS"]:

            # extract selected columns
            cols = selected_columns[method][i]

            # fit model on row-reduced matrix
            t0 = time.perf_counter()
            model = LinearRegression().fit(R_list[i][:, cols], y_list[i].ravel())
            time_model[method].append(time.perf_counter() - t0)

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

    return rmse_train, rmse_test, time_model

def compute_full_rmse(df_train, df_test, y_train, y_test, base, folder):
    """
    Compute benchmark RMSE using the true non-zero coefficients.

    This function loads the true beta vector for each replication, identifies
    the non-zero coefficients, fits a linear model on the corresponding
    columns, and computes the resulting test RMSE. This serves as a benchmark
    for CUR-based approximations.

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
        Base directory containing simulation data.
    folder : str
        Subfolder containing beta files.

    Returns
    -------
    list
        Benchmark RMSE values for each replication.
    """

    # container for benchmark RMSE values
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

        # fit benchmark model
        model = LinearRegression().fit(X_train, y_tr)

        # compute predictions
        preds = model.predict(X_test)

        # compute RMSE
        rmse_full.append(np.sqrt(mean_squared_error(y_te, preds)))

    return rmse_full

def apply_col_after_row_reduction(
        k,
        seed,
        X_train_list,
        X_test_list,
        y_train_list,
        y_test_list,
        base,
        folder,
        gaussian=False):
    """
    Full Row → Column pipeline for a single seed.

    This function performs:
    1. Row reduction via sketching.
    2. Column score computation.
    3. Column reduction via EXPECTED(c).
    4. Linear modeling on reduced matrices.
    5. Benchmark RMSE computation.

    If row/column reduction yields an invalid CUR configuration (t > r),
    the function returns NaN-structured results (soft abort).

    Parameters
    ----------
    k : int
        Target rank.
    seed : int
        Random seed for reproducibility.
    X_train_list : list of ndarrays
        Training matrices for each replication.
    X_test_list : list of ndarrays
        Test matrices for each replication.
    y_train_list : list of ndarrays
        Training responses.
    y_test_list : list of ndarrays
        Test responses.
    base : str
        Base directory.
    folder : str
        Subfolder containing beta files.
    gaussian : bool
        Whether Gaussian sketching is used.

    Returns
    -------
    dict
        Dictionary containing scores, timings, selected indices,
        and RMSE values for all methods.

    Or (soft abort)
    ---------------
    dict with NaN-valued RMSE structures.
    """


    # set seed
    print(f"Setting random seed inside hybrid reduction pipeline (seed = {seed})...")
    random.seed(seed)
    np.random.seed(seed)

    # perform row-first reduction
    print("Performing data reduction...")
    scores, time_scores, selected_columns, selected_rows, R_list, y_list = data_reduction(
        k,
        X_train_list,
        y_train_list,
        gaussian
    )

    # soft abort
    if scores is None:
        print(f"[WARN] Aborting k={k} as p>n detected.")

        # nan structure
        nan_dict = {m: [np.nan] * len(X_train_list) for m in ["LS", "CLS", "RS", "CS"]}

        return {
            "scores": None,
            "time_scores": None,
            "time_model": None,
            "selected_columns": None,
            "selected_rows": None,
            "rmse_train": nan_dict,
            "rmse_test": {**nan_dict, "Full": [np.nan] * len(X_train_list)},
        }

    # fit linear models
    print("Fitting linear models on reduced data...")
    rmse_train, rmse_test, time_model = linear_modeling(
        selected_columns,
        R_list,
        y_list,
        X_train_list,
        X_test_list,
        y_train_list,
        y_test_list
    )

    # compute benchmark RMSE
    print("Computing full model benchmark (RMSE)...")
    rmse_full = compute_full_rmse(
        X_train_list,
        X_test_list,
        y_train_list,
        y_test_list,
        base,
        folder
    )
    rmse_test["Full"] = rmse_full

    print("Data Reduction & Modeling completed.")

    return {
        "scores": scores,
        "time_scores": time_scores,
        "time_model": time_model,
        "selected_columns": selected_columns,
        "selected_rows": selected_rows,
        "rmse_train": rmse_train,
        "rmse_test": rmse_test
    }

def numpy_train_test_split(X, y, test_size, seed):
    """
    NumPy-based deterministic train/test split.

    This function shuffles indices using a fixed random seed and partitions
    the dataset into training and test subsets.

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Design matrix.
    y : ndarray of shape (n,)
        Response vector.
    test_size : float
        Fraction of samples assigned to the test set.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test : ndarray
        Train/test split of the dataset.
    """


    # get length of dataset
    n = X.shape[0]

    # determine train and test sizes
    n_test = int(np.floor(n * test_size))
    n_train = n - n_test

    # shuffle the dataset indices
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    # get the train and test dataset
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    return (
        X[train_idx],
        X[test_idx],
        y[train_idx],
        y[test_idx]
    )

def run_sampling_variance_col_after_row(
    k_vector,
    base,
    folder,
    reps=10,
    outer_reps=10,
    test_size=0.2,
    save_name=None,
    results_folder=None,
    gaussian=False
):
    """
    Sampling variance analysis for the Row→Column pipeline.

    For each outer repetition (seed), this function:
    1. Loads all datasets.
    2. Performs train/test splits.
    3. Runs the full Row→Column pipeline for each k.
    4. Aggregates RMSE, score times, model times, and structural information.
    5. Stores results per seed.

    Soft aborts propagate NaN-valued RMSE structures for invalid CUR
    configurations (t > r).

    Parameters
    ----------
    k_vector : list of int
        List of target ranks to evaluate.
    base : str
        Base directory containing simulation data.
    folder : str
        Subfolder containing simulation files.
    reps : int
        Number of replications per seed.
    outer_reps : int
        Number of outer repetitions (different seeds).
    test_size : float
        Fraction of samples assigned to the test set.
    save_name : str or None
        Optional prefix for saved result files.
    results_folder : str
        Folder where results are stored.
    gaussian : bool
        Whether Gaussian sketching is used.

    Returns
    -------
    dict
        Nested dictionary containing sampling variance results for each seed.
    """


    print("Loading all datasets...")

    # Load all datasets
    X_list = [
        pd.read_csv(f"{base}/{folder}/X{i + 1}.csv").to_numpy()
        for i in range(reps)
    ]
    y_list = [
        pd.read_csv(f"{base}/{folder}/y{i + 1}.csv").to_numpy().reshape(-1)
        for i in range(reps)
    ]

    # initialize the results
    results = {}

    # loop over outer repetitions (different seeds)
    for seed in range(1, outer_reps + 1):

        print(f"\n============================================================")
        print(f"Running seed = {seed}")
        print("============================================================")

        np.random.seed(seed + 42)
        random.seed(seed + 42)

        # initialize Train/Test Split
        X_train_list = []
        X_test_list = []
        y_train_list = []
        y_test_list = []

        # loop over the different datasets
        for i in range(reps):
            # create train test split
            X_tr, X_te, y_tr, y_te = numpy_train_test_split(
                X_list[i],
                y_list[i],
                test_size=test_size,
                seed=seed
            )
            X_train_list.append(X_tr)
            X_test_list.append(X_te)
            y_train_list.append(y_tr)
            y_test_list.append(y_te)

        # initialize seed results
        results[seed] = {}

        # Run for each k
        for k in k_vector:
            print(f"\n--- Running k = {k} ---")

            # perform reduction pipeline
            out = apply_col_after_row_reduction(
                k=k,
                seed=seed,
                X_train_list=X_train_list,
                X_test_list=X_test_list,
                y_train_list=y_train_list,
                y_test_list=y_test_list,
                base=base,
                folder=folder,
                reps=reps,
                gaussian=gaussian
            )

            if out["scores"] is None:
                # Soft abort
                results[seed][k] = {
                    "loss": out["rmse_test"],
                    "train_loss": out["rmse_train"],
                    "selected_columns": None,
                    "selected_rows": None,
                    "scores": None,
                    "time_scores": None,
                    "time_model": None,
                }
                continue

            # extract RMSE values
            rmse_test = out["rmse_test"]
            rmse_train = out["rmse_train"]

            # extract further structural information
            time_scores = out["time_scores"]
            time_model = out["time_model"]

            # summarize the errors
            loss_summary = {
                method: {
                    "raw": rmse_test[method],
                    "mean": float(np.mean(rmse_test[method])),
                    "median": float(np.median(rmse_test[method]))
                }
                for method in rmse_test.keys()
            }

            # same for sanity check of training errors
            train_loss_summary = {
                method: {
                    "raw": rmse_train[method],
                    "mean": float(np.mean(rmse_train[method])),
                    "median": float(np.median(rmse_train[method]))
                }
                for method in rmse_train.keys()
            }

            # summarize the time for score calculation
            score_time_summary = {
                method: {
                    "raw": time_scores[method],
                    "mean": float(np.mean(time_scores[method])),
                    "median": float(np.median(time_scores[method]))
                }
                for method in time_scores.keys()
            }

            # same for modeling time
            model_time_summary = {
                method: {
                    "raw": time_model[method],
                    "mean": float(np.mean(time_model[method])),
                    "median": float(np.median(time_model[method]))
                }
                for method in time_model.keys()
            }

            # store everything
            results[seed][k] = {
                "loss": loss_summary,
                "train_loss": train_loss_summary,
                "selected_columns": out["selected_columns"],
                "selected_rows": out["selected_rows"],
                "scores": out["scores"],
                "time_scores": score_time_summary,
                "time_model": model_time_summary,
            }

        # Save seed results
        save_path = f"{base}/{results_folder}"
        os.makedirs(save_path, exist_ok=True)

        if save_name is None:
            seed_file = f"{save_path}/results_seed_{seed}.pkl"
        else:
            seed_file = f"{save_path}/{save_name}_seed_{seed}.pkl"

        with open(seed_file, "wb") as f:
            pickle.dump(results[seed], f)

        print(f"Saved seed {seed} → {seed_file}")

    # return full sampling variance structure
    return results