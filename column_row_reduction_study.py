import pandas as pd
import numpy as np
import random
import pickle
import gc
import os
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
from sklearn.model_selection import train_test_split

def column_reduction(X, scores, k):
    """
    Column reduction using the EXPECTED(c) importance-based procedure.

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
    scores = np.asarray(scores, dtype=float).reshape(-1)
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

    # ensure at least k columns are selected
    if len(sampled) < k:
        missing = k - len(sampled)
#        print(f"need to fill {missing} columns")
        candidates = np.argsort(scaled_probs)[- (missing + len(sampled)):]
        candidates = [idx for idx in candidates if idx not in sampled]
        sampled = np.concatenate([sampled, candidates[:missing]])
        sampled = sampled.astype(int)

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
    Row reduction using the EXPECTED(r) sampling procedure.

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
    scores = np.asarray(scores, dtype=float).reshape(-1)

    # compute sampling probabilities
    probs = scores / scores.sum()

    # compute expected number of sampled rows
    r = int(np.ceil(c * np.log(c)))

    # compute scaled probabilities
    scaled_probs = np.minimum(r * probs, 1)

    # draw Bernoulli samples
    z = np.random.rand(n)
    sampled = np.where(z <= scaled_probs)[0]

    # ensure at least as many rows as columns
    if len(sampled) < c:
        missing = c - len(sampled)
        #print(f"need to fill {missing} rows, as c = {c} and only {len(sampled)} sampled.")
        order = np.argsort(scaled_probs)[::-1]
        order = [idx for idx in order if idx not in sampled]
        sampled = np.concatenate([sampled, order[:missing]])
        sampled = sampled.astype(int)

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

def compute_scores(k, X_list, y_list):
    """
    Compute all column-based score vectors once per replication for a fixed rank k.

    This function precomputes the four score types used in the CUR pipeline:
    - LS  : Column leverage scores (SVD-based)
    - CLS : Cross-leverage scores (augmented QR-based)
    - RS  : Random scores (uniform reference baseline)
    - CS  : Combined LS/CLS scores (weighted mixture)

    Since X_list and y_list do not change across seeds, these scores can be
    cached and reused for all outer repetitions. This eliminates redundant
    score computation inside the main sampling-variance loop..

    Parameters
    ----------
    k : int
        Target rank for the CUR approximation. Affects LS and CS scores.
    X_list : list of DataFrames
        Full design matrices for all replications (before train/test splitting).
    y_list : list of DataFrames or Series
        Full response vectors for all replications.

    Returns
    -------
    tuple
        scores : dict
            Dictionary with keys {"LS", "CLS", "RS", "CS"}.
            Each entry is a list of score vectors, one per replication.
        time_scores : dict
            Dictionary with the same keys, containing the computation
            time (in seconds) for each score vector.
    """
    print(f"computing scores for k={k}...")
    # Initialize containers for score vectors and timing information
    scores = {"LS": [], "CLS": [], "RS": [], "CS": []}
    time_scores = {"LS": [], "CLS": [], "RS": [], "CS": []}

    # Loop over all replications (X1..X_10)
    for i in range(len(X_list)):
        X = X_list[i]
        y = y_list[i]

        # LS: Column leverage scores
        start = time.perf_counter()
        scores["LS"].append(get_column_leverage_scores(X, k))
        time_scores["LS"].append(time.perf_counter() - start)

        # CLS: Cross-leverage scores
        start = time.perf_counter()
        scores["CLS"].append(get_cross_leverage_scores(X, y))
        time_scores["CLS"].append(time.perf_counter() - start)

        # RS: Random scores
        start = time.perf_counter()
        scores["RS"].append(get_random_scores(X))
        time_scores["RS"].append(time.perf_counter() - start)

        # CS: Combined LS/CLS scores
        scores["CS"].append(get_combined_scores(X, y, k, p_leverage=0.2, ls=scores["LS"][i], cls=np.abs(scores["CLS"][i])))
        time_scores["CS"].append(time_scores["CLS"][i] + time_scores["LS"][i])

    return scores, time_scores

def data_reduction(k, df_train, y_train, row_reduce=True,
                   cached_scores=None, cached_time_scores=None):
    """
    Perform column reduction and optional row reduction using importance scores.

    This function assumes that all score vectors (LS, CLS, RS, CS) have already
    been computed externally and passed via `cached_scores`. It therefore performs
    only the reduction steps:
    - EXPECTED(c) importance-based column reduction for each score type
    - EXPECTED(r) importance-based row reduction (optional)

    Parameters
    ----------
    k : int
        Target rank for CUR approximation.
    df_train : list of DataFrames
        Full training matrices for each replication.
    y_train : list of Series
        Full training response vectors.
    row_reduce : bool
        Whether row reduction should be applied.
    cached_scores : dict
        Precomputed score vectors for all replications and all score types.
        Structure: cached_scores[method][i] → score vector for replication i.
    cached_time_scores : dict
        Precomputed timing information for score computation. Passed through
        unchanged for consistency.

    Returns
    -------
    tuple
        scores : dict
            The same cached score dictionary (passed through unchanged).
        time_scores : dict
            Cached timing information (passed through unchanged).
        C : dict
            Column-reduced matrices and selected column indices.
        R : dict or None
            Row-reduced matrices and selected row indices (if enabled).
    """

    # Pass-through of cached score information
    scores = cached_scores
    time_scores = cached_time_scores

    # Number of replications
    n_reps = len(df_train)

    # Column reduction for each score type
    C = {"LS": [], "CLS": [], "RS": [], "CS": []}

    # iterate over datasets and perform column reduction
    for i in range(n_reps):
        X = df_train[i]
        C["LS"].append(column_reduction(X, scores["LS"][i], k))
        C["CLS"].append(column_reduction(X, np.abs(scores["CLS"][i]), k))
        C["RS"].append(column_reduction(X, scores["RS"][i], k))
        C["CS"].append(column_reduction(X, scores["CS"][i], k))

    # Optional row reduction
    R = None
    if row_reduce:
        R = {"LS": [], "CLS": [], "RS": [], "CS": []}

        # iterate over datasets and perform row reduction
        for i in range(n_reps):
            y = y_train[i]

            R["LS"].append(row_reduction(C["LS"][i]["C"], y, k))
            R["CLS"].append(row_reduction(C["CLS"][i]["C"], y, k))
            R["RS"].append(row_reduction(C["RS"][i]["C"], y, k))
            R["CS"].append(row_reduction(C["CS"][i]["C"], y, k))

    return scores, time_scores, C, R

def linear_modeling(C, R, X_train_list, X_test_list, y_train_list, y_test_list):
    """
    Fit linear regression models on reduced matrices and compute RMSE.

    Parameters
    ----------
    C : dict
        Column-reduced matrices and selected columns.
    R : dict or None
        Row-reduced matrices and selected rows.
    X_train_list : list of ndarray
        Training matrices.
    X_test_list : list of ndarray
        Test matrices.
    y_train_list : list of ndarray
        Training responses.
    y_test_list : list of ndarray
        Test responses.

    Returns
    -------
    tuple
        (rmse_train, rmse_test)
    """

    n_reps = len(X_test_list)

    # initialize results
    rmse_train = {"LS": [], "CLS": [], "RS": [], "CS": []}
    rmse_test = {"LS": [], "CLS": [], "RS": [], "CS": []}

    # initialize time containers
    time_model = {"LS": [], "CLS": [], "RS": [], "CS": []}

    # iterate over datasets
    for i in range(n_reps):

        # catch the optional row reduction
        if R is not None:

            # loop over methods
            for method in ["LS", "CLS", "RS", "CS"]:

                t0 = time.perf_counter()
                model = LinearRegression().fit(R[method][i]["R"], R[method][i]["y"])
                time_model[method].append(time.perf_counter() - t0)

                # get selected columns
                cols = C[method][i]["selected_columns"]

                # compute train RMSE
                X_train_red = X_train_list[i][:, cols]
                preds_train = model.predict(X_train_red)
                rmse_train[method].append(
                    np.sqrt(mean_squared_error(y_train_list[i], preds_train))
                )

                # compute test RMSE
                X_test_red = X_test_list[i][:, cols]
                preds_test = model.predict(X_test_red)
                rmse_test[method].append(
                    np.sqrt(mean_squared_error(y_test_list[i], preds_test))
                )

        else:
            y_tr = y_train_list[i]

            # loop over methods
            for method in ["LS", "CLS", "RS", "CS"]:
                t0 = time.perf_counter()
                model = LinearRegression().fit(C[method][i]["C"], y_tr)
                time_model[method].append(time.perf_counter() - t0)

                cols = C[method][i]["selected_columns"]

                # train
                X_train_red = X_train_list[i][:, cols]
                preds_train = model.predict(X_train_red)
                rmse_train[method].append(
                    np.sqrt(mean_squared_error(y_train_list[i], preds_train))
                )

                # test
                X_test_red = X_test_list[i][:, cols]
                preds_test = model.predict(X_test_red)
                rmse_test[method].append(
                    np.sqrt(mean_squared_error(y_test_list[i], preds_test))
                )

    return rmse_train, rmse_test, time_model

def compute_full_rmse(X_train_list, X_test_list, y_train_list, y_test_list, base, folder):
    """
    Compute benchmark RMSE using the true non-zero coefficients.

    Parameters
    ----------
    X_train_list : list of ndarray
        Training matrices.
    X_test_list : list of ndarray
        Test matrices.
    y_train_list : list of ndarray
        Training responses.
    y_test_list : list of ndarray
        Test responses.
    base : str
        Base directory.
    folder : str
        Subfolder containing beta files.

    Returns
    -------
    list
        Benchmark RMSE values for each replication.
    """

    rmse_full = []

    # iterate over all datasets
    for i in range(len(X_train_list)):

        # load true beta vector
        beta = pd.read_csv(
            f"{base}/{folder}/beta{i + 1}.csv",
            header=0
        ).to_numpy().reshape(-1)

        # identify non-zero coefficients
        selected = np.where(beta != 0)[0]

        # extract relevant columns
        X_train = X_train_list[i][:, selected]
        X_test = X_test_list[i][:, selected]

        # extract responses
        y_tr = y_train_list[i]
        y_te = y_test_list[i]

        # fit benchmark model
        model = LinearRegression().fit(X_train, y_tr)

        # compute predictions
        preds = model.predict(X_test)

        # compute RMSE
        rmse_full.append(np.sqrt(mean_squared_error(y_te, preds)))

        del X_train, X_test, y_tr, y_te, preds
        gc.collect()

    return rmse_full


def apply_row_after_col_reduction(
        k,
        seed,
        X_train_list,
        X_test_list,
        y_train_list,
        y_test_list,
        base,
        folder,
        row_reduction=True,
        cached_scores=None,
        cached_time_scores=None
):
    """
    Execute the importance-based reduction procedure and modeling pipeline.

    This function assumes that all score vectors (LS, CLS, RS, CS) have already
    been computed externally and passed via `cached_scores`. It therefore performs:

        1. Set the random seed for reproducibility.
        2. Perform EXPECTED(c) column reduction using cached score vectors.
        3. Optionally perform EXPECTED(r) row reduction.
        4. Fit linear regression models on the reduced matrices.
        5. Compute train/test RMSE for all CUR variants.
        6. Compute the oracle RMSE using the true non-zero coefficients.
        7. Return all reduction and modeling outputs in a structured dictionary.

    Parameters
    ----------
    k : int
        Target rank for CUR approximation.

    seed : int
        Random seed for reproducibility.

    X_train_list : list of ndarray
        List of training design matrices, one per replication.
        Each element has shape (n_train, p).

    X_test_list : list of ndarray
        List of test design matrices, one per replication.
        Each element has shape (n_test, p).

    y_train_list : list of ndarray
        List of training response vectors, one per replication.
        Each element has shape (n_train,).

    y_test_list : list of ndarray
        List of test response vectors, one per replication.
        Each element has shape (n_test,).

    base : str
        Base directory containing the simulation data.

    folder : str
        Subfolder containing the beta coefficient files.

    row_reduction : bool, default=True
        Whether EXPECTED(r) row reduction should be applied after column reduction.

    cached_scores : dict
        Precomputed score vectors for all replications and score types.
        Structure:
            cached_scores[method][i] → score vector for replication i.

    cached_time_scores : dict
        Precomputed timing information for score computation.
        Passed through unchanged.

    Returns
    -------
    dict
        Dictionary containing:
            - scores : cached score vectors
            - time_scores : cached timing information
            - selected_columns : selected column indices for each method
            - selected_rows : selected row indices (if row reduction enabled)
            - rmse_train : training RMSE values
            - rmse_test : test RMSE values including oracle benchmark
    """

    reps = len(X_train_list)

    # set seed
    print(f"Setting random seed inside importance-based reduction pipeline (seed = {seed})...")
    random.seed(seed)
    np.random.seed(seed)

    # data reduction using scores
    print("Performing data reduction (column + optional row reduction)...")
    scores, time_scores, C, R = data_reduction(
        k=k,
        df_train=X_train_list,
        y_train=y_train_list,
        row_reduce=row_reduction,
        cached_scores=cached_scores,
        cached_time_scores=cached_time_scores
    )

    # Soft abort: check if any C has more columns than rows
    for method in ["LS", "CLS", "RS", "CS"]:
        for i in range(len(X_train_list)):
            Cmat = C[method][i]["C"]
            if Cmat.shape[1] > Cmat.shape[0]:
                print(f"[WARN] Aborting k={k} for method={method}: "
                      f"{Cmat.shape[1]} columns > {Cmat.shape[0]} rows. Skipping.")

                # return a clean "empty" result for this k
                return {
                    "scores": scores,
                    "time_scores": time_scores,
                    "time_model": {m: [np.nan] * len(X_train_list) for m in ["LS", "CLS", "RS", "CS"]},
                    "selected_columns": None,
                    "selected_rows": None,
                    "rmse_train": {m: [np.nan] * len(X_train_list) for m in ["LS", "CLS", "RS", "CS"]},
                    "rmse_test": {m: [np.nan] * len(X_train_list) for m in ["LS", "CLS", "RS", "CS"]},
                }

    # perform linear modeling
    print("Fitting linear models on reduced data...")
    rmse_train, rmse_test, time_model = linear_modeling(
        C=C,
        R=R,
        X_train_list=X_train_list,
        X_test_list=X_test_list,
        y_train_list=y_train_list,
        y_test_list=y_test_list
    )

    # perform benchmark
    print("Computing full-model benchmark (RMSE)...")
    rmse_full = compute_full_rmse(
        X_train_list=X_train_list,
        X_test_list=X_test_list,
        y_train_list=y_train_list,
        y_test_list=y_test_list,
        base=base,
        folder=folder
    )
    rmse_test["Full"] = rmse_full

    print("Data Reduction & Modeling completed.")

    # get selected columns/rows
    selected_columns_clean = {
        method: [C[method][i]["selected_columns"] for i in range(reps)]
        for method in ["LS", "CLS", "RS", "CS"]
    }

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
        "time_model": time_model,
        "selected_columns": selected_columns_clean,
        "selected_rows": selected_rows_clean,
        "rmse_train": rmse_train,
        "rmse_test": rmse_test
    }

def numpy_train_test_split(X, y, test_size, seed):
    """
    self-implemented NumPy-based train/test split.

    Parameters
    ----------
    X : ndarray, shape (n, p)
        design matrix.
    y : ndarray, shape (n,)
        response vector.
    test_size : float
        Fraction of samples assigned to the test set.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test : ndarray
        Deterministic train/test split based on shuffled indices.
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

def run_sampling_variance_row_after_col(
        k_vector,
        base,
        folder,
        reps=10,
        outer_reps=10,
        test_size=0.2,
        save_name=None,
        results_folder=None,
        row_reduction=True,
        seed_from=1,
        seed_to=None
):
    """
    Run the sampling-variance study for the importance-based subsampling procedure.

    Workflow:
    1. Load all full datasets X1..X_10 and y1..y_10.
    2.For each seed:
         - Create train/test splits for all 10 datasets.
         - Compute all score vectors for each k in k_vector for all 10 datasets.
         - For each k and dataset:
               - Apply importance-based data reduction using cached scores.
               - Fit linear models.
               - Compute RMSE and benchmark.
         - Save results per seed.
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

    # default seed_to
    if seed_to is None:
        seed_to = outer_reps

    # iterate over the seeds
    for seed in range(seed_from, seed_to + 1):

        print(f"\n============================================================")
        print(f"Running seed = {seed}")
        print("============================================================")

        np.random.seed(seed+42)
        random.seed(seed+42)

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

        # Compute Scores on training data
        cached_scores = {}
        cached_time_scores = {}

        print("\nComputing scores for current seed...")
        for k in k_vector:
            scores_k, time_scores_k = compute_scores(k, X_train_list, y_train_list)
            cached_scores[k] = scores_k
            cached_time_scores[k] = time_scores_k

        # initialize seed results
        results[seed] = {}

        # Run for each k
        for k in k_vector:
            print(f"\n--- Running k = {k} ---")

            # perform reduction pipeline
            out = apply_row_after_col_reduction(
                k=k,
                seed=seed,
                X_train_list=X_train_list,
                X_test_list=X_test_list,
                y_train_list=y_train_list,
                y_test_list=y_test_list,
                base=base,
                folder=folder,
                row_reduction=row_reduction,
                cached_scores=cached_scores[k],
                cached_time_scores=cached_time_scores[k]
            )

            # get errors
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
                "time_model": model_time_summary
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

    return results