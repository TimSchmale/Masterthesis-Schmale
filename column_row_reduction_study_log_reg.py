import pandas as pd
import random
from scoring_functions import get_column_leverage_scores, get_log_reg_leverage_scores, get_random_scores, get_combined_scores, get_row_leverage_scores, get_cross_leverage_scores
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss
import time
from visualizations import *
import pickle

def column_reduction(X, scores, k):
    """
    Column reduction using the EXPECTED(c) sampling rule.

    Columns are sampled independently using scaled probabilities
    derived from the provided score vector. Selected columns are
    rescaled according to the CUR theorem. The reduced matrix C
    is returned only internally; the sampling variance pipeline
    extracts only the selected column indices.

    Parameters
    ----------
    X : array-like
        Design matrix of shape (n, d).
    scores : array-like
        Column score vector of length d.
    k : int
        Target rank for CUR approximation.

    Returns
    -------
    dict
        Dictionary containing:
            "C"               : reduced matrix (n x t)
            "selected_columns": list of selected column indices
    """

    # convert to numpy
    X = np.asarray(X)
    n, d = X.shape

    # compute sampling probabilities
    probs = scores / scores.sum()

    # expected number of sampled columns
    c = int(np.ceil(k * np.log(k)))

    # scaled probabilities
    scaled = np.minimum(c * probs, 1)

    # Bernoulli sampling
    z = np.random.rand(d)
    sampled = np.where(z <= scaled)[0]

    # ensure at least one column
    if len(sampled) == 0:
        sampled = np.array([np.argmax(scaled)])

    # rescaling factors
    D_inv = 1 / np.sqrt(scaled[sampled])

    # reduced matrix
    C = X[:, sampled] * D_inv

    return {
        "C": C,
        "selected_columns": sampled.tolist()
    }

def estimate_mu(C, y):
    """
    Estimate the logistic coreset imbalance parameter μ.

    A logistic regression model is fitted on the column-reduced
    matrix C. The imbalance ratio μ is computed from the signed
    projection values v = Cβ. This parameter controls the row
    sampling distribution in logistic coreset theory.

    Parameters
    ----------
    C : array-like
        Column-reduced matrix (n x t).
    y : array-like
        Binary response vector.

    Returns
    -------
    float
        Estimated imbalance parameter μ.
    """

    # fit logistic regression
    model = LogisticRegression(
        l1_ratio=0,
        C=np.inf,
        solver='lbfgs',
        max_iter=2000
    )
    model.fit(C, y)

    # compute signed projection
    beta = model.coef_.flatten()
    v = C @ beta

    # positive and negative mass
    pos = np.sum(np.abs(v[v > 0]))
    neg = np.sum(np.abs(v[v < 0]))

    # imbalance ratio
    mu = np.inf if neg == 0 else pos / neg
    mu = max(mu, 1.0001)

    return mu

def row_reduction(C, y, mu, k):
    """
    Row reduction using logistic coreset sampling.

    Rows are sampled using a mixture of:
        - logistic leverage scores
        - row leverage scores
        - uniform component scaled by μ

    The reduced matrix R and reduced response y are returned
    internally. The sampling variance pipeline extracts only
    the selected row indices.

    Parameters
    ----------
    C : array-like
        Column-reduced matrix (n x t).
    y : array-like
        Binary response vector.
    mu : float
        Logistic imbalance parameter.
    k : int
        Target rank.

    Returns
    -------
    dict
        Dictionary containing:
            "R"            : row-reduced matrix
            "y"            : reduced response
            "selected_rows": list of sampled row indices
            "mu"           : μ value
            "r"            : number of sampled rows
    """

    # convert inputs
    y_arr = np.asarray(y).ravel()
    n, d = C.shape

    # expected number of sampled rows
    r = int(np.ceil(mu * d * np.log(mu * d)))
    r = max(1, min(r, n))

    # compute score components
    l1 = get_log_reg_leverage_scores(C)
    l2 = get_row_leverage_scores(C, k)
    uniform = np.ones(n) / n

    # combined score
    scores = mu * l1 + l2 + mu * d * uniform
    probs = scores / scores.sum()

    # sampling
    rng = np.random.default_rng()
    sampled = rng.choice(n, size=r, replace=False, p=probs)

    return {
        "R": C[sampled, :],
        "y": y_arr[sampled],
        "selected_rows": sampled.tolist(),
        "mu": mu,
        "r": r
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

def data_reduction(
    k,
    df_train,
    y_train,
    row_reduce=True,
    cached_scores=None,
    cached_time_scores=None,
    cached_mu_values=None
):
    """
    Corrected and robust version of the data reduction step.
    """

    n_reps = len(df_train)

    # Pass-through of cached score information
    scores = cached_scores
    time_scores = cached_time_scores
    mu_values = cached_mu_values

    # column reduction
    C_mats = {m: [] for m in scores}

    for i in range(n_reps):
        X = df_train[i]
        for method in scores:
            out = column_reduction(X, np.abs(scores[method][i]), k)
            C_mats[method].append({
                "C": out["C"],
                "selected_columns": out["selected_columns"]
            })

    # row reduction
    R_mats = {m: [] for m in scores}

    if row_reduce:
        for i in range(n_reps):
            y = y_train[i]
            for method in scores:

                C = C_mats[method][i]["C"]

                # compute mu
                mu = estimate_mu(C, y)
                mu_values[method].append(mu)

                # row reduction
                R_out = row_reduction(C, y, mu, k)

                R_mats[method].append({
                    "R": R_out["R"],
                    "y": R_out["y"],
                    "selected_rows": R_out["selected_rows"],
                    "mu": mu,
                    "r": R_out["r"]
                })
    else:
        R_mats = None

    return scores, time_scores, mu_values, C_mats, R_mats


def logistic_modeling(C, R, X_train_list, X_test_list, y_train_list, y_test_list):
    """
    Fit logistic regression models on reduced data and compute Brier
    and cross-entropy losses.

    Reduced matrices are reconstructed on the fly using only the
    selected column and row indices. No reduced matrices are stored.

    Parameters
    ----------
    selected_columns : dict
        Selected column indices per method and replication.
    selected_rows : dict or None
        Selected row indices per method and replication.
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
        (brier_train, brier_test, ce_train, ce_test)
    """

    n_reps = len(X_test_list)

    brier_train = {"LS": [], "CLS": [], "RS": [], "CS": []}
    brier_test  = {"LS": [], "CLS": [], "RS": [], "CS": []}

    ce_train = {"LS": [], "CLS": [], "RS": [], "CS": []}
    ce_test  = {"LS": [], "CLS": [], "RS": [], "CS": []}

    # initialize time containers
    time_model = {"LS": [], "CLS": [], "RS": [], "CS": []}

    # loop over replications
    for i in range(n_reps):

        # catch the optional row reduction
        if R is not None:

            # loop over methods
            for method in ["LS", "CLS", "RS", "CS"]:
                t0 = time.perf_counter()
                model = LogisticRegression().fit(R[method][i]["R"], R[method][i]["y"])
                time_model[method].append(time.perf_counter() - t0)

                # get selected columns
                cols = C[method][i]["selected_columns"]

                # compute train errors
                X_train_red = X_train_list[i][:, cols]
                pred_train = model.predict_proba(X_train_red)[:, 1]
                brier_train[method].append(brier_score_loss(y_train_list[i], pred_train))
                ce_train[method].append(log_loss(y_train_list[i], pred_train))

                # compute test errors
                X_test_red = X_test_list[i][:, cols]
                pred_test = model.predict_proba(X_test_red)[:, 1]
                brier_test[method].append(brier_score_loss(y_test_list[i], pred_test))
                ce_test[method].append(log_loss(y_test_list[i], pred_test))

        else:
            # loop over methods
            for method in ["LS", "CLS", "RS", "CS"]:
                t0 = time.perf_counter()
                model = LogisticRegression().fit(C[method][i]["C"], y_train_list[i])
                time_model[method].append(time.perf_counter() - t0)

                # get selected columns
                cols = C[method][i]["selected_columns"]

                # compute train errors
                X_train_red = X_train_list[i][:, cols]
                pred_train = model.predict_proba(X_train_red)[:, 1]
                brier_train[method].append(brier_score_loss(y_train_list[i], pred_train))
                ce_train[method].append(log_loss(y_train_list[i], pred_train))

                # compute test errors
                X_test_red = X_test_list[i][:, cols]
                pred_test = model.predict_proba(X_test_red)[:, 1]
                brier_test[method].append(brier_score_loss(y_test_list[i], pred_test))
                ce_test[method].append(log_loss(y_test_list[i], pred_test))

    return brier_train, brier_test, ce_train, ce_test

def compute_full_model(
        X_train_list,
        X_test_list,
        y_train_list,
        y_test_list,
        base,
        folder):
    """
    Compute oracle Brier loss and cross-entropy loss using the true support of β.

    The true β vector is loaded for each replication. The non-zero entries define
    the oracle support. Logistic regression is fitted on the oracle columns and
    evaluated on the test set using both Brier score and cross-entropy.

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
    tuple
        (brier_full, ce_full)
        where each is a list of length n_reps.
    """

    brier_full = []
    ce_full = []

    for i in range(len(X_train_list)):

        # load true beta
        beta_df = pd.read_csv(f"{base}/{folder}/beta{i + 1}.csv")
        beta = beta_df.select_dtypes(include=[np.number]).to_numpy().reshape(-1)

        # ensure correct length
        p = X_train_list[i].shape[1]
        if len(beta) > p:
            beta = beta[:p]
        elif len(beta) < p:
            raise ValueError(f"Beta length {len(beta)} < number of features {p}.")

        # oracle support
        selected = np.where(beta != 0)[0]

        # subset matrices
        X_train = X_train_list[i][:, selected]
        X_test = X_test_list[i][:, selected]

        y_tr = y_train_list[i].ravel()
        y_te = y_test_list[i].ravel()

        # fit logistic regression
        model = LogisticRegression(
            l1_ratio=0,
            C=np.inf,
            solver='lbfgs',
            max_iter=2000
        ).fit(X_train, y_tr)

        # predict probabilities
        pred = model.predict_proba(X_test)[:, 1]

        # compute metrics
        brier_full.append(brier_score_loss(y_te, pred))
        ce_full.append(log_loss(y_te, pred))

    return brier_full, ce_full

def apply_row_after_col_reduction_log(
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
        cached_time_scores=None,
        cached_mu_values=None,
):
    """
    Full logistic Column→Row reduction pipeline.

    This wrapper loads the simulation data, computes score vectors,
    performs column and optional row reduction, fits logistic models,
    and computes the oracle benchmark. Only selected indices and
    score vectors are returned to keep memory usage minimal.

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
            brier_train
            brier_test
            ce_train
            ce_test
            mu
    """

    reps = len(X_train_list)

    # set seed
    print(f"Setting random seed inside importance-based reduction pipeline for log. regression (seed = {seed})...")
    random.seed(seed)
    np.random.seed(seed)

    print("Performing data reduction...")
    scores, time_scores, mu_values, C, R = data_reduction(
        k=k,
        df_train=X_train_list,
        y_train=y_train_list,
        row_reduce=row_reduction,
        cached_scores=cached_scores,
        cached_time_scores=cached_time_scores,
        cached_mu_values=cached_mu_values,
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
                    "brier_train": {m: [np.nan] * len(X_train_list) for m in ["LS", "CLS", "RS", "CS"]},
                    "brier_test": {m: [np.nan] * len(X_train_list) for m in ["LS", "CLS", "RS", "CS"]},
                    "ce_train": {m: [np.nan] * len(X_train_list) for m in ["LS", "CLS", "RS", "CS"]},
                    "ce_test": {m: [np.nan] * len(X_train_list) for m in ["LS", "CLS", "RS", "CS"]},
                    "mu": mu_values
                }

    print("Building logistic models...")
    brier_train, brier_test, ce_train, ce_test, time_model = logistic_modeling(
        C=C,
        R=R,
        X_train_list=X_train_list,
        X_test_list=X_test_list,
        y_train_list=y_train_list,
        y_test_list=y_test_list
    )

    print("Building Full Model / Benchmark...")
    full_brier, full_ce = compute_full_model(
        X_train_list=X_train_list,
        X_test_list=X_test_list,
        y_train_list=y_train_list,
        y_test_list=y_test_list,
        base=base,
        folder=folder,
    )
    brier_test["Full"] = full_brier
    ce_test["Full"] = full_ce

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
        "brier_train": brier_train,
        "brier_test": brier_test,
        "ce_train": ce_train,
        "ce_test": ce_test,
        "mu": mu_values
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

def run_sampling_variance_row_after_col_log(
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
    Sampling variance analysis for the logistic Column→Row pipeline.

    This optimized version stores only:
        - score vectors
        - score computation times
        - selected column indices
        - selected row indices
        - Brier scores (train/test)
        - Cross-entropy (train/test)
        - μ values

    No reduced matrices (C or R) are stored.

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

    print("Loading all datasets...")

    # Load all datasets
    X_list = [
        pd.read_csv(f"{base}/{folder}/X{i + 1}.csv").to_numpy()
        for i in range(reps)
    ]
    y_list = [
        pd.read_csv(f"{base}/{folder}/y_binary{i + 1}.csv").to_numpy().reshape(-1)
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

        # calculate mu values
        cached_mu_values = {m: [] for m in ["LS", "CLS", "RS", "CS"]}

        # Run for each k
        for k in k_vector:
            print(f"\n--- Running k = {k} ---")

            # perform reduction pipeline
            out = apply_row_after_col_reduction_log(
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
                cached_time_scores=cached_time_scores[k],
                cached_mu_values=cached_mu_values
            )

        # get errors
        rmse_test = out["brier_test"]
        rmse_train = out["brier_train"]
        ce_train = out["ce_train"]
        ce_test = out["ce_test"]

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

        # summarize cross entropy
        ce_summary = {
            method: {
                "raw": ce_test[method],
                "mean": float(np.mean(ce_test[method])),
                "median": float(np.median(ce_test[method]))
            }
            for method in ce_test.keys()
        }

        # same for training
        ce_train_summary = {
            method: {
                "raw": ce_train[method],
                "mean": float(np.mean(ce_train[method])),
                "median": float(np.median(ce_train[method]))
            }
            for method in ce_train.keys()
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

        # store everything for this seed
        results[seed] = {
            "loss": loss_summary,
            "train_loss": train_loss_summary,
            "selected_columns": out["selected_columns"],
            "selected_rows": out["selected_rows"],
            "scores": out["scores"],
            "time_scores": score_time_summary,
            "time_model": model_time_summary,
            "ce": ce_summary,
            "ce_train": ce_train_summary,
            "mu": out["mu"]
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
