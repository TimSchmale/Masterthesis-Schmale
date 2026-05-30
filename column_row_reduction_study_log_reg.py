import pandas as pd
import random
from scoring_functions import get_column_leverage_scores, get_log_reg_leverage_scores, get_random_scores, get_combined_scores, get_row_leverage_scores, get_cross_leverage_scores
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss
import time
from visualizations import *

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

def data_reduction(k, df_train, y_train, row_reduce=True):
    """
    Compute score vectors and perform column and optional row reduction.

    This function computes LS, CLS, RS, and CS scores for each
    replication. Column reduction is performed using EXPECTED(c)
    sampling. If enabled, row reduction is performed using logistic
    coreset sampling. Only selected indices and score vectors are
    returned to keep memory usage minimal.

    Parameters
    ----------
    k : int
        Target rank.
    df_train : list of DataFrames
        Training matrices.
    y_train : list of Series
        Binary response vectors.
    row_reduce : bool
        Whether row reduction should be applied.

    Returns
    -------
    tuple
        (scores, time_scores, selected_columns, selected_rows, mu_values)
    """

    n_reps = len(df_train)

    # containers for scores and times
    scores = {"LS": [], "CLS": [], "RS": [], "CS": []}
    time_scores = {"LS": [], "CLS": [], "RS": [], "CS": []}

    # containers for selected indices
    selected_columns = {"LS": [], "CLS": [], "RS": [], "CS": []}
    selected_rows = {"LS": [], "CLS": [], "RS": [], "CS": []} if row_reduce else None
    mu_values = {"LS": [], "CLS": [], "RS": [], "CS": []}

    # score computation
    for i in range(n_reps):
        X = df_train[i]
        y = y_train[i]

        for method, func in {
            "LS": lambda: get_column_leverage_scores(X, k),
            "CLS": lambda: get_cross_leverage_scores(X, y),
            "RS": lambda: get_random_scores(X),
            "CS": lambda: get_combined_scores(X, y, k, p_leverage=0.2)
        }.items():

            start = time.perf_counter()
            s = func()
            time_scores[method].append(time.perf_counter() - start)
            scores[method].append(s)

    # column reduction
    C_mats = {m: [] for m in scores}
    for i in range(n_reps):
        X = df_train[i]
        for method in scores:
            out = column_reduction(X, np.abs(scores[method][i]), k)
            C_mats[method].append(out["C"])
            selected_columns[method].append(out["selected_columns"])

    # row reduction
    if row_reduce:
        for i in range(n_reps):
            y = y_train[i]
            for method in scores:
                C = C_mats[method][i]

                mu = estimate_mu(C, y)
                mu_values[method].append(mu)

                R_out = row_reduction(C, y, mu, k)
                selected_rows[method].append(R_out["selected_rows"])

    return scores, time_scores, selected_columns, selected_rows, mu_values

def logistic_modeling(selected_columns, selected_rows, df_train, df_test, y_train, y_test):
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

    n_reps = len(df_train)

    brier_train = {"LS": [], "CLS": [], "RS": [], "CS": []}
    brier_test  = {"LS": [], "CLS": [], "RS": [], "CS": []}

    ce_train = {"LS": [], "CLS": [], "RS": [], "CS": []}
    ce_test  = {"LS": [], "CLS": [], "RS": [], "CS": []}

    # loop over replications
    for i in range(n_reps):
        for method in selected_columns:

            cols = selected_columns[method][i]

            # training subset
            if selected_rows is not None:
                rows = selected_rows[method][i]
                X_train = df_train[i].to_numpy()[rows][:, cols]
                y_tr = y_train[i].to_numpy().ravel()[rows]
            else:
                X_train = df_train[i].to_numpy()[:, cols]
                y_tr = y_train[i].to_numpy().ravel()

            # fit logistic regression
            model = LogisticRegression(
                l1_ratio=0,
                C=np.inf,
                solver='lbfgs',
                max_iter=2000
            ).fit(X_train, y_tr)

            # train predictions
            pred_train = model.predict_proba(X_train)[:, 1]
            brier_train[method].append(brier_score_loss(y_tr, pred_train))
            ce_train[method].append(log_loss(y_tr, pred_train))

            # test predictions
            X_test = df_test[i].to_numpy()[:, cols]
            pred_test = model.predict_proba(X_test)[:, 1]
            brier_test[method].append(brier_score_loss(y_test[i], pred_test))
            ce_test[method].append(log_loss(y_test[i], pred_test))

    return brier_train, brier_test, ce_train, ce_test

def compute_full_model(df_train, df_test, y_train, y_test, base, folder):
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

    for i in range(len(df_train)):

        # load true beta
        beta_df = pd.read_csv(f"{base}/{folder}/beta{i+1}.csv")
        beta = beta_df.select_dtypes(include=[np.number]).to_numpy().reshape(-1)

        # ensure correct length
        p = df_train[i].shape[1]
        if len(beta) > p:
            beta = beta[:p]
        elif len(beta) < p:
            raise ValueError(f"Beta length {len(beta)} < number of features {p}.")

        # oracle support
        selected = np.where(beta != 0)[0]

        # subset matrices
        X_train = df_train[i].to_numpy()[:, selected]
        X_test  = df_test[i].to_numpy()[:, selected]

        y_tr = y_train[i].to_numpy().ravel()
        y_te = y_test[i].to_numpy().ravel()

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


def apply_row_after_col_reduction_log(k, seed, base, folder, reps, row_reduction=True):
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

    print("Reading in the simulation data...")
    df_train = [pd.read_csv(f"{base}/{folder}/X_train{i+1}.csv") for i in range(reps)]
    df_test  = [pd.read_csv(f"{base}/{folder}/X_test{i+1}.csv") for i in range(reps)]
    y_train  = [pd.read_csv(f"{base}/{folder}/y_binary_train{i+1}.csv") for i in range(reps)]
    y_test   = [pd.read_csv(f"{base}/{folder}/y_binary_test{i+1}.csv") for i in range(reps)]

    print("Setting the seed...")
    random.seed(seed)
    np.random.seed(seed)

    print("Performing data reduction...")
    scores, time_scores, selected_columns, selected_rows, mu_values = \
        data_reduction(k, df_train, y_train, row_reduction)

    print("Building logistic models...")
    brier_train, brier_test, ce_train, ce_test = logistic_modeling(
        selected_columns,
        selected_rows,
        df_train,
        df_test,
        y_train,
        y_test
    )

    print("Building Full Model / Benchmark...")
    full_brier, full_ce = compute_full_model(
        df_train, df_test, y_train, y_test, base, folder
    )

    brier_test["Full"] = full_brier
    ce_test["Full"] = full_ce

    print("Data Reduction & Modeling completed.")

    return {
        "scores": scores,
        "time_scores": time_scores,
        "selected_columns": selected_columns,
        "selected_rows": selected_rows,
        "brier_train": brier_train,
        "brier_test": brier_test,
        "ce_train": ce_train,
        "ce_test": ce_test,
        "mu": mu_values
    }

def run_sampling_variance_row_after_col_log(
    k,
    base,
    folder,
    reps=10,
    outer_reps=10,
    row_reduction=True
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

    results = {}

    for outer_seed in range(1, outer_reps + 1):

        print(f"Running outer repetition with seed = {outer_seed}")

        out = apply_row_after_col_reduction_log(
            k=k,
            seed=outer_seed,
            base=base,
            folder=folder,
            reps=reps,
            row_reduction=row_reduction
        )

        # extract components
        scores = out["scores"]
        time_scores = out["time_scores"]
        selected_columns = out["selected_columns"]
        selected_rows = out["selected_rows"]
        brier_train = out["brier_train"]
        brier_test = out["brier_test"]
        ce_train = out["ce_train"]
        ce_test = out["ce_test"]
        mu_values = out["mu"]

        # summary containers
        brier_summary = {}
        ce_summary = {}

        # summarize Brier + CE for each method
        for method in brier_test.keys():

            # --- Test Brier ---
            raw_test = brier_test[method]
            brier_summary[method] = {
                "raw": raw_test,
                "mean": float(np.mean(raw_test)),
                "median": float(np.median(raw_test))
            }

            # --- Test CE ---
            raw_ce_test = ce_test[method]
            ce_summary[method] = {
                "raw": raw_ce_test,
                "mean": float(np.mean(raw_ce_test)),
                "median": float(np.median(raw_ce_test))
            }

            # --- Skip training metrics for Full ---
            if method == "Full":
                continue

            # --- Train Brier ---
            raw_train = brier_train[method]
            brier_summary[method]["train_raw"] = raw_train
            brier_summary[method]["train_mean"] = float(np.mean(raw_train))
            brier_summary[method]["train_median"] = float(np.median(raw_train))

            # --- Train CE ---
            raw_ce_train = ce_train[method]
            ce_summary[method]["train_raw"] = raw_ce_train
            ce_summary[method]["train_mean"] = float(np.mean(raw_ce_train))
            ce_summary[method]["train_median"] = float(np.median(raw_ce_train))

        # store everything for this seed
        results[outer_seed] = {
            "scores": scores,
            "time_scores": time_scores,
            "selected_columns": selected_columns,
            "selected_rows": selected_rows,
            "brier": brier_summary,
            "ce": ce_summary,
            "mu": mu_values
        }

    return results
