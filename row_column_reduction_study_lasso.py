import numpy as np
import pandas as pd
import time
from scoring_functions import get_column_leverage_scores, get_row_leverage_scores, get_random_scores, get_combined_scores, get_cross_leverage_scores
from statsmodels.sandbox.distributions.genpareto import shape

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import random
from visualizations import *

def row_reduction(k, X, y, gaussian=False):
    """
    Row reduction using sketching-based dimensionality reduction.

    A sketch matrix of size r × n is generated using either Gaussian
    or Rademacher random vectors. Each row of X contributes to the
    sketch via an outer product with the random vector. The resulting
    sketched matrix R approximates X in a lower-dimensional space.
    The response vector y is sketched in the same manner.

    Parameters
    ----------
    k : int
        Target rank for sketch size.
    X : array-like or DataFrame
        Design matrix of shape (n, d).
    y : array-like or Series
        Response vector of length n.
    gaussian : bool
        If True, Gaussian sketching is used. Otherwise Rademacher.

    Returns
    -------
    R : ndarray
        Row-reduced matrix of shape (r, d).
    y_reduced : ndarray
        Sketched response vector of shape (r, 1).
    """

    # convert inputs
    X = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
    y = y.to_numpy().reshape(-1, 1) if isinstance(y, (pd.Series, pd.DataFrame)) else y.reshape(-1, 1)

    n, d = X.shape
    r = int(np.ceil(k * np.log(d)))

    R = np.zeros((r, d))
    y_reduced = np.zeros((r, 1))

    # iterative sketching
    if r < n:
        for i in range(n):
            sketch = np.random.randn(r, 1) if gaussian else np.random.choice([-1, 1], size=(r, 1)) / np.sqrt(r)
            R += sketch @ X[i:i+1, :]
            y_reduced += sketch * y[i]
    else:
        R = X.copy()
        y_reduced = y.copy()

    return R, y_reduced

def fit_theoretical_lasso(R, y, k):
    """
    Fit Lasso using the theoretical regularization parameter α = 1/sqrt(k).

    The matrix R is standardized before fitting. The number of selected
    features is determined by counting non-zero coefficients.

    Parameters
    ----------
    R : ndarray
        Row-reduced matrix.
    y : ndarray
        Reduced response vector.
    k : int
        Target rank.

    Returns
    -------
    model : Lasso
        Fitted Lasso model.
    scaler : StandardScaler
        Scaler used for standardizing R.
    n_features : int
        Number of non-zero coefficients.
    """

    alpha = 1 / np.sqrt(k)

    scaler = StandardScaler()
    R_scaled = scaler.fit_transform(R)

    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(R_scaled, y)

    n_features = np.sum(model.coef_ != 0)
    return model, scaler, n_features

def find_best_alpha(R, y, k, max_iter=30):
    """
    Perform binary search to identify a regularization parameter α that yields
    approximately k non-zero coefficients in a Lasso model.

    The function standardizes the design matrix, iteratively evaluates Lasso
    fits for candidate α values, and selects the α that minimizes the deviation
    from the target sparsity level. Only the best α and the fitted scaler are
    returned; no final model is produced here.

    Parameters
    ----------
    R : ndarray
        Design matrix of shape (n, d).
    y : ndarray
        Response vector of length n.
    k : int
        Target number of non-zero coefficients.
    max_iter : int
        Number of binary search iterations.

    Returns
    -------
    best_alpha : float
        Regularization parameter yielding sparsity closest to k.
    scaler : StandardScaler
        Scaler fitted on R for later reuse.
    """

    # standardize design matrix
    scaler = StandardScaler()
    R_scaled = scaler.fit_transform(R)

    # initialize search bounds
    alpha_low, alpha_high = 1e-6, 1e2
    best_alpha, best_diff = None, np.inf

    # binary search loop
    for _ in range(max_iter):
        alpha_mid = np.sqrt(alpha_low * alpha_high)
        model = Lasso(alpha=alpha_mid, max_iter=10000)
        model.fit(R_scaled, y)

        n_features = np.sum(model.coef_ != 0)
        diff = abs(n_features - k)

        if diff < best_diff:
            best_diff = diff
            best_alpha = alpha_mid

        if n_features > k:
            alpha_low = alpha_mid
        else:
            alpha_high = alpha_mid

    return best_alpha, scaler

def final_lasso_fit(R_scaled, y, alpha):
    """
    Fit a Lasso model using a fixed regularization parameter α.

    This function performs the final Lasso fit after α has been selected
    via binary search. It is intended to be timed separately to ensure
    fair runtime comparison across methods.

    Parameters
    ----------
    R_scaled : ndarray
        Standardized design matrix.
    y : ndarray
        Response vector.
    alpha : float
        Regularization parameter.

    Returns
    -------
    model : Lasso
        Fitted Lasso model using the provided α.
    """

    # perform final Lasso fit
    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(R_scaled, y)

    return model

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

def lasso_after_cls_expected(R_reduced, df_train, df_test, y_train, y_test, y_reduced, k):
    """
    Apply CLS-based EXPECTED(c) column reduction followed by binary-search Lasso.

    For each replication, CLS scores are computed from the full training matrix.
    Using these scores, the EXPECTED(c) algorithm with c = ceil(k * log(k)) is
    applied to sample informative columns. The reduced design matrix is then used
    to fit binary-search Lasso. RMSE values, coefficient vectors, feature counts,
    and selected α values are returned.

    Parameters
    ----------
    R_reduced : list of ndarray
        Row-reduced matrices (unused here but kept for consistency).
    df_train : list of DataFrames
        Full training matrices.
    df_test : list of DataFrames
        Full test matrices.
    y_train : list of Series
        Full training responses.
    y_test : list of Series
        Full test responses.
    y_reduced : list of ndarray
        Reduced response vectors (unused here).
    k : int
        Target rank and sampling parameter.

    Returns
    -------
    rmse_train_cls_exp : list
        Training RMSE values.
    rmse_test_cls_exp : list
        Test RMSE values.
    betas_cls_exp : list
        Coefficient vectors.
    feats_cls_exp : list
        Number of selected features.
    alphas_cls_exp : list
        Selected α values.
    """

    # prepare containers
    rmse_train_cls_exp, rmse_test_cls_exp = [], []
    betas_cls_exp, feats_cls_exp, alphas_cls_exp = [], [], []

    # iterate over replications
    for i in range(len(df_train)):

        # compute CLS scores
        X_full = df_train[i].to_numpy()
        y_full = y_train[i].to_numpy().ravel()
        cls_scores = np.abs(get_cross_leverage_scores(X_full, y_full))

        # apply EXPECTED(c) column sampling
        col_red = column_reduction(X_full, cls_scores, k)
        X_train_red = col_red["C"]
        selected_cols = col_red["selected_columns"]

        # reduce test matrix accordingly
        X_test_red = df_test[i].to_numpy()[:, selected_cols]

        # fit binary-search Lasso on reduced matrix
        model_c, scaler_c, alpha_c, n_c = fit_lasso_k_binary(X_train_red, y_full, k)

        # store alpha and feature count
        alphas_cls_exp.append(alpha_c)
        feats_cls_exp.append(n_c)
        betas_cls_exp.append(model_c.coef_)

        # compute RMSE train
        preds_train_c = model_c.predict(scaler_c.transform(X_train_red))
        rmse_train_cls_exp.append(np.sqrt(mean_squared_error(y_train[i], preds_train_c)))

        # compute RMSE test
        preds_test_c = model_c.predict(scaler_c.transform(X_test_red))
        rmse_test_cls_exp.append(np.sqrt(mean_squared_error(y_test[i], preds_test_c)))

    return rmse_train_cls_exp, rmse_test_cls_exp, betas_cls_exp, feats_cls_exp, alphas_cls_exp

def lasso_modeling(R_reduced, df_train, df_test, y_train, y_test, y_reduced, k):
    """
    Fit theoretical, binary-search, and CLS-EXPECTED(c) Lasso models and evaluate them.

    Each replication is processed independently. Models are trained on
    row-reduced data (theoretical + binary) or CLS-EXPECTED(c)-reduced data
    (third method). All models are evaluated on the full training and test sets.
    RMSE, coefficient vectors, feature counts, and α values are returned.

    Parameters
    ----------
    R_reduced : list of ndarray
        Row-reduced matrices for each replication.
    df_train : list of DataFrames
        Full training matrices.
    df_test : list of DataFrames
        Full test matrices.
    y_train : list of Series
        Full training responses.
    y_test : list of Series
        Full test responses.
    y_reduced : list of ndarray
        Reduced response vectors.
    k : int
        Target rank.

    Returns
    -------
    rmse_train : dict
        Training RMSE for all Lasso variants.
    rmse_test : dict
        Test RMSE for all Lasso variants.
    betas : dict
        Coefficient vectors for all methods.
    feature_counts : dict
        Number of selected features for all methods.
    alphas : dict
        Selected α values (binary methods only).
    """

    # containers for theoretical Lasso
    rmse_train_theo, rmse_test_theo = [], []
    feats_theo, betas_theo = [], []
    time_theo = []

    # containers for binary Lasso
    rmse_train_bin, rmse_test_bin = [], []
    feats_bin, betas_bin, alphas_bin = [], [], []
    time_bin = []

    # containers for CLS-EXPECTED(c) Lasso
    rmse_train_cls_exp, rmse_test_cls_exp = [], []
    feats_cls_exp, betas_cls_exp, alphas_cls_exp = [], [], []
    time_cls_exp = []

    # loop over replications
    for i in range(len(R_reduced)):

        # extract row-reduced matrix and response
        R = R_reduced[i]
        y_r = y_reduced[i].ravel()

        # ------------------------------------------------------------
        # THEORETICAL LASSO
        # ------------------------------------------------------------
        t0 = time.time()
        model_t, scaler_t, n_t = fit_theoretical_lasso(R, y_r, k)
        time_theo.append(time.time() - t0)
        feats_theo.append(n_t)
        betas_theo.append(model_t.coef_)

        X_train = df_train[i].to_numpy()
        preds_train_t = model_t.predict(scaler_t.transform(X_train))
        rmse_train_theo.append(np.sqrt(mean_squared_error(y_train[i], preds_train_t)))

        X_test = df_test[i].to_numpy()
        preds_test_t = model_t.predict(scaler_t.transform(X_test))
        rmse_test_theo.append(np.sqrt(mean_squared_error(y_test[i], preds_test_t)))

        # ------------------------------------------------------------
        # BINARY-SEARCH LASSO
        # ------------------------------------------------------------
        best_alpha, scaler_b = find_best_alpha(R, y_r, k)
        R_scaled = scaler_b.transform(R)

        t0 = time.time()
        model_b = final_lasso_fit(R_scaled, y_r, best_alpha)
        time_bin.append(time.time() - t0)

        alphas_bin.append(best_alpha)
        n_b = np.sum(model_b.coef_ != 0)
        feats_bin.append(n_b)
        betas_bin.append(model_b.coef_)

        preds_train_b = model_b.predict(scaler_b.transform(X_train))
        rmse_train_bin.append(np.sqrt(mean_squared_error(y_train[i], preds_train_b)))

        preds_test_b = model_b.predict(scaler_b.transform(X_test))
        rmse_test_bin.append(np.sqrt(mean_squared_error(y_test[i], preds_test_b)))

        # ------------------------------------------------------------
        # CLS-EXPECTED(c) + BINARY LASSO
        # ------------------------------------------------------------
        X_full = df_train[i].to_numpy()
        y_full = y_train[i].to_numpy().ravel()

        t0 = time.time()

        # compute CLS scores
        cls_scores = np.abs(get_cross_leverage_scores(X_full, y_full))

        # apply EXPECTED(c) column sampling
        col_red = column_reduction(X_full, cls_scores, k)
        X_train_red = col_red["C"]
        selected_cols = col_red["selected_columns"]

        # reduce test matrix accordingly
        X_test_red = df_test[i].to_numpy()[:, selected_cols]
        t1 = time.time() - t0

        # binary-search for best alpha (not timed)
        t0 = time.time()
        best_alpha_c, scaler_c = find_best_alpha(X_train_red, y_full, k)
        X_train_red_scaled = scaler_c.transform(X_train_red)

        # final Lasso fit (timed)
        model_c = final_lasso_fit(X_train_red_scaled, y_full, best_alpha_c)

        time_cls_exp.append(t1 + time.time() - t0)

        # store results
        alphas_cls_exp.append(best_alpha_c)
        n_c = np.sum(model_c.coef_ != 0)
        feats_cls_exp.append(n_c)
        betas_cls_exp.append(model_c.coef_)

        # compute RMSE train
        preds_train_c = model_c.predict(X_train_red_scaled)
        rmse_train_cls_exp.append(np.sqrt(mean_squared_error(y_train[i], preds_train_c)))

        # compute RMSE test
        preds_test_c = model_c.predict(scaler_c.transform(X_test_red))
        rmse_test_cls_exp.append(np.sqrt(mean_squared_error(y_test[i], preds_test_c)))

    rmse_train = {
        "Lasso_Theoretical": rmse_train_theo,
        "Lasso_Binary": rmse_train_bin,
        "Lasso_CLS_Expected": rmse_train_cls_exp
    }

    rmse_test = {
        "Lasso_Theoretical": rmse_test_theo,
        "Lasso_Binary": rmse_test_bin,
        "Lasso_CLS_Expected": rmse_test_cls_exp
    }

    betas = {
        "Theoretical": betas_theo,
        "Binary": betas_bin,
        "CLS_Expected": betas_cls_exp
    }

    feature_counts = {
        "Theoretical": feats_theo,
        "Binary": feats_bin,
        "CLS_Expected": feats_cls_exp
    }

    alphas = {
        "Binary": alphas_bin,
        "CLS_Expected": alphas_cls_exp
    }

    times = {
        "Lasso_Theoretical": time_theo,
        "Lasso_Binary": time_bin,
        "Lasso_CLS_Expected": time_cls_exp
    }

    return rmse_train, rmse_test, betas, feature_counts, alphas, times

def apply_lasso_after_row_reduction(k, seed, base, folder, reps, gaussian=False):
    """
    Full pipeline for row-reduced Lasso modeling.

    The function loads simulation data, performs row reduction for each
    replication, fits theoretical and binary-search Lasso models, and
    returns RMSE values, coefficient vectors, feature counts, and α values.

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
        If True, Gaussian sketching is used.

    Returns
    -------
    dict
        Dictionary containing:
            R_reduced
            rmse_train
            rmse_test
            betas
            feature_counts
            alphas
    """

    print("Reading in the simulation data...")
    df_train = [pd.read_csv(f"{base}/{folder}/X_train{i+1}.csv") for i in range(reps)]
    df_test  = [pd.read_csv(f"{base}/{folder}/X_test{i+1}.csv") for i in range(reps)]
    y_train  = [pd.read_csv(f"{base}/{folder}/y_train{i+1}.csv") for i in range(reps)]
    y_test   = [pd.read_csv(f"{base}/{folder}/y_test{i+1}.csv") for i in range(reps)]

    print("Setting the seed...")
    random.seed(seed)
    np.random.seed(seed)

    print("Performing row reduction...")
    R_reduced, y_reduced = [], []
    for i in range(reps):
        R_i, y_i = row_reduction(k, df_train[i], y_train[i], gaussian)
        R_reduced.append(R_i)
        y_reduced.append(y_i)

    print("Running Lasso (theoretical + binary)...")
    rmse_train, rmse_test, betas, feature_counts, alphas, times = lasso_modeling(
        R_reduced, df_train, df_test, y_train, y_test, y_reduced, k
    )

    print("Completed.")

    return {
        "R_reduced": R_reduced,
        "rmse_train": rmse_train,
        "rmse_test": rmse_test,
        "betas": betas,
        "feature_counts": feature_counts,
        "alphas": alphas,
        "times": times
    }

def run_sampling_variance_lasso_after_row_reduction(
    k,
    base,
    folder,
    reps=10,
    outer_reps=10,
    gaussian=False
):
    """
    Sampling variance analysis for the row-reduced Lasso pipeline.

    This function runs the full row-reduction → Lasso modeling pipeline
    multiple times using different random seeds. Each outer repetition
    produces 'reps' RMSE values for both theoretical and binary-search
    Lasso. For each method and each seed, the function stores:

        - raw RMSE values (train + test)
        - mean RMSE
        - median RMSE
        - number of selected features
        - selected alpha values (binary only)

    Only essential information is stored. No reduced matrices (R) or
    coefficient vectors are included to keep memory usage minimal.

    Parameters
    ----------
    k : int
        Target rank for sketch size and Lasso regularization.
    base : str
        Base directory containing simulation data.
    folder : str
        Subfolder containing the simulation files.
    reps : int
        Number of replications inside each wrapper run.
    outer_reps : int
        Number of wrapper repetitions with different seeds.
    gaussian : bool
        If True, Gaussian sketching is used.

    Returns
    -------
    dict
        Nested dictionary containing sampling variance results for each seed.
    """

    results = {}

    for outer_seed in range(1, outer_reps + 1):

        print(f"Running outer repetition with seed = {outer_seed}")

        out = apply_lasso_after_row_reduction(
            k=k,
            seed=outer_seed,
            base=base,
            folder=folder,
            reps=reps,
            gaussian=gaussian
        )

        rmse_train = out["rmse_train"]
        rmse_test  = out["rmse_test"]
        feature_counts = out["feature_counts"]
        alphas = out["alphas"]
        times = out["times"]

        # summary containers
        rmse_summary = {}
        feature_summary = {}
        alpha_summary = {}
        time_summary = {}

        # summarize RMSE for all Lasso methods
        for method in rmse_test.keys():

            # test RMSE
            raw_test = rmse_test[method]
            rmse_summary[method] = {
                "test_raw": raw_test,
                "test_mean": float(np.mean(raw_test)),
                "test_median": float(np.median(raw_test))
            }

            # train RMSE
            raw_train = rmse_train[method]
            rmse_summary[method]["train_raw"] = raw_train
            rmse_summary[method]["train_mean"] = float(np.mean(raw_train))
            rmse_summary[method]["train_median"] = float(np.median(raw_train))

        # summarize RMSE for all Lasso methods
        for method in times.keys():
            raw_t = times[method]
            time_summary[method] = {
                "raw": raw_t,
                "mean": float(np.mean(raw_t)),
                "median": float(np.median(raw_t))
            }

        # summarize feature counts
        for method in feature_counts.keys():
            raw_feats = feature_counts[method]
            feature_summary[method] = {
                "raw": raw_feats,
                "mean": float(np.mean(raw_feats)),
                "median": float(np.median(raw_feats))
            }

        # summarize alpha values (binary only)
        if "Binary" in alphas:
            raw_alpha = alphas["Binary"]
            alpha_summary["Binary"] = {
                "raw": raw_alpha,
                "mean": float(np.mean(raw_alpha)),
                "median": float(np.median(raw_alpha))
            }

        # store results for this seed
        results[outer_seed] = {
            "rmse": rmse_summary,
            "feature_counts": feature_summary,
            "alphas": alpha_summary,
            "times": time_summary
        }

    return results