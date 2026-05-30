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

def fit_lasso_k_binary(R, y, k, max_iter=30):
    """
    Fit Lasso using binary search to achieve approximately k non-zero coefficients.

    A binary search is performed over α to match the target sparsity level.
    The best model is selected based on minimal deviation from k.

    Parameters
    ----------
    R : ndarray
        Row-reduced matrix.
    y : ndarray
        Reduced response vector.
    k : int
        Target number of non-zero coefficients.
    max_iter : int
        Number of binary search iterations.

    Returns
    -------
    best_model : Lasso
        Best-fitting Lasso model.
    scaler : StandardScaler
        Scaler used for standardizing R.
    best_alpha : float
        Selected regularization parameter.
    n_features : int
        Number of non-zero coefficients in the best model.
    """

    scaler = StandardScaler()
    R_scaled = scaler.fit_transform(R)

    alpha_low, alpha_high = 1e-6, 1e2
    best_model, best_alpha, best_diff, best_n = None, None, np.inf, None

    # binary search loop
    for _ in range(max_iter):
        alpha_mid = np.sqrt(alpha_low * alpha_high)
        model = Lasso(alpha=alpha_mid, max_iter=10000)
        model.fit(R_scaled, y)

        n_features = np.sum(model.coef_ != 0)
        diff = abs(n_features - k)

        if diff < best_diff:
            best_diff = diff
            best_model = model
            best_alpha = alpha_mid
            best_n = n_features

        if n_features > k:
            alpha_low = alpha_mid
        else:
            alpha_high = alpha_mid

    return best_model, scaler, best_alpha, best_n

def lasso_modeling(R_reduced, df_train, df_test, y_train, y_test, y_reduced, k):
    """
    Fit theoretical and binary-search Lasso models and evaluate them.

    Each replication is processed independently. Models are trained on
    row-reduced data and evaluated on the full training and test sets.
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
        Training RMSE for theoretical and binary Lasso.
    rmse_test : dict
        Test RMSE for theoretical and binary Lasso.
    betas : dict
        Coefficient vectors for both methods.
    feature_counts : dict
        Number of selected features for both methods.
    alphas : dict
        Selected α values (binary only).
    """

    rmse_train_theo, rmse_test_theo = [], []
    rmse_train_bin,  rmse_test_bin  = [], []

    feats_theo, feats_bin = [], []
    betas_theo, betas_bin = [], []
    alphas_bin = []

    # loop over replications
    for i in range(len(R_reduced)):
        R = R_reduced[i]
        y_r = y_reduced[i].ravel()

        # theoretical Lasso
        model_t, scaler_t, n_t = fit_theoretical_lasso(R, y_r, k)
        feats_theo.append(n_t)
        betas_theo.append(model_t.coef_)

        X_train = df_train[i].to_numpy()
        preds_train_t = model_t.predict(scaler_t.transform(X_train))
        rmse_train_theo.append(np.sqrt(mean_squared_error(y_train[i], preds_train_t)))

        X_test = df_test[i].to_numpy()
        preds_test_t = model_t.predict(scaler_t.transform(X_test))
        rmse_test_theo.append(np.sqrt(mean_squared_error(y_test[i], preds_test_t)))

        # binary-search Lasso
        model_b, scaler_b, alpha_b, n_b = fit_lasso_k_binary(R, y_r, k)
        alphas_bin.append(alpha_b)
        feats_bin.append(n_b)
        betas_bin.append(model_b.coef_)

        preds_train_b = model_b.predict(scaler_b.transform(X_train))
        rmse_train_bin.append(np.sqrt(mean_squared_error(y_train[i], preds_train_b)))

        preds_test_b = model_b.predict(scaler_b.transform(X_test))
        rmse_test_bin.append(np.sqrt(mean_squared_error(y_test[i], preds_test_b)))

    rmse_train = {
        "Lasso_Theoretical": rmse_train_theo,
        "Lasso_Binary": rmse_train_bin
    }

    rmse_test = {
        "Lasso_Theoretical": rmse_test_theo,
        "Lasso_Binary": rmse_test_bin
    }

    betas = {"Theoretical": betas_theo, "Binary": betas_bin}
    feature_counts = {"Theoretical": feats_theo, "Binary": feats_bin}
    alphas = {"Binary": alphas_bin}

    return rmse_train, rmse_test, betas, feature_counts, alphas

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
    rmse_train, rmse_test, betas, feature_counts, alphas = lasso_modeling(
        R_reduced, df_train, df_test, y_train, y_test, y_reduced, k
    )

    print("Completed.")

    return {
        "R_reduced": R_reduced,
        "rmse_train": rmse_train,
        "rmse_test": rmse_test,
        "betas": betas,
        "feature_counts": feature_counts,
        "alphas": alphas
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

        # summary containers
        rmse_summary = {}
        feature_summary = {}
        alpha_summary = {}

        # summarize RMSE for both Lasso methods
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
            "alphas": alpha_summary
        }

    return results