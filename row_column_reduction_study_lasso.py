import numpy as np
import pandas as pd
import pickle
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

    n, d = X.shape
    r = int(np.ceil(2 * k * np.log(d)))

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

        X_train = df_train[i]
        preds_train_t = model_t.predict(scaler_t.transform(X_train))
        rmse_train_theo.append(np.sqrt(mean_squared_error(y_train[i], preds_train_t)))

        X_test = df_test[i]
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
        X_full = df_train[i]
        y_full = y_train[i]

        t0 = time.time()

        # compute CLS scores
        cls_scores = np.abs(get_cross_leverage_scores(X_full, y_full))

        # apply EXPECTED(c) column sampling
        col_red = column_reduction(X_full, cls_scores, k)
        X_train_red = col_red["C"]
        selected_cols = col_red["selected_columns"]

        # reduce test matrix accordingly
        X_test_red = df_test[i][:, selected_cols]
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
        "Lasso_Theo": rmse_train_theo,
        "Lasso_Bin": rmse_train_bin,
        "Lasso_CLS": rmse_train_cls_exp
    }

    rmse_test = {
        "Lasso_Theo": rmse_test_theo,
        "Lasso_Bin": rmse_test_bin,
        "Lasso_CLS": rmse_test_cls_exp
    }

    betas = {
        "Lasso_Theo": betas_theo,
        "Lasso_Bin": betas_bin,
        "Lasso_CLS": betas_cls_exp
    }

    feature_counts = {
        "Lasso_Theo": feats_theo,
        "Lasso_Bin": feats_bin,
        "Lasso_CLS": feats_cls_exp
    }

    alphas = {
        "Lasso_Bin": alphas_bin,
        "Lasso_CLS": alphas_cls_exp
    }

    times = {
        "Lasso_Theo": time_theo,
        "Lasso_Binary": time_bin,
        "Lasso_CLS": time_cls_exp
    }

    return rmse_train, rmse_test, betas, feature_counts, alphas, times

def apply_lasso_after_row_reduction(
        k,
        seed,
        X_train_list,
        X_test_list,
        y_train_list,
        y_test_list,
        gaussian=False
):
    """
    Full Row → Lasso pipeline for a single seed.

    This function performs:
    1. Row reduction via sketching (Gaussian or Rademacher).
    2. Lasso modeling on the row‑reduced matrices using three variants:
         - Lasso_Theoretical (α = 1/sqrt(k))
         - Lasso_Binary (binary search for α yielding ~k non‑zero coefficients)
         - Lasso_CLS_Expected (CLS‑based EXPECTED(c) sampling + binary Lasso)
    3. Evaluation of all models on the full train/test matrices.
    4. Aggregation of RMSE values, feature counts, α values, and timing results.

    The function does *not* load data and does *not* perform train/test splits.
    All data must be pre‑split and provided by the wrapper.

    Parameters
    ----------
    k : int
        Target rank controlling sketch size and Lasso sparsity.
    seed : int
        Random seed for reproducibility.
    X_train_list : list of ndarray
        List of training matrices for each replication.
    X_test_list : list of ndarray
        List of test matrices for each replication.
    y_train_list : list of ndarray
        List of training response vectors.
    y_test_list : list of ndarray
        List of test response vectors.
    gaussian : bool
        If True, Gaussian sketching is used. Otherwise Rademacher.

    Returns
    -------
    dict
        Dictionary containing:
            rmse_train : dict
                Training RMSE for all three Lasso variants.
            rmse_test : dict
                Test RMSE for all three Lasso variants.
            feature_counts : dict
                Number of selected features per method and replication.
            alphas : dict
                Selected α values for binary and CLS‑Expected Lasso.
            model_time : dict
                Timing information for all three Lasso variants.
    """


    print("Setting the seed...")
    random.seed(seed)
    np.random.seed(seed)

    reps = len(X_train_list)

    print("Performing row reduction...")
    R_reduced, y_reduced = [], []
    for i in range(reps):
        R_i, y_i = row_reduction(k, X_train_list[i], y_train_list[i], gaussian)
        R_reduced.append(R_i)
        y_reduced.append(y_i)

    print("Running Lasso (theoretical + binary + CLS)...")
    rmse_train, rmse_test, betas, feature_counts, alphas, times = lasso_modeling(
        R_reduced,
        X_train_list,
        X_test_list,
        y_train_list,
        y_test_list,
        y_reduced,
        k
    )

    print("Completed.")

    return {
        "rmse_train": rmse_train,
        "rmse_test": rmse_test,
        "feature_counts": feature_counts,
        "alphas": alphas,
        "model_time": times
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

def run_sampling_variance_lasso_after_row_reduction(
    k_vector,
    base,
    folder,
    reps=10,
    outer_reps=10,
    test_size=0.2,
    save_name=None,
    results_folder=None,
    gaussian=False,
    seed_from=1,
    seed_to=None
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
            out = apply_lasso_after_row_reduction(
                k=k,
                seed=seed,
                X_train_list=X_train_list,
                X_test_list=X_test_list,
                y_train_list=y_train_list,
                y_test_list=y_test_list,
                gaussian=gaussian
            )

            # extract RMSE values
            rmse_test = out["rmse_test"]
            rmse_train = out["rmse_train"]

            # extract time info
            time_scores = None
            time_model = out["model_time"]

            # extract further structural information
            feature_counts = out["feature_counts"]
            alphas = out["alphas"]

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
            #score_time_summary = {
            #    method: {
            #        "raw": time_scores[method],
            #        "mean": float(np.mean(time_scores[method])),
            #        "median": float(np.median(time_scores[method]))
            #    }
            #    for method in time_scores.keys()
            #}

            # same for modeling time
            model_time_summary = {
                method: {
                    "raw": time_model[method],
                    "mean": float(np.mean(time_model[method])),
                    "median": float(np.median(time_model[method]))
                }
                for method in time_model.keys()
            }

            # summarize feature counts
            feature_count_summary = {
                method: {
                    "raw": feature_counts[method],
                    "mean": float(np.mean(feature_counts[method])),
                    "median": float(np.median(feature_counts[method]))
                }
                for method in feature_counts.keys()
            }

            # summarize alpha values (binary only)
            alpha_summary = {}
            if "Lasso_Bin" in alphas:
                raw_alpha = alphas["Lasso_Bin"]
                alpha_summary["Lasso_Bin"] = {
                    "raw": raw_alpha,
                    "mean": float(np.mean(raw_alpha)),
                    "median": float(np.median(raw_alpha))
                }

            # store results for this seed
            results[seed][k] = {
                "loss": loss_summary,
                "train_loss": train_loss_summary,
                "feature_counts": feature_count_summary,
                "alphas": alpha_summary,
                "time_scores": None,
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

    return results