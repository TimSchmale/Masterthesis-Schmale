"""Model fitting and evaluation for CUR reduction experiments.

Provides model-agnostic fitting functions that work on reduced
(or full) data and return standardized evaluation metrics.
"""

import numpy as np
import time
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, brier_score_loss, log_loss


def fit_ols(X_train, y_train, X_test, y_test, selected_columns=None):
    """Fit OLS on (reduced) training data, evaluate on test data.

    The model is trained on X_train (which may be column-/row-reduced).
    For prediction on X_test, only the selected columns are used
    (so the test matrix stays full-dimensional but subsetted).

    Parameters
    ----------
    X_train : ndarray of shape (n_train, t)
        Training matrix (possibly reduced).
    y_train : ndarray of shape (n_train,)
        Training response.
    X_test : ndarray of shape (n_test, p)
        Full test matrix.
    y_test : ndarray of shape (n_test,)
        Test response.
    selected_columns : list of int, optional
        Column indices used during reduction. If provided,
        X_test is subsetted to these columns for prediction.
        If None, X_test is used as-is.

    Returns
    -------
    dict with keys:
        "rmse_train" : float
        "rmse_test" : float
        "coef" : ndarray of shape (t,)
    """
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Train prediction
    y_pred_train = model.predict(X_train)

    # Test prediction (subset columns if needed)
    if selected_columns is not None:
        X_test_sub = X_test[:, selected_columns]
    else:
        X_test_sub = X_test
    y_pred_test = model.predict(X_test_sub)

    return {
        "rmse_train": np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "rmse_test": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "coef": model.coef_
    }


def find_lasso_alpha(X, y, k, max_iter=30):
    """Binary search for Lasso alpha that yields ~k non-zero coefficients.

    Iteratively evaluates Lasso fits for candidate alpha values using
    a geometric midpoint and selects the alpha that minimizes the
    deviation from the target sparsity level k.

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Standardized design matrix.
    y : ndarray of shape (n,)
        Response vector.
    k : int
        Target number of non-zero coefficients.
    max_iter : int
        Number of binary search iterations.

    Returns
    -------
    float
        Best alpha value yielding sparsity closest to k.
    """
    alpha_low, alpha_high = 1e-6, 1e2
    best_alpha, best_diff = alpha_low, np.inf

    for _ in range(max_iter):
        alpha_mid = np.sqrt(alpha_low * alpha_high)  # geometric midpoint
        model = Lasso(alpha=alpha_mid, max_iter=10000)
        model.fit(X, y)

        n_features = np.sum(model.coef_ != 0)
        diff = abs(n_features - k)

        if diff < best_diff:
            best_diff = diff
            best_alpha = alpha_mid

        if n_features > k:
            alpha_low = alpha_mid
        else:
            alpha_high = alpha_mid

    return best_alpha


def fit_lasso(X_train, y_train, X_test, y_test, k,
             mode="binary_search", selected_columns=None):
    """Fit Lasso with specified regularization strategy.

    The training data is standardized before fitting. The same scaler
    is applied to the test data for prediction.

    Parameters
    ----------
    X_train : ndarray of shape (n_train, t)
        Training matrix (possibly reduced).
    y_train : ndarray of shape (n_train,)
        Training response.
    X_test : ndarray of shape (n_test, p)
        Full test matrix.
    y_test : ndarray of shape (n_test,)
        Test response.
    k : int
        Target rank / sparsity level.
    mode : {"theoretical", "binary_search"}
        - "theoretical": alpha = 1/sqrt(k)
        - "binary_search": finds alpha yielding ~k non-zero features
    selected_columns : list of int, optional
        Column indices for subsetting X_test.

    Returns
    -------
    dict with keys:
        "rmse_train" : float
        "rmse_test" : float
        "coef" : ndarray
        "n_features" : int (number of non-zero coefficients)
        "alpha" : float (regularization parameter used)
    """
    # Standardize training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Select alpha (timed separately)
    t0 = time.perf_counter()
    if mode == "theoretical":
        alpha = 1.0 / np.sqrt(k)
    else:
        alpha = find_lasso_alpha(X_train_scaled, y_train, k)
    time_alpha_search = time.perf_counter() - t0

    # Fit final model (timed separately — this is comparable to OLS)
    t0 = time.perf_counter()
    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(X_train_scaled, y_train)
    time_fit = time.perf_counter() - t0

    # Train prediction
    y_pred_train = model.predict(X_train_scaled)

    # Test prediction
    if selected_columns is not None:
        X_test_sub = X_test[:, selected_columns]
    else:
        X_test_sub = X_test
    X_test_scaled = scaler.transform(X_test_sub)
    y_pred_test = model.predict(X_test_scaled)

    return {
        "rmse_train": np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "rmse_test": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "coef": model.coef_,
        "n_features": int(np.sum(model.coef_ != 0)),
        "alpha": alpha,
        "time_fit": time_fit,
        "time_alpha_search": time_alpha_search
    }


def fit_logistic(X_train, y_train, X_test, y_test, selected_columns=None):
    """Fit logistic regression on (reduced) training data.

    Unregularized logistic regression (penalty=None) is used to
    evaluate the quality of the CUR reduction for classification.

    Parameters
    ----------
    X_train : ndarray of shape (n_train, t)
        Training matrix (possibly reduced).
    y_train : ndarray of shape (n_train,)
        Binary response vector.
    X_test : ndarray of shape (n_test, p)
        Full test matrix.
    y_test : ndarray of shape (n_test,)
        Binary test response.
    selected_columns : list of int, optional
        Column indices for subsetting X_test.

    Returns
    -------
    dict with keys:
        "brier_train" : float
        "brier_test" : float
        "ce_train" : float (cross-entropy / log loss)
        "ce_test" : float
        "coef" : ndarray of shape (t,)
    """
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=2000
    )
    model.fit(X_train, y_train)

    # Train probabilities
    p_train = model.predict_proba(X_train)[:, 1]

    # Test probabilities
    if selected_columns is not None:
        X_test_sub = X_test[:, selected_columns]
    else:
        X_test_sub = X_test
    p_test = model.predict_proba(X_test_sub)[:, 1]

    return {
        "brier_train": brier_score_loss(y_train, p_train),
        "brier_test": brier_score_loss(y_test, p_test),
        "ce_train": log_loss(y_train, p_train),
        "ce_test": log_loss(y_test, p_test),
        "coef": model.coef_.ravel()
    }
