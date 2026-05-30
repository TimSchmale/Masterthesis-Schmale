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

# ------------------------------------------------------------
# Row Reduction (Sketching-based)
#
# INPUT:
#   k        : target rank
#   X        : design matrix (n x d)
#   y        : response vector (n x 1)
#   gaussian : True = Gaussian sketch, False = Rademacher
#
# OUTPUT:
#   R         : row-reduced matrix (r x d)
#   y_reduced : reduced response vector (r x 1)
#
# Description:
#   Performs iterative sketching using Gaussian or Rademacher
#   vectors.
# ------------------------------------------------------------
def row_reduction(k, X, y, gaussian=False):
    X = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
    y = y.to_numpy().reshape(-1, 1) if isinstance(y, (pd.Series, pd.DataFrame)) else y.reshape(-1, 1)

    n, d = X.shape
    r = int(np.ceil(k * np.log(d)))  # sketch size

    R = np.zeros((r, d))
    y_reduced = np.zeros((r, 1))

    if r < n:
        for i in range(n):
            sketch = np.random.randn(r, 1) if gaussian else np.random.choice([-1, 1], size=(r, 1)) / np.sqrt(r)
            R += sketch @ X[i:i+1, :]
            y_reduced += sketch * y[i]
    else:
        R = X.copy()
        y_reduced = y.copy()

    return R, y_reduced

# ------------------------------------------------------------
# Theoretical Lasso Fit
#
# INPUT:
#   R : row-reduced matrix
#   y : reduced response
#   k : target rank
#
# OUTPUT:
#   model     : fitted Lasso model
#   scaler    : StandardScaler used for R
#   n_features: number of non-zero coefficients
#
# Description:
#   Fits Lasso with α = 1/sqrt(k) as suggested by theory.
# ------------------------------------------------------------
def fit_theoretical_lasso(R, y, k):
    alpha = 1 / np.sqrt(k)

    scaler = StandardScaler()
    R_scaled = scaler.fit_transform(R)

    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(R_scaled, y)

    n_features = np.sum(model.coef_ != 0)
    return model, scaler, n_features

# ------------------------------------------------------------
# Binary Search Lasso (targeting k non-zero coefficients)
#
# INPUT:
#   R        : row-reduced matrix
#   y        : reduced response
#   k        : target number of non-zero coefficients
#   max_iter : binary search iterations
#
# OUTPUT:
#   best_model  : fitted Lasso model
#   scaler      : StandardScaler used for R
#   best_alpha  : selected α
#   n_features  : number of non-zero coefficients
#
# Description:
#   Searches α via binary search to match target sparsity k.
# ------------------------------------------------------------
def fit_lasso_k_binary(R, y, k, max_iter=30):
    scaler = StandardScaler()
    R_scaled = scaler.fit_transform(R)

    alpha_low, alpha_high = 1e-6, 1e2
    best_model, best_alpha, best_diff, best_n = None, None, np.inf, None

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


# ------------------------------------------------------------
# Lasso Modeling (Theoretical + Binary Search)
#
# INPUT:
#   R_reduced : list of row-reduced matrices
#   df_test   : list of test matrices
#   y_test    : list of test responses
#   y_reduced : list of reduced responses
#   k         : target rank
#
# OUTPUT:
#   rmse      : dict with RMSE for both methods
#   betas     : dict with coefficient vectors
#   feature_counts : dict with number of selected features
#   alphas    : dict with α values (binary only)
#
# Description:
#   Fits both theoretical and binary-search Lasso models and
#   evaluates them on the full test data.
# ------------------------------------------------------------
def lasso_modeling(R_reduced, df_test, y_test, y_reduced, k):

    rmse_theo, rmse_bin = [], []
    feats_theo, feats_bin = [], []
    betas_theo, betas_bin = [], []
    alphas_bin = []

    for i in range(len(R_reduced)):
        R = R_reduced[i]
        y_r = y_reduced[i].ravel()

        # --- theoretical Lasso ---
        model_t, scaler_t, n_t = fit_theoretical_lasso(R, y_r, k)
        feats_theo.append(n_t)
        betas_theo.append(model_t.coef_)

        X_test = df_test[i].to_numpy()
        preds_t = model_t.predict(scaler_t.transform(X_test))
        rmse_theo.append(np.sqrt(mean_squared_error(y_test[i], preds_t)))

        # --- binary-search Lasso ---
        model_b, scaler_b, alpha_b, n_b = fit_lasso_k_binary(R, y_r, k)
        alphas_bin.append(alpha_b)
        feats_bin.append(n_b)
        betas_bin.append(model_b.coef_)

        preds_b = model_b.predict(scaler_b.transform(X_test))
        rmse_bin.append(np.sqrt(mean_squared_error(y_test[i], preds_b)))

    rmse = {"Lasso_Theoretical": rmse_theo, "Lasso_Binary": rmse_bin}
    betas = {"Theoretical": betas_theo, "Binary": betas_bin}
    feature_counts = {"Theoretical": feats_theo, "Binary": feats_bin}
    alphas = {"Binary": alphas_bin}

    return rmse, betas, feature_counts, alphas


# ------------------------------------------------------------
# Wrapper: Row Reduction → Lasso Modeling
#
# INPUT:
#   k        : target rank
#   seed     : random seed
#   base     : base directory
#   folder   : data folder
#   reps     : number of replications
#   gaussian : sketch type
#
# OUTPUT:
#   dict with:
#       "R_reduced"
#       "rmse"
#       "betas"
#       "feature_counts"
#       "alphas"
#
# Description:
#   Full pipeline for row-reduced Lasso modeling.
# ------------------------------------------------------------
def apply_lasso_after_row_reduction(k, seed, base, folder, reps, gaussian=False):

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
    rmse, betas, feature_counts, alphas = lasso_modeling(
        R_reduced, df_test, y_test, y_reduced, k
    )

    print("Completed.")

    return {
        "R_reduced": R_reduced,
        "rmse": rmse,
        "betas": betas,
        "feature_counts": feature_counts,
        "alphas": alphas
    }