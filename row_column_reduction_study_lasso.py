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

def row_reduction(k, X, y, gaussian = False):
    X = X.to_numpy() if isinstance(X, pd.DataFrame) else X

    # define dimensionality
    n, d = X.shape

    # number of rows in sketch
    r = int(np.ceil(k * np.log(d)))

    # y preparation
    y_reduced = np.zeros((r, 1)) if y is not None else None
    y = y.to_numpy().reshape(-1, 1) if isinstance(y, (pd.Series, pd.DataFrame)) else y.reshape(-1, 1)

    # prepare sketched matrix
    R = np.zeros((r,d))

    # created the sketched versions of X and y
    if r <  n:
        for i in range(n):
            if gaussian:
                # Gaussian sketching vector (r x 1)
                sketch_vec = np.random.randn(r, 1)
            else:
                # Rademacher sketching vector (r x 1)
                sketch_vec = np.random.choice([-1,1], size=(r,1)) / np.sqrt(r)

            # Reduce X: Outer product: (r x 1) @ (1 x d) -> (r x d)
            R += sketch_vec @ X[i, :].reshape(1, d)

            # Reduce y in parallel
            if y is not None:
                y_reduced += sketch_vec * y[i]
    else:
        R = X
        y_reduced  = y

    return R, y_reduced

def fit_theoretical_lasso(R,y,k):
    # calculate the theoretical lambda derived from optimal sketching bounds paper
    alpha = 1/np.sqrt(k)

    scaler = StandardScaler()
    R_scaled = scaler.fit_transform(R)

    # fit Lasso model
    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(R_scaled, y)

    # get the number of nonzero coefficients
    n_features = np.sum(model.coef_ != 0)

    return model, scaler, n_features


def fit_lasso_k_binary(R, y, k, max_iter=30):
    scaler = StandardScaler()
    R_scaled = scaler.fit_transform(R)

    # define the grid lambda / alpha is searched in
    alpha_low, alpha_high = 1e-6, 1e2
    best_model = None
    best_diff = np.inf
    best_alpha = None
    best_n_features = None

    # perform a binary search inside of grid to get the best suited lambda / alpha
    for _ in range(max_iter):
        alpha_mid = np.sqrt(alpha_low * alpha_high)
        model = Lasso(alpha=alpha_mid, max_iter=10000)
        model.fit(R_scaled, y)

        n_features = np.sum(model.coef_ != 0)
        diff = abs(n_features - k)

        # Update best
        if diff < best_diff:
            best_diff = diff
            best_model = model
            best_alpha = alpha_mid
            best_n_features = n_features

        # Binary search update
        if n_features > k:
            alpha_low = alpha_mid
        else:
            alpha_high = alpha_mid

    return best_model, scaler, best_alpha, best_n_features

def lasso_modeling(R_reduced, df_test, y_test, y_reduced, k):

    # initialize all different lists
    rmse_theoretical = []
    rmse_binary = []

    selected_features_theoretical = []
    selected_features_binary = []

    beta_theoretical = []
    beta_binary = []

    alpha_binary = []

    for i in range(len(R_reduced)):

        R = R_reduced[i]
        y_r = y_reduced[i].ravel()

        # theoretic Lasso fit
        model_t, scaler_t, n_feat_t = fit_theoretical_lasso(R, y_r, k)

        selected_features_theoretical.append(n_feat_t)
        beta_theoretical.append(model_t.coef_)

        # test procedure
        X_test = df_test[i].to_numpy()
        X_test_scaled_t = scaler_t.transform(X_test)
        preds_t = model_t.predict(X_test_scaled_t)

        rmse_t = np.sqrt(mean_squared_error(y_test[i], preds_t))
        rmse_theoretical.append(rmse_t)

        # binary Lasso fit
        model_b, scaler_b, alpha_b, n_feat_b = fit_lasso_k_binary(R, y_r, k)
        alpha_binary.append(alpha_b)

        selected_features_binary.append(n_feat_b)
        beta_binary.append(model_b.coef_)

        # test procedure
        X_test_scaled_b = scaler_b.transform(X_test)
        preds_b = model_b.predict(X_test_scaled_b)

        rmse_b = np.sqrt(mean_squared_error(y_test[i], preds_b))
        rmse_binary.append(rmse_b)

    return (
        rmse_theoretical,
        rmse_binary,
        selected_features_theoretical,
        selected_features_binary,
        beta_theoretical,
        beta_binary,
        alpha_binary
    )


def apply_lasso_after_row_reduction(k, seed, base, folder, reps, gaussian=False):

    # ---------------------------------------------------------
    # 1. Load Data
    # ---------------------------------------------------------
    df_train, df_test, y_train, y_test = [], [], [], []

    print("Reading in the simulation data...")
    for i in range(reps):
        df_train.append(pd.read_csv(f"{base}/{folder}/X_train{i+1}.csv"))
        df_test.append(pd.read_csv(f"{base}/{folder}/X_test{i+1}.csv"))
        y_train.append(pd.read_csv(f"{base}/{folder}/y_train{i+1}.csv"))
        y_test.append(pd.read_csv(f"{base}/{folder}/y_test{i+1}.csv"))

    # ---------------------------------------------------------
    # 2. Seeding
    # ---------------------------------------------------------
    print("Setting the seed...")
    random.seed(seed)
    np.random.seed(seed)

    # ---------------------------------------------------------
    # 3. Row Reduction
    # ---------------------------------------------------------
    print("Performing row reduction...")
    R_reduced = []
    y_reduced = []

    for i in range(reps):
        R_i, y_i = row_reduction(k, df_train[i], y_train[i], gaussian)
        R_reduced.append(R_i)
        y_reduced.append(y_i)

    # ---------------------------------------------------------
    # 4. Lasso Modeling (theoretical + binary)
    # ---------------------------------------------------------
    print("Running Lasso (theoretical + binary)...")

    (
        rmse_theoretical,
        rmse_binary,
        selected_features_theoretical,
        selected_features_binary,
        beta_theoretical,
        beta_binary,
        alpha_binary
    ) = lasso_modeling(
        R_reduced, df_test, y_test, y_reduced, k
    )

    # ---------------------------------------------------------
    # 5. Return all results (ohne Full Model)
    # ---------------------------------------------------------
    rmse = {
        "Lasso_Theoretical": rmse_theoretical,
        "Lasso_Binary": rmse_binary
    }

    feature_counts = {
        "Theoretical": selected_features_theoretical,
        "Binary": selected_features_binary
    }

    betas = {
        "Theoretical": beta_theoretical,
        "Binary": beta_binary
    }

    alphas = {
        "Binary": alpha_binary
    }

    return R_reduced, rmse, betas, feature_counts, alphas