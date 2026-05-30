import pandas as pd
import numpy as np
import random
import gc
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

# ------------------------------------------------------------
# Function to perform Column Reduction using EXPECTED(c)
#
# INPUT:
#   X      : design matrix of dimension n x p
#   scores : numeric vector of length p containing column scores
#   k      : target rank for CUR approximation
#
# OUTPUT:
#   Dictionary with:
#       "C"               : reduced matrix of dimension n x t
#       "selected_columns": list of selected column indices
#
# Description:
#   Implements the EXPECTED(c) column sampling algorithm from the
#   CUR decomposition. Columns are sampled independently using
#   scaled probabilities and rescaled according to the theorem.
# ------------------------------------------------------------
def column_reduction(X, scores, k):

    # convert to numpy for speed
    X = np.asarray(X)
    n, p = X.shape

    # sampling probabilities
    probs = scores / scores.sum()

    # number of expected sampled columns
    c = int(np.ceil(k * np.log(k)))

    # scaled probabilities (Bernoulli sampling)
    scaled_probs = np.minimum(c * probs, 1)

    # Bernoulli draws
    z = np.random.rand(p)
    sampled = np.where(z <= scaled_probs)[0]

    # fallback: ensure at least one column
    if len(sampled) == 0:
        sampled = np.array([np.argmax(scaled_probs)])

    # rescaling factors
    D_inv = 1 / np.sqrt(scaled_probs[sampled])

    # reduced matrix C = X[:, sampled] * D
    C = X[:, sampled] * D_inv

    return {
        "C": C,
        "selected_columns": sampled.tolist()
    }

# ------------------------------------------------------------
# Function to perform Row Reduction using EXPECTED(r)
#
# INPUT:
#   C : reduced column matrix of dimension n x c
#   y : response vector of length n
#   k : target rank for CUR approximation
#
# OUTPUT:
#   Dictionary with:
#       "R"            : reduced matrix of dimension t x c
#       "y"            : reduced response vector of length t
#       "selected_rows": list of selected row indices
#
# Description:
#   Implements the EXPECTED(r) row sampling algorithm from CUR.
#   Rows are sampled independently using scaled row leverage
#   probabilities and rescaled accordingly.
# ------------------------------------------------------------
def row_reduction(C, y, k):

    # convert to numpy
    C = np.asarray(C)
    y = np.asarray(y).reshape(-1)

    n, c = C.shape

    # compute row leverage scores
    scores = get_row_leverage_scores(C, k)

    # sampling probabilities
    probs = scores / scores.sum()

    # expected number of sampled rows
    r = int(np.ceil(c * np.log(c)))

    # scaled probabilities
    scaled_probs = np.minimum(r * probs, 1)

    # Bernoulli sampling
    z = np.random.rand(n)
    sampled = np.where(z <= scaled_probs)[0]

    # fallback: ensure at least one row
    if len(sampled) == 0:
        sampled = np.array([np.argmax(scaled_probs)])

    # rescaling factors
    D_inv = 1 / np.sqrt(scaled_probs[sampled])

    # reduced matrix R = D * C[sampled, :]
    R = C[sampled, :] * D_inv[:, None]

    # reduced response
    y_reduced = y[sampled]

    return {
        "R": R,
        "y": y_reduced,
        "selected_rows": sampled.tolist()
    }


# ------------------------------------------------------------
# Function to perform full data reduction (column + optional row)
#
# INPUT:
#   k          : target rank for CUR approximation
#   df_train   : list of training matrices (each n x p)
#   y_train    : list of response vectors
#   row_reduce : boolean, whether row reduction should be applied
#
# OUTPUT:
#   scores       : dict with score vectors per method and replication
#   time_scores  : dict with score computation times
#   C            : dict with selected columns per method
#   R            : dict with selected rows per method (or None)
#
# Description:
#   Computes LS, CLS, RS, CS scores for each replication, performs
#   EXPECTED(c) column reduction, and optionally EXPECTED(r) row
#   reduction. Returns only the structures needed for evaluation.
# ------------------------------------------------------------
def data_reduction(k, df_train, y_train, row_reduce=True):

    n_reps = len(df_train)

    # initialize score containers
    scores = {"LS": [], "CLS": [], "RS": [], "CS": []}
    time_scores = {"LS": [], "CLS": [], "RS": [], "CS": []}

    # -------------------------------
    # 1) Score calculation
    # -------------------------------
    for i in range(n_reps):
        X = df_train[i].to_numpy()
        y = y_train[i].to_numpy().reshape(-1)

        # LS
        start = time.perf_counter()
        scores["LS"].append(get_column_leverage_scores(X, k))
        time_scores["LS"].append(time.perf_counter() - start)

        # CLS
        start = time.perf_counter()
        scores["CLS"].append(get_cross_leverage_scores(X, y))
        time_scores["CLS"].append(time.perf_counter() - start)

        # RS
        start = time.perf_counter()
        scores["RS"].append(get_random_scores(X))
        time_scores["RS"].append(time.perf_counter() - start)

        # CS
        start = time.perf_counter()
        scores["CS"].append(get_combined_scores(X, y, k, p_leverage=0.2))
        time_scores["CS"].append(time.perf_counter() - start)

        del X, y
        gc.collect()

    # -------------------------------
    # 2) Column reduction
    # -------------------------------
    C = {"LS": [], "CLS": [], "RS": [], "CS": []}

    for i in range(n_reps):
        X = df_train[i].to_numpy()

        C["LS"].append(column_reduction(X, scores["LS"][i], k))
        C["CLS"].append(column_reduction(X, np.abs(scores["CLS"][i]), k))
        C["RS"].append(column_reduction(X, scores["RS"][i], k))
        C["CS"].append(column_reduction(X, scores["CS"][i], k))

        del X
        gc.collect()

    # -------------------------------
    # 3) Row reduction (optional)
    # -------------------------------
    R = None
    if row_reduce:
        R = {"LS": [], "CLS": [], "RS": [], "CS": []}

        for i in range(n_reps):
            y = y_train[i].to_numpy().reshape(-1)

            R["LS"].append(row_reduction(C["LS"][i]["C"], y, k))
            R["CLS"].append(row_reduction(C["CLS"][i]["C"], y, k))
            R["RS"].append(row_reduction(C["RS"][i]["C"], y, k))
            R["CS"].append(row_reduction(C["CS"][i]["C"], y, k))

            del y
            gc.collect()

    return scores, time_scores, C, R


# ------------------------------------------------------------
# Function to fit linear models on reduced matrices
#
# INPUT:
#   C       : dict with column-reduced matrices and selected columns
#   R       : dict with row-reduced matrices (or None)
#   df_test : list of test matrices
#   y_test  : list of test responses
#   y_train : list of training responses
#
# OUTPUT:
#   rmse : dict with RMSE values per method and replication
#
# Description:
#   Fits linear regression models on reduced matrices (C or R),
#   predicts on test data, and computes RMSE for each method.
# ------------------------------------------------------------
def linear_modeling(C, R, df_test, y_test, y_train):

    n_reps = len(df_test)
    rmse = {"LS": [], "CLS": [], "RS": [], "CS": []}

    for i in range(n_reps):

        if R is not None:
            # row-reduced modeling
            models = {
                "LS": LinearRegression().fit(R["LS"][i]["R"], R["LS"][i]["y"]),
                "CLS": LinearRegression().fit(R["CLS"][i]["R"], R["CLS"][i]["y"]),
                "RS": LinearRegression().fit(R["RS"][i]["R"], R["RS"][i]["y"]),
                "CS": LinearRegression().fit(R["CS"][i]["R"], R["CS"][i]["y"])
            }

            for key in rmse.keys():
                cols = C[key][i]["selected_columns"]
                X_test_red = df_test[i].to_numpy()[:, cols]
                preds = models[key].predict(X_test_red)
                rmse[key].append(np.sqrt(mean_squared_error(y_test[i], preds)))

        else:
            # column-only modeling
            y_tr = y_train[i].to_numpy().reshape(-1)

            models = {
                "LS": LinearRegression().fit(C["LS"][i]["C"], y_tr),
                "CLS": LinearRegression().fit(C["CLS"][i]["C"], y_tr),
                "RS": LinearRegression().fit(C["RS"][i]["C"], y_tr),
                "CS": LinearRegression().fit(C["CS"][i]["C"], y_tr)
            }

            for key in rmse.keys():
                cols = C[key][i]["selected_columns"]
                X_test_red = df_test[i].to_numpy()[:, cols]
                preds = models[key].predict(X_test_red)
                rmse[key].append(np.sqrt(mean_squared_error(y_test[i], preds)))

        gc.collect()

    return rmse

# ------------------------------------------------------------
# Function to compute RMSE of the full model (oracle benchmark)
#
# INPUT:
#   df_train : list of training matrices
#   df_test  : list of test matrices
#   y_train  : list of training responses
#   y_test   : list of test responses
#   base     : base path to simulation folder
#   folder   : subfolder name
#
# OUTPUT:
#   rmse_full : list of RMSE values for each replication
#
# Description:
#   Fits the full oracle model using the true non-zero beta
#   positions and computes RMSE on the test data.
# ------------------------------------------------------------
def compute_full_rmse(df_train, df_test, y_train, y_test, base, folder):

    rmse_full = []

    for i in range(len(df_train)):
        beta = pd.read_csv(
            f"{base}/{folder}/beta{i + 1}.csv",
            header=0  # skip header row
        ).to_numpy().reshape(-1)

        selected = np.where(beta != 0)[0]

        X_train = df_train[i].to_numpy()[:, selected]
        X_test = df_test[i].to_numpy()[:, selected]

        y_tr = y_train[i].to_numpy().reshape(-1)
        y_te = y_test[i].to_numpy().reshape(-1)

        model = LinearRegression().fit(X_train, y_tr)
        preds = model.predict(X_test)

        rmse_full.append(np.sqrt(mean_squared_error(y_te, preds)))

        del X_train, X_test, y_tr, y_te, preds
        gc.collect()

    return rmse_full

# ------------------------------------------------------------
# Wrapper to perform full CUR-based data reduction and modeling
#
# INPUT:
#   k            : target rank
#   seed         : random seed
#   base, folder : paths to simulation data
#   reps         : number of replications
#   row_reduction: boolean, whether row reduction is applied
#
# OUTPUT:
#   Dictionary with:
#       "scores"          : score vectors per method
#       "time_scores"     : score computation times
#       "selected_columns": selected columns per method
#       "selected_rows"   : selected rows per method (or None)
#       "rmse"            : RMSE values per method
#
# Description:
#   Loads simulation data, performs score computation, column and
#   optional row reduction, fits linear models, computes RMSE, and
#   returns only the structures needed for evaluation.
# ------------------------------------------------------------
def apply_row_after_col_reduction(k, seed, base, folder, reps, row_reduction=True):

    print(f"Reading simulation data for {reps} replications...")

    df_train = [pd.read_csv(f"{base}/{folder}/X_train{i+1}.csv") for i in range(reps)]
    df_test  = [pd.read_csv(f"{base}/{folder}/X_test{i+1}.csv") for i in range(reps)]
    y_train  = [pd.read_csv(f"{base}/{folder}/y_train{i+1}.csv") for i in range(reps)]
    y_test   = [pd.read_csv(f"{base}/{folder}/y_test{i+1}.csv") for i in range(reps)]

    print("Setting random seed...")
    random.seed(seed)
    np.random.seed(seed)

    print("Performing data reduction (scores + column reduction)...")

    scores, time_scores, C, R = data_reduction(k, df_train, y_train, row_reduction)

    print("Fitting linear models on reduced data...")

    rmse = linear_modeling(C, R, df_test, y_test, y_train)

    print("Computing full-model benchmark (oracle RMSE)...")

    rmse_full = compute_full_rmse(df_train, df_test, y_train, y_test, base, folder)
    rmse["Full"] = rmse_full

    print("Data Reduction & Modeling completed.")

    # return only relevant structures
    return {
        "scores": scores,
        "time_scores": time_scores,
        "selected_columns": {m: [C[m][i]["selected_columns"] for i in range(reps)] for m in C},
        "selected_rows": None if R is None else {m: [R[m][i]["selected_rows"] for i in range(reps)] for m in R},
        "rmse": rmse
    }
