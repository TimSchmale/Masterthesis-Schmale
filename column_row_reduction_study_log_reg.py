import pandas as pd
import random
from scoring_functions import get_column_leverage_scores, get_log_reg_leverage_scores, get_random_scores, get_combined_scores, get_row_leverage_scores, get_cross_leverage_scores
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import brier_score_loss
import time
from visualizations import *

# ------------------------------------------------------------
# Column Reduction (EXPECTED(c) sampling)
#
# INPUT:
#   X      : data matrix (n x d)
#   scores : column scores (length d)
#   k      : target rank
#
# OUTPUT:
#   dict with:
#       "C"               : reduced matrix (n x t)
#       "selected_columns": list of selected column indices
#
# Description:
#   Implements EXPECTED(c) sampling without building S or D
#   explicitly. Efficient and consistent with CUR theory.
# ------------------------------------------------------------
def column_reduction(X, scores, k):
    X = np.asarray(X)
    n, d = X.shape

    probs = scores / scores.sum()
    c = int(np.ceil(k * np.log(k)))
    scaled = np.minimum(c * probs, 1)

    z = np.random.rand(d)
    sampled = np.where(z <= scaled)[0]
    if len(sampled) == 0:
        sampled = np.array([np.argmax(scaled)])

    D_inv = 1 / np.sqrt(scaled[sampled])
    C = X[:, sampled] * D_inv  # scale selected columns

    return {
        "C": C,
        "selected_columns": sampled.tolist()
    }


# ------------------------------------------------------------
# Estimate μ (logistic coreset imbalance parameter)
#
# INPUT:
#   C : reduced design matrix (n x t)
#   y : binary response vector
#
# OUTPUT:
#   mu : imbalance ratio from logistic coreset theory
# ------------------------------------------------------------
def estimate_mu(C, y):
    model = LogisticRegression(
        l1_ratio=0,
        C=np.inf,
        solver='lbfgs',
        max_iter=2000
    )
    model.fit(C, y)

    beta = model.coef_.flatten()  # only used internally
    v = C @ beta

    pos = np.sum(np.abs(v[v > 0]))
    neg = np.sum(np.abs(v[v < 0]))

    mu = np.inf if neg == 0 else pos / neg
    mu = max(mu, 1.0001)

    return mu

# ------------------------------------------------------------
# Row Reduction (logistic coreset sampling)
#
# INPUT:
#   C  : reduced matrix (n x t)
#   y  : binary response vector
#   mu : imbalance parameter
#   k  : target rank
#
# OUTPUT:
#   dict with:
#       "R"            : row-reduced matrix
#       "y"            : reduced response
#       "selected_rows": sampled row indices
#       "mu"           : μ value
#       "r"            : number of sampled rows
# ------------------------------------------------------------
def row_reduction(C, y, mu, k):
    y_arr = np.asarray(y).ravel()
    n, d = C.shape

    r = int(np.ceil(mu * d * np.log(mu * d)))
    r = max(1, min(r, n))

    l1 = get_log_reg_leverage_scores(C)
    l2 = get_row_leverage_scores(C, k)
    uniform = np.ones(n) / n

    scores = mu * l1 + l2 + mu * d * uniform
    probs = scores / scores.sum()

    rng = np.random.default_rng()
    sampled = rng.choice(n, size=r, replace=False, p=probs)

    return {
        "R": C[sampled, :],
        "y": y_arr[sampled],
        "selected_rows": sampled.tolist(),
        "mu": mu,
        "r": r
    }

# ------------------------------------------------------------
# Data Reduction (Column → optional Row)
#
# INPUT:
#   k          : target rank
#   df_train   : list of training matrices
#   y_train    : list of binary response vectors
#   row_reduce : whether to perform row reduction
#
# OUTPUT:
#   scores           : dict of score vectors
#   time_scores      : dict of score computation times
#   selected_columns : dict of selected column indices
#   selected_rows    : dict of selected row indices (or None)
#   mu_values        : dict of μ values per method
# ------------------------------------------------------------
def data_reduction(k, df_train, y_train, row_reduce=True):

    n_reps = len(df_train)

    scores = {"LS": [], "CLS": [], "RS": [], "CS": []}
    time_scores = {"LS": [], "CLS": [], "RS": [], "CS": []}
    selected_columns = {"LS": [], "CLS": [], "RS": [], "CS": []}
    selected_rows = {"LS": [], "CLS": [], "RS": [], "CS": []} if row_reduce else None
    mu_values = {"LS": [], "CLS": [], "RS": [], "CS": []}

    # --- score computation ---
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

    # --- column reduction ---
    C_mats = {m: [] for m in scores}
    for i in range(n_reps):
        X = df_train[i]
        for method in scores:
            out = column_reduction(X, np.abs(scores[method][i]), k)
            C_mats[method].append(out["C"])
            selected_columns[method].append(out["selected_columns"])

    # --- row reduction + μ ---
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


# ------------------------------------------------------------
# Logistic Modeling on Reduced Data
#
# INPUT:
#   selected_columns : dict of selected column indices per method
#   selected_rows    : dict of selected row indices per method (or None)
#   df_train         : list of full training matrices
#   df_test          : list of full test matrices
#   y_train          : list of training response vectors
#   y_test           : list of test response vectors
#
# OUTPUT:
#   brier : dict of Brier scores per method
#
# Description:
#   Fits logistic regression models on reduced data (column-only
#   or column+row) and evaluates prediction performance via the
#   Brier score.
# ------------------------------------------------------------
def logistic_modeling(selected_columns, selected_rows, df_train, df_test, y_train, y_test):

    n_reps = len(df_train)
    brier = {"LS": [], "CLS": [], "RS": [], "CS": []}

    for i in range(n_reps):
        for method in selected_columns:

            cols = selected_columns[method][i]  # selected columns

            if selected_rows is not None:
                rows = selected_rows[method][i]  # selected rows
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

            # test data
            X_test = df_test[i].to_numpy()[:, cols]
            pred = model.predict_proba(X_test)[:, 1]

            # compute Brier score
            brier[method].append(brier_score_loss(y_test[i], pred))

    return brier


# ------------------------------------------------------------
# Full Model Benchmark (Oracle Brier Score)
#
# INPUT:
#   df_train : list of full training matrices
#   df_test  : list of full test matrices
#   y_train  : list of training response vectors
#   y_test   : list of test response vectors
#   base     : base directory for simulation data
#   folder   : subfolder containing beta-files
#
# OUTPUT:
#   brierloss_full : list of Brier scores (one per replication)
#
# Description:
#   Fits logistic regression using the true support (non-zero
#   entries of beta) and computes the Brier score on the test set.
# ------------------------------------------------------------
def compute_full_brierloss(df_train, df_test, y_train, y_test, base, folder):

    brierloss_full = []

    for i in range(len(df_train)):
        # load true beta
        beta_df = pd.read_csv(f"{base}/{folder}/beta{i+1}.csv")
        beta = beta_df.select_dtypes(include=[np.number]).to_numpy().reshape(-1)

        # ensure correct length
        p = df_train[i].shape[1]
        if len(beta) > p:
            beta = beta[:p]
        elif len(beta) < p:
            raise ValueError(f"Beta length {len(beta)} < number of features {p} in replication {i+1}.")

        # oracle support
        selected = np.where(beta != 0)[0]

        # subset matrices
        X_train = df_train[i].to_numpy()[:, selected]
        X_test = df_test[i].to_numpy()[:, selected]

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

        # compute Brier score
        brierloss_full.append(brier_score_loss(y_te, pred))

    return brierloss_full

# ------------------------------------------------------------
# Wrapper: Column → optional Row Reduction + Logistic Modeling
#
# OUTPUT:
#   dict with:
#       "scores"
#       "time_scores"
#       "selected_columns"
#       "selected_rows"
#       "brierloss"
#       "mu"
# ------------------------------------------------------------
def apply_row_after_col_reduction_log(k, seed, base, folder, reps, row_reduction=True):

    # ------------------------------------------------------------
    # 1. Data Load
    # ------------------------------------------------------------
    print("Reading in the simulation data...")
    df_train = [pd.read_csv(f"{base}/{folder}/X_train{i+1}.csv") for i in range(reps)]
    df_test  = [pd.read_csv(f"{base}/{folder}/X_test{i+1}.csv") for i in range(reps)]
    y_train  = [pd.read_csv(f"{base}/{folder}/y_binary_train{i+1}.csv") for i in range(reps)]
    y_test   = [pd.read_csv(f"{base}/{folder}/y_binary_test{i+1}.csv") for i in range(reps)]

    # ------------------------------------------------------------
    # 2. Seeding
    # ------------------------------------------------------------
    print("Setting the seed...")
    random.seed(seed)
    np.random.seed(seed)

    # ------------------------------------------------------------
    # 3. Data Reduction
    # ------------------------------------------------------------
    print("Performing data reduction...")
    scores, time_scores, selected_columns, selected_rows, mu_values = \
        data_reduction(k, df_train, y_train, row_reduction)

    # ------------------------------------------------------------
    # 4. Logistic Modeling
    # ------------------------------------------------------------
    print("Building logistic models...")
    brierloss = logistic_modeling(
        selected_columns,
        selected_rows,
        df_train,
        df_test,
        y_train,
        y_test
    )

    # ------------------------------------------------------------
    # 5. Full Model (Benchmark)
    # ------------------------------------------------------------
    print("Building Full Model / Benchmark...")
    full = compute_full_brierloss(df_train, df_test, y_train, y_test, base, folder)
    brierloss["Full"] = full

    print("Data Reduction & Modeling completed.")

    # ------------------------------------------------------------
    # 6. Return results
    # ------------------------------------------------------------
    return {
        "scores": scores,
        "time_scores": time_scores,
        "selected_columns": selected_columns,
        "selected_rows": selected_rows,
        "brierloss": brierloss,
        "mu": mu_values
    }
