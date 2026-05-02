import pandas as pd
import random
from scoring_functions import get_column_leverage_scores, get_log_reg_leverage_scores, get_random_scores, get_combined_scores, get_cross_leverage_scores
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import log_loss
import time
from visualizations import *

# ------------------------------------------------------------
# Function to perform a column reduction given score values for each column as calculation basis
#
# INPUT:
#   X         : data matrix of dimension n x p
#   scores    : scoring vector of length p
#   k         : rank parameter (desired rank of approximation)
#
# OUTPUT:
#   reduced data matrix as well as vector of selected variable indices
# ------------------------------------------------------------
def column_reduction(X, scores, k):
    # get dimensions
    n, p = np.shape(X)
    # get probabilities
    probs = scores / np.sum(scores)

    # get the number of desired columns
    c = int(np.ceil(k * np.log(k)))

    # scale probs and set maximum to 1
    scaled_probs = np.minimum(c * probs, 1)

    # sample the columns and fill S and D
    t = 0
    sampled_cols = []
    D_diag = []

    for j in range(p):
        z = np.random.uniform(0, 1)
        if z <= scaled_probs[j]:
            sampled_cols.append(j)
            D_diag.append(1 / np.sqrt(scaled_probs[j]))
            t += 1

    # create S and D
    t = len(sampled_cols)
    S = np.zeros((p, t))
    D = np.zeros((t, t))

    for idx, j in enumerate(sampled_cols):
        S[j, idx] = 1
        D[idx, idx] = D_diag[idx]

    # get C
    X = np.array(X)
    C = X @ S @ D

    return {
        "C": C,
        "selected_columns": sampled_cols,
        "probs": scaled_probs
    }

# ------------------------------------------------------------
# Function to perform a row reduction based on a score vector
#
# INPUT:
#   C         : reduced data matrix of dimension n x c
#   y         : response vector of length n
#   k         : rank parameter (desired rank of approximation)
#
# OUTPUT:
#   reduced data matrix as well as vector of selected variable indices
# ------------------------------------------------------------
def row_reduction(C, y):
    # ensure numpy arrays
    C = np.asarray(C)
    y = np.asarray(y)

    n, d = C.shape

    # --- estimate mu via logistic regression ---
    def estimate_mu_via_logreg(C, y):
        model = LogisticRegression(
            penalty='l2',
            C=1e6,
            solver='lbfgs',
            max_iter=2000
        )
        model.fit(C, y)
        beta = model.coef_.flatten()

        v = C @ beta
        pos = np.sum(np.abs(v[v > 0]))
        neg = np.sum(np.abs(v[v < 0]))

        if neg == 0:
            return np.inf
        return pos / neg

    mu = estimate_mu_via_logreg(C, y)

    # --- compute r from theorem ---
    if mu == np.inf or mu <= 1:
        mu = max(mu, 1.0001)

    r_float = mu * d * np.log(mu * d)
    r = int(np.ceil(r_float))

    # cap r
    r = max(1, min(r, n))

    # --- compute leverage scores ---
    scores = get_log_reg_leverage_scores(C)
    probs = scores / np.sum(scores)

    # --- sample rows ---
    rng = np.random.default_rng()
    sampled_rows = rng.choice(
        n,
        size=r,
        replace=False,
        p=probs
    )

    R = C[sampled_rows, :]
    y_reduced = y[sampled_rows]

    return {
        "R": R,
        "y": y_reduced,
        "selected_rows": sampled_rows,
        "probs": probs,
        "mu": mu,
        "r": r
    }

# ==============================================================================================
# data_reduction: Function for data reduction of Simulation Study
# ==============================================================================================

def data_reduction(k, df_train, y_train, row_reduce = True):
    """
    Perform data reduction on simulated datasets using multiple scoring methods.

    This function calculates column- and row-reduction sets based on four different
    scoring methods (Leverage Scores, Cross-Leverage Scores, Random Scores, Combined Scores)
    for each replication of the simulation study. It returns the raw scores,
    the column-reduced datasets, and the row-reduced datasets.

    Parameters
    ----------
    k : int
        Number of features/columns to select per reduction step.
    df_train : list of pd.DataFrame
        List of training datasets, one per replication.
    y_train : list of array-like
        List of target vectors corresponding to each training dataset.

    Returns
    -------
    scores : dict
        Dictionary of calculated scores per method and replication:
        {
            "LS": list of leverage scores per replication,
            "CLS": list of cross-leverage scores per replication,
            "RS": list of random scores per replication,
            "CS": list of combined scores per replication
        }

    C : dict
        Dictionary of column-reduced datasets per method:
        {
            "C_ls": list of column-reduction results for LS,
            "C_cls": list for CLS,
            "C_rs": list for RS,
            "C_cs": list for CS
        }

    R : dict
        Dictionary of row-reduced datasets per method:
        {
            "R_ls": list of row-reduction results for LS,
            "R_cls": list for CLS,
            "R_rs": list for RS,
            "R_cs": list for CS
        }

    Notes
    -----
    - Each list in the output dictionaries corresponds to one replication of the simulation.
    - The row and column reduction functions should return dictionaries containing
      the reduced matrices and any metadata needed for downstream modeling.
    """
    # 1. Perform the score calculations
    column_ls = []
    column_cls = []
    column_rs = []
    column_cs = []

    time_ls = []
    time_cls = []
    time_rs = []
    time_cs = []
    for i in range(len(df_train)):
        start = time.perf_counter()
        column_ls.append(get_column_leverage_scores(df_train[i], k))
        time_ls.append(time.perf_counter() - start)

        start = time.perf_counter()
        column_cls.append(get_cross_leverage_scores(df_train[i], y_train[i]))
        time_cls.append(time.perf_counter() - start)

        start = time.perf_counter()
        column_rs.append(get_random_scores(df_train[i]))
        time_rs.append(time.perf_counter() - start)

        start = time.perf_counter()
        column_cs.append(get_combined_scores(df_train[i], y_train[i], k, p_leverage=0.2))
        time_cs.append(time.perf_counter() - start)

    timing_scores = {
        "LS": time_ls,
        "CLS": time_cls,
        "RS": time_rs,
        "CS": time_cs
    }
    scores = {
        "LS": column_ls,
        "CLS": column_cls,
        "RS": column_rs,
        "CS": column_cs
    }

    # 2. Calculation of column reduction
    C_ls = []
    C_cls = []
    C_rs = []
    C_cs = []

    for i in range(len(df_train)):
        C_ls.append(column_reduction(df_train[i], column_ls[i], k))
        C_cls.append(column_reduction(df_train[i], np.abs(column_cls[i]), k))
        C_rs.append(column_reduction(df_train[i], column_rs[i], k))
        C_cs.append(column_reduction(df_train[i], column_cs[i], k))

    C = {
        "C_ls": C_ls,
        "C_cls": C_cls,
        "C_rs": C_rs,
        "C_cs": C_cs
    }

    if row_reduce:
        # 3. Calculation of row reduction
        R_ls = []
        R_cls = []
        R_rs = []
        R_cs = []

        for i in range(len(df_train)):
            R_ls.append(row_reduction(C_ls[i]['C'], y_train[i]))
            R_cls.append(row_reduction(C_cls[i]['C'], y_train[i]))
            R_rs.append(row_reduction(C_rs[i]['C'], y_train[i]))
            R_cs.append(row_reduction(C_cs[i]['C'], y_train[i]))

        R = {
            "R_ls": R_ls,
            "R_cls": R_cls,
            "R_rs": R_rs,
            "R_cs": R_cs
        }
    else:
        R = None

    return scores, timing_scores, C, R

# ==============================================================================================
# logistic_modeling: Function for application of logistic model to reduced data sets, and accuracy calculation
# ==============================================================================================

def logistic_modeling(C, R, df_test, y_test, y_train):
    """
    Apply logistic regression to reduced datasets and compute log-loss per replication and method.
    """

    # Models per method
    model_ls = []
    model_cls = []
    model_rs = []
    model_cs = []

    if R is not None:
        n_reps = len(R['R_ls'])
    else:
        n_reps = len(C['C_ls'])

    # --- Fit models ---
    for i in range(n_reps):

        if R is not None:
            # LS
            model = LogisticRegression(penalty='l2', C=1e6, solver='lbfgs', max_iter=2000)
            model.fit(R['R_ls'][i]['R'], R['R_ls'][i]['y'])
            model_ls.append(model)

            # CLS
            model = LogisticRegression(penalty='l2', C=1e6, solver='lbfgs', max_iter=2000)
            model.fit(R['R_cls'][i]['R'], R['R_cls'][i]['y'])
            model_cls.append(model)

            # RS
            model = LogisticRegression(penalty='l2', C=1e6, solver='lbfgs', max_iter=2000)
            model.fit(R['R_rs'][i]['R'], R['R_rs'][i]['y'])
            model_rs.append(model)

            # CS
            model = LogisticRegression(penalty='l2', C=1e6, solver='lbfgs', max_iter=2000)
            model.fit(R['R_cs'][i]['R'], R['R_cs'][i]['y'])
            model_cs.append(model)

        else:
            # Only column reduction
            model = LogisticRegression(penalty='l2', C=1e6, solver='lbfgs', max_iter=2000)
            model.fit(C['C_ls'][i]['C'], y_train[i])
            model_ls.append(model)

            model = LogisticRegression(penalty='l2', C=1e6, solver='lbfgs', max_iter=2000)
            model.fit(C['C_cls'][i]['C'], y_train[i])
            model_cls.append(model)

            model = LogisticRegression(penalty='l2', C=1e6, solver='lbfgs', max_iter=2000)
            model.fit(C['C_rs'][i]['C'], y_train[i])
            model_rs.append(model)

            model = LogisticRegression(penalty='l2', C=1e6, solver='lbfgs', max_iter=2000)
            model.fit(C['C_cs'][i]['C'], y_train[i])
            model_cs.append(model)

    # --- Predictions ---
    pred_ls = []
    pred_cls = []
    pred_rs = []
    pred_cs = []

    for i in range(n_reps):
        # LS
        X_test_ls = df_test[i].iloc[:, C['C_ls'][i]['selected_columns']]
        pred_ls.append(model_ls[i].predict_proba(X_test_ls)[:, 1])

        # CLS
        X_test_cls = df_test[i].iloc[:, C['C_cls'][i]['selected_columns']]
        pred_cls.append(model_cls[i].predict_proba(X_test_cls)[:, 1])

        # RS
        X_test_rs = df_test[i].iloc[:, C['C_rs'][i]['selected_columns']]
        pred_rs.append(model_rs[i].predict_proba(X_test_rs)[:, 1])

        # CS
        X_test_cs = df_test[i].iloc[:, C['C_cs'][i]['selected_columns']]
        pred_cs.append(model_cs[i].predict_proba(X_test_cs)[:, 1])

    # --- Compute log-loss ---
    ll_ls = []
    ll_cls = []
    ll_rs = []
    ll_cs = []

    for i in range(n_reps):
        ll_ls.append(log_loss(y_test[i], pred_ls[i]))
        ll_cls.append(log_loss(y_test[i], pred_cls[i]))
        ll_rs.append(log_loss(y_test[i], pred_rs[i]))
        ll_cs.append(log_loss(y_test[i], pred_cs[i]))

    return {
        "LL_LS": ll_ls,
        "LL_CLS": ll_cls,
        "LL_RS": ll_rs,
        "LL_CS": ll_cs
    }


# ==============================================================================================
# Full Model: Application of Linear Model to the optimal set of columns
# ==============================================================================================
def compute_full_logloss(df_train, df_test, y_train, y_test, base, folder):
    """
    Compute log-loss per replication for the Full Model using the true beta selection.

    Parameters
    ----------
    df_train : list of pd.DataFrame
        Training datasets per replication
    df_test : list of pd.DataFrame
        Test datasets per replication
    y_train : list of arrays
        Training targets per replication
    y_test : list of arrays
        Test targets per replication
    base : str
        Base path to replication folders
    folder : str
        Folder name inside base containing the beta.csv files

    Returns
    -------
    logloss_full : list of float
        Log-loss per replication
    """
    logloss_full = []

    for i in range(len(df_train)):
        # Load beta for this replication
        beta_df = pd.read_csv(f"{base}/{folder}/beta{i + 1}.csv")
        beta = np.array(beta_df).reshape(-1)

        # Identify selected columns (beta != 0)
        selected_cols = np.where(beta != 0)[0]

        # Subset training and test data
        X_train_sel = df_train[i].iloc[:, selected_cols]
        X_test_sel = df_test[i].iloc[:, selected_cols]

        # Make sure target vectors are 1D
        y_tr = np.ravel(y_train[i])
        y_te = np.ravel(y_test[i])

        # Fit logistic model
        model = LogisticRegression(
            penalty='l2',
            C=1e6,
            solver='lbfgs',
            max_iter=2000
        )
        model.fit(X_train_sel.to_numpy(), y_tr)

        # Predict probabilities on test set
        pred = model.predict_proba(X_test_sel.to_numpy())[:, 1]

        # Compute log-loss
        ll = log_loss(y_te, pred)
        logloss_full.append(ll)

    return logloss_full

def apply_row_after_col_reduction_log(k, seed, base, folder, reps, row_reduction=True):
    """
    Full simulation workflow for logistic regression:
    - Load data
    - Perform column and row reduction
    - Fit logistic models on reduced datasets
    - Fit Full Model using true beta
    - Compute log-loss for all methods
    """

    # ------------------------------------------------------------
    # 1. Data Load
    # ------------------------------------------------------------
    df_train = []
    df_test = []
    y_train = []
    y_test = []

    print("Reading in the simulation data...")
    for i in range(reps):
        df_train.append(pd.read_csv(f"{base}/{folder}/X_train{i + 1}.csv"))
        df_test.append(pd.read_csv(f"{base}/{folder}/X_test{i + 1}.csv"))
        y_train.append(pd.read_csv(f"{base}/{folder}/y_binary_train{i + 1}.csv"))
        y_test.append(pd.read_csv(f"{base}/{folder}/y_binary_test{i + 1}.csv"))

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
    scores, time_scores, C, R = data_reduction(k, df_train, y_train, row_reduction)

    # ------------------------------------------------------------
    # 4. Logistic Modeling
    # ------------------------------------------------------------
    print("Building logistic models...")
    logloss = logistic_modeling(C, R, df_test, y_test, y_train)

    # ------------------------------------------------------------
    # 5. Full Model (Benchmark)
    # ------------------------------------------------------------
    print("Building Full Model / Benchmark...")
    logloss_full = compute_full_logloss(df_train, df_test, y_train, y_test, base, folder)
    logloss["Full"] = logloss_full

    print("Data Reduction & Modeling completed.")

    return scores, time_scores, C, R, logloss

