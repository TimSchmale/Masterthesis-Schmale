import numpy as np
import pandas as pd
import time
from scoring_functions import get_column_leverage_scores, get_row_leverage_scores, get_random_scores, get_combined_scores, get_cross_leverage_scores
from statsmodels.sandbox.distributions.genpareto import shape

from sklearn.preprocessing import StandardScaler

from scoring_functions import *
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
import random
from visualizations import *

# ------------------------------------------------------------
# Row Reduction (Sketching-based)
#
# INPUT:
#   k        : target rank for the sketch size
#   X        : design matrix (n x d), DataFrame or ndarray
#   y        : response vector (n x 1)
#   gaussian : boolean, True = Gaussian sketch, False = Rademacher
#
# OUTPUT:
#   R         : row-reduced matrix (r x d)
#   y_reduced : reduced response vector (r x 1)
#
# Description:
#   Performs sketching-based row reduction using either Gaussian
#   or Rademacher sketch vectors.
# ------------------------------------------------------------
def row_reduction(k, X, y, gaussian=False):
    X = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
    y = y.to_numpy().reshape(-1, 1) if isinstance(y, (pd.Series, pd.DataFrame)) else y.reshape(-1, 1)

    n, d = X.shape
    r = int(np.ceil(k * np.log(d)))  # sketch size

    # allocate sketch matrices
    R = np.zeros((r, d))
    y_reduced = np.zeros((r, 1))

    # iterative sketching loop
    if r < n:
        for i in range(n):
            if gaussian:
                sketch_vec = np.random.randn(r, 1)
            else:
                sketch_vec = np.random.choice([-1, 1], size=(r, 1)) / np.sqrt(r)

            R += sketch_vec @ X[i:i+1, :]
            y_reduced += sketch_vec * y[i]
    else:
        R = X.copy()
        y_reduced = y.copy()

    return R, y_reduced

# ------------------------------------------------------------
# Column Reduction (EXPECTED(c) sampling)
#
# INPUT:
#   R      : row-reduced matrix (r x d)
#   scores : column scores (length d)
#   k      : target rank
#
# OUTPUT:
#   Dictionary with:
#       "C"               : reduced matrix (r x t)
#       "selected_columns": list of selected column indices
#
# Description:
#   Implements the EXPECTED(c) column sampling algorithm from CUR.
#   Uses Bernoulli sampling with scaled probabilities and rescales
#   selected columns accordingly.
# ------------------------------------------------------------
def column_reduction(R, scores, k):
    R = np.asarray(R)
    r, d = R.shape

    probs = scores / scores.sum()
    c = int(np.ceil(k * np.log(k)))
    scaled_probs = np.minimum(c * probs, 1)

    # Bernoulli sampling
    z = np.random.rand(d)
    sampled = np.where(z <= scaled_probs)[0]

    if len(sampled) == 0:
        sampled = np.array([np.argmax(scaled_probs)])

    # rescaling
    D_inv = 1 / np.sqrt(scaled_probs[sampled])
    C = R[:, sampled] * D_inv

    return {
        "C": C,
        "selected_columns": sampled.tolist()
    }

# ------------------------------------------------------------
# Data Reduction (Row → Column)
#
# INPUT:
#   k        : target rank
#   df_train : list of training matrices
#   y_train  : list of response vectors
#   gaussian : boolean, whether Gaussian sketching is used
#
# OUTPUT:
#   scores           : dict of score vectors per method
#   time_scores      : dict of score computation times
#   selected_columns : dict of selected columns per method
#   selected_rows    : list of selected row indices per replication
#   R_list           : list of row-reduced matrices
#   y_list           : list of row-reduced response vectors
#
# Description:
#   Performs row reduction first (sketching), then computes
#   column scores and applies EXPECTED(c) column sampling.
#   Only selected indices and scores are returned.
# ------------------------------------------------------------
def data_reduction(k, df_train, y_train, gaussian=False):

    n_reps = len(df_train)

    selected_rows = []
    selected_columns = {"LS": [], "CLS": [], "RS": [], "CS": []}
    scores = {"LS": [], "CLS": [], "RS": [], "CS": []}
    time_scores = {"LS": [], "CLS": [], "RS": [], "CS": []}

    R_list = []
    y_list = []

    # 1) Row reduction
    for i in range(n_reps):
        R, y_red = row_reduction(k, df_train[i], y_train[i], gaussian)
        R_list.append(R)
        y_list.append(y_red)
        selected_rows.append(list(range(R.shape[0])))

    # 2) Score computation
    for i in range(n_reps):
        R = R_list[i]
        y = y_list[i]

        start = time.perf_counter()
        scores["LS"].append(get_column_leverage_scores(R, k))
        time_scores["LS"].append(time.perf_counter() - start)

        start = time.perf_counter()
        scores["CLS"].append(get_cross_leverage_scores(R, y))
        time_scores["CLS"].append(time.perf_counter() - start)

        start = time.perf_counter()
        scores["RS"].append(get_random_scores(R))
        time_scores["RS"].append(time.perf_counter() - start)

        start = time.perf_counter()
        scores["CS"].append(get_combined_scores(R, y, k, p_leverage=0.2))
        time_scores["CS"].append(time.perf_counter() - start)

    # 3) Column reduction
    for i in range(n_reps):
        R = R_list[i]

        selected_columns["LS"].append(column_reduction(R, scores["LS"][i], k)["selected_columns"])
        selected_columns["CLS"].append(column_reduction(R, np.abs(scores["CLS"][i]), k)["selected_columns"])
        selected_columns["RS"].append(column_reduction(R, scores["RS"][i], k)["selected_columns"])
        selected_columns["CS"].append(column_reduction(R, scores["CS"][i], k)["selected_columns"])

    return scores, time_scores, selected_columns, selected_rows, R_list, y_list


# ------------------------------------------------------------
# Linear Modeling on Reduced Data
#
# INPUT:
#   selected_columns : dict of selected columns per method
#   R_list           : list of row-reduced matrices
#   y_list           : list of reduced response vectors
#   df_test          : list of test matrices
#   y_test           : list of test responses
#
# OUTPUT:
#   rmse : dict of RMSE values per method
#
# Description:
#   Fits linear regression models on the reduced matrices and
#   evaluates prediction performance on the test data.
# ------------------------------------------------------------
def linear_modeling(selected_columns, R_list, y_list, df_test, y_test):

    n_reps = len(df_test)
    rmse = {"LS": [], "CLS": [], "RS": [], "CS": []}

    for i in range(n_reps):
        for method in ["LS", "CLS", "RS", "CS"]:
            cols = selected_columns[method][i]

            model = LinearRegression().fit(R_list[i][:, cols], y_list[i])
            preds = model.predict(df_test[i].to_numpy()[:, cols])

            rmse[method].append(np.sqrt(mean_squared_error(y_test[i], preds)))

    return rmse

# ------------------------------------------------------------
# Full Model Benchmark (Oracle RMSE)
#
# INPUT:
#   df_train : list of training design matrices (per replication)
#   df_test  : list of test design matrices (per replication)
#   y_train  : list of training response vectors
#   y_test   : list of test response vectors
#   base     : base directory for simulation data
#   folder   : subfolder containing beta-files
#
# OUTPUT:
#   rmse_full : list of RMSE values (one per replication)
#
# Description:
#   Uses the true non-zero beta positions (oracle support) to fit
#   a full linear model per replication and computes the RMSE on
#   the corresponding test data. Serves as benchmark.
# ------------------------------------------------------------
def compute_full_rmse(df_train, df_test, y_train, y_test, base, folder):

    rmse_full = []

    for i in range(len(df_train)):
        # load beta for this replication (assume single numeric column or row)
        beta_df = pd.read_csv(f"{base}/{folder}/beta{i+1}.csv")
        beta = beta_df.select_dtypes(include=[np.number]).to_numpy().reshape(-1)

        # ensure beta length matches number of features
        p = df_train[i].shape[1]
        if len(beta) > p:
            beta = beta[:p]
        elif len(beta) < p:
            raise ValueError(f"Beta length {len(beta)} < number of features {p} in replication {i+1}.")

        # oracle support: indices of non-zero coefficients
        selected = np.where(beta != 0)[0]

        # subset design matrices
        X_train = df_train[i].to_numpy()[:, selected]
        X_test = df_test[i].to_numpy()[:, selected]

        # response vectors as 1D
        y_tr = y_train[i].to_numpy().reshape(-1)
        y_te = y_test[i].to_numpy().reshape(-1)

        # fit oracle model
        model = LinearRegression().fit(X_train, y_tr)
        preds = model.predict(X_test)

        # compute RMSE
        rmse_full.append(np.sqrt(mean_squared_error(y_te, preds)))

    return rmse_full


# ------------------------------------------------------------
# Wrapper: Row → Column Reduction + Modeling
#
# INPUT:
#   k        : target rank
#   seed     : random seed
#   base     : base directory for data
#   folder   : subfolder containing simulation files
#   reps     : number of replications
#   gaussian : boolean, whether Gaussian sketching is used
#
# OUTPUT:
#   Dictionary with:
#       "scores"
#       "time_scores"
#       "selected_columns"
#       "selected_rows"
#       "rmse"
#
# Description:
#   Loads simulation data, performs row reduction followed by
#   column reduction, fits linear models, computes RMSE, and
#   returns only the relevant structures.
# ------------------------------------------------------------
def apply_col_after_row_reduction(k, seed, base, folder, reps, gaussian=False):

    print("Reading simulation data...")
    df_train = [pd.read_csv(f"{base}/{folder}/X_train{i+1}.csv") for i in range(reps)]
    df_test  = [pd.read_csv(f"{base}/{folder}/X_test{i+1}.csv") for i in range(reps)]
    y_train  = [pd.read_csv(f"{base}/{folder}/y_train{i+1}.csv") for i in range(reps)]
    y_test   = [pd.read_csv(f"{base}/{folder}/y_test{i+1}.csv") for i in range(reps)]

    random.seed(seed)
    np.random.seed(seed)

    print("Performing data reduction...")
    scores, time_scores, selected_columns, selected_rows, R_list, y_list = \
        data_reduction(k, df_train, y_train, gaussian)

    print("Fitting linear models...")
    rmse = linear_modeling(selected_columns, R_list, y_list, df_test, y_test)

    print("Computing full model benchmark...")
    rmse_full = compute_full_rmse(df_train, df_test, y_train, y_test, base, folder)
    rmse["Full"] = rmse_full

    print("Completed.")

    return {
        "scores": scores,
        "time_scores": time_scores,
        "selected_columns": selected_columns,
        "selected_rows": selected_rows,
        "rmse": rmse
    }