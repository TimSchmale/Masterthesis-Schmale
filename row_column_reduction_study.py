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

def row_reduction(k, X, y, gaussian = False):
    X = X.to_numpy() if isinstance(X, pd.DataFrame) else X

    # define dimensionality of the data matrix
    n, d = X.shape

    # number of rows in sketch given by the optimal sketching bound paper as a variation of the upper bound for L2 regression
    r = int(np.ceil(k * np.log(d)))

    # preparation of the sketched y vector
    y_reduced = np.zeros((r, 1)) if y is not None else None
    y = y.to_numpy().reshape(-1, 1) if isinstance(y, (pd.Series, pd.DataFrame)) else y.reshape(-1, 1)

    # prepare the sketched matrix with reduced number of rows r
    R = np.zeros((r,d))

    # created the sketched versions of X and y iteratively (keeping the comp effort low - matrix mults.)
    if r <  n:
        for i in range(n):
            if gaussian:
                # Gaussian sketching vector (r x 1)
                sketch_vec = np.random.randn(r, 1)
            else:
                # Rademacher sketching vector (r x 1) -> important: scaling by factor sqrt(r)
                sketch_vec = np.random.choice([-1,1], size=(r,1)) / np.sqrt(r)

            # Reduce X: Outer product: (r x 1) @ (1 x d) -> (r x d)
            R += sketch_vec @ X[i, :].reshape(1, d)

            # Reduce y in parallel
            if y is not None:
                y_reduced += sketch_vec * y[i]
    else:
        # case when now reduction is needed
        R = X
        y_reduced  = y

    return R, y_reduced

def column_reduction(R, scores, k):

    # get the dimensions of the row-reduced matrix R
    r,d = np.shape(R)

    # get probabilities by normalizing the scores
    probs = scores / np.sum(scores)

    # get the number of desired columns from CUR paper by assuming the matrix R is given
    c = int(np.ceil(k * np.log(k)))

    # scale probs and set maximum to 1
    scaled_probs = np.minimum(c * probs, 1)

    # initialize the sampling matrix S and rescaling matrix D
    t = 0
    sampled_cols = []
    S_cols = []
    D_diag = []

    # sample the columns through uniform random variable -> EXPECTED(c) algorithm
    for j in range(d):
        z = np.random.uniform(0, 1)
        if z <= scaled_probs[j]:
            sampled_cols.append(j)
            S_cols.append(j)
            D_diag.append(1 / np.sqrt(scaled_probs[j]))
            t += 1

    # create S and D in matrix form
    t = len(sampled_cols)
    S = np.zeros((d, t))
    D = np.zeros((t, t))

    for idx, j in enumerate(sampled_cols):
        S[j, idx] = 1
        D[idx, idx] = D_diag[idx]

    # determine C as final matrix
    R = np.array(R)
    C = R @ S @ D

    return {
        "C": C,
        "selected_columns": sampled_cols,
        "probs": scaled_probs
    }

def data_reduction(k, df_train, y_train, gaussian = False):

    # 1. Calculate the row reduction
    R_reduced = []
    y_reduced = []
    for i in range(len(df_train)):
        R_temp, y_temp = row_reduction(k, df_train[i], y_train[i], gaussian)
        R_reduced.append(R_temp)
        y_reduced.append(y_temp)

    # 2. Perform the score calculations
    column_ls = []
    column_cls = []
    column_rs = []
    column_cs = []

    time_ls = []
    time_cls = []
    time_rs = []
    time_cs = []
    for i in range(len(R_reduced)):
        start = time.perf_counter()
        column_ls.append(get_column_leverage_scores(R_reduced[i], k))
        time_ls.append(time.perf_counter() - start)

        start = time.perf_counter()
        column_cls.append(get_cross_leverage_scores(R_reduced[i], y_reduced[i]))
        time_cls.append(time.perf_counter() - start)

        start = time.perf_counter()
        column_rs.append(get_random_scores(R_reduced[i]))
        time_rs.append(time.perf_counter() - start)

        start = time.perf_counter()
        column_cs.append(get_combined_scores(R_reduced[i], y_reduced[i], k, p_leverage=0.2))
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

    # 3. Calculation of column reduction
    C_ls = []
    C_cls = []
    C_rs = []
    C_cs = []

    for i in range(len(R_reduced)):
        C_ls.append(column_reduction(R_reduced[i], column_ls[i], k))
        C_cls.append(column_reduction(R_reduced[i], np.abs(column_cls[i]), k))
        C_rs.append(column_reduction(R_reduced[i], column_rs[i], k))
        C_cs.append(column_reduction(R_reduced[i], column_cs[i], k))

    C = {
        "C_ls": C_ls,
        "C_cls": C_cls,
        "C_rs": C_rs,
        "C_cs": C_cs
    }

    return scores, timing_scores, C, R_reduced, y_reduced

def linear_modeling(C, R, df_test, y_test, y_reduced):

    # Fit linear models to the reduced data matrix
    model_ls = []
    model_cls = []
    model_rs = []
    model_cs = []

    n_reps = len(C['C_ls'])

    for i in range(n_reps):
        model = LinearRegression()
        model.fit(C['C_ls'][i]['C'], y_reduced[i])
        model_ls.append(model)
        model = LinearRegression()
        model.fit(C['C_cls'][i]['C'], y_reduced[i])
        model_cls.append(model)
        model = LinearRegression()
        model.fit(C['C_rs'][i]['C'], y_reduced[i])
        model_rs.append(model)
        model = LinearRegression()
        model.fit(C['C_cs'][i]['C'], y_reduced[i])
        model_cs.append(model)


    # Build predictions
    predictions_ls = []
    predictions_cls = []
    predictions_rs = []
    predictions_cs = []
    df_test_reduced_ls = []
    df_test_reduced_cls = []
    df_test_reduced_rs = []
    df_test_reduced_cs = []

    for i in range(n_reps):
        df_test_reduced_ls.append(df_test[i].iloc[:, C['C_ls'][i]['selected_columns']])
        predictions_ls.append(model_ls[i].predict(df_test_reduced_ls[i]))
        df_test_reduced_cls.append(df_test[i].iloc[:, C['C_cls'][i]['selected_columns']])
        predictions_cls.append(model_cls[i].predict(df_test_reduced_cls[i]))
        df_test_reduced_rs.append(df_test[i].iloc[:, C['C_rs'][i]['selected_columns']])
        predictions_rs.append(model_rs[i].predict(df_test_reduced_rs[i]))
        df_test_reduced_cs.append(df_test[i].iloc[:, C['C_cs'][i]['selected_columns']])
        predictions_cs.append(model_cs[i].predict(df_test_reduced_cs[i]))

    # Calculate RMSE
    rmse_ls = []
    rmse_cls = []
    rmse_rs = []
    rmse_cs = []
    for i in range(n_reps):
        rmse_ls.append(np.sqrt(mean_squared_error(y_test[i], predictions_ls[i])))
        rmse_cls.append(np.sqrt(mean_squared_error(y_test[i], predictions_cls[i])))
        rmse_rs.append(np.sqrt(mean_squared_error(y_test[i], predictions_rs[i])))
        rmse_cs.append(np.sqrt(mean_squared_error(y_test[i], predictions_cs[i])))
    rmse = {"LS": rmse_ls, "CLS": rmse_cls, "RS": rmse_rs, "CS": rmse_cs, }

    return rmse

# ==============================================================================================
# Full Model: Application of Linear Model to the optimal set of columns
# ==============================================================================================
def compute_full_rmse(df_train, df_test, y_train, y_test, base, folder):
    rmse_full = []

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

        # Fit linear model
        model = LinearRegression()
        model.fit(X_train_sel.to_numpy(), y_tr)

        # Predict on test set
        predictions = model.predict(X_test_sel.to_numpy())

        # Compute RMSE
        rmse = np.sqrt(mean_squared_error(y_te, predictions))
        rmse_full.append(rmse)

    return rmse_full

def apply_col_after_row_reduction(k, seed, base, folder, reps, gaussian = False):
    ## 1. Data Load
    # initialize lists to store data
    df_train = []
    df_test = []
    y_train = []
    y_test = []

    print("Reading in the simulation data...")
    # read in the simulation data
    for i in range(reps):
        df_train.append(
            pd.read_csv(f"{base}/{folder}/X_train{i + 1}.csv")
        )
        df_test.append(
            pd.read_csv(f"{base}/{folder}/X_test{i + 1}.csv")
        )
        y_train.append(
            pd.read_csv(f"{base}/{folder}/y_train{i + 1}.csv")
        )
        y_test.append(
            pd.read_csv(f"{base}/{folder}/y_test{i + 1}.csv")
        )

    print("Setting the seed...")
    ## 2. Seeding
    random.seed(seed)
    np.random.seed(seed)

    print("Performing data reduction...")
    ## 3. Data Reduction
    scores, timing_scores, C, R_reduced, y_reduced = data_reduction(k, df_train, y_train, gaussian)

    print("Building linear models...")
    ## 4. Linear Modeling
    rmse = linear_modeling(C, R_reduced, df_test, y_test, y_reduced)


    print("Building Full Model / Benchmark...")
    ## 6. Full Model
    rmse_full = compute_full_rmse(df_train, df_test, y_train, y_test, base, folder)
    rmse["Full"] = rmse_full

    print("Data Reduction & Modeling completed.")

    return scores, timing_scores, C, R_reduced, rmse