import pandas as pd
import numpy as np
import random
from scoring_functions import get_column_leverage_scores, get_row_leverage_scores, get_random_scores, get_combined_scores, get_cross_leverage_scores
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import time
from visualizations import *

def column_reduction(X, scores, k):

    # determine the dimensions of the data matrix
    n, d = np.shape(X)
    # get probabilities
    probs = scores / np.sum(scores)

    # get the number of required columns according to CUR theorem
    c = int(np.ceil(k * np.log(k)))

    # scale probs and set maximum possible probability to 1
    scaled_probs = np.minimum(c * probs, 1)

    # initialize sampling matrix S and rescaling matrix D
    t = 0
    sampled_cols = []
    D_diag = []

    # EXPECTED(c) column algorithm
    for j in range(d):
        # draw a uniform value and compare the probability against it
        z = np.random.uniform(0, 1)
        if z <= scaled_probs[j]:
            # fill rescaling and sampling matrix components as defined in algorithm
            sampled_cols.append(j)
            D_diag.append(1 / np.sqrt(scaled_probs[j]))
            t += 1

    # create S and D in matrix form
    t = len(sampled_cols)
    S = np.zeros((d, t))
    D = np.zeros((t, t))
    for idx, j in enumerate(sampled_cols):
        S[j, idx] = 1
        D[idx, idx] = D_diag[idx]

    # calculate C according to CUR paper as S and D are now given
    X = np.array(X)
    C = X @ S @ D

    return {
        "C": C,
        "selected_columns": sampled_cols,
        "probs": scaled_probs
    }

def row_reduction(C, y, k):

    # get dimensions of the reduced data matrix
    n, c = np.shape(C)

    # get the row leverage scores
    scores = get_row_leverage_scores(C, k)

    # determine the probabilities
    probs = scores / np.sum(scores)

    # get the number of required rows according to the CUR theorem
    r = int(np.ceil(c * np.log(c)))

    # scale probs and set their maximum to 1
    scaled_probs = np.minimum(r * probs, 1)

    # initialize sampling matrix S and rescaling matrix D
    t = 0
    sampled_rows = []
    D_diag = []

    # EXPECTED(r) row algorithm
    for i in range(n):
        z = np.random.uniform(0, 1)
        if z <= scaled_probs[i]:
            sampled_rows.append(i)
            D_diag.append(1 / np.sqrt(scaled_probs[i]))
            t += 1

    # create S and D in matrix form
    t = len(sampled_rows)
    S = np.zeros((n, t))
    D = np.zeros((t, t))

    for idx, i in enumerate(sampled_rows):
        S[i, idx] = 1
        D[idx, idx] = D_diag[idx]

    # determine C according to the CUR paper
    C = np.array(C)
    R = D @ S.T @ C

    #print(f"k = {k} resulting in shape(C): {np.shape(R)}")

    # get the reduced y
    y_reduced = y.iloc[sampled_rows]
    return {
        "R": R,
        "y": y_reduced,
        "selected_rows": sampled_rows,
        "probs": scaled_probs
    }

def data_reduction(k, df_train, y_train, row_reduce = True):
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
            R_ls.append(row_reduction(C_ls[i]['C'], y_train[i], k))
            R_cls.append(row_reduction(C_cls[i]['C'], y_train[i], k))
            R_rs.append(row_reduction(C_rs[i]['C'], y_train[i], k))
            R_cs.append(row_reduction(C_cs[i]['C'], y_train[i], k))

        R = {
            "R_ls": R_ls,
            "R_cls": R_cls,
            "R_rs": R_rs,
            "R_cs": R_cs
        }
    else:
        R = None

    return scores, timing_scores, C, R

def linear_modeling(C, R, df_test, y_test, y_train):
    # Fit linear models to the reduced data matrix
    model_ls = []
    model_cls = []
    model_rs = []
    model_cs = []

    if R is not None:
        n_reps = len(R['R_ls'])
    else:
        n_reps = len(C['C_ls'])

    for i in range(n_reps):
        if R is not None:
            model = LinearRegression()
            model.fit(R['R_ls'][i]['R'], R['R_ls'][i]['y'])
            model_ls.append(model)
            model = LinearRegression()
            model.fit(R['R_cls'][i]['R'], R['R_cls'][i]['y'])
            model_cls.append(model)
            model = LinearRegression()
            model.fit(R['R_rs'][i]['R'], R['R_rs'][i]['y'])
            model_rs.append(model)
            model = LinearRegression()
            model.fit(R['R_cs'][i]['R'], R['R_cs'][i]['y'])
            model_cs.append(model)
        else:
            model = LinearRegression()
            model.fit(C['C_ls'][i]['C'], y_train[i])
            model_ls.append(model)
            model = LinearRegression()
            model.fit(C['C_cls'][i]['C'], y_train[i])
            model_cls.append(model)
            model = LinearRegression()
            model.fit(C['C_rs'][i]['C'], y_train[i])
            model_rs.append(model)
            model = LinearRegression()
            model.fit(C['C_cs'][i]['C'], y_train[i])
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

def apply_row_after_col_reduction(k, seed, base, folder, reps, row_reduction = True):
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
    scores, time_scores, C, R = data_reduction(k, df_train, y_train, row_reduction)

    print("Building linear models...")
    ## 4. Linear Modeling
    rmse = linear_modeling(C, R, df_test, y_test, y_train)

    print("Building Full Model / Benchmark...")
    ## 5. Full Model
    rmse_full = compute_full_rmse(df_train, df_test, y_train, y_test, base, folder)
    rmse["Full"] = rmse_full

    print("Data Reduction & Modeling completed.")

    return scores, time_scores, C, R, rmse
