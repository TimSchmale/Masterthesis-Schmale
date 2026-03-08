import numpy as np
import pandas as pd
import time
from scoring_functions import *

def row_reduction(X, k, y, gaussian = False):
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
    for i in range(n):
        if gaussian:
            # Gaussian sketching vector (r x 1)
            sketch_vec = np.random.randn(r, 1)
        else:
            # Rademacher sketching vector (r x 1)
            sketch_vec = np.random.choice([-1, 1], size=(r, 1))

        # Reduce X: Outer product: (r x 1) @ (1 x d) -> (r x d)
        R += sketch_vec @ X[i, :].reshape(1, d)

        # Reduce y in parallel
        if y is not None:
            y_reduced += sketch_vec * y[i]

    # scale R
    R = R / np.sqrt(r)

    # scale y_reduced
    y_reduced = y_reduced / np.sqrt(r)

    return R, y_reduced

def column_reduction(R, scores, k):
    # get dimensions
    n, p = np.shape(R)
    # get probabilities
    probs = scores / np.sum(scores)

    # get the number of desired columns
    c = int(np.ceil(k * np.log(k)))

    # scale probs and set maximum to 1
    scaled_probs = np.minimum(c * probs, 1)

    # sample the columns and fill S and D
    t = 0
    sampled_cols = []
    S_cols = []
    D_diag = []

    for j in range(p):
        z = np.random.uniform(0, 1)
        if z <= scaled_probs[j]:
            sampled_cols.append(j)
            S_cols.append(j)
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
        column_ls.append(get_leverage_scores(R_reduced[i], k))
        time_ls.append(time.perf_counter() - start)

        start = time.perf_counter()
        column_cls.append(get_cross_leverage_scores(R_reduced[i], y_train[i]))
        time_cls.append(time.perf_counter() - start)

        start = time.perf_counter()
        column_rs.append(get_random_scores(R_reduced[i]))
        time_rs.append(time.perf_counter() - start)

        start = time.perf_counter()
        column_cs.append(get_combined_scores(R_reduced[i], y_train[i], k, p_leverage=0.2))
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


## Linear Modeling needed

## compute full rmse needed

## method for comparison needed