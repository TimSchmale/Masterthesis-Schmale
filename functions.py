import random
import numpy as np
import pandas as pd
from numpy.ma.core import shape


# ------------------------------------------------------------
# Function to generate a random score (uniformly distributed) per column
#
# INPUT:
#   X         : design matrix of dimension n x p
#
# OUTPUT:
#   Numeric vector of length p containing random scores between 0 and 1
# ------------------------------------------------------------
def get_random_scores(X):
    n_cols = X.shape[1]
    return np.random.uniform(0, 1, size=n_cols)

# ------------------------------------------------------------
# Function to compute Leverage Scores (LS) for the case n<p
#
# INPUT:
#   X         : design matrix of dimension n x p
#   k         : desired rank of final approximation
#
# OUTPUT:
#   Numeric vector of length p containing the LS values
# ------------------------------------------------------------
def get_leverage_scores(X, k):
    # check n < p
    if X.shape[0] >= X.shape[1]:
        print("This function is designed for the case n < p. The matrix is now getting transposed.")
        X = X.T

    # check k
    if k > X.shape[0]:
        print(k, X.shape[0])
        raise ValueError("k must be <= nrow(X)")

    # perform singular value decomposition to get V matrix
    U, S, Vh = np.linalg.svd(X)

    return np.sum(Vh.T[:,1:k] ** 2, axis = 1)

# ------------------------------------------------------------
# Function to compute Column Cross Leverage Scores (CLS) for the case n<p
#
# INPUT:
#   X         : design matrix of dimension n x p
#   y         : response vector of length n
#
# OUTPUT:
#   Numeric vector of length p containing the CLS values
# ------------------------------------------------------------
def get_cross_leverage_scores(X, y):

    # combine X and y to one matrix
    Xy = pd.concat([X, y], axis=1)

    # check n < p
    if X.shape[0] >= X.shape[1]:
        raise ValueError("This function is designed for the case n < p.")

    # perform the QR decomposition
    Q, R = np.linalg.qr(Xy.T)

    # compute CLS using inner products of rows of Q
    return np.sum(Q[:-1, :] * Q[-1, :], axis = 1)

# ------------------------------------------------------------
# Function to compute Column Cross Leverage Scores (CLS) for the case n<p
#
# INPUT:
#   X         : design matrix of dimension n x p
#   y         : response vector of length n
#   k         : desired rank of final approximation
#   p_leverage: percentage of leverage scores in calculation of combined scores
# OUTPUT:
#   Numeric vector of length p containing the combined score values
# ------------------------------------------------------------
def get_combined_scores(X, y, k, p_leverage):

    # calculate leverage scores and normalize to 1
    ls = get_leverage_scores(X, k)
    ls = ls / np.sqrt(np.sum(ls ** 2))

    # calculate cross leverage scores and normalize to 1
    cls = np.abs(get_cross_leverage_scores(X, y))
    cls = cls / np.sqrt(np.sum(ls ** 2))

    return (1-p_leverage) * cls + p_leverage * ls

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
#   k         : rank parameter (desired rank of approximation)
#
# OUTPUT:
#   reduced data matrix as well as vector of selected variable indices
# ------------------------------------------------------------
def row_reduction(C, k):
    # get dimensions
    n, c = np.shape(C)
    # get the leverage scores for the rows
    scores = get_leverage_scores(C, k)
    # get probabilities
    probs = scores / np.sum(scores)

    # get the number of desired columns
    r = int(np.ceil(c * np.log(c)))

    # scale probs and set maximum to 1
    scaled_probs = np.minimum(r * probs, 1)

    # build mechanism when r > row number
    if n < r:
        print("No row reduction needed. Original Matrix C kept.")
        return {
            "R": C,
            "selected_rows": list(range(n)),
            "probs": scaled_probs
        }

    # sample the rows and fill S and D
    t = 0
    sampled_rows = []
    S_rows = []
    D_diag = []

    for j in range(n):
        z = np.random.uniform(0, 1)
        if z <= scaled_probs[j]:
            sampled_rows.append(j)
            S_rows.append(j)
            D_diag.append(1 / np.sqrt(scaled_probs[j]))
            t += 1

    # create S and D
    t = len(sampled_rows)
    S = np.zeros((t, n))
    D = np.zeros((t, t))

    for idx, j in enumerate(sampled_rows):
        S[idx, j] = 1
        D[idx, idx] = D_diag[idx]

    # get C
    C = np.array(C)
    R = D @ S @ C

    return {
        "R": R,
        "selected_rows": sampled_rows,
        "probs": scaled_probs
    }