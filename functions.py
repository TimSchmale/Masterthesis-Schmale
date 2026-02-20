import random
import numpy as np
import pandas as pd

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
    # get probabilities
    probs = scores / np.sum(scores)

    # get the number of desired columns
    c = np.ceil(k * np.log(k))

    # scale probs and set maximum to 1
    scaled_probs = np.minimum(c * probs, 1)

    # S matrix
    S = np.zeros((X.shape[1], c))

    # D matrix
    D = np.zeros((c, c))

    # sample the columns and fill S and D
    sampled_cols = np.random.choice(X.shape[1], size = c, replace = False,p = scaled_probs)
    for i in range(c):
        S[sampled_cols[i+1], i+1] = 1
        D[i+1, i+1] = 1 / min(1, np.sqrt(scaled_probs[sampled_cols[i+1]]))

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
    print("calculation of leverage scores...")
    # get the leverage scores for the rows
    scores = get_cross_leverage_scores(C, k)
    print(np.shape(scores))
    # get probabilities
    probs = scores / np.sum(scores)

    # get the number of desired columns
    c = np.ceil(k * np.log(k))
    r = np.ceil(c * np.log(c))

    # scale probs and set maximum to 1
    scaled_probs = np.minimum(r * probs, 1)

    # S matrix
    S = np.zeros((r, C.shape[0]))

    # D matrix
    D = np.zeros((r, r))

    # sample the columns and fill S and D
    sampled_rows = np.random.choice(C.shape[0], size = c, replace = False,p = scaled_probs)
    for i in range(r):
        S[i+1,sampled_rows[i+1]] = 1
        D[i+1, i+1] = 1 / min(1, np.sqrt(scaled_probs[sampled_rows[i+1]]))

    # get C
    C = np.array(C)
    R = D @ S @ C

    return {
        "R": R,
        "selected_rows": sampled_rows,
        "probs": scaled_probs
    }