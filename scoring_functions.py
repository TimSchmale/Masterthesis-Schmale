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
        #print("This function is designed for the case n < p. The matrix is now getting transposed.")
        X = X.T

    # check k
    if k > X.shape[0]:
        print(k, X.shape[0])
        raise ValueError("k must be <= nrow(X)")

    # perform singular value decomposition to get V matrix
    U, S, Vh = np.linalg.svd(X)

    return np.sum(Vh.T[:,0:k-1] ** 2, axis = 1)

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