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
def get_column_leverage_scores(X, k):
    # check k
    if k > min(X.shape):
        print(k, X.shape[0])
        raise ValueError("k must be <= min(n,p)")

    # perform singular value decomposition to get V matrix
    U, S, Vh = np.linalg.svd(X, full_matrices=False)

    return np.sum(Vh.T[:,0:k] ** 2, axis = 1)

def get_row_leverage_scores(X, k):

    n, p = X.shape
    if k > min(n,p):
        raise ValueError("k must be <= min(n,p)")

    U, S, Vh = np.linalg.svd(X, full_matrices=False)

    return np.sum(U[:, 0:k]**2, axis=1)

def get_log_reg_leverage_scores(X: np.ndarray, p=1):
    """
        Computes leverage scores.
    """
    if not len(X.shape) == 2:
        raise ValueError("X must be 2D!")

    Q, *_ = np.linalg.qr(X)

    leverage_scores = np.power(np.linalg.norm(Q, axis=1, ord=p), p)

    return leverage_scores

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

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    if isinstance(y, np.ndarray):
        y = pd.DataFrame(y)

    XY = pd.concat([X, y], axis=1)

    if XY.shape[0] < XY.shape[1]:
        C = XY.T
    else:
        C = XY

    Q, R = np.linalg.qr(C, mode="reduced")

    cls = Q[:-1, :] @ Q[-1, :]

    return cls

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
    ls = get_column_leverage_scores(X, k)
    ls = ls / np.sqrt(np.sum(ls ** 2))

    # calculate cross leverage scores and normalize to 1
    cls = np.abs(get_cross_leverage_scores(X, y))
    cls = cls / np.sqrt(np.sum(cls ** 2))

    return (1-p_leverage) * cls + p_leverage * ls