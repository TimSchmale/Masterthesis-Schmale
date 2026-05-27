import numpy as np

# ------------------------------------------------------------
# Function to generate a random score (uniformly distributed) per column
#
# INPUT:
#   X : design matrix of dimension n x p
#
# OUTPUT:
#   Numeric vector of length p containing random scores between 0 and 1
# ------------------------------------------------------------
def get_random_scores(X):
    n_cols = X.shape[1]
    return np.random.uniform(0.0, 1.0, size=n_cols)

# ------------------------------------------------------------
# Function to compute Column Leverage Scores (LS) for the case n < p
#
# INPUT:
#   X : design matrix of dimension n x p
#   k : desired rank of final approximation (k ≤ min(n,p))
#
# OUTPUT:
#   Numeric vector of length p containing the LS values
# ------------------------------------------------------------
def get_column_leverage_scores(X, k):
    n, p = X.shape
    if k > min(n, p):
        raise ValueError("k must be <= min(n,p)")

    U, S, Vh = np.linalg.svd(X, full_matrices=False)
    V = Vh.T[:, :k]

    return np.sum(V * V, axis=1)

# ------------------------------------------------------------
# Function to compute Row Leverage Scores (LS)
#
# INPUT:
#   X : design matrix of dimension n x p
#   k : desired rank of final approximation
#
# OUTPUT:
#   Numeric vector of length n containing the LS values
# ------------------------------------------------------------
def get_row_leverage_scores(X, k):
    n, p = X.shape
    if k > min(n, p):
        raise ValueError("k must be <= min(n,p)")

    U, S, Vh = np.linalg.svd(X, full_matrices=False)
    return np.sum(U[:, :k] ** 2, axis=1)

# ------------------------------------------------------------
# Function to compute L_p leverage-like scores via QR decomposition
#
# INPUT:
#   X : design matrix of dimension n x p
#   p : norm parameter (default p = 1)
#
# OUTPUT:
#   Numeric vector of length n containing ||Q[i,:]||_p^p
# ------------------------------------------------------------
def get_log_reg_leverage_scores(X, p=1):
    if X.ndim != 2:
        raise ValueError("X must be 2D!")

    Q, _ = np.linalg.qr(X, mode="reduced")
    return np.linalg.norm(Q, ord=p, axis=1) ** p

# ------------------------------------------------------------
# Function to compute Column Cross Leverage Scores (CLS)
#
# INPUT:
#   X : design matrix of dimension n x p
#   y : response vector of length n
#
# OUTPUT:
#   Numeric vector of length p containing the CLS values
# ------------------------------------------------------------
def get_cross_leverage_scores(X, y):
    y = y.reshape(-1, 1) if isinstance(y, np.ndarray) else y
    X_tilde = np.concatenate([X, y], axis=1)

    if X_tilde.shape[0] < X_tilde.shape[1]:
        X_tilde = X_tilde.T

    Q, R = np.linalg.qr(X_tilde, mode="reduced")
    return Q[:-1, :] @ Q[-1, :]

# ------------------------------------------------------------
# Function to compute combined LS + CLS scores
#
# INPUT:
#   X          : design matrix of dimension n x p
#   y          : response vector of length n
#   k          : rank parameter for LS
#   p_leverage : weight for leverage scores in [0,1]
#
# OUTPUT:
#   Numeric vector of length p containing combined scores
# ------------------------------------------------------------
def get_combined_scores(X, y, k, p_leverage):
    ls = get_column_leverage_scores(X, k)
    ls = ls / np.linalg.norm(ls)

    cls = np.abs(get_cross_leverage_scores(X, y))
    cls = cls / np.linalg.norm(cls)

    return (1 - p_leverage) * cls + p_leverage * ls