import numpy as np

def get_random_scores(X):
    """
    Generate uniformly random column scores.

    A random score in [0,1] is assigned independently to each column
    of the design matrix. This baseline scoring method is used as a
    non-informative reference in reduction pipelines.

    Parameters
    ----------
    X : array-like
        Design matrix of shape (n, p).

    Returns
    -------
    ndarray
        Vector of length p containing random scores.
    """

    n_cols = X.shape[1]
    return np.random.uniform(0.0, 1.0, size=n_cols)


def get_column_leverage_scores(X, k, rank_reduce = True):
    """
    Compute column leverage scores using the top-k right singular vectors.

    The design matrix X is decomposed via thin SVD. The leverage score
    of column j is the squared Euclidean norm of the j-th row of V_k,
    where V_k contains the top-k right singular vectors. These scores
    quantify the influence of each column on the row space of X.

    Parameters
    ----------
    X : array-like
        Design matrix of shape (n, p).
    k : int
        Target rank, must satisfy k ≤ min(n, p).

    Returns
    -------
    ndarray
        Vector of length p containing column leverage scores.
    """

    n, p = X.shape
    if k > min(n, p):
        raise ValueError("k must be <= min(n,p)")

    U, S, Vh = np.linalg.svd(X, full_matrices=False)

    # only if rank reduction is needed or not
    if rank_reduce:
        V = Vh.T[:, :k]
    else:
        V = Vh.T

    return np.sum(V * V, axis=1)

def get_row_leverage_scores(X, k, rank_reduce = True):
    """
    Compute row leverage scores using the top-k left singular vectors.

    The leverage score of row i is the squared Euclidean norm of the
    i-th row of U_k, where U_k contains the top-k left singular vectors.
    These scores quantify the influence of each row on the column space
    of X.

    Parameters
    ----------
    X : array-like
        Design matrix of shape (n, p).
    k : int
        Target rank, must satisfy k ≤ min(n, p).

    Returns
    -------
    ndarray
        Vector of length n containing row leverage scores.
    """

    n, p = X.shape
    if k > min(n, p):
        raise ValueError("k must be <= min(n,p)")

    U, S, Vh = np.linalg.svd(X, full_matrices=False)

    # only if rank reduction is needed
    if rank_reduce:
        U = U[:, :k]

    return np.sum(U ** 2, axis=1)


def get_log_reg_leverage_scores(X, k, p = 1, rank_reduce = True):
    """
    Compute L_p leverage-like scores using the QR decomposition.

    The reduced QR factorization X = QR is computed. The leverage-like
    score of row i is defined as ||Q[i,:]||_p^p. For p = 1, this yields
    the ℓ₁-based leverage scores used in logistic coreset constructions.

    Parameters
    ----------
    X : array-like
        Design matrix of shape (n, p).
    p : int or float
        Norm parameter for the row-wise ℓ_p norm.

    Returns
    -------
    ndarray
        Vector of length n containing leverage-like scores.
    """

    if X.ndim != 2:
        raise ValueError("X must be 2D!")

    Q, _ = np.linalg.qr(X, mode="reduced")

    if rank_reduce:
        Q = Q[:, :k]
    return np.linalg.norm(Q, ord=p, axis=1) ** p


def get_cross_leverage_scores(X, y, k, rank_reduce = True):
    """
    Compute column cross-leverage scores between X and the response y.

    The augmented matrix [X | y] is formed and QR-decomposed. The
    cross-leverage score for column j is the inner product between
    the j-th row of Q and the final row of Q. These scores quantify
    the alignment between each column of X and the response vector.

    Parameters
    ----------
    X : array-like
        Design matrix of shape (n, p).
    y : array-like
        Response vector of length n.

    Returns
    -------
    ndarray
        Vector of length p containing cross-leverage scores.
    """

    y = y.reshape(-1, 1) if isinstance(y, np.ndarray) else y
    X_tilde = np.concatenate([X, y], axis=1)

    if X_tilde.shape[0] < X_tilde.shape[1]:
        X_tilde = X_tilde.T

    Q, R = np.linalg.qr(X_tilde, mode="reduced")

    if rank_reduce:
        Q = Q[:, :k]

    return Q[:-1, :] @ Q[-1, :]


import numpy as np


def get_cross_leverage_scores_svd(X, y, k=None, rank_reduce=True):
    """
    SVD-basierte Cross-Leverage Scores nach Teschke.

    Die augmentierte Matrix [X | y] wird gebildet und per SVD zerlegt:
        X_tilde = U S V^T.

    Der Cross-Leverage Score für Spalte j ist das innere Produkt zwischen
    der j-ten Zeile von U und der letzten Zeile von U (die zu y gehört).

    Parameters
    ----------
    X : array-like, shape (n, p)
        Design matrix.
    y : array-like, shape (n,)
        Response vector.
    k : int, optional
        Rank for reduction (Top-k left singular vectors).
    rank_reduce : bool
        If True, use only the first k columns of U.

    Returns
    -------
    ndarray
        Vector of length p containing cross-leverage scores.
    """

    # y als Spalte
    y = y.reshape(-1, 1)

    # augmentierte Matrix: [X | y]
    X_tilde = np.concatenate([X, y], axis=1)

    # falls n < p+1 → transponieren (wie in deiner QR-Version)
    if X_tilde.shape[0] < X_tilde.shape[1]:
        X_tilde = X_tilde.T

    # SVD der erweiterten Matrix
    U, S, Vt = np.linalg.svd(X_tilde, full_matrices=False)

    # rank reduction
    if rank_reduce:
        if k is None:
            raise ValueError("k must be provided when rank_reduce=True")
        U = U[:, :k]

    # letzte Zeile ist die y-Zeile
    u_tilde = U[-1, :]

    # Cross-Leverage Scores: inneres Produkt jeder Zeile mit der y-Zeile
    # aber nur die ersten p Zeilen (die zu X gehören)
    return U[:-1, :] @ u_tilde


def get_combined_scores(X, y, k, p_leverage, ls = None, cls = None):
    """
    Compute combined column scores using LS and CLS components.

    Column leverage scores (LS) and cross-leverage scores (CLS) are
    normalized and combined via a convex mixture. The parameter
    p_leverage ∈ [0,1] controls the relative weight of LS versus CLS.

    Parameters
    ----------
    X : array-like
        Design matrix of shape (n, p).
    y : array-like
        Response vector of length n.
    k : int
        Rank parameter for LS computation.
    p_leverage : float
        Weight for leverage scores in the convex combination.
    ls : ndarray of shape (p,), optional
        Precomputed (unnormalized) column leverage scores. If None,
        the scores are computed internally.

    cls : ndarray of shape (p,), optional
        Precomputed (unnormalized) cross-leverage scores. If None,
        the scores are computed internally.

    Returns
    -------
    ndarray
        Vector of length p containing combined scores.
    """

    if ls is None:
        ls = get_column_leverage_scores(X, k)
    ls = ls / np.linalg.norm(ls)

    if cls is None:
        cls = np.abs(get_cross_leverage_scores(X, y))

    cls = cls / np.linalg.norm(cls)

    return (1 - p_leverage) * cls + p_leverage * ls