import numpy as np
from sklearn.utils.extmath import randomized_svd

# =============================================================================
# Column Scores
# =============================================================================

def get_column_leverage_scores(X, k, rank_reduce=True):
    """Compute column leverage scores using right singular vectors.

    The leverage score of column j is ||V_k[j, :]||^2, where V_k
    contains the top-k right singular vectors of X.

    When rank_reduce=True, only the top-k singular vectors are computed
    via randomized SVD (fast). When False, all singular vectors are used
    via full SVD (exact, but slower).

    Parameters
    ----------
    X : array-like of shape (n, p)
        Design matrix.
    k : int
        Target rank, must satisfy k <= min(n, p).
    rank_reduce : bool
        If True, use only top-k singular vectors (truncated SVD).
        If False, use all singular vectors (full SVD).

    Returns
    -------
    ndarray of shape (p,)
        Column leverage scores.
    """
    n, p = X.shape
    if k > min(n, p):
        raise ValueError(f"k={k} must be <= min(n,p)={min(n, p)}")

    if rank_reduce:
        # Truncated SVD: O(n * p * k) — only computes top-k components
        _, _, Vt = randomized_svd(X, n_components=k, random_state=None)
        V = Vt.T  # shape (p, k)
    else:
        # Full SVD: O(n * p * min(n,p)) — all components
        _, _, Vh = np.linalg.svd(X, full_matrices=False)
        V = Vh.T  # shape (p, min(n,p))

    return np.sum(V ** 2, axis=1)


def get_cross_leverage_scores(X, y, k, rank_reduce=True):
    """Compute column cross-leverage scores between X and response y.

    The augmented matrix [X | y] is formed and decomposed via SVD.
    The cross-leverage score for column j is the inner product between
    the j-th row of U (or V, depending on orientation) and the last
    row (corresponding to y). This quantifies the alignment between
    each column of X and the response vector.

    Uses truncated SVD when rank_reduce=True for efficiency.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Design matrix.
    y : array-like of shape (n,) or (n, 1)
        Response vector.
    k : int
        Target rank for truncated decomposition.
    rank_reduce : bool
        If True, use only top-k components (truncated SVD).
        If False, use all components (full SVD).

    Returns
    -------
    ndarray of shape (p,)
        Cross-leverage scores.
    """
    y = np.asarray(y).reshape(-1, 1)
    X_tilde = np.concatenate([X, y], axis=1)

    # Ensure tall matrix for SVD (transpose if n < p+1)
    if X_tilde.shape[0] < X_tilde.shape[1]:
        X_tilde = X_tilde.T

    if rank_reduce:
        U, _, _ = randomized_svd(X_tilde, n_components=k, random_state=None)
    else:
        U, _, _ = np.linalg.svd(X_tilde, full_matrices=False)

    # Cross-leverage: inner product of each row with the last row (y-row)
    u_y = U[-1, :]
    return U[:-1, :] @ u_y


def get_random_scores(X):
    """Generate uniformly random column scores (non-informative baseline).

    A random score in [0,1] is assigned independently to each column.
    Used as a reference to evaluate whether informative scoring methods
    outperform random selection.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Design matrix (only shape is used).

    Returns
    -------
    ndarray of shape (p,)
        Uniform random scores in [0, 1].
    """
    return np.random.uniform(0.0, 1.0, size=X.shape[1])


def get_combined_scores(X, y, k, p_leverage=0.2, ls=None, cls=None):
    """Compute combined column scores as convex mixture of LS and |CLS|.

    CS = (1 - p_leverage) * norm(|CLS|) + p_leverage * norm(LS)

    Both components are L2-normalized before mixing so that p_leverage
    controls the relative weight independent of scale.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Design matrix.
    y : array-like of shape (n,) or (n, 1)
        Response vector.
    k : int
        Target rank.
    p_leverage : float in [0, 1]
        Weight for leverage scores. 0 = pure CLS, 1 = pure LS.
    ls : ndarray of shape (p,), optional
        Precomputed column leverage scores. Computed internally if None.
    cls : ndarray of shape (p,), optional
        Precomputed |cross-leverage scores|. Computed internally if None.

    Returns
    -------
    ndarray of shape (p,)
        Combined scores.
    """
    if ls is None:
        ls = get_column_leverage_scores(X, k)
    ls_norm = ls / np.linalg.norm(ls)

    if cls is None:
        cls = np.abs(get_cross_leverage_scores(X, y, k))
    cls_norm = cls / np.linalg.norm(cls)

    return (1 - p_leverage) * cls_norm + p_leverage * ls_norm


# =============================================================================
# Compute All Column Scores
# =============================================================================

def compute_all_scores(X, y, k, rank_reduce=True):
    """Compute all four column score vectors for a single replication.

    Returns dictionaries of scores and computation times for:
    - LS  : Column leverage scores
    - CLS : Cross-leverage scores (absolute values)
    - RS  : Random scores (baseline)
    - CS  : Combined LS/CLS scores (p_leverage=0.2)

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Design matrix.
    y : ndarray of shape (n,)
        Response vector.
    k : int
        Target rank.
    rank_reduce : bool
        If True, use truncated SVD (top-k). If False, use full SVD.
        Set to False when X is already a sketch (row-first pipeline).

    Returns
    -------
    scores : dict {"LS", "CLS", "RS", "CS"} -> ndarray of shape (p,)
    timings : dict {"LS", "CLS", "RS", "CS"} -> float (seconds)
    """
    import time

    scores = {}
    timings = {}

    # LS: Column leverage scores
    t0 = time.perf_counter()
    scores["LS"] = get_column_leverage_scores(X, k, rank_reduce=rank_reduce)
    timings["LS"] = time.perf_counter() - t0

    # CLS: Cross-leverage scores (absolute values)
    t0 = time.perf_counter()
    scores["CLS"] = get_cross_leverage_scores(X, y, k, rank_reduce=rank_reduce)
    timings["CLS"] = time.perf_counter() - t0

    # RS: Random scores
    t0 = time.perf_counter()
    scores["RS"] = get_random_scores(X)
    timings["RS"] = time.perf_counter() - t0

    # CS: Combined scores (reuses already-computed LS and |CLS| arrays).
    # Timing attributed as full LS+CLS cost, since both SVDs are required
    # to produce CS — reflects end-to-end cost of choosing CS as a method.
    scores["CS"] = get_combined_scores(
        X, y, k, p_leverage=0.2,
        ls=scores["LS"], cls=np.abs(scores["CLS"])
    )
    timings["CS"] = timings["LS"] + timings["CLS"]

    return scores, timings


# =============================================================================
# Row Scores
# =============================================================================

def get_row_leverage_scores(X, k, rank_reduce=True):
    """Compute row leverage scores using left singular vectors.

    The leverage score of row i is ||U_k[i, :]||^2, where U_k
    contains the top-k left singular vectors of X. These scores
    quantify the influence of each row on the column space of X.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Design matrix.
    k : int
        Target rank, must satisfy k <= min(n, p).
    rank_reduce : bool
        If True, use only top-k left singular vectors (truncated SVD).
        If False, use all left singular vectors (full SVD).

    Returns
    -------
    ndarray of shape (n,)
        Row leverage scores.
    """
    n, p = X.shape
    if k > min(n, p):
        raise ValueError(f"k={k} must be <= min(n,p)={min(n, p)}")

    if rank_reduce:
        U, _, _ = randomized_svd(X, n_components=k, random_state=None)
    else:
        U, _, _ = np.linalg.svd(X, full_matrices=False)

    return np.sum(U ** 2, axis=1)


def get_log_reg_leverage_scores(X, k, p=1, rank_reduce=True):
    """Compute L_p leverage-like scores via QR decomposition.

    The reduced QR factorization X = QR is computed. The score of
    row i is ||Q[i, :]||_p^p. For p=1 this yields the l1-based
    leverage scores used in logistic coreset constructions.

    Parameters
    ----------
    X : array-like of shape (n, d)
        Design matrix.
    k : int
        Target rank for optional rank reduction of Q.
    p : int or float
        Norm parameter for the row-wise l_p norm.
    rank_reduce : bool
        If True, use only first k columns of Q.

    Returns
    -------
    ndarray of shape (n,)
        Leverage-like scores.
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D")

    Q, _ = np.linalg.qr(X, mode="reduced")

    if rank_reduce:
        Q = Q[:, :k]

    return np.linalg.norm(Q, ord=p, axis=1) ** p
