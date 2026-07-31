"""Unified reduction primitives for CUR-based dimensionality reduction.

Consolidates all column and row reduction methods into a single module.
Key improvements:
- Single column_reduction() replacing 4 duplicated versions
- Vectorized sketching (matrix multiply instead of Python loop)
- Consistent minimum-column guarantee (always >= k)
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from scoring import get_row_leverage_scores, get_log_reg_leverage_scores

def column_reduction(X, scores, k):
    """Column reduction using EXPECTED(c) importance-based Bernoulli sampling.

    Columns are sampled independently with scaled probabilities derived
    from the score vector. Selected columns are rescaled according to
    the CUR theorem to preserve unbiasedness. At least k columns are
    guaranteed (filled from top-scoring if Bernoulli yields fewer).

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Design matrix (full or row-reduced).
    scores : ndarray of shape (p,)
        Column score vector (non-negative; absolute values taken internally).
    k : int
        Target rank. Controls expected sample size c = k * log(k)
        and minimum column guarantee.

    Returns
    -------
    dict with keys:
        "C" : ndarray of shape (n, t)
            Column-reduced and rescaled matrix (t >= k).
        "selected_columns" : list of int
            Indices of sampled columns.
    """
    X = np.asarray(X)
    scores = np.abs(np.asarray(scores, dtype=float).ravel())
    n, p = X.shape

    # Compute sampling probabilities
    score_sum = scores.sum()
    if score_sum == 0:
        probs = np.ones(p) / p
    else:
        probs = scores / score_sum

    # Expected number of sampled columns
    c = int(np.ceil(k * np.log(max(k, 2))))

    # Scaled probabilities (capped at 1)
    scaled_probs = np.minimum(c * probs, 1.0)

    # Bernoulli sampling
    z = np.random.rand(p)
    sampled = np.where(z <= scaled_probs)[0]

    # Guarantee at least k columns
    if len(sampled) < k:
        missing = k - len(sampled)
        candidates = np.argsort(scaled_probs)[::-1]
        candidates = np.array([idx for idx in candidates if idx not in sampled])
        sampled = np.concatenate([sampled, candidates[:missing]]).astype(int)

    # Rescaling (CUR expected algorithm, Drineas et al.): D_tt = 1/sqrt(c * p_j).
    # Uses the uncapped c*p_j as per the theoretical framework.
    # At test time, predictions use raw (unscaled) columns — this is intentional:
    # we treat CUR as a feature selection mechanism, not as a matrix approximation.
    raw_probs = c * probs[sampled]  # uncapped c * p_j
    raw_probs = np.maximum(raw_probs, 1e-10)  # numerical safety only
    D_inv = 1.0 / np.sqrt(raw_probs)

    C = X[:, sampled] * D_inv

    return {
        "C": C,
        "selected_columns": sampled.tolist()
    }


def row_reduction_sketch(X, y, k):
    """Row reduction via Rademacher sketching (VECTORIZED).

    Generates a Rademacher sketch matrix S of shape (r, n) with entries
    drawn uniformly from {-1/sqrt(r), +1/sqrt(r)} and computes:
        R = S @ X       (shape r x p)
        y_red = S @ y   (shape r,)

    This is mathematically equivalent to the original iterative approach
    but runs as a single BLAS matrix multiply — no Python loop over rows.

    The sketch size is r = 2k * log(p), preserving the subspace structure
    of X in expectation (Johnson-Lindenstrauss property).

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Design matrix.
    y : ndarray of shape (n,) or (n, 1)
        Response vector.
    k : int
        Target rank controlling sketch size.

    Returns
    -------
    R : ndarray of shape (r, p)
        Row-reduced (sketched) design matrix.
    y_reduced : ndarray of shape (r,)
        Sketched response vector.
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    n, p = X.shape

    # Sketch size: r = 2k * log(p)
    r = int(np.ceil(2 * k * np.log(max(p, 2))))

    # If r >= n, no reduction needed
    if r >= n:
        return X.copy(), y.copy()

    # Rademacher sketch matrix: entries ~ {-1/sqrt(r), +1/sqrt(r)}
    S = np.random.choice([-1, 1], size=(r, n)).astype(float) / np.sqrt(r)

    # Single matrix multiply — uses BLAS
    R = S @ X          # shape (r, p)
    y_reduced = S @ y  # shape (r,)

    return R, y_reduced


def row_reduction_leverage(C, y, k):
    """Row reduction using EXPECTED(r) leverage-based sampling.

    Rows are sampled independently using row leverage scores computed
    on the column-reduced matrix C. Selected rows are rescaled according
    to the CUR theorem. At least t rows are guaranteed (where t is the
    number of columns in C) to ensure a full-rank system for OLS.

    Used in the Column→Row pipeline after column_reduction().

    Parameters
    ----------
    C : ndarray of shape (n, t)
        Column-reduced matrix.
    y : ndarray of shape (n,)
        Response vector.
    k : int
        Target rank for leverage score computation.

    Returns
    -------
    dict with keys:
        "R" : ndarray of shape (s, t)
            Row-reduced and rescaled matrix (s >= t).
        "y" : ndarray of shape (s,)
            Reduced response vector.
        "selected_rows" : list of int
            Indices of sampled rows.
    """
    C = np.asarray(C)
    y = np.asarray(y).ravel()
    n, t = C.shape

    # Row leverage scores (no rank reduction — C is already column-reduced)
    scores = get_row_leverage_scores(C, k, rank_reduce=False)
    scores = np.asarray(scores, dtype=float).ravel()

    # Sampling probabilities
    probs = scores / scores.sum()

    # Expected number of sampled rows: r = t * log(t)
    r = int(np.ceil(t * np.log(max(t, 2))))

    # Scaled probabilities (capped at 1)
    scaled_probs = np.minimum(r * probs, 1.0)

    # Bernoulli sampling
    z = np.random.rand(n)
    sampled = np.where(z <= scaled_probs)[0]

    # Guarantee at least t rows (full-rank condition for OLS)
    if len(sampled) < t:
        missing = t - len(sampled)
        order = np.argsort(scaled_probs)[::-1]
        order = np.array([idx for idx in order if idx not in sampled])
        sampled = np.concatenate([sampled, order[:missing]]).astype(int)

    # Rescaling (CUR expected algorithm, Drineas et al.): D_tt = 1/sqrt(r * q_i).
    # Uses the uncapped r*q_i as per the theoretical framework.
    raw_probs = r * probs[sampled]  # uncapped r * q_i
    raw_probs = np.maximum(raw_probs, 1e-10)  # numerical safety only
    D_inv = 1.0 / np.sqrt(raw_probs)

    # Row reduction rescales both C and y with the same 1/sqrt(r * q_i) factors.
    # This ensures the sketched OLS objective S(Cβ - y) = SCβ - Sy is preserved,
    # yielding an unbiased estimator of the full-data solution.
    # At evaluation, raw (unscaled) test data is used with the resulting β̂.
    R = C[sampled, :] * D_inv[:, None]
    y_reduced = y[sampled] * D_inv

    return {
        "R": R,
        "y": y_reduced,
        "selected_rows": sampled.tolist()
    }


def estimate_mu(C, y):
    """Estimate the logistic coreset imbalance parameter mu.

    A logistic regression model is fitted on the column-reduced matrix C.
    The imbalance ratio mu is computed from the signed projection v = C @ beta.
    mu controls the row sampling distribution in logistic coreset theory.

    Parameters
    ----------
    C : ndarray of shape (n, t)
        Column-reduced matrix.
    y : ndarray of shape (n,)
        Binary response vector.

    Returns
    -------
    float
        Estimated mu (>= 1.0001).
    """
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=2000
    )
    model.fit(C, y)

    # Signed projection
    beta = model.coef_.flatten()
    v = C @ beta

    # Imbalance ratio
    pos = np.sum(np.abs(v[v > 0]))
    neg = np.sum(np.abs(v[v < 0]))

    mu = np.inf if neg == 0 else pos / neg
    return max(mu, 1.0001)


def row_reduction_coreset(C, y, mu, k):
    """Row reduction using logistic coreset sampling.

    Sampling distribution is a weighted mixture of:
    - L1 leverage scores (logistic-specific, from QR)
    - L2 row leverage scores (standard, from SVD)
    - Uniform component scaled by mu

    The sample size is r = mu * d * log(mu * d), derived from
    logistic coreset theory. Sampling is without replacement.

    Used in the Column->Row pipeline for logistic regression.

    Parameters
    ----------
    C : ndarray of shape (n, t)
        Column-reduced matrix.
    y : ndarray of shape (n,)
        Binary response vector.
    mu : float
        Logistic imbalance parameter (>= 1). Estimated via
        estimate_mu().
    k : int
        Target rank for leverage score computation.

    Returns
    -------
    dict with keys:
        "R" : ndarray of shape (r, t)
            Row-reduced matrix.
        "y" : ndarray of shape (r,)
            Reduced binary response.
        "selected_rows" : list of int
            Sampled row indices.
        "mu" : float
            The mu value used.
        "r" : int
            Number of sampled rows.
    """
    C = np.asarray(C)
    y_arr = np.asarray(y).ravel()
    n, d = C.shape

    # Expected sample size from coreset theory
    r = int(np.ceil(mu * d * np.log(max(mu * d, 2))))
    r = max(1, min(r, n))

    # Score components
    l1 = get_log_reg_leverage_scores(C, k, p=1, rank_reduce=False)
    l2 = get_row_leverage_scores(C, k, rank_reduce=False)
    uniform = np.ones(n) / n

    # Combined sampling distribution
    scores = mu * l1 + l2 + mu * d * uniform
    probs = scores / scores.sum()

    # Weighted sampling without replacement (uses global np.random state for reproducibility)
    sampled = np.random.choice(n, size=r, replace=False, p=probs)

    return {
        "R": C[sampled, :],
        "y": y_arr[sampled],
        "selected_rows": sampled.tolist(),
        "mu": mu,
        "r": r
    }
