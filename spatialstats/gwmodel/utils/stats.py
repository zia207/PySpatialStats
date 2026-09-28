"""
spatialstats.gwmodel.utils.stats
=====================
Statistical helper functions.
"""

import numpy as np
from scipy import stats as scipy_stats


def aicc(rss: float, n: int, k: float) -> float:
    """
    Corrected Akaike Information Criterion.

    Parameters
    ----------
    rss : float  Residual sum of squares.
    n   : int    Number of observations.
    k   : float  Effective degrees of freedom (trace of hat matrix).
    """
    sigma2 = rss / n
    if sigma2 <= 0:
        sigma2 = 1e-10
    aic = n * np.log(sigma2) + n + k
    correction = (2 * k * (k + 1)) / max(n - k - 1, 1e-6)
    return aic + correction


def aic(rss: float, n: int, k: float) -> float:
    """Standard AIC."""
    sigma2 = rss / n
    if sigma2 <= 0:
        sigma2 = 1e-10
    return n * np.log(sigma2) + 2 * k


def bic(rss: float, n: int, k: float) -> float:
    """Bayesian Information Criterion."""
    sigma2 = rss / n
    if sigma2 <= 0:
        sigma2 = 1e-10
    return n * np.log(sigma2) + k * np.log(n)


def hat_matrix_gwr(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Compute n×n hat matrix row for a single GWR calibration point.

    For the full hat matrix, call for each i and assemble.
    Returns the i-th row of the influence matrix.

    Parameters
    ----------
    X : ndarray (n, p)  Design matrix.
    W : ndarray (n,)    Kernel weights for focal point i.
    """
    Wmat = np.diag(W)
    XtW = X.T @ Wmat
    try:
        XtWX_inv = np.linalg.inv(XtW @ X)
    except np.linalg.LinAlgError:
        XtWX_inv = np.linalg.pinv(XtW @ X)
    return X @ XtWX_inv @ XtW  # (n, n) row


def effective_df(hat_matrix_trace: float) -> float:
    """Effective degrees of freedom from hat matrix trace."""
    return hat_matrix_trace


def local_r2(y: np.ndarray, yhat: np.ndarray, weights: np.ndarray) -> float:
    """Weighted local R² at a calibration point."""
    ybar = np.average(y, weights=weights)
    ss_tot = np.sum(weights * (y - ybar) ** 2)
    ss_res = np.sum(weights * (y - yhat) ** 2)
    return 1.0 - ss_res / max(ss_tot, 1e-10)


def bh_correction(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Benjamini-Hochberg false discovery rate correction.

    Returns boolean array: True = significant after correction.
    """
    n = len(p_values)
    order = np.argsort(p_values)
    p_sorted = p_values[order]
    threshold = (np.arange(1, n + 1) / n) * alpha
    sig_sorted = p_sorted <= threshold
    # All ranks up to the largest significant rank are significant
    if sig_sorted.any():
        last_sig = np.where(sig_sorted)[0][-1]
        sig_sorted[:last_sig + 1] = True
    result = np.zeros(n, dtype=bool)
    result[order] = sig_sorted
    return result


def t_to_p(t_values: np.ndarray, df: float) -> np.ndarray:
    """Convert t-statistics to two-tailed p-values."""
    return 2 * scipy_stats.t.sf(np.abs(t_values), df=df)
