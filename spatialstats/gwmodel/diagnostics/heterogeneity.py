"""
spatialstats.gwmodel.diagnostics.heterogeneity
=====================================
Spatial heterogeneity tests for GW model coefficient surfaces.
"""

import numpy as np
from typing import Optional


def f3_test(coef: np.ndarray, residuals: np.ndarray,
            hat_diag: np.ndarray) -> dict:
    """
    F3 test for spatial non-stationarity of coefficient surfaces
    (Leung et al. 2000).

    Parameters
    ----------
    coef      : ndarray (n, p)  Local coefficients.
    residuals : ndarray (n,)    GWR residuals.
    hat_diag  : ndarray (n,)    Hat matrix diagonal.

    Returns
    -------
    dict with 'statistic', 'p_value' per covariate.
    """
    from scipy import stats
    n, p = coef.shape
    rss = np.sum(residuals ** 2)
    df_resid = max(n - np.sum(hat_diag), 1)

    f_stats = []
    p_vals = []
    for j in range(p):
        # Variance of local coefficients
        var_coef = np.var(coef[:, j])
        # Under stationarity: expect near-zero variance
        f = var_coef / max(rss / df_resid, 1e-12)
        f_stats.append(f)
        p_vals.append(1 - stats.f.cdf(f, df_resid, n - 1))

    return {"statistic": np.array(f_stats), "p_value": np.array(p_vals),
            "df_resid": df_resid, "n": n}


def monte_carlo_variability(coef: np.ndarray, n_sim: int = 499,
                             seed: int = 42) -> dict:
    """
    Monte Carlo test for spatial variability of coefficient surfaces.

    Null: coefficient surface is constant (no spatial variation).

    Returns
    -------
    dict with 'statistic' (IQR-based), 'p_value' arrays of shape (p,).
    """
    rng = np.random.default_rng(seed)
    n, p = coef.shape
    obs_stat = np.percentile(coef, 75, axis=0) - np.percentile(coef, 25, axis=0)
    count = np.zeros(p)
    for _ in range(n_sim):
        for j in range(p):
            perm_col = rng.permutation(coef[:, j])
            sim_stat = np.percentile(perm_col, 75) - np.percentile(perm_col, 25)
            if sim_stat >= obs_stat[j]:
                count[j] += 1
    p_values = (count + 1) / (n_sim + 1)
    return {"statistic": obs_stat, "p_value": p_values, "n_sim": n_sim}


def moran_test(residuals: np.ndarray, coords: np.ndarray,
               k: int = 8) -> dict:
    """
    Moran's I test for spatial autocorrelation in residuals.

    Parameters
    ----------
    residuals : ndarray (n,)
    coords    : ndarray (n, 2)
    k         : int  Number of nearest neighbours for weights.

    Returns
    -------
    dict with 'I', 'z_score', 'p_value'.
    """
    from scipy.spatial.distance import cdist
    from scipy import stats

    n = len(residuals)
    D = cdist(coords, coords)
    np.fill_diagonal(D, np.inf)

    # k-nearest-neighbour weights
    W = np.zeros((n, n))
    for i in range(n):
        nn_idx = np.argsort(D[i])[:k]
        W[i, nn_idx] = 1.0
    W = (W + W.T) / 2  # symmetrise
    row_sum = W.sum(axis=1)
    row_sum[row_sum == 0] = 1
    W_norm = W / row_sum[:, None]

    S0 = W.sum()
    z = residuals - residuals.mean()
    numerator = n * (z @ W_norm @ z)
    denominator = np.sum(z ** 2) * S0
    I = numerator / max(denominator, 1e-12)

    # Approximate normal distribution
    E_I = -1.0 / (n - 1)
    Var_I = (n ** 2 * S0 + n * S0) / (S0 ** 2 * (n ** 2 - 1)) - E_I ** 2
    z_score = (I - E_I) / max(np.sqrt(Var_I), 1e-12)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

    return {"I": float(I), "E_I": float(E_I), "z_score": float(z_score),
            "p_value": float(p_value)}
