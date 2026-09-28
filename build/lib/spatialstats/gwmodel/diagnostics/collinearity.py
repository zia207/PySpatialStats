"""
spatialstats.gwmodel.diagnostics.collinearity
====================================
Local collinearity diagnostics for GW models.
"""

import numpy as np
from typing import Optional


def local_condition_number(coef: np.ndarray, X: np.ndarray,
                            D: np.ndarray, kernel) -> np.ndarray:
    """
    Compute local condition number of the weighted design matrix X'W(i)X
    at each calibration point.

    Parameters
    ----------
    coef   : ndarray (n, p)
    X      : ndarray (n, p)  Design matrix (with intercept).
    D      : ndarray (n, n)  Distance matrix.
    kernel : SpatialKernel

    Returns
    -------
    cn : ndarray (n,)  Local condition numbers.
    """
    n = len(X)
    cn = np.zeros(n)
    for i in range(n):
        w = kernel(D[i])
        W = np.diag(w)
        A = X.T @ W @ X
        try:
            cn[i] = np.linalg.cond(A)
        except np.linalg.LinAlgError:
            cn[i] = np.inf
    return cn


def local_vif(X: np.ndarray, D: np.ndarray, kernel) -> np.ndarray:
    """
    Compute local variance inflation factors (VIF) at each calibration point.

    Returns
    -------
    vif : ndarray (n, p)  Local VIF for each covariate.
    """
    n, p = X.shape
    vif = np.zeros((n, p))
    for i in range(n):
        w = kernel(D[i])
        for j in range(p):
            X_j = np.delete(X, j, axis=1)
            W = np.diag(w)
            XtW_j = X_j.T @ W
            A_j = XtW_j @ X_j + 1e-10 * np.eye(p - 1)
            b_j = XtW_j @ X[:, j]
            try:
                coef_j = np.linalg.solve(A_j, b_j)
            except np.linalg.LinAlgError:
                coef_j = np.linalg.lstsq(A_j, b_j, rcond=None)[0]
            fitted_j = X_j @ coef_j
            ss_res = np.sum(w * (X[:, j] - fitted_j) ** 2)
            ss_tot = np.sum(w * (X[:, j] - np.average(X[:, j], weights=w)) ** 2)
            r2_j = 1.0 - ss_res / max(ss_tot, 1e-12)
            vif[i, j] = 1.0 / max(1.0 - r2_j, 1e-10)
    return vif
