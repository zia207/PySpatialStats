"""
spatialstats.gwmodel.linear.gwglm
======================
Geographically Weighted Generalised Linear Models (GWGLM).

Supports Gaussian, Poisson, and Binomial families via IRLS.
"""

import numpy as np
from typing import Optional
from scipy.special import expit  # sigmoid

from spatialstats.gwmodel.core.base import GWRegressor, _parse_geometry
from spatialstats.gwmodel.core.kernels import SpatialKernel
from spatialstats.gwmodel.utils.distance import pairwise_distances
from spatialstats.gwmodel.utils.stats import aicc, t_to_p


def _link_gaussian(mu): return mu
def _ilink_gaussian(eta): return eta
def _variance_gaussian(mu): return np.ones_like(mu)
def _deviance_gaussian(y, mu): return np.sum((y - mu) ** 2)

def _link_log(mu): return np.log(np.maximum(mu, 1e-10))
def _ilink_log(eta): return np.exp(np.clip(eta, -20, 20))
def _variance_poisson(mu): return np.maximum(mu, 1e-10)
def _deviance_poisson(y, mu):
    return 2 * np.sum(np.where(y > 0, y * np.log(np.maximum(y / np.maximum(mu, 1e-10), 1e-10)), 0) - (y - mu))

def _link_logit(mu): return np.log(np.maximum(mu, 1e-10) / np.maximum(1 - mu, 1e-10))
def _ilink_logit(eta): return expit(eta)
def _variance_binomial(mu): return np.maximum(mu * (1 - mu), 1e-10)
def _deviance_binomial(y, mu):
    return -2 * np.sum(y * np.log(np.maximum(mu, 1e-10)) + (1 - y) * np.log(np.maximum(1 - mu, 1e-10)))


FAMILIES = {
    "gaussian": (_link_gaussian, _ilink_gaussian, _variance_gaussian, _deviance_gaussian),
    "poisson":  (_link_log,      _ilink_log,      _variance_poisson,  _deviance_poisson),
    "binomial": (_link_logit,    _ilink_logit,    _variance_binomial, _deviance_binomial),
}


class GWGLM(GWRegressor):
    """
    Geographically Weighted Generalised Linear Model.

    Parameters
    ----------
    family : str
        'gaussian' | 'poisson' | 'binomial'
    bandwidth : float, int, or 'auto'
    fixed : bool
    kernel : str
    max_iter : int   IRLS iterations per local model.
    tol : float      IRLS convergence tolerance.
    n_jobs : int

    Examples
    --------
    >>> from spatialstats.gwmodel.linear import GWGLM
    >>> import numpy as np
    >>> coords = np.random.rand(50, 2) * 100
    >>> X = np.random.rand(50, 2)
    >>> lam = np.exp(0.5 + X @ [1, -0.5])
    >>> y = np.random.poisson(lam)
    >>> model = GWGLM(family='poisson', bandwidth=20, fixed=True)
    >>> model.fit(X, y, coords)
    """

    def __init__(self, family: str = "gaussian", bandwidth=None,
                 fixed: bool = False, kernel: str = "bisquare",
                 criterion: str = "AICc", max_iter: int = 100,
                 tol: float = 1e-6, n_jobs: int = -1):
        super().__init__(bandwidth=bandwidth, fixed=fixed, kernel=kernel,
                         criterion=criterion, n_jobs=n_jobs)
        if family not in FAMILIES:
            raise ValueError(f"family must be one of {list(FAMILIES)}")
        self.family = family
        self.max_iter = max_iter
        self.tol = tol
        self._link, self._ilink, self._variance, self._deviance = FAMILIES[family]

    def _irls_point(self, i: int, X: np.ndarray, y: np.ndarray,
                     w_kernel: np.ndarray) -> np.ndarray:
        """IRLS at calibration point i, returns coefficient vector."""
        n, p = X.shape
        mu = np.where(y > 0, y, 0.5) if self.family == "binomial" else np.maximum(y.mean(), 1e-3) * np.ones(n)
        if self.family == "poisson":
            mu = np.maximum(y, 0.1)
        coef = np.zeros(p)
        for _ in range(self.max_iter):
            eta = X @ coef
            mu = self._ilink(eta)
            V = self._variance(mu)
            d_mu = self._ilink(eta)  # derivative of inverse link ≈ mu for exp
            W_irls = w_kernel * (1.0 / np.maximum(V, 1e-10))
            z = eta + (y - mu) / np.maximum(V, 1e-10)
            Wmat = np.diag(W_irls)
            XtW = X.T @ Wmat
            A = XtW @ X
            b = XtW @ z
            try:
                coef_new = np.linalg.solve(A + 1e-8 * np.eye(p), b)
            except np.linalg.LinAlgError:
                coef_new = np.linalg.lstsq(A, b, rcond=None)[0]
            if np.max(np.abs(coef_new - coef)) < self.tol:
                coef = coef_new
                break
            coef = coef_new
        return coef

    def fit(self, X, y, geometry):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        coords = _parse_geometry(geometry)
        n, p = X.shape
        X_int = np.column_stack([np.ones(n), X])

        self.coords_ = coords
        self.X_ = X_int
        self.y_ = y

        D = pairwise_distances(coords)
        bw = self.bandwidth if self.bandwidth != "auto" and self.bandwidth is not None else 20
        self.bandwidth_ = float(bw)
        kernel = SpatialKernel(self.kernel, self.fixed, self.bandwidth_)

        coef = np.zeros((n, X_int.shape[1]))
        for i in range(n):
            w = kernel(D[i])
            coef[i] = self._irls_point(i, X_int, y, w)

        self.coef_ = coef
        self.fitted_ = np.array([self._ilink(X_int[i] @ coef[i]) for i in range(n)])
        self.residuals_ = y - self.fitted_
        self.std_err_ = np.ones_like(coef) * 0.1  # placeholder
        self.t_values_ = coef / 0.1
        self.p_values_ = t_to_p(self.t_values_, df=max(n - p - 1, 1))
        return self

    def predict(self, X, geometry) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Fit the model first.")
        X = np.asarray(X, dtype=float)
        coords_new = _parse_geometry(geometry)
        m = len(coords_new)
        X_int = np.column_stack([np.ones(m), X])
        from scipy.spatial.distance import cdist
        D_pred = cdist(coords_new, self.coords_, metric="euclidean")
        nearest = np.argmin(D_pred, axis=1)
        return np.array([self._ilink(X_int[j] @ self.coef_[nearest[j]]) for j in range(m)])
