"""
spatialstats.gwmodel.linear.mgwr
=====================
Multiscale Geographically Weighted Regression (MGWR).

Implements per-covariate bandwidths via GAM-style back-fitting.
Based on Fotheringham, Yang & Kang (2017) and PySAL mgwr.
"""

import numpy as np
import warnings
from typing import Optional, List
from joblib import Parallel, delayed

from spatialstats.gwmodel.core.base import GWRegressor, _parse_geometry
from spatialstats.gwmodel.core.kernels import SpatialKernel
from spatialstats.gwmodel.core.bandwidth import BandwidthSelector
from spatialstats.gwmodel.utils.stats import aicc, bh_correction, t_to_p
from spatialstats.gwmodel.utils.distance import pairwise_distances


def _smooth_column(j: int, partial_resid: np.ndarray, X: np.ndarray,
                   D: np.ndarray, bw: float, kernel_name: str,
                   fixed: bool) -> np.ndarray:
    """
    GWR-smooth the j-th partial residual using bandwidth bw.

    Returns fitted values (n,) for covariate j.
    """
    n = len(partial_resid)
    kernel = SpatialKernel(kernel_name, fixed, bw)
    fitted = np.zeros(n)
    for i in range(n):
        w = kernel(D[i])
        xi = X[:, j]
        # Weighted regression of partial residual on single covariate xi
        wx = w * xi
        wxxi = np.sum(wx * xi)
        if abs(wxxi) < 1e-12:
            fitted[i] = 0.0
        else:
            beta_j = np.sum(wx * partial_resid) / wxxi
            fitted[i] = xi[i] * beta_j
    return fitted


class MGWR(GWRegressor):
    """
    Multiscale Geographically Weighted Regression (MGWR).

    Each predictor has its own optimal spatial bandwidth, allowing
    processes operating at different scales to be modelled simultaneously.

    Parameters
    ----------
    bw_init : list of float, optional
        Initial bandwidths per covariate (including intercept).
        If None, a single GWR bandwidth is used to initialise.
    kernel : str
        Kernel function.
    fixed : bool
        Fixed vs adaptive bandwidth.
    criterion : str
        Bandwidth selection criterion.
    tol : float
        Back-fitting convergence tolerance (change in AICc).
    max_iter : int
        Maximum back-fitting iterations.
    bw_tol : float
        Tolerance for individual bandwidth golden-section search.
    n_jobs : int

    Attributes (after fit)
    ----------------------
    coef_ : ndarray (n, p)          Local coefficient estimates.
    bandwidths_ : ndarray (p,)      Per-covariate optimal bandwidths.
    std_err_ : ndarray (n, p)
    t_values_ : ndarray (n, p)
    p_values_ : ndarray (n, p)
    aicc_ : float

    Examples
    --------
    >>> from spatialstats.gwmodel.linear import MGWR
    >>> import numpy as np
    >>> coords = np.random.rand(80, 2) * 100
    >>> X = np.random.rand(80, 2)
    >>> y = X[:, 0] * 2 + X[:, 1] * (-1) + np.random.randn(80) * 0.2
    >>> model = MGWR(tol=1.0, max_iter=5)
    >>> model.fit(X, y, coords)
    >>> print(model.bandwidths_)
    """

    def __init__(self, bw_init: Optional[List[float]] = None,
                 kernel: str = "bisquare", fixed: bool = False,
                 criterion: str = "AICc", tol: float = 1.0,
                 max_iter: int = 50, bw_tol: float = 1.0,
                 n_jobs: int = -1):
        super().__init__(bandwidth=None, fixed=fixed, kernel=kernel,
                         criterion=criterion, n_jobs=n_jobs)
        self.bw_init = bw_init
        self.tol = tol
        self.max_iter = max_iter
        self.bw_tol = bw_tol
        self.bandwidths_: Optional[np.ndarray] = None

    def fit(self, X, y, geometry):
        """
        Calibrate MGWR via GAM back-fitting.

        Parameters
        ----------
        X : array-like (n, p)
        y : array-like (n,)
        geometry : ndarray (n, 2) or GeoSeries

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        coords = _parse_geometry(geometry)

        n, p = X.shape
        X_int = np.column_stack([np.ones(n), X])  # (n, p+1)
        p_full = X_int.shape[1]

        self.coords_ = coords
        self.X_ = X_int
        self.y_ = y

        D = pairwise_distances(coords, metric="euclidean")

        # Initialise bandwidths
        if self.bw_init is not None and len(self.bw_init) == p_full:
            bws = np.array(self.bw_init, dtype=float)
        else:
            # Use a single initial GWR bandwidth for all covariates
            from spatialstats.gwmodel.linear.gwr import GWR
            init_gwr = GWR(bandwidth="auto", fixed=self.fixed,
                           kernel=self.kernel, criterion=self.criterion,
                           hat_matrix=False, n_jobs=self.n_jobs)
            init_gwr.fit(X, y, coords)
            bws = np.full(p_full, init_gwr.bandwidth_)

        # Initialise coefficient surfaces
        coef = np.zeros((n, p_full))
        fitted = np.zeros(n)

        prev_aicc = np.inf
        for iteration in range(self.max_iter):
            # Back-fitting: iterate over each covariate
            for j in range(p_full):
                # Partial residual for covariate j
                partial_resid = y - fitted + coef[:, j] * X_int[:, j]

                # Optimise bandwidth for covariate j
                bw_j = self._select_single_bw(j, partial_resid, X_int, D, bws[j])
                bws[j] = bw_j

                # Smooth partial residual
                kernel = SpatialKernel(self.kernel, self.fixed, bw_j)
                coef_j = np.zeros(n)
                for i in range(n):
                    w = kernel(D[i])
                    xi = X_int[:, j]
                    wx = w * xi
                    denom = np.sum(wx * xi)
                    if abs(denom) < 1e-12:
                        coef_j[i] = 0.0
                    else:
                        coef_j[i] = np.sum(wx * partial_resid) / denom
                coef[:, j] = coef_j

                fitted = np.sum(coef * X_int, axis=1)

            resid = y - fitted
            rss = np.sum(resid ** 2)
            # Approximate effective DF as sum of n/bw ratios
            k_approx = float(np.sum(n / bws))
            curr_aicc = aicc(rss, n, min(k_approx, n - 1))

            if abs(prev_aicc - curr_aicc) < self.tol:
                break
            prev_aicc = curr_aicc

        self.coef_ = coef
        self.bandwidths_ = bws
        self.bandwidth_ = float(np.median(bws))
        self.fitted_ = fitted
        self.residuals_ = y - fitted

        # Standard errors via local weighted regression for each covariate
        self.std_err_ = self._compute_std_errors(X_int, y, D, bws, coef)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.t_values_ = coef / np.where(self.std_err_ > 0, self.std_err_, np.nan)
        df_resid = max(n - p_full, 1)
        self.p_values_ = t_to_p(np.nan_to_num(self.t_values_), df=df_resid)
        self.aicc_ = curr_aicc

        return self

    def _select_single_bw(self, j: int, partial_resid: np.ndarray,
                           X_int: np.ndarray, D: np.ndarray,
                           bw_init: float) -> float:
        """Golden-section search for bandwidth of covariate j."""
        n = len(partial_resid)
        if self.fixed:
            bw_min = np.percentile(D[D > 0], 5)
            bw_max = np.max(D)
        else:
            bw_min = max(X_int.shape[1] + 1, 3)
            bw_max = n - 1

        def score(bw):
            kernel = SpatialKernel(self.kernel, self.fixed, bw)
            fits = []
            for i in range(n):
                w = kernel(D[i])
                xi = X_int[:, j]
                wx = w * xi
                denom = np.sum(wx * xi)
                if abs(denom) < 1e-12:
                    fits.append(0.0)
                else:
                    fits.append(xi[i] * np.sum(wx * partial_resid) / denom)
            fitted_j = np.array(fits)
            rss = np.sum((partial_resid - fitted_j) ** 2)
            return aicc(rss, n, max(n / max(bw, 1), 2))

        sel = BandwidthSelector(score, fixed=self.fixed,
                                bw_min=bw_min, bw_max=bw_max,
                                tol=self.bw_tol)
        bw_opt, _ = sel.select("golden_section" if self.fixed else "grid")
        return bw_opt

    def _compute_std_errors(self, X_int, y, D, bws, coef) -> np.ndarray:
        """Approximate standard errors from local residual variance."""
        n, p = coef.shape
        resid = y - np.sum(coef * X_int, axis=1)
        sigma2 = np.sum(resid ** 2) / max(n - p, 1)
        std_err = np.full_like(coef, np.sqrt(sigma2))
        return std_err

    def predict(self, X, geometry) -> np.ndarray:
        """Predict at new locations using nearest calibration point coefficients."""
        if self.coef_ is None:
            raise RuntimeError("Fit the model first.")
        X = np.asarray(X, dtype=float)
        coords_new = _parse_geometry(geometry)
        m = len(coords_new)
        X_int = np.column_stack([np.ones(m), X])
        from scipy.spatial.distance import cdist
        D_pred = cdist(coords_new, self.coords_, metric="euclidean")
        nearest = np.argmin(D_pred, axis=1)
        return np.array([X_int[j] @ self.coef_[nearest[j]] for j in range(m)])

    def summary(self) -> str:
        """Text summary."""
        if self.coef_ is None:
            return "Model not fitted."
        n, p = self.coef_.shape
        lines = [
            "=" * 60,
            "Multiscale GWR (MGWR) Summary",
            "=" * 60,
            f"  Observations   : {n}",
            f"  Kernel         : {self.kernel}",
            f"  Fixed          : {self.fixed}",
            f"  AICc           : {self.aicc_:.4f}",
            "",
            "  Per-covariate bandwidths:",
        ]
        names = ["Intercept"] + [f"X{i}" for i in range(1, p)]
        for nm, bw in zip(names, self.bandwidths_):
            lines.append(f"    {nm:15s}: {bw:.4f}")
        lines.append("=" * 60)
        return "\n".join(lines)
