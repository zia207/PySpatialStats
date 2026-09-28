"""
spatialstats.gwmodel.linear.mixed_gwr
===========================
Mixed (Semiparametric) Geographically Weighted Regression.

Some coefficients are estimated globally; others vary spatially.
"""

import numpy as np
from typing import List, Optional
from spatialstats.gwmodel.core.base import GWRegressor, _parse_geometry
from spatialstats.gwmodel.core.kernels import SpatialKernel
from spatialstats.gwmodel.utils.distance import pairwise_distances
from spatialstats.gwmodel.utils.stats import aicc, t_to_p


class MixedGWR(GWRegressor):
    """
    Mixed (Semiparametric) GWR.

    Parameters
    ----------
    fixed_vars : list of int
        Column indices (in X) that have globally fixed coefficients.
    varying_vars : list of int
        Column indices that vary spatially.
    bandwidth : float or int
    fixed : bool
    kernel : str

    Attributes (after fit)
    ----------------------
    global_coef_ : ndarray (len(fixed_vars),)  Globally estimated coefficients.
    coef_ : ndarray (n, len(varying_vars)+1)   Local coefficients (incl. intercept).

    Examples
    --------
    >>> from spatialstats.gwmodel.linear import MixedGWR
    >>> import numpy as np
    >>> coords = np.random.rand(60, 2) * 100
    >>> X = np.random.rand(60, 3)
    >>> y = X @ [1, -1, 0.5] + np.random.randn(60) * 0.2
    >>> model = MixedGWR(fixed_vars=[0], varying_vars=[1, 2],
    ...                   bandwidth=20, fixed=True)
    >>> model.fit(X, y, coords)
    """

    def __init__(self, fixed_vars: List[int] = None,
                 varying_vars: List[int] = None,
                 bandwidth=None, fixed: bool = False,
                 kernel: str = "bisquare", n_jobs: int = -1):
        super().__init__(bandwidth=bandwidth, fixed=fixed, kernel=kernel, n_jobs=n_jobs)
        self.fixed_vars = fixed_vars or []
        self.varying_vars = varying_vars or []

    def fit(self, X, y, geometry):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        coords = _parse_geometry(geometry)
        n, p = X.shape

        X_fix = X[:, self.fixed_vars] if self.fixed_vars else np.zeros((n, 0))
        X_var = X[:, self.varying_vars] if self.varying_vars else X

        # Step 1: Backfitting — start with global estimate for fixed vars
        global_coef = np.zeros(X_fix.shape[1]) if X_fix.shape[1] > 0 else np.array([])
        partial_resid = y.copy()

        from spatialstats.gwmodel.linear.gwr import GWR
        for _ in range(5):  # backfitting iterations
            # Local step: GWR on partial residuals for varying vars
            gwr_local = GWR(bandwidth=self.bandwidth or n // 3,
                            fixed=self.fixed, kernel=self.kernel,
                            hat_matrix=False, n_jobs=self.n_jobs)
            gwr_local.fit(X_var, partial_resid, coords)
            local_fitted = gwr_local.fitted_

            if X_fix.shape[1] > 0:
                # Global step: OLS on remaining residuals
                global_resid = y - local_fitted
                try:
                    global_coef = np.linalg.lstsq(X_fix, global_resid, rcond=None)[0]
                except np.linalg.LinAlgError:
                    global_coef = np.zeros(X_fix.shape[1])
                global_fitted = X_fix @ global_coef
                partial_resid = y - global_fitted

        self.global_coef_ = global_coef
        self.coef_ = gwr_local.coef_
        self.bandwidth_ = gwr_local.bandwidth_
        self.fitted_ = gwr_local.fitted_ + (X_fix @ global_coef if X_fix.shape[1] > 0 else 0)
        self.residuals_ = y - self.fitted_
        self.coords_ = coords
        self.X_ = X_var
        self.y_ = y
        self.std_err_ = gwr_local.std_err_
        self.t_values_ = gwr_local.t_values_
        self.p_values_ = gwr_local.p_values_
        return self

    def predict(self, X, geometry):
        from scipy.spatial.distance import cdist
        X = np.asarray(X, dtype=float)
        coords_new = _parse_geometry(geometry)
        m = len(coords_new)
        X_var = X[:, self.varying_vars] if self.varying_vars else X
        X_fix = X[:, self.fixed_vars] if self.fixed_vars else np.zeros((m, 0))
        X_var_int = np.column_stack([np.ones(m), X_var])
        D_pred = cdist(coords_new, self.coords_)
        nearest = np.argmin(D_pred, axis=1)
        local_pred = np.array([X_var_int[j] @ self.coef_[nearest[j]] for j in range(m)])
        global_pred = X_fix @ self.global_coef_ if X_fix.shape[1] > 0 else np.zeros(m)
        return local_pred + global_pred
