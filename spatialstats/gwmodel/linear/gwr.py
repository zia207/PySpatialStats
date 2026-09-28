"""
spatialstats.gwmodel.linear.gwr
====================
Geographically Weighted Regression (GWR).

Based on Fotheringham, Brunsdon & Charlton (2002) and the R GWmodel
gwr.basic() function. Uses iteratively-weighted least squares per
calibration point via spatial kernel weights.
"""

import numpy as np
import warnings
from typing import Optional, Union
from joblib import delayed

from spatialstats.gwmodel.core.base import GWRegressor, _parse_geometry
from spatialstats.gwmodel.core.kernels import SpatialKernel
from spatialstats.gwmodel.core.bandwidth import BandwidthSelector
from spatialstats.gwmodel.utils.stats import aicc, local_r2, t_to_p, bh_correction
from spatialstats.gwmodel.utils.distance import pairwise_distances
from spatialstats.gwmodel.utils.parallel import parallel_map


def _fit_single_point(i: int, X: np.ndarray, y: np.ndarray,
                       D: np.ndarray, kernel: SpatialKernel,
                       hat_matrix: bool) -> dict:
    """
    Fit WLS at calibration point i.

    Returns dict with coef, std_err, fitted_i, hat_ii.
    """
    w = kernel(D[i])
    W = np.diag(w)
    XtW = X.T @ W
    A = XtW @ X
    b = XtW @ y
    try:
        coef = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(A, b, rcond=None)[0]

    yhat_i = X[i] @ coef
    residual_i = y[i] - yhat_i

    # Local sigma² for std errors
    fitted = X @ coef
    resid = y - fitted
    sigma2_local = np.sum(w * resid ** 2) / max(np.sum(w) - X.shape[1], 1e-10)

    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        A_inv = np.linalg.pinv(A)

    var_coef = sigma2_local * np.diag(A_inv)
    std_err = np.sqrt(np.maximum(var_coef, 0))

    # Hat matrix diagonal element
    if hat_matrix:
        xi = X[i]
        hat_ii = float(xi @ A_inv @ XtW[:, i])
    else:
        hat_ii = np.nan

    return {
        "coef": coef,
        "std_err": std_err,
        "fitted_i": yhat_i,
        "residual_i": residual_i,
        "hat_ii": hat_ii,
        "weights": w,
    }


class GWR(GWRegressor):
    """
    Geographically Weighted Regression.

    Parameters
    ----------
    bandwidth : float, int, or 'auto'
        If 'auto', bandwidth is selected by minimising ``criterion``.
        For fixed kernels: distance threshold.
        For adaptive kernels: number of nearest neighbours.
    fixed : bool
        True = fixed distance; False = adaptive k-NN.
    kernel : str
        Kernel function name (see spatialstats.gwmodel.core.kernels).
    criterion : str
        Score used for bandwidth selection: 'AICc', 'AIC', 'BIC', 'CV'.
    hat_matrix : bool
        Whether to compute hat matrix diagonal (needed for AICc and inference).
    n_jobs : int
        Parallel workers (-1 = all CPUs).

    Attributes (after fit)
    ----------------------
    coef_ : ndarray (n, p)
    std_err_ : ndarray (n, p)
    t_values_ : ndarray (n, p)
    p_values_ : ndarray (n, p)
    residuals_ : ndarray (n,)
    fitted_ : ndarray (n,)
    r2_local_ : ndarray (n,)
    bandwidth_ : float
    aicc_ : float
    effective_df_ : float

    Examples
    --------
    >>> from spatialstats.gwmodel.linear import GWR
    >>> import numpy as np
    >>> coords = np.random.rand(50, 2) * 100
    >>> X = np.random.rand(50, 3)
    >>> y = X @ [1, -2, 0.5] + np.random.randn(50) * 0.3
    >>> model = GWR(bandwidth=20, fixed=True, kernel='bisquare')
    >>> model.fit(X, y, coords)
    >>> print(model.coef_.shape)
    (50, 3)
    """

    def __init__(self, bandwidth=None, fixed: bool = False,
                 kernel: str = "bisquare", criterion: str = "AICc",
                 hat_matrix: bool = True, n_jobs: int = -1):
        super().__init__(bandwidth=bandwidth, fixed=fixed, kernel=kernel,
                         criterion=criterion, n_jobs=n_jobs)
        self.hat_matrix = hat_matrix

    def fit(self, X, y, geometry):
        """
        Calibrate GWR model.

        Parameters
        ----------
        X : array-like (n, p)   Feature matrix (WITHOUT intercept; added internally).
        y : array-like (n,)     Target vector.
        geometry : ndarray (n,2) or GeoSeries
                                 Spatial coordinates of observations.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        coords = _parse_geometry(geometry)

        n, p = X.shape
        # Prepend intercept
        X_int = np.column_stack([np.ones(n), X])
        p_int = X_int.shape[1]

        self.coords_ = coords
        self.X_ = X_int
        self.y_ = y

        # Distance matrix
        D = pairwise_distances(coords, metric="euclidean")

        # Bandwidth selection
        bw = self.bandwidth
        if bw == "auto" or bw is None:
            bw = self._select_bandwidth(X_int, y, D)
        self.bandwidth_ = bw

        kernel = SpatialKernel(self.kernel, self.fixed, bw)

        # Fit at all calibration points
        results = parallel_map(
            (delayed(_fit_single_point)(
                i, X_int, y, D, kernel, self.hat_matrix
            ) for i in range(n)),
            n_jobs=self.n_jobs,
        )

        self.coef_ = np.array([r["coef"] for r in results])       # (n, p+1)
        self.std_err_ = np.array([r["std_err"] for r in results])
        self.fitted_ = np.array([r["fitted_i"] for r in results])
        self.residuals_ = y - self.fitted_

        hat_diag = np.array([r["hat_ii"] for r in results])
        self.effective_df_ = float(np.nansum(hat_diag))

        rss = float(np.sum(self.residuals_ ** 2))
        self.aicc_ = aicc(rss, n, self.effective_df_)

        # t-values and p-values
        df_resid = max(n - self.effective_df_, 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.t_values_ = self.coef_ / np.where(self.std_err_ > 0, self.std_err_, np.nan)
        self.p_values_ = t_to_p(np.nan_to_num(self.t_values_), df=df_resid)

        # Local R²
        self.r2_local_ = np.array([
            local_r2(y, X_int @ self.coef_[i], results[i]["weights"])
            for i in range(n)
        ])

        return self

    def _select_bandwidth(self, X, y, D) -> float:
        """Select optimal bandwidth by minimising criterion."""
        n = len(y)
        if self.fixed:
            flat_d = D[D > 0].ravel()
            bw_min = np.percentile(flat_d, 5) if (self.bandwidth is None or self.bandwidth == 'auto') else self.bandwidth
            bw_max = np.max(D)
        else:
            bw_min = max(X.shape[1] + 1, 5)
            bw_max = n - 1

        def score_fn(bw):
            kernel = SpatialKernel(self.kernel, self.fixed, bw)
            fits, hats = [], []
            for i in range(n):
                w = kernel(D[i])
                W = np.diag(w)
                XtW = X.T @ W
                A = XtW @ X
                b_vec = XtW @ y
                try:
                    coef = np.linalg.solve(A, b_vec)
                except np.linalg.LinAlgError:
                    coef = np.linalg.lstsq(A, b_vec, rcond=None)[0]
                fits.append(X[i] @ coef)
                try:
                    A_inv = np.linalg.inv(A)
                    hats.append(float(X[i] @ A_inv @ XtW[:, i]))
                except np.linalg.LinAlgError:
                    hats.append(0.0)
            fitted = np.array(fits)
            rss = np.sum((y - fitted) ** 2)
            k = np.sum(hats)
            return aicc(rss, n, k)

        sel = BandwidthSelector(score_fn, fixed=self.fixed,
                                bw_min=bw_min, bw_max=bw_max,
                                criterion=self.criterion)
        bw_opt, _ = sel.select("golden_section" if self.fixed else "grid")
        return bw_opt

    def predict(self, X, geometry) -> np.ndarray:
        """
        Predict at new locations using the nearest calibration point's coefficients.

        Parameters
        ----------
        X : array-like (m, p)
        geometry : ndarray (m, 2) or GeoSeries

        Returns
        -------
        yhat : ndarray (m,)
        """
        if self.coef_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X = np.asarray(X, dtype=float)
        coords_new = _parse_geometry(geometry)
        m = len(coords_new)
        X_int = np.column_stack([np.ones(m), X])

        from scipy.spatial.distance import cdist
        D_pred = cdist(coords_new, self.coords_, metric="euclidean")
        nearest = np.argmin(D_pred, axis=1)

        return np.array([X_int[j] @ self.coef_[nearest[j]] for j in range(m)])

    def summary(self) -> str:
        """Return a text summary of model fit."""
        if self.coef_ is None:
            return "Model not fitted."
        n, p = self.coef_.shape
        lines = [
            "=" * 60,
            "Geographically Weighted Regression (GWR) Summary",
            "=" * 60,
            f"  Observations            : {n}",
            f"  Predictors (incl. int.) : {p}",
            f"  Bandwidth               : {self.bandwidth_:.4f}",
            f"  Kernel                  : {self.kernel}",
            f"  Fixed                   : {self.fixed}",
            f"  AICc                    : {self.aicc_:.4f}",
            f"  Effective DF            : {self.effective_df_:.4f}",
            f"  Global R²               : {1 - np.sum(self.residuals_**2)/np.sum((self.y_ - self.y_.mean())**2):.4f}",
            "",
            "  Local coefficient summary (min / median / max):",
        ]
        names = ["Intercept"] + [f"X{i}" for i in range(1, p)]
        for j, nm in enumerate(names):
            c = self.coef_[:, j]
            lines.append(f"    {nm:15s}: {c.min():.4f} / {np.median(c):.4f} / {c.max():.4f}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def local_significance(self, alpha: float = 0.05,
                            correction: str = "bh") -> np.ndarray:
        """
        Return boolean mask (n, p) of significant local coefficients.

        Parameters
        ----------
        alpha      : float   Significance level.
        correction : str     'bh' (Benjamini-Hochberg) | 'none'.
        """
        if self.p_values_ is None:
            raise RuntimeError("Fit the model first.")
        n, p = self.p_values_.shape
        sig = np.zeros((n, p), dtype=bool)
        for j in range(p):
            pv = self.p_values_[:, j]
            if correction == "bh":
                sig[:, j] = bh_correction(pv, alpha)
            else:
                sig[:, j] = pv < alpha
        return sig

    def monte_carlo_test(self, n_sim: int = 99, seed: int = 42) -> dict:
        """
        Monte Carlo test for spatial variability of each coefficient surface.

        Null hypothesis: coefficient is spatially stationary.

        Returns
        -------
        dict with 'statistic' and 'p_value' arrays of shape (p,).
        """
        if self.coef_ is None:
            raise RuntimeError("Fit the model first.")
        rng = np.random.default_rng(seed)
        n, p = self.coef_.shape
        obs_var = np.var(self.coef_, axis=0)
        count = np.zeros(p)
        for _ in range(n_sim):
            perm_idx = rng.permutation(n)
            temp = GWR(bandwidth=self.bandwidth_, fixed=self.fixed,
                       kernel=self.kernel, hat_matrix=False, n_jobs=self.n_jobs)
            temp.fit(self.X_[:, 1:], self.y_[perm_idx], self.coords_)
            sim_var = np.var(temp.coef_, axis=0)
            count += (sim_var >= obs_var).astype(float)
        p_values = (count + 1) / (n_sim + 1)
        return {"statistic": obs_var, "p_value": p_values,
                "n_sim": n_sim,
                "covariate_names": [f"X{i}" for i in range(p)]}
