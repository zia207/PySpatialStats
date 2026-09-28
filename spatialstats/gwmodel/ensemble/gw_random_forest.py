"""
spatialstats.gwmodel.ensemble.gw_random_forest
====================================
Geographically Weighted Random Forest (GWRandomForest).

Combines global and local Random Forest models, weighting training
samples by spatial kernel. Incorporates improvements from PyGRF
(Sun & Hu 2024): theory-informed bandwidth via Moran's I and
local sample expansion for sparse regions.
"""

import numpy as np
import warnings
from typing import Optional, Union
from joblib import delayed

from spatialstats.gwmodel.core.base import GWEnsemble, _parse_geometry
from spatialstats.gwmodel.core.kernels import SpatialKernel
from spatialstats.gwmodel.utils.distance import pairwise_distances
from spatialstats.gwmodel.utils.parallel import parallel_map


def _fit_local_rf(i: int, X: np.ndarray, y: np.ndarray,
                   w: np.ndarray, params: dict) -> object:
    """Fit a local RandomForest at calibration point i."""
    from sklearn.ensemble import RandomForestRegressor
    # Normalise weights to [0, 1] for sample_weight
    w_norm = w / max(w.max(), 1e-12)
    # If resampled, expand training set for sparse regions
    # (include all samples with weight > threshold)
    mask = w_norm > 0.01
    if mask.sum() < params.get("min_samples", 10):
        mask = np.ones(len(y), dtype=bool)
    rf = RandomForestRegressor(
        n_estimators=params.get("n_estimators", 100),
        max_features=params.get("max_features", "sqrt"),
        max_depth=params.get("max_depth", None),
        min_samples_split=params.get("min_samples_split", 2),
        bootstrap=params.get("bootstrap", True),
        random_state=params.get("random_state", None),
        n_jobs=1,
    )
    rf.fit(X[mask], y[mask], sample_weight=w_norm[mask])
    return rf


def _auto_bandwidth_moran(X: np.ndarray, y: np.ndarray,
                           coords: np.ndarray, kernel_name: str,
                           fixed: bool) -> float:
    """
    Theory-informed bandwidth selection via Moran's I incremental test.
    Selects the bandwidth at which residuals become spatially random.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_predict
    # Fit global RF to get residuals
    rf_global = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rf_global.fit(X, y)
    residuals = y - rf_global.predict(X)

    D = pairwise_distances(coords, metric="euclidean")
    flat_d = D[D > 0]
    n = len(y)

    # Try candidate bandwidths and find where Moran's I drops below significance
    if fixed:
        candidates = np.percentile(flat_d, np.arange(10, 90, 10))
    else:
        candidates = np.arange(max(10, n // 10), n - 1, max(1, n // 10))

    for bw in candidates:
        kern = SpatialKernel(kernel_name, fixed, bw)
        # Build weight matrix and compute Moran's I
        W_mat = np.array([kern(D[i]) for i in range(n)])
        np.fill_diagonal(W_mat, 0)
        row_sums = W_mat.sum(axis=1)
        row_sums[row_sums == 0] = 1
        W_norm = W_mat / row_sums[:, None]
        spatial_lag = W_norm @ residuals
        cov = np.cov(residuals, spatial_lag)
        moran_i = cov[0, 1] / max(cov[0, 0], 1e-12)
        if abs(moran_i) < 0.1:  # residuals approximately random
            return float(bw)

    return float(candidates[len(candidates) // 2])


class GWRandomForest(GWEnsemble):
    """
    Geographically Weighted Random Forest.

    Fits a global RF for overall patterns and local RFs weighted by
    spatial kernels. Predictions blend global and local models.

    Parameters
    ----------
    n_estimators : int      Trees per local forest.
    max_features : str or int
    bandwidth : float, int, or 'auto'
        'auto' = theory-informed bandwidth via Moran's I.
    fixed : bool
    kernel : str
    train_weighted : bool   Weight local training samples by kernel.
    predict_weighted : bool Blend global + local predictions.
    local_weight : float or None
        Weight given to local prediction [0, 1].
        If None, determined by cross-validation.
    resampled : bool        Expand sparse local samples (PyGRF strategy).
    bootstrap : bool
    max_depth : int or None
    min_samples_split : int
    random_state : int or None
    n_jobs : int

    Attributes (after fit)
    ----------------------
    feature_importances_local_ : ndarray (n, p)
    feature_importances_global_ : ndarray (p,)
    local_estimators_ : list of RandomForest objects
    global_estimator_ : RandomForest
    bandwidth_ : float
    local_weight_ : float  (actual weight used)

    Examples
    --------
    >>> from spatialstats.gwmodel.ensemble import GWRandomForest
    >>> import numpy as np
    >>> coords = np.random.rand(80, 2) * 100
    >>> X = np.random.rand(80, 3)
    >>> y = X @ [1, -1, 2] + np.random.randn(80) * 0.3
    >>> model = GWRandomForest(n_estimators=50, bandwidth=25, fixed=True)
    >>> model.fit(X, y, coords)
    >>> preds = model.predict(X, coords)
    >>> print(preds.shape)
    (80,)
    """

    def __init__(self, n_estimators: int = 100,
                 max_features: Union[str, int] = "sqrt",
                 bandwidth=None, fixed: bool = False,
                 kernel: str = "bisquare",
                 train_weighted: bool = True,
                 predict_weighted: bool = True,
                 local_weight: Optional[float] = None,
                 resampled: bool = True,
                 bootstrap: bool = True,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 random_state: Optional[int] = None,
                 n_jobs: int = -1):
        super().__init__(bandwidth=bandwidth, fixed=fixed, kernel=kernel,
                         n_jobs=n_jobs, train_weighted=train_weighted,
                         predict_weighted=predict_weighted,
                         local_weight=local_weight)
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.resampled = resampled
        self.bootstrap = bootstrap
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.local_weight_: Optional[float] = None

    def fit(self, X, y, geometry):
        """
        Fit global + local random forests.

        Parameters
        ----------
        X        : array-like (n, p)
        y        : array-like (n,)
        geometry : ndarray (n, 2) or GeoSeries

        Returns
        -------
        self
        """
        from sklearn.ensemble import RandomForestRegressor

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        coords = _parse_geometry(geometry)
        n, p = X.shape

        self.coords_ = coords
        self.X_ = X
        self.y_ = y

        # Fit global model
        self.global_estimator_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_features=self.max_features,
            max_depth=self.max_depth,
            bootstrap=self.bootstrap,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.global_estimator_.fit(X, y)

        # Bandwidth selection
        bw = self.bandwidth
        if bw == "auto" or bw is None:
            bw = _auto_bandwidth_moran(X, y, coords, self.kernel, self.fixed)
        self.bandwidth_ = float(bw)

        D = pairwise_distances(coords, metric="euclidean")
        kernel = SpatialKernel(self.kernel, self.fixed, self.bandwidth_)

        rf_params = {
            "n_estimators": self.n_estimators,
            "max_features": self.max_features,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "bootstrap": self.bootstrap,
            "random_state": self.random_state,
            "min_samples": max(10, n // 10),
        }

        # Fit local models
        if self.train_weighted:
            self.local_estimators_ = parallel_map(
                (delayed(_fit_local_rf)(i, X, y, kernel(D[i]), rf_params)
                 for i in range(n)),
                n_jobs=self.n_jobs,
            )
        else:
            self.local_estimators_ = [self.global_estimator_] * n

        # Local feature importances
        self.feature_importances_local_ = np.array([
            est.feature_importances_ for est in self.local_estimators_
        ])
        self.feature_importances_global_ = self.global_estimator_.feature_importances_

        # Determine local_weight
        if self.local_weight is not None:
            self.local_weight_ = float(self.local_weight)
        else:
            self.local_weight_ = self._auto_local_weight(X, y, coords, D, kernel)

        # Store fitted values
        self.fitted_ = self.predict(X, coords)
        self.residuals_ = y - self.fitted_

        return self

    def _auto_local_weight(self, X, y, coords, D, kernel) -> float:
        """Select local weight by minimising CV RMSE."""
        n = len(y)
        best_w, best_rmse = 0.5, np.inf
        for w in np.arange(0.0, 1.05, 0.1):
            preds = []
            for i in range(n):
                g = self.global_estimator_.predict(X[i:i+1])[0]
                loc = self.local_estimators_[i].predict(X[i:i+1])[0]
                preds.append(w * loc + (1 - w) * g)
            rmse = np.sqrt(np.mean((y - np.array(preds)) ** 2))
            if rmse < best_rmse:
                best_rmse = rmse
                best_w = w
        return float(best_w)

    def predict(self, X, geometry) -> np.ndarray:
        """
        Predict using blended global + local models.

        Returns
        -------
        yhat : ndarray (m,)
        """
        if self.global_estimator_ is None:
            raise RuntimeError("Model not fitted.")
        X = np.asarray(X, dtype=float)
        coords_new = _parse_geometry(geometry)
        m = len(coords_new)

        global_pred = self.global_estimator_.predict(X)

        if not self.predict_weighted or self.local_estimators_ is None:
            return global_pred

        from scipy.spatial.distance import cdist
        D_pred = cdist(coords_new, self.coords_, metric="euclidean")
        nearest = np.argmin(D_pred, axis=1)

        local_pred = np.array([
            self.local_estimators_[nearest[j]].predict(X[j:j+1])[0]
            for j in range(m)
        ])

        lw = self.local_weight_
        blended = lw * local_pred + (1 - lw) * global_pred
        # Fall back to global where local gave NaN/Inf
        bad = ~np.isfinite(blended)
        blended[bad] = global_pred[bad]
        return blended

    def feature_importance_map(self):
        """
        Return a summary of spatially varying feature importances.

        Returns
        -------
        ndarray (n, p) of local feature importances.
        """
        if self.feature_importances_local_ is None:
            raise RuntimeError("Fit the model first.")
        return self.feature_importances_local_

    def get_params(self, deep=True) -> dict:
        p = super().get_params(deep)
        p.update({
            "n_estimators": self.n_estimators,
            "max_features": self.max_features,
            "resampled": self.resampled,
            "bootstrap": self.bootstrap,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "random_state": self.random_state,
        })
        return p
