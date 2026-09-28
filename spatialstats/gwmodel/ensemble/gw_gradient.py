"""
spatialstats.gwmodel.ensemble.gw_gradient
================================
Geographically Weighted Gradient Boosting models.

Wraps XGBoost, LightGBM, and scikit-learn HistGradientBoosting
with spatial kernel weighting.
"""

import numpy as np
from typing import Optional, Union
from joblib import delayed

from spatialstats.gwmodel.core.base import GWEnsemble, _parse_geometry
from spatialstats.gwmodel.core.kernels import SpatialKernel
from spatialstats.gwmodel.utils.distance import pairwise_distances
from spatialstats.gwmodel.utils.parallel import parallel_map
from spatialstats.gwmodel.utils.compute import get_compute_config


def _fit_local_boosting(i, X, y, w, estimator_cls, params):
    """Fit a local boosting model at point i."""
    w_norm = w / max(w.max(), 1e-12)
    mask = w_norm > 0.01
    if mask.sum() < 10:
        mask = np.ones(len(y), dtype=bool)
    est = estimator_cls(**params)
    est.fit(X[mask], y[mask], sample_weight=w_norm[mask])
    return est


class GWXGBoost(GWEnsemble):
    """
    Geographically Weighted XGBoost.

    Parameters
    ----------
    n_estimators : int
    max_depth : int
    learning_rate : float
    subsample : float
    bandwidth : float, int, or None
    fixed : bool
    kernel : str
    train_weighted : bool
    predict_weighted : bool
    local_weight : float or None
    n_jobs : int

    Examples
    --------
    >>> from spatialstats.gwmodel.ensemble import GWXGBoost
    >>> import numpy as np
    >>> coords = np.random.rand(60, 2) * 100
    >>> X = np.random.rand(60, 3)
    >>> y = X @ [1, -1, 0.5] + np.random.randn(60) * 0.3
    >>> model = GWXGBoost(n_estimators=30, bandwidth=20, fixed=True)
    >>> model.fit(X, y, coords)
    >>> print(model.predict(X, coords).shape)
    (60,)
    """

    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1,
                 subsample=0.8, colsample_bytree=0.8,
                 bandwidth=None, fixed=False, kernel="bisquare",
                 train_weighted=True, predict_weighted=True,
                 local_weight=None, n_jobs=-1):
        super().__init__(bandwidth=bandwidth, fixed=fixed, kernel=kernel,
                         n_jobs=n_jobs, train_weighted=train_weighted,
                         predict_weighted=predict_weighted, local_weight=local_weight)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.local_weight_: Optional[float] = None

    def fit(self, X, y, geometry):
        try:
            from xgboost import XGBRegressor
        except ImportError:
            raise ImportError("xgboost is required: pip install xgboost")

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        coords = _parse_geometry(geometry)
        n, p = X.shape
        self.coords_ = coords

        params = dict(n_estimators=self.n_estimators, max_depth=self.max_depth,
                      learning_rate=self.learning_rate, subsample=self.subsample,
                      colsample_bytree=self.colsample_bytree,
                      eval_metric="rmse", verbosity=0,
                      **get_compute_config().xgboost_params())

        self.global_estimator_ = XGBRegressor(**params)
        self.global_estimator_.fit(X, y)

        bw = self.bandwidth if self.bandwidth else n // 3
        self.bandwidth_ = float(bw)
        D = pairwise_distances(coords)
        kernel = SpatialKernel(self.kernel, self.fixed, self.bandwidth_)

        self.local_estimators_ = parallel_map(
            (delayed(_fit_local_boosting)(i, X, y, kernel(D[i]), XGBRegressor, params)
             for i in range(n)),
            n_jobs=self.n_jobs,
        )

        self.feature_importances_local_ = np.array([
            est.feature_importances_ for est in self.local_estimators_
        ])
        self.local_weight_ = self.local_weight if self.local_weight else 0.5
        self.fitted_ = self.predict(X, coords)
        self.residuals_ = y - self.fitted_
        return self

    def predict(self, X, geometry):
        if self.global_estimator_ is None:
            raise RuntimeError("Fit model first.")
        X = np.asarray(X, dtype=float)
        coords_new = _parse_geometry(geometry)
        m = len(coords_new)
        g_pred = self.global_estimator_.predict(X)
        if not self.predict_weighted or self.local_estimators_ is None:
            return g_pred
        from scipy.spatial.distance import cdist
        nearest = np.argmin(cdist(coords_new, self.coords_), axis=1)
        l_pred = np.array([self.local_estimators_[nearest[j]].predict(X[j:j+1])[0]
                           for j in range(m)])
        lw = self.local_weight_
        return lw * l_pred + (1 - lw) * g_pred


class GWLightGBM(GWEnsemble):
    """
    Geographically Weighted LightGBM.

    Examples
    --------
    >>> from spatialstats.gwmodel.ensemble import GWLightGBM
    >>> import numpy as np
    >>> coords = np.random.rand(60, 2) * 100
    >>> X = np.random.rand(60, 3)
    >>> y = X @ [1, -1, 0.5] + np.random.randn(60) * 0.3
    >>> model = GWLightGBM(n_estimators=30, bandwidth=20, fixed=True)
    >>> model.fit(X, y, coords)
    """

    def __init__(self, n_estimators=100, num_leaves=31, learning_rate=0.05,
                 bandwidth=None, fixed=False, kernel="bisquare",
                 train_weighted=True, predict_weighted=True,
                 local_weight=None, n_jobs=-1):
        super().__init__(bandwidth=bandwidth, fixed=fixed, kernel=kernel,
                         n_jobs=n_jobs, train_weighted=train_weighted,
                         predict_weighted=predict_weighted, local_weight=local_weight)
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.local_weight_: Optional[float] = None

    def fit(self, X, y, geometry):
        try:
            from lightgbm import LGBMRegressor
        except ImportError:
            raise ImportError("lightgbm is required: pip install lightgbm")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        coords = _parse_geometry(geometry)
        n = len(y)
        self.coords_ = coords

        params = dict(n_estimators=self.n_estimators, num_leaves=self.num_leaves,
                      learning_rate=self.learning_rate, verbose=-1,
                      **get_compute_config().lightgbm_params())
        self.global_estimator_ = LGBMRegressor(**params)
        self.global_estimator_.fit(X, y)

        bw = self.bandwidth if self.bandwidth else n // 3
        self.bandwidth_ = float(bw)
        D = pairwise_distances(coords)
        kernel = SpatialKernel(self.kernel, self.fixed, self.bandwidth_)

        self.local_estimators_ = parallel_map(
            (delayed(_fit_local_boosting)(i, X, y, kernel(D[i]), LGBMRegressor, params)
             for i in range(n)),
            n_jobs=self.n_jobs,
        )
        self.feature_importances_local_ = np.array([
            est.feature_importances_ / max(est.feature_importances_.sum(), 1)
            for est in self.local_estimators_
        ])
        self.local_weight_ = self.local_weight if self.local_weight else 0.5
        self.fitted_ = self.predict(X, coords)
        self.residuals_ = y - self.fitted_
        return self

    def predict(self, X, geometry):
        if self.global_estimator_ is None:
            raise RuntimeError("Fit model first.")
        X = np.asarray(X, dtype=float)
        coords_new = _parse_geometry(geometry)
        m = len(coords_new)
        g_pred = self.global_estimator_.predict(X)
        if not self.predict_weighted or self.local_estimators_ is None:
            return g_pred
        from scipy.spatial.distance import cdist
        nearest = np.argmin(cdist(coords_new, self.coords_), axis=1)
        l_pred = np.array([self.local_estimators_[nearest[j]].predict(X[j:j+1])[0]
                           for j in range(m)])
        lw = self.local_weight_
        return lw * l_pred + (1 - lw) * g_pred
