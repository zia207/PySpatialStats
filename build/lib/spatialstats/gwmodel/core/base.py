"""
spatialstats.gwmodel.core.base
===================
Abstract base classes for all PyGWmodel estimators.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union
import warnings


def _parse_geometry(geometry) -> np.ndarray:
    """
    Convert geometry argument to (n, 2) coordinate array.

    Accepts:
    - ndarray of shape (n, 2)
    - GeoSeries / GeoDataFrame geometry column
    """
    if isinstance(geometry, np.ndarray):
        if geometry.ndim == 1:
            raise ValueError("geometry array must be 2-D (n, 2)")
        return geometry.astype(float)
    # Try GeoSeries / GeoDataFrame
    try:
        import geopandas as gpd
        if hasattr(geometry, "geometry"):
            geometry = geometry.geometry
        pts = geometry.representative_point()
        return np.column_stack([pts.x.values, pts.y.values])
    except Exception as e:
        raise ValueError(
            f"Cannot parse geometry: {e}. Pass an ndarray of shape (n, 2) "
            "or a GeoSeries/GeoDataFrame."
        )


class BaseGWModel(ABC):
    """
    Abstract base for all geographically weighted models.

    Subclasses must implement ``fit``, ``predict``, and ``_compute_score``.
    """

    def __init__(self, bandwidth=None, fixed: bool = False,
                 kernel: str = "bisquare", criterion: str = "AICc",
                 n_jobs: int = -1):
        self.bandwidth = bandwidth
        self.fixed = fixed
        self.kernel = kernel
        self.criterion = criterion
        self.n_jobs = n_jobs

    @abstractmethod
    def fit(self, X, y, geometry):
        """Fit the model to training data."""
        ...

    @abstractmethod
    def predict(self, X, geometry):
        """Predict at new locations."""
        ...

    def score(self, X, y, geometry) -> float:
        """Return default goodness-of-fit score (R² for regression)."""
        yhat = self.predict(X, geometry)
        y = np.asarray(y)
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1.0 - ss_res / max(ss_tot, 1e-10)

    def get_params(self, deep: bool = True) -> dict:
        """Return model hyperparameters (scikit-learn compatible)."""
        return {
            "bandwidth": self.bandwidth,
            "fixed": self.fixed,
            "kernel": self.kernel,
            "criterion": self.criterion,
            "n_jobs": self.n_jobs,
        }

    def set_params(self, **params):
        """Set model hyperparameters (scikit-learn compatible)."""
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def _get_kernel(self):
        from spatialstats.gwmodel.core.kernels import SpatialKernel
        return SpatialKernel(self.kernel, self.fixed, self.bandwidth)

    def _get_coords(self, geometry) -> np.ndarray:
        return _parse_geometry(geometry)

    def __repr__(self):
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{self.__class__.__name__}({params})"


class GWRegressor(BaseGWModel, ABC):
    """Base class for geographically weighted regression models."""

    def __init__(self, bandwidth=None, fixed=False, kernel="bisquare",
                 criterion="AICc", n_jobs=-1):
        super().__init__(bandwidth, fixed, kernel, criterion, n_jobs)
        # Set in fit()
        self.coef_: Optional[np.ndarray] = None
        self.std_err_: Optional[np.ndarray] = None
        self.t_values_: Optional[np.ndarray] = None
        self.p_values_: Optional[np.ndarray] = None
        self.residuals_: Optional[np.ndarray] = None
        self.fitted_: Optional[np.ndarray] = None
        self.r2_local_: Optional[np.ndarray] = None
        self.bandwidth_: Optional[float] = None
        self.aicc_: Optional[float] = None
        self.effective_df_: Optional[float] = None
        self.coords_: Optional[np.ndarray] = None
        self.X_: Optional[np.ndarray] = None
        self.y_: Optional[np.ndarray] = None


class GWClassifier(BaseGWModel, ABC):
    """Base class for geographically weighted classifiers."""

    def __init__(self, bandwidth=None, fixed=False, kernel="bisquare",
                 criterion="AICc", n_jobs=-1):
        super().__init__(bandwidth, fixed, kernel, criterion, n_jobs)
        self.classes_: Optional[np.ndarray] = None

    @abstractmethod
    def predict_proba(self, X, geometry) -> np.ndarray:
        """Return class probability estimates."""
        ...

    def score(self, X, y, geometry) -> float:
        """Return classification accuracy."""
        yhat = self.predict(X, geometry)
        return np.mean(np.asarray(y) == yhat)


class GWEnsemble(GWRegressor, ABC):
    """Base class for geographically weighted ensemble models."""

    def __init__(self, bandwidth=None, fixed=False, kernel="bisquare",
                 criterion="AICc", n_jobs=-1,
                 train_weighted=True, predict_weighted=True,
                 local_weight=None):
        super().__init__(bandwidth, fixed, kernel, criterion, n_jobs)
        self.train_weighted = train_weighted
        self.predict_weighted = predict_weighted
        self.local_weight = local_weight
        self.feature_importances_local_: Optional[np.ndarray] = None
        self.local_estimators_: Optional[list] = None
        self.global_estimator_ = None
