"""
spatialstats.gwmodel.linear.gwss
=====================
Geographically Weighted Summary Statistics (GWSS).

Computes spatially varying descriptive statistics including local mean,
variance, skewness, kurtosis, and correlation coefficients.
"""

import numpy as np
from typing import List, Optional, Union
import pandas as pd

from spatialstats.gwmodel.core.base import _parse_geometry
from spatialstats.gwmodel.core.kernels import SpatialKernel
from spatialstats.gwmodel.utils.distance import pairwise_distances


class GWSS:
    """
    Geographically Weighted Summary Statistics.

    Parameters
    ----------
    bandwidth : float or int
    fixed : bool
    kernel : str
    quantile : bool
        If True, also compute local median and IQR.

    Attributes (after fit)
    ----------------------
    local_mean_ : ndarray (n, k)
    local_std_  : ndarray (n, k)
    local_var_  : ndarray (n, k)
    local_skew_ : ndarray (n, k)
    local_kurt_ : ndarray (n, k)
    local_corr_ : dict of ndarray (n,) for each pair

    Examples
    --------
    >>> from spatialstats.gwmodel.linear import GWSS
    >>> import numpy as np
    >>> coords = np.random.rand(60, 2) * 100
    >>> X = np.random.rand(60, 3)
    >>> ss = GWSS(bandwidth=25, fixed=True)
    >>> ss.fit(X, coords)
    >>> print(ss.local_mean_.shape)
    (60, 3)
    """

    def __init__(self, bandwidth: Union[float, int] = None, fixed: bool = False,
                 kernel: str = "bisquare", quantile: bool = False):
        self.bandwidth = bandwidth
        self.fixed = fixed
        self.kernel = kernel
        self.quantile = quantile

    def fit(self, X, geometry, var_names: Optional[List[str]] = None):
        """
        Compute local summary statistics.

        Parameters
        ----------
        X : array-like (n, k)
        geometry : ndarray or GeoSeries
        var_names : list of str, optional

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        n, k = X.shape
        coords = _parse_geometry(geometry)

        self.var_names_ = var_names or [f"Var{i}" for i in range(k)]
        self.coords_ = coords
        self.X_ = X

        D = pairwise_distances(coords, metric="euclidean")
        kern = SpatialKernel(self.kernel, self.fixed, self.bandwidth)

        lmean = np.zeros((n, k))
        lvar  = np.zeros((n, k))
        lstd  = np.zeros((n, k))
        lskew = np.zeros((n, k))
        lkurt = np.zeros((n, k))

        # Pairwise correlations
        pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
        lcorr = {f"{self.var_names_[a]}-{self.var_names_[b]}": np.zeros(n)
                 for a, b in pairs}

        lmedian = np.zeros((n, k)) if self.quantile else None
        liqr    = np.zeros((n, k)) if self.quantile else None

        for i in range(n):
            w = kern(D[i])
            w_sum = max(w.sum(), 1e-12)
            w_norm = w / w_sum

            # Weighted mean
            mean_i = np.sum(w_norm[:, None] * X, axis=0)
            lmean[i] = mean_i

            # Weighted variance
            diff = X - mean_i
            var_i = np.sum(w_norm[:, None] * diff ** 2, axis=0)
            lvar[i] = var_i
            lstd[i] = np.sqrt(np.maximum(var_i, 0))

            # Weighted skewness and kurtosis
            std_i = lstd[i]
            for j in range(k):
                if std_i[j] > 0:
                    z = diff[:, j] / std_i[j]
                    lskew[i, j] = np.sum(w_norm * z ** 3)
                    lkurt[i, j] = np.sum(w_norm * z ** 4) - 3.0

            # Weighted Pearson correlations
            for a, b in pairs:
                key = f"{self.var_names_[a]}-{self.var_names_[b]}"
                sa, sb = std_i[a], std_i[b]
                if sa > 0 and sb > 0:
                    cov_ab = np.sum(w_norm * diff[:, a] * diff[:, b])
                    lcorr[key][i] = cov_ab / (sa * sb)

            # Local quantile statistics
            if self.quantile:
                order = np.argsort(w)[::-1]
                cumw = np.cumsum(w[order]) / w_sum
                for j in range(k):
                    xj_sorted = X[order, j]
                    med_idx = np.searchsorted(cumw, 0.5)
                    q25_idx = np.searchsorted(cumw, 0.25)
                    q75_idx = np.searchsorted(cumw, 0.75)
                    lmedian[i, j] = xj_sorted[min(med_idx, n - 1)]
                    liqr[i, j] = (xj_sorted[min(q75_idx, n - 1)] -
                                  xj_sorted[min(q25_idx, n - 1)])

        self.local_mean_ = lmean
        self.local_var_  = lvar
        self.local_std_  = lstd
        self.local_skew_ = lskew
        self.local_kurt_ = lkurt
        self.local_corr_ = lcorr
        if self.quantile:
            self.local_median_ = lmedian
            self.local_iqr_    = liqr

        return self

    def to_dataframe(self) -> pd.DataFrame:
        """Return local statistics as a DataFrame."""
        data = {}
        for j, nm in enumerate(self.var_names_):
            data[f"mean_{nm}"]  = self.local_mean_[:, j]
            data[f"std_{nm}"]   = self.local_std_[:, j]
            data[f"var_{nm}"]   = self.local_var_[:, j]
            data[f"skew_{nm}"]  = self.local_skew_[:, j]
            data[f"kurt_{nm}"]  = self.local_kurt_[:, j]
        for key, arr in self.local_corr_.items():
            data[f"corr_{key}"] = arr
        if self.quantile and hasattr(self, "local_median_"):
            for j, nm in enumerate(self.var_names_):
                data[f"median_{nm}"] = self.local_median_[:, j]
                data[f"iqr_{nm}"]    = self.local_iqr_[:, j]
        return pd.DataFrame(data)
