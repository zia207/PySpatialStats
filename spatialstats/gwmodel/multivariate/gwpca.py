"""
spatialstats.gwmodel.multivariate.gwpca
=============================
Geographically Weighted Principal Component Analysis (GWPCA).

Computes spatially varying eigenvectors (loadings), eigenvalues,
and scores. Ported from R GWmodel gwpca().
"""

import numpy as np
from typing import Optional

from spatialstats.gwmodel.core.base import _parse_geometry
from spatialstats.gwmodel.core.kernels import SpatialKernel
from spatialstats.gwmodel.utils.distance import pairwise_distances


class GWPCA:
    """
    Geographically Weighted PCA.

    Parameters
    ----------
    n_components : int or None
        Number of components. If None, all retained.
    bandwidth : float or int
    fixed : bool
    kernel : str
    robust : bool
        If True, use median-based (robust) local covariance.

    Attributes (after fit)
    ----------------------
    local_loadings_    : ndarray (n, p, k)  Spatially varying eigenvectors.
    local_eigenvalues_ : ndarray (n, k)
    local_scores_      : ndarray (n, k)
    local_pve_         : ndarray (n, k)  Proportion of variance explained.

    Examples
    --------
    >>> from spatialstats.gwmodel.multivariate import GWPCA
    >>> import numpy as np
    >>> coords = np.random.rand(50, 2) * 100
    >>> X = np.random.rand(50, 4)
    >>> gwpca = GWPCA(n_components=2, bandwidth=20, fixed=True)
    >>> gwpca.fit(X, coords)
    >>> print(gwpca.local_loadings_.shape)
    (50, 4, 2)
    """

    def __init__(self, n_components=None, bandwidth=None,
                 fixed=False, kernel="bisquare", robust=False):
        self.n_components = n_components
        self.bandwidth = bandwidth
        self.fixed = fixed
        self.kernel = kernel
        self.robust = robust

    def fit(self, X, geometry):
        X = np.asarray(X, dtype=float)
        n, p = X.shape
        k = self.n_components or p
        k = min(k, p)
        coords = _parse_geometry(geometry)

        self.coords_ = coords
        D = pairwise_distances(coords)
        kern = SpatialKernel(self.kernel, self.fixed, self.bandwidth or n // 3)

        loadings = np.zeros((n, p, k))
        eigenvalues = np.zeros((n, k))
        scores = np.zeros((n, k))
        pve = np.zeros((n, k))

        for i in range(n):
            w = kern(D[i])
            w_norm = w / max(w.sum(), 1e-12)

            # Weighted mean centring
            mean_i = np.average(X, axis=0, weights=w_norm)
            Xc = X - mean_i

            # Weighted covariance
            if self.robust:
                # Median-based robust covariance
                med_i = np.median(X, axis=0)
                Xc = X - med_i

            Wmat = np.diag(w_norm)
            cov_i = Xc.T @ Wmat @ Xc

            # Eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(cov_i)
            # Sort descending
            order = np.argsort(eigvals)[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]

            loadings[i] = eigvecs[:, :k]
            eigenvalues[i] = eigvals[:k]
            scores[i] = (Xc[i] @ eigvecs[:, :k])
            total_var = max(eigvals.sum(), 1e-12)
            pve[i] = eigvals[:k] / total_var

        self.local_loadings_ = loadings
        self.local_eigenvalues_ = eigenvalues
        self.local_scores_ = scores
        self.local_pve_ = pve
        return self
