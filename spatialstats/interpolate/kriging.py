"""
spatialstats.interpolate.kriging
================================
Variogram estimation / modelling and ordinary kriging with prediction variance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import linalg, optimize
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

from spatialstats.interpolate.base import Interpolator, as_xy


# ---------------------------------------------------------------------------
# Variogram models (range = practical range where the model reaches the sill)
# ---------------------------------------------------------------------------

def _spherical(h, psill, rng):
    r = np.minimum(h / rng, 1.0)
    return psill * (1.5 * r - 0.5 * r ** 3)


def _exponential(h, psill, rng):
    return psill * (1.0 - np.exp(-3.0 * h / rng))


def _gaussian(h, psill, rng):
    return psill * (1.0 - np.exp(-3.0 * (h / rng) ** 2))


_MODELS = {"spherical": _spherical, "exponential": _exponential, "gaussian": _gaussian}


@dataclass
class VariogramModel:
    """
    Parametric semivariogram ``γ(h) = nugget + psill · g(h / range)`` for ``h > 0`` (γ(0) = 0).

    ``range`` is the *practical* range (distance at which the spherical model reaches
    the sill, and the exponential/Gaussian models reach 95% of it).
    """
    model: str = "spherical"
    nugget: float = 0.0
    psill: float = 1.0
    range: float = 1.0

    def __post_init__(self):
        if self.model not in _MODELS:
            raise ValueError(f"model must be one of {list(_MODELS)}")
        if self.nugget < 0 or self.psill < 0 or self.range <= 0:
            raise ValueError("nugget and psill must be >= 0 and range > 0")

    @property
    def sill(self) -> float:
        return self.nugget + self.psill

    def gamma(self, h) -> np.ndarray:
        h = np.asarray(h, dtype=float)
        g = self.nugget + _MODELS[self.model](h, self.psill, self.range)
        return np.where(h > 0, g, 0.0)

    __call__ = gamma

    def plot(self, empirical: Optional[pd.DataFrame] = None, ax=None, hmax: Optional[float] = None):
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots()
        if empirical is not None:
            ax.plot(empirical["lag"], empirical["gamma"], "o", label="empirical")
        hmax = hmax or (empirical["lag"].max() if empirical is not None else 1.5 * self.range)
        h = np.linspace(0, hmax, 200)
        ax.plot(h, self.gamma(h), label=f"{self.model} (nugget={self.nugget:.3g}, "
                                         f"psill={self.psill:.3g}, range={self.range:.3g})")
        ax.set_xlabel("distance")
        ax.set_ylabel("semivariance")
        ax.legend()
        return ax


def empirical_variogram(coords, values, n_lags: int = 15, max_dist: Optional[float] = None,
                        min_pairs: int = 10) -> pd.DataFrame:
    """
    Matheron's classical estimator ``γ̂(h) = 1/(2 N(h)) Σ (z_i - z_j)²`` in equal-width distance bins.

    ``max_dist`` defaults to half the largest pairwise distance (the usual guideline).
    Returns a DataFrame with ``lag`` (mean pair distance in the bin), ``gamma`` and ``n_pairs``;
    bins with fewer than ``min_pairs`` pairs are dropped.
    """
    xy = as_xy(coords)
    z = np.asarray(values, dtype=float).ravel()
    d = pdist(xy)
    diff2 = pdist(z[:, None], metric="sqeuclidean")
    if max_dist is None:
        max_dist = 0.5 * d.max()
    edges = np.linspace(0, max_dist, n_lags + 1)
    which = np.digitize(d, edges) - 1
    rows = []
    for b in range(n_lags):
        m = which == b
        if m.sum() >= min_pairs:
            rows.append((d[m].mean(), 0.5 * diff2[m].mean(), int(m.sum())))
    if len(rows) < 3:
        raise ValueError("too few populated distance bins to describe a variogram; use fewer lags")
    return pd.DataFrame(rows, columns=["lag", "gamma", "n_pairs"])


def fit_variogram(empirical: pd.DataFrame, model: str = "spherical", fit_nugget: bool = True) -> VariogramModel:
    """
    Weighted least-squares fit of a variogram model to an empirical variogram
    (Cressie's weights ``N(h) / γ_model(h)²``, which favour short lags with many pairs).
    """
    if model not in _MODELS:
        raise ValueError(f"model must be one of {list(_MODELS)}")
    h = empirical["lag"].to_numpy()
    g = empirical["gamma"].to_numpy()
    n = empirical["n_pairs"].to_numpy(dtype=float)
    sill0 = float(np.max(g))
    range0 = float(h.max() * 0.6)
    x0 = [0.1 * sill0 if fit_nugget else 0.0, 0.9 * sill0, range0]
    lo = [0.0, 1e-12, h.min() * 0.1]
    hi = [sill0 * 2 + 1e-9, sill0 * 5 + 1e-9, h.max() * 3]

    def resid(p):
        nug, ps, rg = (p[0] if fit_nugget else 0.0), p[1], p[2]
        gm = nug + _MODELS[model](h, ps, rg)
        return np.sqrt(n) * (g - gm) / np.maximum(gm, 1e-12)

    res = optimize.least_squares(resid, x0, bounds=(lo, hi))
    nug = float(res.x[0]) if fit_nugget else 0.0
    return VariogramModel(model=model, nugget=nug, psill=float(res.x[1]), range=float(res.x[2]))


def fit_variogram_auto(coords, values, models=("spherical", "exponential", "gaussian"),
                       n_lags: int = 15, max_dist: Optional[float] = None) -> Tuple[VariogramModel, pd.DataFrame]:
    """Fit each candidate model and keep the one with the smallest weighted residual sum of squares."""
    emp = empirical_variogram(coords, values, n_lags=n_lags, max_dist=max_dist)
    best, best_sse = None, np.inf
    for m in models:
        vm = fit_variogram(emp, m)
        gm = vm.gamma(emp["lag"].to_numpy())
        sse = float(np.sum(emp["n_pairs"] * ((emp["gamma"] - gm) / np.maximum(gm, 1e-12)) ** 2))
        if sse < best_sse:
            best, best_sse = vm, sse
    return best, emp


# ---------------------------------------------------------------------------
# Ordinary kriging
# ---------------------------------------------------------------------------

class OrdinaryKriging(Interpolator):
    """
    Ordinary kriging: the best linear unbiased predictor under an unknown constant mean,
    with the kriging (prediction) variance.

    ::

        ẑ(s0) = Σ λ_i z_i,   Σ λ_i = 1,   λ = Γ⁻¹ γ0   (with Lagrange multiplier)
        σ²_K(s0) = Σ λ_i γ(s_i, s0) + μ

    Unlike IDW the weights come from the *estimated spatial correlation structure* (the
    variogram), so clustered observations are down-weighted and a prediction variance is
    available; the variance depends only on the sampling configuration and the variogram,
    not on the observed values.

    Parameters
    ----------
    variogram : VariogramModel, or {'auto', 'spherical', 'exponential', 'gaussian'}
        A fitted model, or a model name / 'auto' to estimate it from the data in ``fit``.
    n_neighbors : int or None
        Local kriging with the nearest ``n_neighbors`` observations (recommended above ~500
        observations); None uses all observations (one factorisation, fastest for small n).
    n_lags : int
        Number of distance bins when the variogram is estimated automatically.
    """

    name = "ordinary kriging"

    def __init__(self, variogram: Union[VariogramModel, str] = "auto", n_neighbors: Optional[int] = None,
                 n_lags: int = 15):
        self.variogram, self.n_neighbors, self.n_lags = variogram, n_neighbors, n_lags

    def _fit(self, xy, v):
        vg = self.variogram
        if isinstance(vg, str):
            if vg == "auto":
                vg, emp = fit_variogram_auto(xy, v, n_lags=self.n_lags)
            else:
                emp = empirical_variogram(xy, v, n_lags=self.n_lags)
                vg = fit_variogram(emp, vg)
            self.empirical_ = emp
        self.variogram_ = vg
        self._tree = cKDTree(xy)
        if self.n_neighbors is None:
            n = len(xy)
            G = np.zeros((n + 1, n + 1))
            G[:n, :n] = vg.gamma(np.linalg.norm(xy[:, None] - xy[None], axis=2))
            G[:n, n] = G[n, :n] = 1.0
            self._lu = linalg.lu_factor(G)

    def _predict(self, xy):
        mean, _ = self._krige(xy)
        return mean

    def predict(self, coords, return_std: bool = False):
        """Predictions; with ``return_std=True`` also the kriging standard deviation."""
        if not hasattr(self, "coords_"):
            raise RuntimeError("call fit() first")
        mean, var = self._krige(as_xy(coords))
        return (mean, np.sqrt(np.maximum(var, 0.0))) if return_std else mean

    def _krige(self, xy):
        vg, X, z = self.variogram_, self.coords_, self.values_
        n = len(X)
        if self.n_neighbors is None or self.n_neighbors >= n:
            g0 = vg.gamma(np.linalg.norm(X[:, None] - xy[None], axis=2))       # (n, m)
            if self.n_neighbors is None:
                sol = linalg.lu_solve(self._lu, np.vstack([g0, np.ones((1, g0.shape[1]))]))
            else:
                G = np.zeros((n + 1, n + 1))
                G[:n, :n] = vg.gamma(np.linalg.norm(X[:, None] - X[None], axis=2))
                G[:n, n] = G[n, :n] = 1.0
                sol = np.linalg.solve(G, np.vstack([g0, np.ones((1, g0.shape[1]))]))
            lam, mu = sol[:n], sol[n]
            return lam.T @ z, np.sum(lam * g0, axis=0) + mu
        k = self.n_neighbors
        _, nbr = self._tree.query(xy, k=k)
        mean = np.empty(len(xy))
        var = np.empty(len(xy))
        for i in range(len(xy)):
            idx = nbr[i]
            Xi = X[idx]
            G = np.zeros((k + 1, k + 1))
            G[:k, :k] = vg.gamma(np.linalg.norm(Xi[:, None] - Xi[None], axis=2))
            G[:k, k] = G[k, :k] = 1.0
            g0 = np.append(vg.gamma(np.linalg.norm(Xi - xy[i], axis=1)), 1.0)
            sol = np.linalg.solve(G, g0)
            mean[i] = sol[:k] @ z[idx]
            var[i] = sol[:k] @ g0[:k] + sol[k]
        return mean, var

    def predict_grid(self, region, resolution: Optional[float] = None, n_cells: int = 100,
                     return_std: bool = False):
        from spatialstats.interpolate.base import make_grid
        grid = make_grid(region, resolution, n_cells)
        out = self.predict(grid.coords, return_std=return_std)
        return (grid, *out) if return_std else (grid, out)
