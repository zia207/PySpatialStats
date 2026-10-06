"""
spatialstats.interpolate.kriging
================================
Variogram estimation and kriging with prediction variance.

* ordinary, simple and universal kriging
* ordinary cokriging (one or several secondaries) and colocated cokriging
* regression kriging: a trend model plus ordinary kriging of the residuals
* indicator variograms, indicator kriging and E-type estimates
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Optional, Sequence, Tuple, Union

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


def indicator_transform(values, threshold: float, greater_equal: bool = True) -> np.ndarray:
    """
    Code a continuous sample as an indicator.

    ``greater_equal=True`` (the default) is ``I(Z ≥ threshold)``, the survival
    indicator that indicator kriging turns into ``P(Z ≥ threshold)``.
    ``greater_equal=False`` is ``I(Z ≤ threshold)``. The two codings sum to 1,
    so they share one variogram.
    """
    v = np.asarray(values, dtype=float).ravel()
    if greater_equal:
        return (v >= threshold).astype(float)
    return (v <= threshold).astype(float)


def empirical_indicator_variogram(coords, values, threshold: float, n_lags: int = 15,
                                  max_dist: Optional[float] = None, min_pairs: int = 10,
                                  greater_equal: bool = True) -> pd.DataFrame:
    """
    Empirical variogram of the indicator at ``threshold``.

    Same Matheron estimator and distance bins as :func:`empirical_variogram`,
    applied to ``I(Z ≥ threshold)`` (or ``I(Z ≤ threshold)`` when
    ``greater_equal`` is False). Because ``(I − I')² = ((1−I) − (1−I'))²``,
    the two codings produce the same variogram. Indicator semivariances lie
    in ``[0, 1/2]``; the sill of a stationary indicator is ``p(1−p) ≤ 1/4``.
    """
    ind = indicator_transform(values, threshold, greater_equal=greater_equal)
    if np.unique(ind).size < 2:
        raise ValueError(
            "the indicator is constant at this threshold, so the variogram is identically zero"
        )
    return empirical_variogram(coords, ind, n_lags=n_lags, max_dist=max_dist, min_pairs=min_pairs)


def fit_indicator_variogram(coords, values, threshold: float, model: str = "spherical",
                            n_lags: int = 15, max_dist: Optional[float] = None,
                            greater_equal: bool = True,
                            models: Sequence[str] = ("spherical", "exponential", "gaussian")):
    """
    Fit a variogram model to the indicator at ``threshold``.

    Returns ``(VariogramModel, empirical DataFrame)``. ``model='auto'`` tries
    each name in ``models`` and keeps the weighted-least-squares winner, as
    :func:`fit_variogram_auto` does for a continuous variable. ``range`` is the
    practical range used by :class:`VariogramModel`.
    """
    emp = empirical_indicator_variogram(
        coords, values, threshold, n_lags=n_lags, max_dist=max_dist, greater_equal=greater_equal,
    )
    if model == "auto":
        best, best_sse = None, np.inf
        for m in models:
            vm = fit_variogram(emp, m)
            gm = vm.gamma(emp["lag"].to_numpy())
            sse = float(np.sum(emp["n_pairs"] * ((emp["gamma"] - gm) / np.maximum(gm, 1e-12)) ** 2))
            if sse < best_sse:
                best, best_sse = vm, sse
        return best, emp
    return fit_variogram(emp, model), emp


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


# ---------------------------------------------------------------------------
# Shared linear algebra (covariance form used by simple / universal / co-kriging)
# ---------------------------------------------------------------------------

def _c0(model) -> float:
    """Variance at the origin, C(0) = nugget + partial sill."""
    return float(model.nugget) + float(model.psill)


def _cov(model, dist) -> np.ndarray:
    """Covariance C(h) = C(0) − γ(h).

    ``gamma(0)`` is 0, so a pair of coincident locations gets C(0) on the
    diagonal and kriging reproduces the observation.
    """
    return _c0(model) - model.gamma(dist)


def _dist(a, b) -> np.ndarray:
    """Euclidean distances between the rows of ``a`` and the rows of ``b``."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)


def _solve(a, b):
    """Solve ``a x = b``, falling back to least squares if ``a`` is singular."""
    try:
        return linalg.solve(a, b)
    except linalg.LinAlgError:
        return linalg.lstsq(a, b)[0]


def _factor(a):
    """LU factorisation, ridging a singular covariance diagonal.

    Structural zeros (the Lagrange block of a kriging matrix) are left alone,
    so the unbiasedness constraints stay exact.
    """
    try:
        return linalg.lu_factor(a)
    except linalg.LinAlgError:
        diag = np.diag(a)
        scale = float(np.nanmax(np.abs(diag)))
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        bumped = np.array(a, dtype=float, copy=True)
        idx = np.flatnonzero(np.abs(diag) > 0)
        bumped[idx, idx] += 1e-10 * scale
        return linalg.lu_factor(bumped)


def _neighbors(tree: cKDTree, xy, k: int) -> np.ndarray:
    k = min(int(k), int(tree.n))
    _, nbr = tree.query(np.asarray(xy, dtype=float), k=k)
    nbr = np.asarray(nbr, dtype=int)
    if k == 1:
        nbr = nbr.reshape(-1, 1)
    return nbr


def _resolve_variogram(variogram, xy, values, n_lags: int,
                       max_dist: Optional[float] = None) -> Tuple[VariogramModel, Optional[pd.DataFrame]]:
    """A fitted ``VariogramModel`` from a model, a model name, or ``'auto'``."""
    if isinstance(variogram, VariogramModel):
        return variogram, None
    if not isinstance(variogram, str):
        raise TypeError(
            "variogram must be a VariogramModel or one of "
            "'auto', 'spherical', 'exponential', 'gaussian'"
        )
    if variogram == "auto":
        return fit_variogram_auto(xy, values, n_lags=n_lags, max_dist=max_dist)
    emp = empirical_variogram(xy, values, n_lags=n_lags, max_dist=max_dist)
    return fit_variogram(emp, variogram), emp


def _xy_values(coords, values, what: str, min_n: int = 1):
    xy = as_xy(coords)
    v = np.asarray(values, dtype=float).ravel()
    if len(xy) != len(v):
        raise ValueError(f"{what}: {len(xy)} coordinates but {len(v)} values")
    keep = np.isfinite(v) & np.isfinite(xy).all(axis=1)
    xy, v = xy[keep], v[keep]
    if len(v) < min_n:
        raise ValueError(f"{what}: need at least {min_n} finite observations")
    return xy, v


def _n_drift(degree: int) -> int:
    """Number of monomials in a 2-D polynomial of total degree ``degree``."""
    return (degree + 1) * (degree + 2) // 2


def _drift(xy, degree: int, center, scale) -> np.ndarray:
    """Polynomial drift, centered and scaled so the columns stay O(1).

    The column space is the same as the unscaled monomials of total degree
    ``degree`` (the constant column is included).
    """
    u = (np.asarray(xy, dtype=float) - center) / scale
    x, y = u[:, 0], u[:, 1]
    cols = [np.ones(len(u))]
    for total in range(1, int(degree) + 1):
        for j in range(total + 1):
            i = total - j
            cols.append((x ** i) * (y ** j))
    return np.column_stack(cols)


def _grid_predict(model, region, resolution, n_cells, return_std):
    from spatialstats.interpolate.base import make_grid
    grid = make_grid(region, resolution, n_cells)
    out = model.predict(grid.coords, return_std=return_std)
    return (grid, *out) if return_std else (grid, out)


def _returned(mean, var, return_std: bool):
    if return_std:
        return mean, np.sqrt(np.maximum(var, 0.0))
    return mean


# ---------------------------------------------------------------------------
# Simple kriging
# ---------------------------------------------------------------------------

class SimpleKriging(Interpolator):
    """
    Simple kriging: the best linear predictor when the mean is known.

    ::

        ẑ(s0) = m + Σ λ_i (z_i − m),   C λ = c
        σ²_SK(s0) = C(0) − Σ λ_i C(s_i, s0)

    There is no unbiasedness constraint, so the weights need not sum to 1.
    Beyond the variogram range the covariances vanish and the prediction
    falls back to ``m`` with variance C(0). With ``mean=None`` the mean is
    the sample average of the fitted values — a fixed number, not the
    locally estimated mean of ordinary kriging.

    The predictor is exact at the sample locations (the kriging variance is
    zero there), including when the variogram has a nugget.

    Parameters
    ----------
    variogram : VariogramModel, or {'auto', 'spherical', 'exponential', 'gaussian'}
        A fitted model, or a model name / ``'auto'`` to estimate it in ``fit``.
    mean : float or None
        Known mean. ``None`` uses the mean of the observations passed to ``fit``.
    n_neighbors : int or None
        Local simple kriging using the nearest observations. ``None`` uses
        every observation (one factorisation).
    n_lags : int
        Distance bins when the variogram is estimated from the data.
    """

    name = "simple kriging"

    def __init__(self, variogram: Union[VariogramModel, str] = "auto", mean: Optional[float] = None,
                 n_neighbors: Optional[int] = None, n_lags: int = 15):
        self.variogram, self.mean, self.n_neighbors, self.n_lags = variogram, mean, n_neighbors, n_lags

    def _fit(self, xy, v):
        vg, emp = _resolve_variogram(self.variogram, xy, v, self.n_lags)
        self.variogram_ = vg
        if emp is not None:
            self.empirical_ = emp
        if self.mean is None:
            self.mean_ = float(v.mean())
        else:
            self.mean_ = float(self.mean)
            if not np.isfinite(self.mean_):
                raise ValueError("mean must be finite")
        self._tree = cKDTree(xy)
        if self.n_neighbors is None:
            self._lu = _factor(_cov(vg, _dist(xy, xy)))

    def _predict(self, xy):
        mean, _ = self._krige(xy)
        return mean

    def predict(self, coords, return_std: bool = False):
        """Predictions; with ``return_std=True`` also the kriging standard deviation."""
        if not hasattr(self, "coords_"):
            raise RuntimeError("call fit() first")
        mean, var = self._krige(as_xy(coords))
        return _returned(mean, var, return_std)

    def _krige(self, xy):
        vg, X, z = self.variogram_, self.coords_, self.values_
        m = self.mean_
        n = len(X)
        c0 = _c0(vg)
        if self.n_neighbors is None or self.n_neighbors >= n:
            c = _cov(vg, _dist(xy, X))                                          # (m, n)
            if self.n_neighbors is None:
                lam = linalg.lu_solve(self._lu, c.T).T
            else:
                lam = _solve(_cov(vg, _dist(X, X)), c.T).T
            return m + lam @ (z - m), c0 - np.sum(lam * c, axis=1)
        k = self.n_neighbors
        nbr = _neighbors(self._tree, xy, k)
        mean = np.empty(len(xy))
        var = np.empty(len(xy))
        for i in range(len(xy)):
            idx = nbr[i]
            Xi = X[idx]
            c = _cov(vg, _dist(xy[i:i + 1], Xi)).ravel()
            lam = _solve(_cov(vg, _dist(Xi, Xi)), c)
            mean[i] = m + lam @ (z[idx] - m)
            var[i] = c0 - lam @ c
        return mean, var

    def predict_grid(self, region, resolution: Optional[float] = None, n_cells: int = 100,
                     return_std: bool = False):
        return _grid_predict(self, region, resolution, n_cells, return_std)


# ---------------------------------------------------------------------------
# Universal kriging
# ---------------------------------------------------------------------------

class UniversalKriging(Interpolator):
    """
    Universal kriging: ordinary kriging with a polynomial drift in the coordinates.

    The unknown mean is a polynomial of total degree ``degree``
    (``degree=0`` is a constant, and then the predictor matches ordinary
    kriging). Coordinates are centered at the sample mean and scaled by
    their standard deviation before the monomials are built (``center_``,
    ``scale_``); that is the same column space as the raw polynomial and
    keeps the kriging matrix well conditioned in projected coordinates.

    ::

        [ C  F ] [λ]   [ c ]
        [ Fᵀ 0 ] [μ] = [ f ]
        ẑ(s0) = Σ λ_i z_i
        σ²_UK(s0) = C(0) − Σ λ_i C(s_i, s0) − Σ μ_k f_k(s0)

    ``F`` holds the drift at the samples and ``f`` the drift at the
    prediction location. The predictor is exact at the sample locations.
    Where the data really do lie on a polynomial of this degree, the
    surface equals that polynomial everywhere, not only at the samples.

    Parameters
    ----------
    variogram : VariogramModel, or {'auto', 'spherical', 'exponential', 'gaussian'}
    degree : int
        Total degree of the drift. 1 is a plane (1, x, y).
    n_neighbors : int or None
        Local universal kriging. The neighbourhood has to be larger than
        the number of drift terms.
    n_lags : int
        Distance bins when the variogram is estimated from the data.
    """

    name = "universal kriging"

    def __init__(self, variogram: Union[VariogramModel, str] = "auto", degree: int = 1,
                 n_neighbors: Optional[int] = None, n_lags: int = 15):
        if isinstance(degree, bool) or not isinstance(degree, (int, np.integer)) or int(degree) < 0:
            raise ValueError("degree must be an integer >= 0")
        self.variogram = variogram
        self.degree = int(degree)
        self.n_neighbors, self.n_lags = n_neighbors, n_lags

    def _fit(self, xy, v):
        n_terms = _n_drift(self.degree)
        if len(v) <= n_terms:
            raise ValueError(
                f"universal kriging of degree {self.degree} needs more than {n_terms} observations"
            )
        if (self.n_neighbors is not None and self.n_neighbors < len(xy)
                and self.n_neighbors <= n_terms):
            raise ValueError(
                f"n_neighbors must exceed the {n_terms} drift terms"
            )
        vg, emp = _resolve_variogram(self.variogram, xy, v, self.n_lags)
        self.variogram_ = vg
        if emp is not None:
            self.empirical_ = emp
        self.center_ = xy.mean(axis=0)
        scale = xy.std(axis=0)
        scale[scale < 1e-12] = 1.0
        self.scale_ = scale
        self.F_ = _drift(xy, self.degree, self.center_, self.scale_)
        self._tree = cKDTree(xy)
        if self.n_neighbors is None:
            n, p = len(xy), n_terms
            K = np.zeros((n + p, n + p))
            K[:n, :n] = _cov(vg, _dist(xy, xy))
            K[:n, n:] = self.F_
            K[n:, :n] = self.F_.T
            self._lu = _factor(K)

    def _predict(self, xy):
        mean, _ = self._krige(xy)
        return mean

    def predict(self, coords, return_std: bool = False):
        """Predictions; with ``return_std=True`` also the kriging standard deviation."""
        if not hasattr(self, "coords_"):
            raise RuntimeError("call fit() first")
        mean, var = self._krige(as_xy(coords))
        return _returned(mean, var, return_std)

    def _krige(self, xy):
        vg, X, z = self.variogram_, self.coords_, self.values_
        n = len(X)
        p = self.F_.shape[1]
        c0 = _c0(vg)
        if self.n_neighbors is None or self.n_neighbors >= n:
            c = _cov(vg, _dist(xy, X))                                          # (m, n)
            F0 = _drift(xy, self.degree, self.center_, self.scale_)             # (m, p)
            aug = np.hstack([c, F0])
            if self.n_neighbors is None:
                sol = linalg.lu_solve(self._lu, aug.T).T
            else:
                K = np.zeros((n + p, n + p))
                K[:n, :n] = _cov(vg, _dist(X, X))
                K[:n, n:] = self.F_
                K[n:, :n] = self.F_.T
                sol = _solve(K, aug.T).T
            lam, mu = sol[:, :n], sol[:, n:]
            var = c0 - np.sum(lam * c, axis=1) - np.sum(mu * F0, axis=1)
            return lam @ z, var
        k = self.n_neighbors
        nbr = _neighbors(self._tree, xy, k)
        mean = np.empty(len(xy))
        var = np.empty(len(xy))
        for i in range(len(xy)):
            idx = nbr[i]
            Xi = X[idx]
            Fi = self.F_[idx]
            kk = len(idx)
            K = np.zeros((kk + p, kk + p))
            K[:kk, :kk] = _cov(vg, _dist(Xi, Xi))
            K[:kk, kk:] = Fi
            K[kk:, :kk] = Fi.T
            c = _cov(vg, _dist(xy[i:i + 1], Xi)).ravel()
            f0 = _drift(xy[i:i + 1], self.degree, self.center_, self.scale_).ravel()
            sol = _solve(K, np.concatenate([c, f0]))
            mean[i] = sol[:kk] @ z[idx]
            var[i] = c0 - sol[:kk] @ c - sol[kk:] @ f0
        return mean, var

    def predict_grid(self, region, resolution: Optional[float] = None, n_cells: int = 100,
                     return_std: bool = False):
        return _grid_predict(self, region, resolution, n_cells, return_std)


# ---------------------------------------------------------------------------
# Cross-variograms and the linear model of coregionalization
# ---------------------------------------------------------------------------

@dataclass
class CrossVariogramModel:
    """
    Cross-semivariogram ``γ₁₂(h) = nugget + psill · g(h / range)`` for ``h > 0``.

    Unlike :class:`VariogramModel`, the nugget and the partial cross-sill
    ``psill`` (often written c₁₂) may be negative. ``range`` is the practical
    range, shared with the marginal models under a linear model of
    coregionalization. C₁₂(0) = nugget + psill.
    """
    model: str = "spherical"
    nugget: float = 0.0
    psill: float = 1.0
    range: float = 1.0

    def __post_init__(self):
        if self.model not in _MODELS:
            raise ValueError(f"model must be one of {list(_MODELS)}")
        if not np.isfinite(self.nugget) or not np.isfinite(self.psill) or self.range <= 0:
            raise ValueError("nugget and psill must be finite and range > 0")

    @property
    def sill(self) -> float:
        """C₁₂(0), the lag-zero cross-covariance."""
        return self.nugget + self.psill

    def gamma(self, h) -> np.ndarray:
        h = np.asarray(h, dtype=float)
        g = self.nugget + _MODELS[self.model](h, self.psill, self.range)
        return np.where(h > 0, g, 0.0)

    __call__ = gamma


def empirical_cross_variogram(coords, primary, secondary, n_lags: int = 15,
                              max_dist: Optional[float] = None, min_pairs: int = 10) -> pd.DataFrame:
    """
    Cross-semivariogram ``γ̂₁₂(h) = 1/(2 N(h)) Σ (z₁ᵢ − z₁ⱼ)(z₂ᵢ − z₂ⱼ)``.

    The two variables must be measured at the same locations. Binning matches
    :func:`empirical_variogram` (equal-width bins, ``max_dist`` defaults to
    half the largest pairwise distance). The cross-semivariance may be negative.
    """
    xy = as_xy(coords)
    z1 = np.asarray(primary, dtype=float).ravel()
    z2 = np.asarray(secondary, dtype=float).ravel()
    if not (len(xy) == len(z1) == len(z2)):
        raise ValueError("coords, primary and secondary must have the same length")
    keep = np.isfinite(z1) & np.isfinite(z2) & np.isfinite(xy).all(axis=1)
    xy, z1, z2 = xy[keep], z1[keep], z2[keep]
    if len(z1) < 2:
        raise ValueError("need at least 2 finite co-located pairs")
    iu, ju = np.triu_indices(len(xy), k=1)
    d = np.linalg.norm(xy[iu] - xy[ju], axis=1)
    cross = 0.5 * (z1[iu] - z1[ju]) * (z2[iu] - z2[ju])
    if max_dist is None:
        max_dist = 0.5 * float(d.max())
    edges = np.linspace(0, max_dist, n_lags + 1)
    which = np.digitize(d, edges) - 1
    rows = []
    for b in range(n_lags):
        m = which == b
        if m.sum() >= min_pairs:
            rows.append((d[m].mean(), cross[m].mean(), int(m.sum())))
    if len(rows) < 3:
        raise ValueError("too few populated distance bins to describe a cross-variogram")
    return pd.DataFrame(rows, columns=["lag", "gamma", "n_pairs"])


def fit_cross_variogram(empirical: pd.DataFrame, model: str = "spherical",
                        range_: Optional[float] = None, cov0: Optional[float] = None) -> CrossVariogramModel:
    """
    Least-squares fit of a cross-variogram. ``psill`` is unbounded, so a
    negative cross-correlation is allowed.

    ``range_`` locks the range (the shared-range step of an LMC fit).
    ``cov0`` is the initial partial cross-sill; it defaults to the empirical
    lag with the largest absolute semivariance.
    """
    if model not in _MODELS:
        raise ValueError(f"model must be one of {list(_MODELS)}")
    h = empirical["lag"].to_numpy(dtype=float)
    g = empirical["gamma"].to_numpy(dtype=float)
    ok = np.isfinite(h) & np.isfinite(g)
    h, g = h[ok], g[ok]
    if len(h) < 3:
        raise ValueError("need at least 3 finite empirical lags")
    if cov0 is None:
        cov0 = float(g[np.argmax(np.abs(g))])
    if range_ is None:
        range0 = float(h.max() * 0.6)

        def resid(p):
            return g - (p[0] + _MODELS[model](h, p[1], p[2]))

        lo = [-np.inf, -np.inf, max(float(h.min()) * 0.1, 1e-6)]
        hi = [np.inf, np.inf, float(h.max()) * 3]
        res = optimize.least_squares(resid, [0.0, float(cov0), range0], bounds=(lo, hi))
        nug, ps, rg = (float(x) for x in res.x)
    else:
        rg = float(range_)
        if rg <= 0:
            raise ValueError("range_ must be > 0")

        def resid(p, _rg=rg):
            return g - (p[0] + _MODELS[model](h, p[1], _rg))

        res = optimize.least_squares(resid, [0.0, float(cov0)])
        nug, ps = float(res.x[0]), float(res.x[1])
    return CrossVariogramModel(model=model, nugget=nug, psill=ps, range=rg)


def check_lmc_validity(sill_matrix, nugget_matrix=None, tol: float = 1e-10) -> dict:
    """
    Positive-semidefinite check for the coregionalization matrices of an LMC.

    Returns ``valid``, ``sill_eigs`` and ``nugget_eigs`` (the last is ``None``
    when ``nugget_matrix`` is omitted). Eigenvalues above ``-tol`` count as
    non-negative.
    """
    B1 = np.asarray(sill_matrix, dtype=float)
    B1 = 0.5 * (B1 + B1.T)
    eigs1 = np.linalg.eigvalsh(B1)
    ok = bool(np.all(eigs1 >= -tol))
    eigs0 = None
    if nugget_matrix is not None:
        B0 = np.asarray(nugget_matrix, dtype=float)
        B0 = 0.5 * (B0 + B0.T)
        eigs0 = np.linalg.eigvalsh(B0)
        ok = ok and bool(np.all(eigs0 >= -tol))
    return {"valid": ok, "sill_eigs": eigs1, "nugget_eigs": eigs0}


def _refit_locked_range(emp: pd.DataFrame, model: str, range_: float) -> Tuple[float, float]:
    """Weighted least squares for nugget and partial sill with the range held fixed."""
    h = emp["lag"].to_numpy(dtype=float)
    g = emp["gamma"].to_numpy(dtype=float)
    n = emp["n_pairs"].to_numpy(dtype=float)
    sill0 = max(float(np.max(g)), 0.0)
    hi = [max(sill0 * 2, 1e-6), max(sill0 * 5, 1e-6)]
    x0 = [min(0.1 * sill0, hi[0] / 2), min(max(0.9 * sill0, 1e-8), hi[1] / 2)]

    def resid(p):
        gm = p[0] + _MODELS[model](h, p[1], range_)
        return np.sqrt(n) * (g - gm) / np.maximum(gm, 1e-12)

    res = optimize.least_squares(resid, x0, bounds=([0.0, 1e-12], hi))
    return float(res.x[0]), float(res.x[1])


def fit_lmc(coords, values_list, names: Optional[Sequence[str]] = None, model: str = "spherical",
            n_lags: int = 12, max_dist: Optional[float] = None, shared_range: bool = True):
    """
    Fit auto- and cross-variograms under a one-structure linear model of coregionalization.

    Auto-variograms are fitted first. With ``shared_range=True`` (the default)
    every structure is given the median auto-variogram range and the sills are
    refitted. Partial cross-sills are then clipped by the Cauchy–Schwarz
    bound ``|c₁₂| ≤ √(c₁ c₂)`` so the coregionalization matrices can be
    positive semidefinite.

    The variables must be co-located: ``values_list[k]`` is the k-th variable
    at ``coords``. Heterotopic cokriging can still use the returned models
    with samples that do not share locations.

    Returns
    -------
    variograms : list of VariogramModel
        Index 0 is the first variable.
    cross : dict
        ``{(i, j): CrossVariogramModel}`` for ``i < j``.
    info : dict
        ``names``, ``model``, ``range``, ``sill_matrix``, ``nugget_matrix``,
        ``lmc_check``.
    """
    if model not in _MODELS:
        raise ValueError(f"model must be one of {list(_MODELS)}")
    xy = as_xy(coords)
    values_list = [np.asarray(v, dtype=float).ravel() for v in values_list]
    p = len(values_list)
    if p < 2:
        raise ValueError("need at least two variables")
    if names is None:
        names = [f"Z{i}" for i in range(p)]
    if len(names) != p:
        raise ValueError("names must have one entry per variable")
    if any(len(v) != len(xy) for v in values_list):
        raise ValueError("each variable must be co-located with coords")
    if max_dist is None:
        max_dist = 0.5 * float(pdist(xy).max())

    emps, variograms = [], []
    for vals in values_list:
        emp = empirical_variogram(xy, vals, n_lags=n_lags, max_dist=max_dist)
        emps.append(emp)
        variograms.append(fit_variogram(emp, model))

    common_range = float(np.median([vg.range for vg in variograms]))
    if shared_range:
        refit = []
        for emp in emps:
            nug, ps = _refit_locked_range(emp, model, common_range)
            refit.append(VariogramModel(model=model, nugget=nug, psill=ps, range=common_range))
        variograms = refit

    cross = {}
    for i, j in combinations(range(p), 2):
        emp = empirical_cross_variogram(
            xy, values_list[i], values_list[j], n_lags=n_lags, max_dist=max_dist,
        )
        cov0 = float(np.cov(values_list[i], values_list[j])[0, 1])
        locked = common_range if shared_range else None
        cv = fit_cross_variogram(emp, model=model, range_=locked, cov0=cov0)
        cap = float(np.sqrt(abs(variograms[i].psill * variograms[j].psill)))
        nug_cap = float(np.sqrt(abs(variograms[i].nugget * variograms[j].nugget)))
        rng = common_range if shared_range else cv.range
        cross[(i, j)] = CrossVariogramModel(
            model=model,
            nugget=float(np.clip(cv.nugget, -nug_cap, nug_cap)),
            psill=float(np.clip(cv.psill, -cap, cap)),
            range=float(rng),
        )

    B0 = np.zeros((p, p))
    B1 = np.zeros((p, p))
    for k, vg in enumerate(variograms):
        B0[k, k] = vg.nugget
        B1[k, k] = vg.psill
    for (i, j), cv in cross.items():
        B0[i, j] = B0[j, i] = cv.nugget
        B1[i, j] = B1[j, i] = cv.psill
    info = {
        "names": list(names),
        "model": model,
        "range": common_range if shared_range else None,
        "sill_matrix": B1,
        "nugget_matrix": B0,
        "lmc_check": check_lmc_validity(B1, B0),
    }
    return variograms, cross, info


def _constraints(n_per) -> np.ndarray:
    """Unbiasedness design: 1 on each variable's own samples, one column per variable."""
    n_per = list(n_per)
    n = int(np.sum(n_per))
    F = np.zeros((n, len(n_per)))
    offset = np.cumsum([0, *n_per])
    for k, nk in enumerate(n_per):
        F[offset[k]:offset[k] + nk, k] = 1.0
    return F


def _kriging_matrix(C, F):
    n, p = F.shape
    K = np.zeros((n + p, n + p))
    K[:n, :n] = C
    K[:n, n:] = F
    K[n:, :n] = F.T
    return K


# ---------------------------------------------------------------------------
# Cokriging
# ---------------------------------------------------------------------------

class Cokriging:
    """
    Ordinary cokriging of one primary variable and one secondary.

    The two unbiasedness constraints are Σ λ₁ = 1 on the primary weights and
    Σ λ₂ = 0 on the secondary weights, so the secondary is used only through
    its cross-covariance with the primary. Sampling may be heterotopic: the
    secondary does not have to be measured at the primary locations.

    With a cross-covariance that is identically zero the secondary weights
    vanish and the predictor matches ordinary kriging of the primary. At a
    primary sample location the prediction equals that sample.

    Parameters
    ----------
    primary_var, secondary_var : VariogramModel
        Fitted direct variograms.
    cross_var : CrossVariogramModel
        Fitted cross-variogram. ``psill`` may be negative.
    """

    def __init__(self, primary_var: VariogramModel, secondary_var: VariogramModel,
                 cross_var: CrossVariogramModel):
        self.primary_var = primary_var
        self.secondary_var = secondary_var
        self.cross_var = cross_var

    def fit(self, coords_primary, primary, coords_secondary, secondary) -> "Cokriging":
        """Store the two samples and factor the cokriging system."""
        self.coords_primary_, self.primary_ = _xy_values(coords_primary, primary, "primary", min_n=2)
        self.coords_secondary_, self.secondary_ = _xy_values(
            coords_secondary, secondary, "secondary", min_n=1,
        )
        self.n_primary_ = len(self.primary_)
        self.n_secondary_ = len(self.secondary_)
        self.all_values_ = np.hstack([self.primary_, self.secondary_])
        C = self._covariance()
        F = _constraints([self.n_primary_, self.n_secondary_])
        self._n = self.n_primary_ + self.n_secondary_
        self._lu = _factor(_kriging_matrix(C, F))
        return self

    def _covariance(self) -> np.ndarray:
        n1, n2 = self.n_primary_, self.n_secondary_
        C = np.zeros((n1 + n2, n1 + n2))
        C[:n1, :n1] = _cov(self.primary_var, _dist(self.coords_primary_, self.coords_primary_))
        C[n1:, n1:] = _cov(self.secondary_var, _dist(self.coords_secondary_, self.coords_secondary_))
        Cps = _cov(self.cross_var, _dist(self.coords_primary_, self.coords_secondary_))
        C[:n1, n1:] = Cps
        C[n1:, :n1] = Cps.T
        return C

    def _rhs(self, xy) -> np.ndarray:
        n1 = self.n_primary_
        rhs = np.zeros((len(xy), self._n))
        rhs[:, :n1] = _cov(self.primary_var, _dist(xy, self.coords_primary_))
        rhs[:, n1:] = _cov(self.cross_var, _dist(xy, self.coords_secondary_))
        return rhs

    def predict(self, coords, return_std: bool = False):
        """Predict the primary variable. ``return_std=True`` adds the cokriging standard deviation."""
        if not hasattr(self, "_lu"):
            raise RuntimeError("call fit() first")
        xy = as_xy(coords)
        rhs = self._rhs(xy)
        constraints = np.zeros((len(xy), 2))
        constraints[:, 0] = 1.0
        weights = linalg.lu_solve(self._lu, np.hstack([rhs, constraints]).T).T
        pred = weights[:, :self._n] @ self.all_values_
        if not return_std:
            return pred
        var = _c0(self.primary_var) - np.sum(weights[:, :self._n] * rhs, axis=1) - weights[:, self._n]
        return _returned(pred, var, True)

    def predict_grid(self, region, resolution: Optional[float] = None, n_cells: int = 100,
                     return_std: bool = False):
        return _grid_predict(self, region, resolution, n_cells, return_std)


class MultivariateCokriging:
    """
    Ordinary cokriging of a primary variable with any number of secondaries.

    Each variable may be observed at its own locations. The unbiasedness
    constraints are Σ λ₀ = 1 for the primary (index 0) and Σ λₖ = 0 for
    every secondary. ``cross_variograms`` maps ``(i, j)`` with ``i < j``
    (either order is accepted) to a :class:`CrossVariogramModel`.

    Parameters
    ----------
    variograms : sequence of VariogramModel
        Direct variograms, index 0 = primary.
    cross_variograms : dict
        One cross-variogram for every pair of variables.
    """

    def __init__(self, variograms: Sequence[VariogramModel], cross_variograms: dict):
        self.variograms = list(variograms)
        self.n_var = len(self.variograms)
        if self.n_var < 2:
            raise ValueError("need a primary and at least one secondary")
        self.cross_variograms = {}
        for key, cv in dict(cross_variograms).items():
            i, j = int(key[0]), int(key[1])
            if i == j:
                raise ValueError("a cross-variogram needs two different variables")
            self.cross_variograms[(min(i, j), max(i, j))] = cv
        for i, j in combinations(range(self.n_var), 2):
            if (i, j) not in self.cross_variograms:
                raise ValueError(f"missing cross-variogram for pair {(i, j)}")

    def _cross(self, i, j) -> CrossVariogramModel:
        return self.cross_variograms[(min(i, j), max(i, j))]

    def fit(self, coords_list, values_list) -> "MultivariateCokriging":
        """
        Store one sample per variable.

        ``coords_list[k]`` has shape ``(n_k, 2)`` and ``values_list[k]``
        length ``n_k``. Index 0 is the primary.
        """
        if len(coords_list) != self.n_var or len(values_list) != self.n_var:
            raise ValueError("coords_list and values_list must have one entry per variable")
        coords, values = [], []
        for k, (c, v) in enumerate(zip(coords_list, values_list)):
            xy, vk = _xy_values(c, v, f"variable {k}", min_n=2 if k == 0 else 1)
            coords.append(xy)
            values.append(vk)
        self.coords_list_ = coords
        self.values_list_ = values
        self.n_per_var_ = [len(v) for v in values]
        self.all_values_ = np.hstack(values)
        self._offset = np.cumsum([0, *self.n_per_var_])
        self._n = int(self._offset[-1])
        C = np.zeros((self._n, self._n))
        for i, j in combinations(range(self.n_var), 2):
            block = self._cov_block(i, j)
            a, b = self._offset[i], self._offset[i + 1]
            c, d = self._offset[j], self._offset[j + 1]
            C[a:b, c:d] = block
            C[c:d, a:b] = block.T
        for i in range(self.n_var):
            a, b = self._offset[i], self._offset[i + 1]
            C[a:b, a:b] = self._cov_block(i, i)
        self._lu = _factor(_kriging_matrix(C, _constraints(self.n_per_var_)))
        return self

    def _cov_block(self, i, j) -> np.ndarray:
        d = _dist(self.coords_list_[i], self.coords_list_[j])
        model = self.variograms[i] if i == j else self._cross(i, j)
        return _cov(model, d)

    def predict(self, coords, return_std: bool = False):
        """Predict the primary variable."""
        if not hasattr(self, "_lu"):
            raise RuntimeError("call fit() first")
        xy = as_xy(coords)
        rhs = np.zeros((len(xy), self._n))
        a, b = self._offset[0], self._offset[1]
        rhs[:, a:b] = _cov(self.variograms[0], _dist(xy, self.coords_list_[0]))
        for k in range(1, self.n_var):
            c, d = self._offset[k], self._offset[k + 1]
            rhs[:, c:d] = _cov(self._cross(0, k), _dist(xy, self.coords_list_[k]))
        constraints = np.zeros((len(xy), self.n_var))
        constraints[:, 0] = 1.0
        weights = linalg.lu_solve(self._lu, np.hstack([rhs, constraints]).T).T
        pred = weights[:, :self._n] @ self.all_values_
        if not return_std:
            return pred
        var = _c0(self.variograms[0]) - np.sum(weights[:, :self._n] * rhs, axis=1) - weights[:, self._n]
        return _returned(pred, var, True)

    def predict_grid(self, region, resolution: Optional[float] = None, n_cells: int = 100,
                     return_std: bool = False):
        return _grid_predict(self, region, resolution, n_cells, return_std)


class ColocatedCokriging:
    """
    Colocated cokriging under the Markov model MM1 (Xu et al., 1992).

    Only the secondary values sitting at the prediction location enter the
    system, together with every primary sample. The screening hypothesis
    replaces the cross-covariance by a scaled copy of the primary covariance,

    ::

        C₀ₖ(h) = C₀ₖ(0) / C₀₀(0) · C₀₀(h)

    (set ``mm1=False`` to evaluate the fitted cross-variogram instead).
    Secondary means are treated as known, and a single ordinary-kriging
    constraint Σ λᵢ = 1 is imposed on the primary weights. Secondaries are
    entered as residuals from ``secondary_means``.

    Parameters
    ----------
    primary_var : VariogramModel
    secondary_vars : sequence of VariogramModel
        Direct variograms of each secondary, used for Cₖₖ(0).
    cross_vars : sequence of CrossVariogramModel
        Primary–secondary cross-variograms, one per secondary, used for C₀ₖ(0).
    secondary_cross_vars : dict, optional
        ``{(i, j): CrossVariogramModel}`` among secondaries (0-based secondary
        indices). Omitted pairs are treated as uncorrelated.
    mm1 : bool
        Use the Markov screening approximation above.
    """

    def __init__(self, primary_var: VariogramModel, secondary_vars: Sequence[VariogramModel],
                 cross_vars: Sequence[CrossVariogramModel], secondary_cross_vars: Optional[dict] = None,
                 mm1: bool = True):
        self.primary_var = primary_var
        self.secondary_vars = list(secondary_vars)
        self.cross_vars = list(cross_vars)
        self.n_sec = len(self.secondary_vars)
        if self.n_sec < 1:
            raise ValueError("need at least one secondary")
        if len(self.cross_vars) != self.n_sec:
            raise ValueError("need one primary–secondary cross-variogram per secondary")
        self.secondary_cross_vars = {} if secondary_cross_vars is None else dict(secondary_cross_vars)
        for key in self.secondary_cross_vars:
            i, j = int(key[0]), int(key[1])
            if i == j or not (0 <= i < self.n_sec and 0 <= j < self.n_sec):
                raise ValueError(f"secondary cross-variogram index {key} is out of range")
        self.mm1 = bool(mm1)

    def fit(self, coords_primary, primary, secondary_means) -> "ColocatedCokriging":
        """
        Store the primary sample and the known secondary means.

        ``secondary_means`` has one value per secondary (usually the mean of
        that secondary's training sample).
        """
        self.coords_primary_, self.primary_ = _xy_values(coords_primary, primary, "primary", min_n=2)
        means = np.asarray(secondary_means, dtype=float).reshape(-1)
        if means.size != self.n_sec or not np.isfinite(means).all():
            raise ValueError("secondary_means must have one finite value per secondary")
        self.secondary_means_ = means
        self.n_primary_ = len(self.primary_)
        self.C00_ = _c0(self.primary_var)
        if self.mm1 and self.C00_ <= 0:
            raise ValueError("MM1 colocated cokriging needs a positive primary sill")
        self.C0k_ = np.array([_c0(cv) for cv in self.cross_vars], dtype=float)
        self.Ckk_ = np.array([_c0(vg) for vg in self.secondary_vars], dtype=float)
        Cyy = np.diag(self.Ckk_)
        for (i, j), cv in self.secondary_cross_vars.items():
            Cyy[int(i), int(j)] = Cyy[int(j), int(i)] = _c0(cv)
        self._Cyy = Cyy
        n = self.n_primary_
        Czz = _cov(self.primary_var, _dist(self.coords_primary_, self.coords_primary_))
        A = np.zeros((n + 1, n + 1))
        A[:n, :n] = Czz
        A[:n, n] = 1.0
        A[n, :n] = 1.0
        self._lu = _factor(A)
        return self

    def _c0k_of_h(self, dist) -> np.ndarray:
        """Primary–secondary covariance, shape ``(n_sec, n_pred, n_primary)``."""
        C00_h = _cov(self.primary_var, dist)
        out = np.zeros((self.n_sec, dist.shape[0], dist.shape[1]))
        for k, cv in enumerate(self.cross_vars):
            if self.mm1:
                out[k] = (self.C0k_[k] / self.C00_) * C00_h
            else:
                out[k] = _cov(cv, dist)
        return out

    def predict(self, coords, secondary_at_pred, return_std: bool = False):
        """
        Predict the primary at ``coords`` using secondaries known at those same locations.

        ``secondary_at_pred`` has shape ``(n_pred, n_sec)`` (a single secondary
        may be passed as a 1-D array).
        """
        if not hasattr(self, "_lu"):
            raise RuntimeError("call fit() first")
        xy = as_xy(coords)
        Y = np.asarray(secondary_at_pred, dtype=float)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        m = len(xy)
        if Y.shape != (m, self.n_sec):
            raise ValueError(
                f"secondary_at_pred must have shape ({m}, {self.n_sec}), got {Y.shape}"
            )
        n = self.n_primary_
        dist = _dist(xy, self.coords_primary_)                                  # (m, n)
        C00_p0 = _cov(self.primary_var, dist)
        Czy = np.transpose(self._c0k_of_h(dist), (2, 0, 1))                     # (n, k, m)

        rhs_a = np.zeros((n + 1, m))
        rhs_a[:n, :] = C00_p0.T
        rhs_a[n, :] = 1.0
        w_a0 = linalg.lu_solve(self._lu, rhs_a)                                 # (n + 1, m)

        B = np.zeros((n + 1, self.n_sec * m))
        B[:n, :] = Czy.reshape(n, self.n_sec * m)
        P = linalg.lu_solve(self._lu, B).reshape(n + 1, self.n_sec, m)

        Y_res = Y - self.secondary_means_[None, :]
        S = self._Cyy[None, :, :] - np.einsum("iat,ibt->tab", Czy, P[:n])
        rhs_y = self.C0k_[None, :] - np.einsum("iat,it->ta", Czy, w_a0[:n])
        try:
            w_y = np.linalg.solve(S, rhs_y[..., None])[..., 0]
        except np.linalg.LinAlgError:
            w_y = np.stack([linalg.lstsq(S[t], rhs_y[t])[0] for t in range(m)])
        w_a = w_a0.T - np.einsum("abt,tb->ta", P, w_y)                          # (m, n + 1)
        pred = w_a[:, :n] @ self.primary_ + np.sum(w_y * Y_res, axis=1)
        if not return_std:
            return pred
        var = self.C00_ - np.sum(w_a[:, :n] * C00_p0, axis=1) - w_y @ self.C0k_ - w_a[:, n]
        return _returned(pred, var, True)


# ---------------------------------------------------------------------------
# Regression kriging
# ---------------------------------------------------------------------------

# Shortcuts for RegressionKriging(regressor=...). Anything else with
# fit(X, y) / predict(X) can be passed in directly.
ENSEMBLE_REGRESSORS = ("linear", "random_forest", "gradient_boosting", "bagging", "stacking")
STATISTICAL_REGRESSORS = ("gam", "glm", "bayesian_ridge", "quantile")
AVAILABLE_REGRESSORS = ENSEMBLE_REGRESSORS + STATISTICAL_REGRESSORS

_DEFAULT_REGRESSOR_PARAMS = {
    "random_forest": {"n_estimators": 300, "max_depth": None, "n_jobs": -1, "random_state": 42},
    "gradient_boosting": {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 3, "random_state": 42},
    "bagging": {"n_estimators": 50, "random_state": 42},
    "quantile": {"quantile": 0.5, "alpha": 0.0},
}


class GAMRegressor:
    """
    Additive model ``f(X) = β₀ + Σⱼ sⱼ(Xⱼ)`` with one spline smooth per column.

    scikit-learn's ``fit`` / ``predict`` interface, so it can be handed to
    :class:`RegressionKriging` as ``regressor='gam'``. The ``pygam`` model is
    built in ``fit``, once the number of columns is known.

    Parameters
    ----------
    distribution, link : str
        Noise model and link passed to ``pygam.GAM`` (``'normal'`` /
        ``'identity'`` by default).
    n_splines : int
        Splines in each smooth.
    lam : float
        Smoothing penalty (larger is smoother).

    Requires the optional ``pygam`` package.
    """

    def __init__(self, distribution: str = "normal", link: str = "identity",
                 n_splines: int = 10, lam: float = 0.6):
        self.distribution = distribution
        self.link = link
        self.n_splines = n_splines
        self.lam = lam
        self.gam_ = None

    def get_params(self, deep: bool = True) -> dict:
        return {
            "distribution": self.distribution, "link": self.link,
            "n_splines": self.n_splines, "lam": self.lam,
        }

    def set_params(self, **params) -> "GAMRegressor":
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y) -> "GAMRegressor":
        try:
            from pygam import GAM, s
        except ImportError as exc:
            raise ImportError(
                "regressor='gam' requires the optional 'pygam' package (pip install pygam)"
            ) from exc
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim != 2 or X.shape[1] < 1:
            raise ValueError("GAMRegressor needs X with shape (n_samples, n_features)")
        terms = s(0, n_splines=self.n_splines, lam=self.lam)
        for j in range(1, X.shape[1]):
            terms = terms + s(j, n_splines=self.n_splines, lam=self.lam)
        self.gam_ = GAM(terms, distribution=self.distribution, link=self.link)
        self.gam_.fit(X, y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X) -> np.ndarray:
        if self.gam_ is None:
            raise RuntimeError("call fit() first")
        return np.asarray(self.gam_.predict(np.asarray(X, dtype=float)), dtype=float)

    def __repr__(self) -> str:
        return (f"GAMRegressor(distribution={self.distribution!r}, link={self.link!r}, "
                f"n_splines={self.n_splines}, lam={self.lam})")


def _default_stacking_estimator(**overrides):
    """Random forest, gradient boosting and a linear model, combined by ridge."""
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
    from sklearn.linear_model import LinearRegression, RidgeCV
    overrides = dict(overrides)
    estimators = overrides.pop("estimators", None) or [
        ("random_forest", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
        ("gradient_boosting", GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, random_state=42)),
        ("linear", LinearRegression()),
    ]
    final = overrides.pop("final_estimator", None) or RidgeCV()
    return StackingRegressor(estimators=estimators, final_estimator=final, **overrides)


def _build_regressor(regressor, regressor_kwargs):
    """Turn ``None``, a shortcut name, or an estimator into a regressor."""
    if regressor is None:
        from sklearn.linear_model import LinearRegression
        return LinearRegression(**(regressor_kwargs or {}))
    if not isinstance(regressor, str):
        return regressor
    key = regressor.lower()
    kwargs = dict(regressor_kwargs or {})
    if key in ("linear", "ols"):
        from sklearn.linear_model import LinearRegression
        return LinearRegression(**kwargs)
    if key in ("random_forest", "randomforest", "rf"):
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(**{**_DEFAULT_REGRESSOR_PARAMS["random_forest"], **kwargs})
    if key in ("gradient_boosting", "gradientboosting", "gbm", "gbr"):
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(**{**_DEFAULT_REGRESSOR_PARAMS["gradient_boosting"], **kwargs})
    if key == "bagging":
        from sklearn.ensemble import BaggingRegressor
        return BaggingRegressor(**{**_DEFAULT_REGRESSOR_PARAMS["bagging"], **kwargs})
    if key == "stacking":
        return _default_stacking_estimator(**kwargs)
    if key == "gam":
        return GAMRegressor(**kwargs)
    if key == "glm":
        from sklearn.linear_model import TweedieRegressor
        return TweedieRegressor(**kwargs)
    if key in ("bayesian_ridge", "bayesianridge", "bayesian", "bayes"):
        from sklearn.linear_model import BayesianRidge
        return BayesianRidge(**kwargs)
    if key == "quantile":
        from sklearn.linear_model import QuantileRegressor
        return QuantileRegressor(**{**_DEFAULT_REGRESSOR_PARAMS["quantile"], **kwargs})
    raise ValueError(
        f"unknown regressor {regressor!r}; pass an estimator, or one of {list(AVAILABLE_REGRESSORS)}"
    )


class RegressionKriging:
    """
    Regression kriging: fit a trend in the covariates, then ordinary-krige the residuals.

    The trend may be any object with ``fit(X, y)`` and ``predict(X)``, or one
    of :data:`AVAILABLE_REGRESSORS`:

    * ``'linear'`` — ordinary least squares (the default)
    * ``'random_forest'``, ``'gradient_boosting'``, ``'bagging'``, ``'stacking'``
    * ``'glm'`` — ``TweedieRegressor``; ``'bayesian_ridge'``; ``'quantile'``
    * ``'gam'`` — :class:`GAMRegressor` (needs the optional ``pygam`` package)

    ``regressor_kwargs`` is passed to that constructor (ignored when
    ``regressor`` is already an estimator). For ``'stacking'``, the keys
    ``estimators`` and ``final_estimator`` replace the default base learners
    and the ridge meta-learner.

    The residual variogram is fitted on the training residuals unless
    ``variogram`` is already a :class:`VariogramModel`. Only ordinary kriging
    of the residuals is implemented (``kriging_type='ordinary'``). The
    standard deviation from ``predict(..., return_std=True)`` is the residual
    kriging standard deviation; it does not include uncertainty in the trend.

    Separately fitted ``'quantile'`` models can cross. Pass their predictions
    through :func:`enforce_quantile_monotonicity` to sort them at each location.
    """

    def __init__(self, regressor=None, kriging_type: str = "ordinary",
                 variogram: Union[VariogramModel, str] = "spherical",
                 n_neighbors: Optional[int] = None, n_lags: int = 15,
                 max_dist: Optional[float] = None, regressor_kwargs: Optional[dict] = None):
        if kriging_type != "ordinary":
            raise ValueError("only 'ordinary' kriging of the residuals is supported")
        self.regressor = _build_regressor(regressor, regressor_kwargs)
        self.kriging_type = kriging_type
        self.variogram = variogram
        self.n_neighbors = n_neighbors
        self.n_lags = n_lags
        self.max_dist = max_dist
        self.regressor_kwargs = regressor_kwargs
        self.is_fitted_ = False

    def fit(self, X, y, coords) -> "RegressionKriging":
        """Fit the trend on ``X``, then krige ``y − trend`` at ``coords``."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        y = np.asarray(y, dtype=float).ravel()
        xy = as_xy(coords)
        if not (len(X) == len(y) == len(xy)):
            raise ValueError("X, y and coords must have the same number of rows")
        keep = np.isfinite(y) & np.isfinite(xy).all(axis=1) & np.isfinite(X).all(axis=1)
        X, y, xy = X[keep], y[keep], xy[keep]
        if len(y) < 2:
            raise ValueError("need at least 2 finite observations")
        self.regressor.fit(X, y)
        residuals = y - np.asarray(self.regressor.predict(X), dtype=float).ravel()
        self.variogram_, emp = _resolve_variogram(
            self.variogram, xy, residuals, self.n_lags, self.max_dist,
        )
        if emp is not None:
            self.empirical_ = emp
        self.krige_ = OrdinaryKriging(self.variogram_, n_neighbors=self.n_neighbors).fit(xy, residuals)
        self.coords_, self.X_, self.y_ = xy, X, y
        self.residuals_ = residuals
        self.is_fitted_ = True
        return self

    def predict(self, X, coords, return_std: bool = False):
        """Trend plus kriged residual. ``X`` and ``coords`` must have the same number of rows."""
        if not self.is_fitted_:
            raise RuntimeError("call fit() first")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        xy = as_xy(coords)
        if len(X) != len(xy):
            raise ValueError("X and coords must have the same number of rows")
        trend = np.asarray(self.regressor.predict(X), dtype=float).ravel()
        if return_std:
            resid, std = self.krige_.predict(xy, return_std=True)
            return trend + resid, std
        return trend + self.krige_.predict(xy)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Importances from a tree ensemble trend. Linear trends do not have them."""
        if not self.is_fitted_:
            raise RuntimeError("call fit() first")
        if hasattr(self.regressor, "feature_importances_"):
            return self.regressor.feature_importances_
        raise AttributeError(
            f"{type(self.regressor).__name__} does not expose feature_importances_"
        )

    def __repr__(self) -> str:
        name = self.variogram_.model if hasattr(self, "variogram_") else self.variogram
        if not isinstance(name, str):
            name = getattr(name, "model", name)
        return (f"RegressionKriging(regressor={type(self.regressor).__name__}, "
                f"variogram={name!r})")


def enforce_quantile_monotonicity(predictions):
    """
    Sort quantile-regression predictions so a lower quantile stays below a higher one.

    Independently fitted quantile models (including regression kriging of each
    quantile's residuals) can cross. Rearrangement — sorting the predicted
    quantiles at each location — restores monotonicity (Chernozhukov,
    Fernández-Val and Galichon, 2010).

    Parameters
    ----------
    predictions : dict or array
        A dict ``{quantile: array of length n}``, or an array of shape
        ``(n_quantiles, n)`` whose rows are already in increasing quantile order.

    Returns
    -------
    The same type, sorted along the quantile axis at every location.
    """
    if isinstance(predictions, dict):
        levels = sorted(predictions)
        stacked = np.stack([np.asarray(predictions[q], dtype=float) for q in levels], axis=0)
        ordered = np.sort(stacked, axis=0)
        return {q: ordered[i] for i, q in enumerate(levels)}
    return np.sort(np.asarray(predictions, dtype=float), axis=0)


# ---------------------------------------------------------------------------
# Indicator kriging and E-type estimates
# ---------------------------------------------------------------------------

def _as_variogram(spec, family: str = "spherical") -> VariogramModel:
    """A VariogramModel, or ``(nugget, psill, range)`` / ``(..., model name)``."""
    if isinstance(spec, VariogramModel):
        return spec
    try:
        seq = tuple(spec)
    except TypeError as exc:
        raise TypeError(
            "a variogram must be a VariogramModel or (nugget, psill, range)"
        ) from exc
    if len(seq) == 3:
        nug, ps, rg = seq
        model = family
    elif len(seq) == 4:
        nug, ps, rg, model = seq
    else:
        raise TypeError("variogram parameters must be (nugget, psill, range) or (nugget, psill, range, model)")
    return VariogramModel(model=str(model), nugget=float(nug), psill=float(ps), range=float(rg))


def _ok_weights_batch(neighbor_coords, targets, variogram: VariogramModel, regularization: float):
    """
    Ordinary-kriging weights for many targets that share a neighbour count.

    ``neighbor_coords`` has shape ``(n_pred, k, 2)`` and ``targets`` shape
    ``(n_pred, 2)``. One stacked solve, the same system as ordinary kriging
    written in covariances. A ridge of ``regularization`` is added to the
    covariance diagonal only.
    """
    m, k, _ = neighbor_coords.shape
    diff = neighbor_coords[:, :, None, :] - neighbor_coords[:, None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    C = _cov(variogram, dists)
    if regularization:
        C = C + np.eye(k) * float(regularization)
    K = np.zeros((m, k + 1, k + 1))
    K[:, :k, :k] = C
    K[:, :k, k] = 1.0
    K[:, k, :k] = 1.0
    d0 = np.linalg.norm(neighbor_coords - targets[:, None, :], axis=-1)
    rhs = np.ones((m, k + 1, 1))
    rhs[:, :k, 0] = _cov(variogram, d0)
    return np.linalg.solve(K, rhs)[:, :k, 0]


class IndicatorKriging:
    """
    Ordinary indicator kriging.

    Each threshold ``t`` is coded as ``I(Z ≥ t)`` and ordinary-kriged. The
    result is read as the conditional probability ``P(Z ≥ t)`` and clipped
    to ``[0, 1]``. Thresholds are kriged independently, so a set of
    probabilities need not be a decreasing survival function.

    The variogram is the indicator variogram at that threshold (sill at most
    ``p(1−p)``), not the variogram of ``Z``. Pass one model for every
    threshold, a dict ``{threshold: model}``, or ``'auto'`` / a model name
    to fit each indicator when predicting. A model may also be the tuple
    ``(nugget, psill, range)``.

    Parameters
    ----------
    variogram : VariogramModel, dict, tuple, or {'auto', 'spherical', 'exponential', 'gaussian'}
    n_neighbors : int or None
        Nearest neighbours used at each target. ``None`` uses every sample.
        The default, 12, is a local search.
    search_radius : float or None
        When set, neighbours are those inside this radius, still capped at
        ``n_neighbors``. Targets with no neighbour are left as NaN.
    n_lags : int
        Distance bins when an indicator variogram is fitted automatically.
    regularization : float
        Ridge added to the covariance diagonal. ``0`` leaves the system
        unregularized, and the predictor then reproduces the indicator at
        the sample locations.
    """

    def __init__(self, variogram: Union[VariogramModel, dict, str, tuple] = "auto",
                 n_neighbors: Optional[int] = 12, search_radius: Optional[float] = None,
                 n_lags: int = 15, regularization: float = 1e-8):
        if n_neighbors is not None and int(n_neighbors) < 1:
            raise ValueError("n_neighbors must be >= 1")
        if search_radius is not None and float(search_radius) <= 0:
            raise ValueError("search_radius must be positive")
        if regularization < 0:
            raise ValueError("regularization must be >= 0")
        self.variogram = variogram
        self.n_neighbors = None if n_neighbors is None else int(n_neighbors)
        self.search_radius = None if search_radius is None else float(search_radius)
        self.n_lags = n_lags
        self.regularization = float(regularization)

    def fit(self, coords, values) -> "IndicatorKriging":
        """Store the sample. Indicator variograms are fitted in ``predict`` when requested."""
        self.coords_, self.values_ = _xy_values(coords, values, "indicator kriging", min_n=2)
        self._tree = cKDTree(self.coords_)
        self.variograms_ = {}
        self.empirical_ = {}
        self.is_fitted_ = True
        return self

    def _family(self) -> str:
        if isinstance(self.variogram, str) and self.variogram != "auto":
            return self.variogram
        return "spherical"

    def _model_for(self, threshold: float, override) -> VariogramModel:
        spec = self.variogram if override is None else override
        if isinstance(spec, dict):
            for key, model in spec.items():
                if np.isclose(float(key), threshold):
                    return _as_variogram(model, self._family())
            raise KeyError(f"no variogram for threshold {threshold}")
        if isinstance(spec, str):
            if override is None and threshold in self.variograms_:
                return self.variograms_[threshold]
            vg, emp = fit_indicator_variogram(
                self.coords_, self.values_, threshold, model=spec, n_lags=self.n_lags,
            )
            if override is None:
                self.variograms_[threshold] = vg
                self.empirical_[threshold] = emp
            return vg
        return _as_variogram(spec, self._family())

    def _krige_indicator(self, xy, indicator, variogram: VariogramModel) -> np.ndarray:
        X = self.coords_
        n = len(X)
        n_pred = len(xy)
        if self.search_radius is None:
            k = n if self.n_neighbors is None else min(self.n_neighbors, n)
            nbr = _neighbors(self._tree, xy, k)
            try:
                weights = _ok_weights_batch(X[nbr], xy, variogram, self.regularization)
                prob = np.einsum("ij,ij->i", weights, indicator[nbr])
            except np.linalg.LinAlgError:
                prob = np.empty(n_pred)
                for i in range(n_pred):
                    idx = nbr[i]
                    try:
                        w = _ok_weights_batch(X[idx][None], xy[i:i + 1], variogram, self.regularization)
                        prob[i] = w[0] @ indicator[idx]
                    except np.linalg.LinAlgError:
                        prob[i] = float(np.mean(indicator[idx]))
            return np.clip(prob, 0.0, 1.0)

        prob = np.full(n_pred, np.nan)
        cap = n if self.n_neighbors is None else self.n_neighbors
        for i, target in enumerate(xy):
            idxs = np.asarray(self._tree.query_ball_point(target, r=self.search_radius), dtype=int)
            if idxs.size == 0:
                continue
            if idxs.size > cap:
                dist = np.linalg.norm(X[idxs] - target, axis=1)
                idxs = idxs[np.argsort(dist)[:cap]]
            try:
                w = _ok_weights_batch(X[idxs][None], target.reshape(1, 2), variogram, self.regularization)
                prob[i] = float(np.clip(w[0] @ indicator[idxs], 0.0, 1.0))
            except np.linalg.LinAlgError:
                prob[i] = float(np.mean(indicator[idxs]))
        return prob

    def predict(self, coords, thresholds, variogram=None) -> dict:
        """
        ``P(Z ≥ t)`` at ``coords`` for each threshold.

        Returns a dict ``{threshold: array of length n_pred}``. A threshold
        that every sample falls on the same side of is returned as that
        constant indicator, without solving a kriging system.
        """
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("call fit() first")
        xy = as_xy(coords)
        levels = [float(t) for t in np.atleast_1d(thresholds).ravel()]
        if not levels:
            raise ValueError("thresholds must be non-empty")
        out = {}
        for t in levels:
            ind = indicator_transform(self.values_, t)
            if np.all(ind == ind[0]):
                out[t] = np.full(len(xy), float(ind[0]))
                continue
            out[t] = self._krige_indicator(xy, ind, self._model_for(t, variogram))
        return out

    def etype(self, coords, thresholds, z_min: float = 0.0, z_max: Optional[float] = None,
              variogram=None) -> np.ndarray:
        """
        E-type estimate: the conditional expectation from the kriged survival function.

        Thresholds are sorted, each one is indicator-kriged, and the
        probabilities are integrated by :func:`etype_estimate`.
        """
        levels = np.sort(np.atleast_1d(thresholds).astype(float).ravel())
        probs = self.predict(coords, levels, variogram=variogram)
        return etype_estimate(levels, [probs[float(t)] for t in levels], z_min=z_min, z_max=z_max)

    def predict_grid(self, region, thresholds, resolution: Optional[float] = None, n_cells: int = 100,
                     variogram=None):
        """Indicator probabilities on a regular grid. Returns ``(grid, {threshold: values})``."""
        from spatialstats.interpolate.base import make_grid
        grid = make_grid(region, resolution, n_cells)
        return grid, self.predict(grid.coords, thresholds, variogram=variogram)


def etype_estimate(thresholds, prob_arrays, z_min: float = 0.0, z_max: Optional[float] = None):
    """
    E-type (conditional expectation) from indicator-kriging probabilities.

    For a variable supported on ``[z_min, z_max]``,
    ``E[Z] = ∫ P(Z ≥ z) dz``. The integral uses the trapezoid rule between
    the supplied thresholds, a linear lower tail from ``z_min`` (where the
    probability is taken to be 1) up to the first threshold, and a linear
    upper tail from the last threshold down to 0 at ``z_max``. When
    ``z_max`` is omitted and there are at least two thresholds, the last
    interval width is reused for that upper tail.

    Parameters
    ----------
    thresholds : array-like
        Cutoffs, sorted ascending.
    prob_arrays : sequence of array-like
        ``P(Z ≥ threshold)`` for each cutoff, all the same shape.
    z_min, z_max : float
        Ends of the support. ``z_max=None`` reuses the last gap.

    Returns
    -------
    ndarray
        E-type estimate, same shape as one probability array.
    """
    thresholds = np.asarray(thresholds, dtype=float).ravel()
    if thresholds.size == 0:
        raise ValueError("thresholds must be non-empty")
    if not np.all(thresholds[:-1] <= thresholds[1:]):
        raise ValueError("thresholds must be sorted in ascending order")
    probs = [np.asarray(p, dtype=float) for p in prob_arrays]
    if len(probs) != len(thresholds):
        raise ValueError("prob_arrays must have one array per threshold")
    shape = probs[0].shape
    if any(p.shape != shape for p in probs):
        raise ValueError("all probability arrays must have the same shape")

    etype = np.zeros(shape)
    z0, p0 = thresholds[0], probs[0]
    if z0 > z_min:
        etype = etype + 0.5 * (1.0 + p0) * (z0 - z_min)
    for i in range(len(thresholds) - 1):
        dz = thresholds[i + 1] - thresholds[i]
        etype = etype + 0.5 * (probs[i] + probs[i + 1]) * dz
    p_last, zk = probs[-1], thresholds[-1]
    if z_max is not None:
        if z_max < zk:
            raise ValueError("z_max must be >= the last threshold")
        etype = etype + 0.5 * p_last * (z_max - zk)
    elif len(thresholds) > 1:
        etype = etype + 0.5 * p_last * (thresholds[-1] - thresholds[-2])
    return etype
