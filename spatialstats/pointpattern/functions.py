"""
spatialstats.pointpattern.functions
===================================
Summary functions of a point pattern: nearest-neighbour statistics
(Clark–Evans, G, F, J) and second-order functions (Ripley's K, L, pair
correlation g, cross-K, inhomogeneous K).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree

from spatialstats.core.result import SpatialResult
from spatialstats.pointpattern.pattern import PointPattern
from spatialstats.pointpattern.intensity import intensity_at_points


# ---------------------------------------------------------------------------
# Result type for functions of distance
# ---------------------------------------------------------------------------

class FunctionResult(SpatialResult):
    """
    A summary function ``f(r)`` with its CSR-theoretical curve.

    Attributes
    ----------
    r : ndarray
    estimate : ndarray        primary estimate (the recommended correction)
    theoretical : ndarray     value expected under CSR
    estimates : dict          all computed corrections
    """

    def __init__(self, name: str, r, estimates: Dict[str, np.ndarray], theoretical,
                 primary: str, **stats_):
        super().__init__(name=name, correction=primary, **stats_)
        self.r = np.asarray(r, dtype=float)
        self.estimates = {k: np.asarray(v, dtype=float) for k, v in estimates.items()}
        self.primary = primary
        self.theoretical = np.asarray(theoretical, dtype=float)

    @property
    def estimate(self) -> np.ndarray:
        return self.estimates[self.primary]

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame({"r": self.r, "theo": self.theoretical})
        for k, v in self.estimates.items():
            df[k] = v
        return df

    def plot(self, ax=None, detrend: bool = False, **kwargs):
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "plot requires matplotlib. Install with: pip install pyspatialstats[viz]"
            ) from e
        if ax is None:
            _, ax = plt.subplots()
        off = self.r if detrend else 0.0
        ax.plot(self.r, self.estimate - off, label=f"observed ({self.primary})", **kwargs)
        ax.plot(self.r, self.theoretical - off, "k--", lw=1, label="CSR")
        ax.set_xlabel("r")
        ax.set_ylabel(f"{self.name}(r)" + (" - r" if detrend else ""))
        ax.legend()
        return ax


def _default_r(pp: PointPattern, rmax: Optional[float] = None, n: int = 128) -> np.ndarray:
    if rmax is None:
        rmax = 0.25 * pp.window.short_side
    return np.linspace(0.0, rmax, n)


# ---------------------------------------------------------------------------
# Nearest-neighbour distances, Clark–Evans, G / F / J
# ---------------------------------------------------------------------------

def nearest_neighbor_distances(pp: PointPattern) -> np.ndarray:
    """Distance from each event to its nearest other event."""
    if pp.n < 2:
        raise ValueError("need at least 2 events")
    d, _ = cKDTree(pp.coords).query(pp.coords, k=2)
    return d[:, 1]


class ClarkEvansResult(SpatialResult):
    pass


def clark_evans(pp: PointPattern, correction: str = "donnelly") -> ClarkEvansResult:
    """
    Clark–Evans nearest-neighbour test of CSR.

    ``R = observed mean NN distance / expected under CSR``: R < 1 indicates
    clustering, R > 1 regularity.

    Parameters
    ----------
    correction : {'donnelly', 'none'}
        ``'none'`` uses the infinite-plane expectation ``0.5 / sqrt(λ)``, which
        is biased upward in a finite window. ``'donnelly'`` (Donnelly 1978)
        adds the boundary correction for the mean and variance and needs a
        window perimeter.
    """
    d = nearest_neighbor_distances(pp)
    n, A = pp.n, pp.area
    dbar = float(d.mean())
    if correction == "none":
        expected = 0.5 * np.sqrt(A / n)
        se = 0.26136 * np.sqrt(A) / n
    elif correction == "donnelly":
        P = float(pp.window.polygon.length)
        expected = 0.5 * np.sqrt(A / n) + (0.0514 + 0.041 / np.sqrt(n)) * P / n
        var = 0.070 * A / n ** 2 + 0.037 * P * np.sqrt(A / n ** 5)
        se = np.sqrt(var)
    else:
        raise ValueError("correction must be 'donnelly' or 'none'")
    z = (dbar - expected) / se
    return ClarkEvansResult(
        name="Clark-Evans test", R=dbar / expected, mean_nn=dbar, expected_nn=float(expected),
        z_score=float(z), p_value=float(2 * stats.norm.sf(abs(z))),
        p_clustered=float(stats.norm.cdf(z)), p_regular=float(stats.norm.sf(z)),
        correction=correction, n=n,
    )


def _km_cdf(t: np.ndarray, event: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Kaplan–Meier estimate of the distribution function at ``r`` from right-censored distances."""
    order = np.argsort(t, kind="stable")
    t, event = t[order], event[order]
    n = len(t)
    at_risk = n - np.arange(n)
    surv = np.cumprod(1.0 - event / at_risk)
    idx = np.searchsorted(t, r, side="right") - 1
    s = np.where(idx >= 0, surv[np.clip(idx, 0, n - 1)], 1.0)
    return 1.0 - s


def _reduced_sample_cdf(d: np.ndarray, b: np.ndarray, r: np.ndarray) -> np.ndarray:
    out = np.empty_like(r)
    for k, rk in enumerate(r):
        ok = b >= rk
        out[k] = np.mean(d[ok] <= rk) if ok.any() else np.nan
    return out


def _nn_function(d, b, r, correction):
    est = {}
    if correction in ("none", "all"):
        est["none"] = np.searchsorted(np.sort(d), r, side="right") / len(d)
    if correction in ("border", "all"):
        est["border"] = _reduced_sample_cdf(d, b, r)
    if correction in ("km", "all"):
        t = np.minimum(d, b)
        est["km"] = _km_cdf(t, (d <= b).astype(float), r)
    return est


def g_function(pp: PointPattern, r: Optional[np.ndarray] = None, correction: str = "km") -> FunctionResult:
    """
    Nearest-neighbour distance distribution ``G(r) = P(d_i ≤ r)``.

    Under CSR ``G(r) = 1 - exp(-λ π r²)``; clustering pushes G above it.

    Parameters
    ----------
    correction : {'km', 'border', 'none', 'all'}
        ``'km'`` = Kaplan–Meier (censoring by distance to the window boundary),
        ``'border'`` = reduced-sample.
    """
    r = _default_r(pp, n=128) if r is None else np.asarray(r, dtype=float)
    d = nearest_neighbor_distances(pp)
    b = pp.window.distance_to_boundary(pp.coords)
    est = _nn_function(d, b, r, correction)
    theo = 1.0 - np.exp(-pp.intensity * np.pi * r ** 2)
    primary = "km" if "km" in est else next(iter(est))
    return FunctionResult("G", r, est, theo, primary)


def _grid_in_window(pp: PointPattern, n_grid: int):
    mask, xs, ys, _ = pp.window.raster(n_grid)
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([gx[mask], gy[mask]])


def f_function(pp: PointPattern, r: Optional[np.ndarray] = None, correction: str = "km",
               n_grid: int = 100) -> FunctionResult:
    """
    Empty-space function ``F(r) = P(distance from a random location to the nearest event ≤ r)``.

    Under CSR ``F(r) = 1 - exp(-λ π r²)``; clustering pulls F *below* it.
    """
    r = _default_r(pp, n=128) if r is None else np.asarray(r, dtype=float)
    pts = _grid_in_window(pp, n_grid)
    d, _ = cKDTree(pp.coords).query(pts)
    b = pp.window.distance_to_boundary(pts)
    est = _nn_function(d, b, r, correction)
    theo = 1.0 - np.exp(-pp.intensity * np.pi * r ** 2)
    primary = "km" if "km" in est else next(iter(est))
    return FunctionResult("F", r, est, theo, primary)


def j_function(pp: PointPattern, r: Optional[np.ndarray] = None, n_grid: int = 100) -> FunctionResult:
    """
    ``J(r) = (1 - G(r)) / (1 - F(r))``. J = 1 under CSR, J < 1 clustering, J > 1 regularity.
    """
    r = _default_r(pp, n=128) if r is None else np.asarray(r, dtype=float)
    G = g_function(pp, r, "km").estimate
    F = f_function(pp, r, "km", n_grid=n_grid).estimate
    with np.errstate(divide="ignore", invalid="ignore"):
        J = np.where(1 - F > 1e-3, (1 - G) / (1 - F), np.nan)
    return FunctionResult("J", r, {"km": J}, np.ones_like(r), "km")


# ---------------------------------------------------------------------------
# Second-order functions
# ---------------------------------------------------------------------------

def _resolve_correction(pp: PointPattern, correction: Optional[str]) -> str:
    if correction is None:
        return "translation"
    if correction not in ("translation", "border", "none"):
        raise ValueError("correction must be 'translation', 'border' or 'none'")
    return correction


def _pair_geometry(coords_a: np.ndarray, coords_b: np.ndarray, rmax: float, same: bool):
    """All pairs (a, b) closer than ``rmax``; returns index arrays, distances and displacement vectors."""
    ta, tb = cKDTree(coords_a), cKDTree(coords_b)
    pairs = ta.sparse_distance_matrix(tb, rmax, output_type="ndarray")
    i, j, d = pairs["i"], pairs["j"], pairs["v"]
    if same:
        keep = i != j
        i, j, d = i[keep], j[keep], d[keep]
    v = coords_a[i] - coords_b[j]
    return i, j, d, v


def _k_engine(pp, coords_a, coords_b, lam_a, lam_b, r, correction, same):
    """
    Generic K estimator: ``Σ_{a,b} 1(d ≤ r) / (λ_a λ_b γ_ab)`` (translation),
    with ``γ`` the window set covariance. ``lam_*`` are per-event intensities.
    Returns (K, pair_d, pair_weight) so callers can reuse the pair data for g(r).
    """
    rmax = float(r.max())
    i, j, d, v = _pair_geometry(coords_a, coords_b, rmax, same)
    A = pp.area
    if correction == "translation":
        gamma = pp.window.set_covariance(v)
        floor = 0.01 * A
        wgt = 1.0 / (lam_a[i] * lam_b[j] * np.maximum(gamma, floor))
        K = np.array([wgt[d <= rk].sum() for rk in r])
        return K, d, wgt
    if correction == "none":
        wgt = 1.0 / (lam_a[i] * lam_b[j] * A)
        K = np.array([wgt[d <= rk].sum() for rk in r])
        return K, d, wgt
    # border (reduced-sample): only anchors a with boundary distance >= r contribute,
    # normalised by the area of the window eroded by r
    b = pp.window.distance_to_boundary(coords_a)
    K = np.empty(len(r))
    base = 1.0 / (lam_a[i] * lam_b[j])
    for k, rk in enumerate(r):
        eroded = pp.window.polygon.buffer(-rk).area if rk > 0 else A
        if eroded <= 0 or not (b >= rk).any():
            K[k] = np.nan
            continue
        inc = (d <= rk) & (b[i] >= rk)
        K[k] = base[inc].sum() / eroded
    return K, d, None


def ripley_k(pp: PointPattern, r: Optional[np.ndarray] = None,
             correction: Optional[str] = None) -> FunctionResult:
    """
    Ripley's K function: the expected number of further events within distance
    ``r`` of a typical event, divided by the intensity.

    Under CSR ``K(r) = π r²``; larger values mean clustering at that scale.

    Parameters
    ----------
    r : array, optional
        Evaluation distances. Default: 128 values up to a quarter of the
        window's short side (the usual rule of thumb).
    correction : {'translation', 'border', 'none'}
        Edge correction (default ``'translation'``). Translation uses the exact
        set covariance for rectangles and a raster approximation for other
        windows.
    """
    correction = _resolve_correction(pp, correction)
    r = _default_r(pp) if r is None else np.asarray(r, dtype=float)
    lam = np.full(pp.n, pp.n / pp.area)
    # bias-corrected λ² = n(n-1)/A²  (replaces the plug-in n²/A²)
    K, _, _ = _k_engine(pp, pp.coords, pp.coords, lam, lam, r, correction, same=True)
    K = K * (pp.n / (pp.n - 1.0))
    return FunctionResult("K", r, {correction: K}, np.pi * r ** 2, correction)


def ripley_l(pp: PointPattern, r: Optional[np.ndarray] = None,
             correction: Optional[str] = None) -> FunctionResult:
    """Variance-stabilised ``L(r) = sqrt(K(r)/π)``; equals ``r`` under CSR (plot ``L(r) - r``)."""
    k = ripley_k(pp, r, correction)
    L = np.sqrt(np.maximum(k.estimate, 0) / np.pi)
    return FunctionResult("L", k.r, {k.primary: L}, k.r.copy(), k.primary)


def pair_correlation(pp: PointPattern, r: Optional[np.ndarray] = None, bandwidth: Optional[float] = None,
                     correction: Optional[str] = None) -> FunctionResult:
    """
    Pair correlation function ``g(r) = K'(r) / (2π r)``, estimated by kernel
    smoothing of pair distances (Epanechnikov kernel).

    ``g(r) = 1`` under CSR; ``g > 1`` means more pairs at distance ``r`` than
    expected (clustering), ``g < 1`` fewer (inhibition). Unlike K it is not
    cumulative, so it localises the scale of interaction.

    Parameters
    ----------
    bandwidth : float, optional
        Kernel half-width; default ``0.15 / sqrt(λ)`` (Stoyan's rule).
    """
    correction = "none" if correction == "none" else "translation"
    r = _default_r(pp) if r is None else np.asarray(r, dtype=float)
    r = r[r > 0] if r[0] == 0 else r
    h = 0.15 / np.sqrt(pp.intensity) if bandwidth is None else float(bandwidth)
    lam = np.full(pp.n, pp.n / pp.area)
    rmax = float(r.max()) + h
    i, j, d, v = _pair_geometry(pp.coords, pp.coords, rmax, True)
    if correction == "translation":
        gamma = np.maximum(pp.window.set_covariance(v), 0.01 * pp.area)
    else:
        gamma = np.full(len(d), pp.area)
    wgt = (pp.n / (pp.n - 1.0)) / (lam[i] * lam[j] * gamma)
    g = np.empty(len(r))
    for k, rk in enumerate(r):
        u = (rk - d) / h
        kern = np.where(np.abs(u) <= 1, 0.75 * (1 - u * u) / h, 0.0)
        g[k] = (kern * wgt).sum() / (2 * np.pi * rk)
    return FunctionResult("g", r, {correction: g}, np.ones_like(r), correction, bandwidth=h)


def cross_k(pp: PointPattern, mark_i, mark_j, r: Optional[np.ndarray] = None,
            correction: Optional[str] = None) -> FunctionResult:
    """
    Bivariate (cross) K function between two mark types: the expected number of
    ``mark_j`` events within ``r`` of a typical ``mark_i`` event, divided by the
    ``mark_j`` intensity. ``K_ij(r) = π r²`` if the two types are independent.
    """
    if pp.marks is None:
        raise ValueError("pattern has no marks")
    correction = _resolve_correction(pp, correction)
    r = _default_r(pp) if r is None else np.asarray(r, dtype=float)
    a = pp.coords[pp.marks == mark_i]
    b = pp.coords[pp.marks == mark_j]
    if len(a) == 0 or len(b) == 0:
        raise ValueError("both marks must occur in the pattern")
    la = np.full(len(a), len(a) / pp.area)
    lb = np.full(len(b), len(b) / pp.area)
    K, _, _ = _k_engine(pp, a, b, la, lb, r, correction, same=False)
    return FunctionResult(f"K[{mark_i},{mark_j}]", r, {correction: K}, np.pi * r ** 2, correction)


def k_inhom(pp: PointPattern, intensity=None, r: Optional[np.ndarray] = None,
            bandwidth: Optional[float] = None, correction: Optional[str] = None) -> FunctionResult:
    """
    Inhomogeneous K function (Baddeley–Møller–Waagepetersen): K after removing the
    effect of a *known or estimated* spatially varying intensity, so that
    ``K_inhom(r) = π r²`` for an inhomogeneous Poisson process.

    Parameters
    ----------
    intensity : array (n,) or KDEResult, optional
        Intensity at each event, or a :class:`KDEResult` (e.g. population
        density) to evaluate at the events. Default: a leave-one-out Gaussian
        kernel estimate from the pattern itself (``bandwidth``).
    """
    correction = _resolve_correction(pp, correction)
    r = _default_r(pp) if r is None else np.asarray(r, dtype=float)
    if intensity is None:
        lam = intensity_at_points(pp, bandwidth, leave_one_out=True)
    elif hasattr(intensity, "at"):
        lam = intensity.at(pp.coords)
    else:
        lam = np.asarray(intensity, dtype=float)
    if len(lam) != pp.n or not np.all(np.isfinite(lam)) or np.any(lam <= 0):
        raise ValueError("intensity must be positive and finite at every event")
    K, _, _ = _k_engine(pp, pp.coords, pp.coords, lam, lam, r, correction, same=True)
    # Σ 1/(λ_i λ_j γ_ij) already estimates K_inhom directly (units: area)
    return FunctionResult("K_inhom", r, {correction: K}, np.pi * r ** 2, correction)
