"""
spatialstats.esda.global_
=========================
Global spatial autocorrelation: Moran's I, Geary's C, Getis-Ord General G.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from spatialstats.core.result import SpatialResult
from spatialstats.esda.inference import conditional_permutation
from spatialstats.esda._utils import as_float_vector, check_y_w, check_xy_w
from spatialstats.weights.w import W


def _z(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float).ravel()
    return y - y.mean()

class MoranResult(SpatialResult):
    """Result of a global Moran's I test."""

    def __init__(self, **stats):
        super().__init__(name="Moran's I", **stats)
        self.I = stats.get("I")
        self.p_value = stats.get("p_value")
        self.z_score = stats.get("z_score")

    def plot(self, ax=None, **kwargs):
        from spatialstats.viz.moran import moran_scatter
        return moran_scatter(self, ax=ax, **kwargs)


class GearyResult(SpatialResult):
    def __init__(self, **stats):
        super().__init__(name="Geary's C", **stats)
        self.C = stats.get("C")
        self.p_value = stats.get("p_value")


class GeneralGResult(SpatialResult):
    def __init__(self, **stats):
        super().__init__(name="Getis-Ord General G", **stats)
        self.G = stats.get("G")
        self.p_value = stats.get("p_value")


def _moran_I(y: np.ndarray, w: W) -> float:
    z = _z(y)
    n = len(z)
    Wmat = w.sparse
    num = float(z @ (Wmat @ z))
    den = float(z @ z)
    S0 = float(Wmat.sum())
    if den == 0 or S0 == 0:
        return np.nan
    return (n / S0) * (num / den)


def _moran_expectation_variance(y: np.ndarray, w: W):
    """Analytical E[I] and Var[I] under normality assumption."""
    n = w.n
    EI = -1.0 / (n - 1)
    Wmat = w.sparse
    S0 = float(Wmat.sum())
    S1 = float(0.5 * ((Wmat + Wmat.T).power(2)).sum())
    s2 = np.asarray(Wmat.sum(axis=1)).ravel() + np.asarray(Wmat.sum(axis=0)).ravel()
    S2 = float(np.sum(s2 ** 2))
    z = _z(y)
    m2 = float(np.mean(z ** 2))
    m4 = float(np.mean(z ** 4))
    b2 = m4 / (m2 ** 2) if m2 > 0 else 3.0

    n2, n3, n4 = n * n, n ** 3, n ** 4
    A = n * ((n2 - 3 * n + 3) * S1 - n * S2 + 3 * S0 * S0)
    B = b2 * ((n2 - n) * S1 - 2 * n * S2 + 6 * S0 * S0)
    C = (n - 1) * (n - 2) * (n - 3) * S0 * S0
    VI = (A - B) / C - EI ** 2
    return EI, max(VI, 0.0)


def Moran(
    y,
    w: W,
    permutations: int = 999,
    seed: Optional[int] = None,
) -> MoranResult:
    """
    Global Moran's I.

    Parameters
    ----------
    y : array-like
    w : W
        Spatial weights (typically row-standardized).
    permutations : int
        Conditional permutations for p-value (0 = analytical only).
    """
    y = as_float_vector(y)
    check_y_w(y, w)
    I = _moran_I(y, w)
    EI, VI = _moran_expectation_variance(y, w)
    se = np.sqrt(VI) if VI > 0 else np.nan
    z_score = (I - EI) / se if se and se > 0 else np.nan
    # Analytical two-tailed p (normal approx)
    from scipy.stats import norm
    p_norm = float(2 * (1 - norm.cdf(abs(z_score)))) if np.isfinite(z_score) else np.nan

    p_perm = None
    if permutations and permutations > 0:
        def stat(yp):
            return _moran_I(yp, w)
        p_perm, _ = conditional_permutation(y, stat, permutations, seed=seed)

    res = MoranResult(
        I=float(I),
        expected_I=float(EI),
        var_I=float(VI),
        z_score=float(z_score) if np.isfinite(z_score) else np.nan,
        p_value=float(p_perm) if p_perm is not None else p_norm,
        p_norm=p_norm,
        p_sim=float(p_perm) if p_perm is not None else None,
        permutations=permutations,
        n=w.n,
    )
    res.y = y
    res.w = w
    return res


def _geary_C(y: np.ndarray, w: W) -> float:
    z = _z(y)
    n = len(z)
    Wmat = w.sparse.tocoo()
    num = 0.0
    for i, j, wij in zip(Wmat.row, Wmat.col, Wmat.data):
        num += wij * (y[i] - y[j]) ** 2
    den = float(2 * Wmat.data.sum() * np.sum(z ** 2) / (n - 1)) if n > 1 else np.nan
    if den == 0:
        return np.nan
    return num / den


def Geary(
    y,
    w: W,
    permutations: int = 999,
    seed: Optional[int] = None,
) -> GearyResult:
    """Global Geary's C."""
    y = as_float_vector(y)
    check_y_w(y, w)
    C = _geary_C(y, w)
    p_perm = None
    if permutations and permutations > 0:
        p_perm, _ = conditional_permutation(
            y, lambda yp: _geary_C(yp, w), permutations, seed=seed
        )
    return GearyResult(
        C=float(C),
        expected_C=1.0,
        p_value=float(p_perm) if p_perm is not None else np.nan,
        permutations=permutations,
        n=w.n,
    )


def _general_G(y: np.ndarray, w: W) -> float:
    y = np.asarray(y, dtype=float).ravel()
    Wmat = w.sparse.tocoo()
    num = 0.0
    for i, j, wij in zip(Wmat.row, Wmat.col, Wmat.data):
        if i != j:
            num += wij * y[i] * y[j]
    den = float(np.sum(y) ** 2 - np.sum(y ** 2))
    if den == 0:
        return np.nan
    return num / den


def GeneralG(
    y,
    w: W,
    permutations: int = 999,
    seed: Optional[int] = None,
) -> GeneralGResult:
    """Getis-Ord General G statistic."""
    y = as_float_vector(y)
    check_y_w(y, w)
    G = _general_G(y, w)
    p_perm = None
    if permutations and permutations > 0:
        p_perm, _ = conditional_permutation(
            y, lambda yp: _general_G(yp, w), permutations, seed=seed
        )
    return GeneralGResult(
        G=float(G),
        p_value=float(p_perm) if p_perm is not None else np.nan,
        permutations=permutations,
        n=w.n,
    )


def MoranBV(
    x,
    y,
    w: W,
    permutations: int = 999,
    seed: Optional[int] = None,
) -> MoranResult:
    """
    Bivariate Moran's I: correlation of x with spatial lag of y.
    """
    x = as_float_vector(x, "x")
    y = as_float_vector(y, "y")
    check_xy_w(x, y, w)
    zx = _z(x)
    zy = _z(y)
    n = len(zx)
    Wmat = w.sparse
    S0 = float(Wmat.sum())
    num = float(zx @ (Wmat @ zy))
    den = float(zx @ zx)
    I = (n / S0) * (num / den) if den and S0 else np.nan

    def stat(yp):
        # permute y, keep x fixed (standard bivariate Moran permutation)
        zyp = yp - yp.mean()
        num_p = float(zx @ (Wmat @ zyp))
        return (n / S0) * (num_p / den) if den and S0 else np.nan

    p_perm = None
    if permutations and permutations > 0:
        p_perm, _ = conditional_permutation(y, stat, permutations, seed=seed)

    res = MoranResult(
        I=float(I),
        p_value=float(p_perm) if p_perm is not None else np.nan,
        permutations=permutations,
        n=w.n,
        bivariate=True,
    )
    res.y = x
    res.w = w
    res.y2 = y
    return res


def Moran_Diff(
    y_t,
    y_t0,
    w: W,
    permutations: int = 999,
    seed: Optional[int] = None,
) -> MoranResult:
    """
    Differential Moran's I on ``y_t - y_t0`` (e.g. change between two periods).

    Parameters
    ----------
    y_t, y_t0 : array-like
        Later and earlier period attributes (same length as ``w``).
    w : W
        Spatial weights.
    """
    y_t = as_float_vector(y_t, "y_t")
    y_t0 = as_float_vector(y_t0, "y_t0")
    check_xy_w(y_t, y_t0, w)
    diff = y_t - y_t0
    res = Moran(diff, w, permutations=permutations, seed=seed)
    res.name = "Differential Moran's I"
    res._stats["differential"] = True
    res.y_t = y_t
    res.y_t0 = y_t0
    return res
