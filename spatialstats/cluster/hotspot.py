"""
spatialstats.cluster.hotspot
============================
Local cluster detection for a variable observed on areas: Getis–Ord Gi / Gi* hot spots
and Local Moran (LISA) cluster–outlier maps, with conditional-permutation inference and
multiple-testing control.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse, stats

from spatialstats.cluster._perm import (
    conditional_lag_simulations, permutation_pvalues, adjust_pvalues, check_resolution,
)
from spatialstats.core.result import SpatialResult


class HotSpotResult(SpatialResult):
    """
    Result of :func:`getis_ord_gi`.

    Attributes
    ----------
    z : ndarray            standardised Gi / Gi* (analytical, Ord & Getis 1995)
    p_analytic, p_sim      two-sided p-values from the normal approximation / conditional permutation
    p_adjusted : ndarray   p-values (of the chosen ``inference``) after multiple-testing correction
    labels : ndarray of str  'Hot spot 99%' … 'Cold spot 99%' / 'Not significant'
    hot, cold : boolean masks
    """

    def __init__(self, gi, z, p_analytic, p_sim, p_adjusted, labels, **stats_):
        super().__init__(name="Getis-Ord Gi*" if stats_.get("star") else "Getis-Ord Gi", **stats_)
        self.gi, self.z = gi, z
        self.p_analytic, self.p_sim, self.p_adjusted = p_analytic, p_sim, p_adjusted
        self.labels = labels
        self.hot = np.char.startswith(labels.astype(str), "Hot")
        self.cold = np.char.startswith(labels.astype(str), "Cold")
        self._stats.update(n_hot=int(self.hot.sum()), n_cold=int(self.cold.sum()))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"gi": self.gi, "z": self.z, "p_analytic": self.p_analytic,
                             "p_sim": self.p_sim, "p_adjusted": self.p_adjusted, "label": self.labels})

    def plot(self, gdf=None, ax=None, **kwargs):
        return _plot_labels(self.labels, gdf, ax, _HOT_COLORS, "Hot-spot analysis", **kwargs)


_HOT_COLORS = {
    "Hot spot 99%": "#b2182b", "Hot spot 95%": "#ef8a62", "Hot spot 90%": "#fddbc7",
    "Not significant": "#f0f0f0",
    "Cold spot 90%": "#d1e5f0", "Cold spot 95%": "#67a9cf", "Cold spot 99%": "#2166ac",
}
_LISA_COLORS = {"HH": "#d7191c", "LL": "#2c7bb6", "LH": "#abd9e9", "HL": "#fdae61", "Not significant": "#eeeeee"}


def _plot_labels(labels, gdf, ax, colors, title, **kwargs):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError as e:
        raise ImportError("plotting requires matplotlib. Install with: pip install pyspatialstats[viz]") from e
    if gdf is None:
        raise ValueError("pass the GeoDataFrame the statistic was computed on (gdf=...)")
    if ax is None:
        _, ax = plt.subplots(figsize=kwargs.pop("figsize", (7, 6)))
    g = gdf.copy()
    g["_lab"] = np.asarray(labels).astype(str)
    for lab, col in colors.items():
        sub = g[g["_lab"] == lab]
        if len(sub):
            sub.plot(ax=ax, color=col, edgecolor="k", linewidth=0.2)
    ax.legend(handles=[Patch(facecolor=c, edgecolor="k", label=l) for l, c in colors.items()
                       if (g["_lab"] == l).any()], loc="best", fontsize=8)
    ax.set_axis_off()
    ax.set_title(kwargs.pop("title", title))
    return ax


def _w_matrix(w, star: bool) -> sparse.csr_matrix:
    """
    Weights matrix for Gi (zero diagonal) or Gi* (area added to its own neighbourhood).

    For Gi* the self-weight is the area's mean neighbour weight (1 for binary weights;
    isolated areas get 1), and rows are renormalised if the input was row-standardised, so
    ``Gi*`` keeps the same weighting scheme as the user's ``W``.
    """
    A = w.sparse.tocsr().astype(float).copy()
    A.setdiag(0.0)
    A.eliminate_zeros()
    if not star:
        return A
    deg = np.diff(A.indptr)
    rowsum = np.asarray(A.sum(axis=1)).ravel()
    self_w = np.where(deg > 0, rowsum / np.maximum(deg, 1), 1.0)
    B = A.tolil()
    B.setdiag(self_w)
    B = B.tocsr()
    if w.transform_type == "row":
        B = sparse.diags(1.0 / np.asarray(B.sum(axis=1)).ravel()) @ B
    return B.tocsr()


def _classify(z, p_adj, alpha):
    """Confidence tiers from adjusted p-values: 99% (<= alpha/5), 95% (<= alpha), 90% (<= 2 alpha)."""
    lab = np.full(len(z), "Not significant", dtype=object)
    for tier, thr in (("90%", 2 * alpha), ("95%", alpha), ("99%", alpha / 5.0)):
        lab[(p_adj <= thr) & (z > 0)] = f"Hot spot {tier}"
        lab[(p_adj <= thr) & (z < 0)] = f"Cold spot {tier}"
    return lab


def getis_ord_gi(y, w, star: bool = True, inference: str = "analytic", permutations: int = 999,
                 correction: str = "fdr", alpha: float = 0.05, seed: Optional[int] = None) -> HotSpotResult:
    """
    Getis–Ord Gi* (or Gi) hot-spot analysis.

    ``Gi*_i = Σ_j w_ij y_j / Σ_j y_j`` compares the sum of values in the neighbourhood of *i*
    (including *i* when ``star``) with what randomness would give. Large positive standardised
    scores mark **hot spots** (a high-value area surrounded by high values), large negative ones
    **cold spots**. Unlike Local Moran it does not flag outliers, only clusters of highs or lows.

    Parameters
    ----------
    y : array (n,)  a non-negative ratio/count-like variable is customary, but any variable works
        (the z-score form used here is location/scale invariant).
    w : W
        Spatial weights. With ``star=True`` the area itself is added to its neighbourhood.
    inference : {'analytic', 'permutation'}
        Which p-value drives the classification. ``'analytic'`` uses the normal distribution of the
        standardised score (Ord & Getis 1995; what GIS packages report). ``'permutation'`` uses
        conditional permutation (y_i fixed, neighbours resampled). Both are always returned.
    correction : {'fdr', 'bonferroni', 'none'}
        Multiple-testing correction across the n local tests (a real issue: at α = 0.05 about
        5% of areas would be flagged by chance).
    alpha : float
        Significance level of the adjusted p-value. Tiers: 99% (α/5), 95% (α), 90% (2α).
    """
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    if w.n != n:
        raise ValueError(f"w has {w.n} observations but y has {n}")
    if np.std(y) == 0:
        raise ValueError("y is constant")
    W = _w_matrix(w, star)
    Wsum = np.asarray(W.sum(axis=1)).ravel()
    W2 = np.asarray(W.multiply(W).sum(axis=1)).ravel()
    lag = W @ y

    # Standardised score (valid for arbitrary weights; equals Ord & Getis for Gi*)
    if star:
        Nset = n
        mean_y, sd_y = y.mean(), y.std()
        mean_s = np.full(n, mean_y)
        sd_s = np.full(n, sd_y)
        gi = lag / y.sum()
    else:
        Nset = n - 1
        tot_others = y.sum() - y
        mean_s = tot_others / Nset
        sd_s = np.sqrt(np.maximum((np.sum(y ** 2) - y ** 2) / Nset - mean_s ** 2, 1e-300))
        gi = lag / tot_others
    denom = sd_s * np.sqrt(np.maximum((Nset * W2 - Wsum ** 2) / (Nset - 1), 1e-300))
    z = (lag - mean_s * Wsum) / denom
    p_an = 2.0 * stats.norm.sf(np.abs(z))

    rng = np.random.default_rng(seed)
    sims = conditional_lag_simulations(y, W, permutations, rng) if permutations > 0 else None
    p_sim = permutation_pvalues(lag, sims) if sims is not None else np.full(n, np.nan)

    if inference == "analytic":
        p_use = p_an
    elif inference == "permutation":
        if sims is None:
            raise ValueError("inference='permutation' needs permutations > 0")
        check_resolution(permutations, n, alpha, correction)
        p_use = p_sim
    else:
        raise ValueError("inference must be 'analytic' or 'permutation'")
    p_adj = adjust_pvalues(p_use, correction)
    labels = _classify(z, p_adj, alpha)
    return HotSpotResult(gi, z, p_an, p_sim, p_adj, labels, star=star, inference=inference,
                         correction=correction, alpha=alpha, permutations=permutations, n=n)


class LisaClusterResult(SpatialResult):
    """Result of :func:`local_moran_clusters`."""

    def __init__(self, Ii, p_sim, p_adjusted, quadrant, labels, **stats_):
        super().__init__(name="Local Moran clusters (conditional permutation)", **stats_)
        self.Ii, self.p_sim, self.p_adjusted = Ii, p_sim, p_adjusted
        self.quadrant, self.labels = quadrant, labels
        counts = pd.Series(labels).value_counts()
        for k in ("HH", "LL", "HL", "LH"):
            self._stats[f"n_{k}"] = int(counts.get(k, 0))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"Ii": self.Ii, "p_sim": self.p_sim, "p_adjusted": self.p_adjusted,
                             "quadrant": self.quadrant, "label": self.labels})

    def plot(self, gdf=None, ax=None, **kwargs):
        return _plot_labels(self.labels, gdf, ax, _LISA_COLORS, "Local Moran clusters", **kwargs)


def local_moran_clusters(y, w, permutations: int = 999, correction: str = "fdr", alpha: float = 0.05,
                         seed: Optional[int] = None) -> LisaClusterResult:
    """
    Local Moran's I (LISA) cluster / outlier map with proper *conditional* permutation inference.

    ``I_i = z_i · Σ_j w_ij z_j / m₂`` with ``z = y - ȳ``. For each area ``y_i`` is held fixed
    and the neighbours' values are resampled from the other ``n - 1`` values, which is the
    reference distribution for a *local* statistic. Areas with an adjusted p ≤ ``alpha`` are
    labelled by Moran-scatterplot quadrant: **HH** (high among high), **LL**, **HL**
    (high outlier among lows), **LH** (low outlier among highs).

    Note: with FDR the smallest attainable p-value (2 / (permutations + 1)) limits what can be
    detected; a warning is issued when ``permutations`` is too small for ``n``.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    if w.n != n:
        raise ValueError(f"w has {w.n} observations but y has {n}")
    W = w.sparse.tocsr().astype(float).copy()
    W.setdiag(0.0)
    W.eliminate_zeros()
    z = y - y.mean()
    m2 = float(np.mean(z ** 2))
    if m2 == 0:
        raise ValueError("y is constant")
    lag = W @ z
    Ii = z * lag / m2
    rng = np.random.default_rng(seed)
    sims_lag = conditional_lag_simulations(z, W, permutations, rng)
    sims_I = z[None, :] * sims_lag / m2
    p = permutation_pvalues(Ii, sims_I)
    check_resolution(permutations, n, alpha, correction)
    p_adj = adjust_pvalues(p, correction)
    sig = p_adj <= alpha
    quad = np.zeros(n, dtype=int)
    labels = np.full(n, "Not significant", dtype=object)
    for code, name, cond in ((1, "HH", (z > 0) & (lag > 0)), (2, "LH", (z < 0) & (lag > 0)),
                             (3, "LL", (z < 0) & (lag < 0)), (4, "HL", (z > 0) & (lag < 0))):
        m = sig & cond
        quad[m] = code
        labels[m] = name
    return LisaClusterResult(Ii, p, p_adj, quad, labels, permutations=permutations, correction=correction,
                             alpha=alpha, n=n)
