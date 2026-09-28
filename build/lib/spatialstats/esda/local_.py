"""
spatialstats.esda.local_
========================
Local indicators: Local Moran (LISA), Local Geary, Getis-Ord Gi/Gi*.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from spatialstats.core.result import SpatialResult
from spatialstats.esda.inference import conditional_permutation, fdr_bh
from spatialstats.esda._utils import as_float_vector, check_y_w
from spatialstats.weights.w import W


def _z(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float).ravel()
    return y - y.mean()


class LocalMoranResult(SpatialResult):
    """Local Moran's I (LISA) result."""

    def __init__(self, Is, p_sim, q, p_adjusted=None, labels=None, **stats):
        super().__init__(name="Local Moran (LISA)", **stats)
        self.Is = np.asarray(Is, dtype=float)
        self.p_sim = np.asarray(p_sim, dtype=float)
        self.q = np.asarray(q, dtype=int)  # quadrant 1=HH,2=LH,3=LL,4=HL
        self.p_adjusted = p_adjusted
        self.labels = labels

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "Ii": self.Is,
            "p_sim": self.p_sim,
            "q": self.q,
            "label": self.labels if self.labels is not None else self._labels_from_q(),
        })
        if self.p_adjusted is not None:
            df["p_adjusted"] = self.p_adjusted
        return df

    def _labels_from_q(self):
        mapping = {1: "HH", 2: "LH", 3: "LL", 4: "HL", 0: "NS"}
        return np.array([mapping.get(int(q), "NS") for q in self.q])

    def plot(self, gdf=None, ax=None, **kwargs):
        from spatialstats.viz.lisa import lisa_cluster_map
        return lisa_cluster_map(self, gdf=gdf, ax=ax, **kwargs)


class LocalGearyResult(SpatialResult):
    def __init__(self, local_C, p_sim, p_adjusted=None, **stats):
        super().__init__(name="Local Geary", **stats)
        self.local_C = np.asarray(local_C, dtype=float)
        self.p_sim = np.asarray(p_sim, dtype=float)
        self.p_adjusted = p_adjusted

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame({"Ci": self.local_C, "p_sim": self.p_sim})
        if self.p_adjusted is not None:
            df["p_adjusted"] = self.p_adjusted
        return df


class GLocalResult(SpatialResult):
    def __init__(self, Gs, p_sim, star: bool = False, p_adjusted=None, **stats):
        name = "Getis-Ord Gi*" if star else "Getis-Ord Gi"
        super().__init__(name=name, **stats)
        self.Gs = np.asarray(Gs, dtype=float)
        self.p_sim = np.asarray(p_sim, dtype=float)
        self.star = star
        self.p_adjusted = p_adjusted

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame({"Gi": self.Gs, "p_sim": self.p_sim})
        if self.p_adjusted is not None:
            df["p_adjusted"] = self.p_adjusted
        return df


def _local_moran_I(y: np.ndarray, w: W) -> np.ndarray:
    z = _z(y)
    den = float(np.sum(z ** 2) / len(z))
    if den == 0:
        return np.full(len(z), np.nan)
    lag = w.lag(z)
    return z * lag / den


def _lisa_quadrants(y: np.ndarray, w: W, Is: np.ndarray, p_sim: np.ndarray, alpha: float):
    z = _z(y)
    lag = w.lag(z)
    q = np.zeros(len(z), dtype=int)
    sig = p_sim <= alpha
    hh = sig & (z > 0) & (lag > 0)
    ll = sig & (z < 0) & (lag < 0)
    lh = sig & (z < 0) & (lag > 0)
    hl = sig & (z > 0) & (lag < 0)
    q[hh] = 1
    q[lh] = 2
    q[ll] = 3
    q[hl] = 4
    labels = np.array(["NS"] * len(z), dtype=object)
    labels[hh] = "HH"
    labels[lh] = "LH"
    labels[ll] = "LL"
    labels[hl] = "HL"
    return q, labels


def Moran_Local(
    y,
    w: W,
    permutations: int = 999,
    seed: Optional[int] = None,
    alpha: float = 0.05,
    fdr: bool = True,
) -> LocalMoranResult:
    """Local Moran's I (Anselin LISA)."""
    y = as_float_vector(y)
    check_y_w(y, w)
    Is = _local_moran_I(y, w)

    if permutations and permutations > 0:
        p_sim, _ = conditional_permutation(
            y, lambda yp: _local_moran_I(yp, w), permutations, seed=seed
        )
    else:
        p_sim = np.full(len(y), np.nan)

    p_adjusted = None
    alpha_use = alpha
    if fdr and np.isfinite(p_sim).all():
        _, p_adjusted = fdr_bh(p_sim, alpha=alpha)
        # Use adjusted p for cluster labeling
        q, labels = _lisa_quadrants(y, w, Is, p_adjusted, alpha)
    else:
        q, labels = _lisa_quadrants(y, w, Is, p_sim, alpha_use)

    res = LocalMoranResult(
        Is=Is,
        p_sim=p_sim,
        q=q,
        p_adjusted=p_adjusted,
        labels=labels,
        alpha=alpha,
        permutations=permutations,
        n=w.n,
    )
    res.y = y
    res.w = w
    return res


def _local_geary(y: np.ndarray, w: W) -> np.ndarray:
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    out = np.zeros(n)
    for i, id_i in enumerate(w.id_order):
        s = 0.0
        for nbr, wij in zip(w.neighbors[id_i], w.weights[id_i]):
            j = w.id_to_index[nbr]
            s += wij * (y[i] - y[j]) ** 2
        out[i] = s
    return out


def Geary_Local(
    y,
    w: W,
    permutations: int = 999,
    seed: Optional[int] = None,
    alpha: float = 0.05,
    fdr: bool = True,
) -> LocalGearyResult:
    """Local Geary's C."""
    y = as_float_vector(y)
    check_y_w(y, w)
    local_C = _local_geary(y, w)
    if permutations and permutations > 0:
        p_sim, _ = conditional_permutation(
            y, lambda yp: _local_geary(yp, w), permutations, seed=seed
        )
    else:
        p_sim = np.full(len(y), np.nan)
    p_adjusted = None
    # FDR only on finite p-values
    finite = np.isfinite(p_sim)
    if fdr and finite.any() and finite.all():
        _, p_adjusted = fdr_bh(p_sim, alpha=alpha)
    elif fdr and finite.any():
        p_adjusted = np.full(len(p_sim), np.nan)
        _, adj = fdr_bh(p_sim[finite], alpha=alpha)
        p_adjusted[finite] = adj
    return LocalGearyResult(
        local_C=local_C,
        p_sim=p_sim,
        p_adjusted=p_adjusted,
        permutations=permutations,
        n=w.n,
    )


def _gi_statistic(y: np.ndarray, w: W, star: bool = False) -> np.ndarray:
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    out = np.full(n, np.nan)
    y_sum = float(y.sum())
    y2_sum = float(np.sum(y ** 2))
    for i, id_i in enumerate(w.id_order):
        nbrs = list(w.neighbors[id_i])
        ws = list(w.weights[id_i])
        if star:
            nbrs = nbrs + [id_i]
            ws = ws + [1.0]
        if len(nbrs) == 0:
            continue  # island → NaN
        wi = np.array(ws, dtype=float)
        yi = np.array(
            [y[w.id_to_index[j]] if j != id_i else y[i] for j in nbrs],
            dtype=float,
        )
        num = float(np.dot(wi, yi))
        W_i = float(wi.sum())
        if W_i == 0:
            continue
        if star:
            ybar = y_sum / n
            s2 = y2_sum / n - ybar ** 2
            denom = np.sqrt(s2 * ((n * np.sum(wi ** 2) - W_i ** 2) / (n - 1)))
            out[i] = (num - ybar * W_i) / denom if denom > 0 else np.nan
        else:
            ybar = (y_sum - y[i]) / (n - 1)
            s2 = ((y2_sum - y[i] ** 2) / (n - 1)) - ybar ** 2
            denom = np.sqrt(
                s2 * (((n - 1) * np.sum(wi ** 2) - W_i ** 2) / (n - 2))
            )
            out[i] = (num - ybar * W_i) / denom if denom > 0 else np.nan
    return out


def G_Local(
    y,
    w: W,
    star: bool = False,
    permutations: int = 999,
    seed: Optional[int] = None,
    alpha: float = 0.05,
    fdr: bool = True,
) -> GLocalResult:
    """Getis-Ord Gi (star=False) or Gi* (star=True)."""
    y = as_float_vector(y)
    check_y_w(y, w)
    Gs = _gi_statistic(y, w, star=star)
    if permutations and permutations > 0:
        p_sim, _ = conditional_permutation(
            y, lambda yp: _gi_statistic(yp, w, star=star), permutations, seed=seed
        )
    else:
        p_sim = np.full(len(y), np.nan)
    p_adjusted = None
    # FDR only on finite p-values (islands / NaNs excluded)
    finite = np.isfinite(p_sim)
    if fdr and finite.any() and finite.all():
        _, p_adjusted = fdr_bh(p_sim, alpha=alpha)
    elif fdr and finite.any():
        p_adjusted = np.full(len(p_sim), np.nan)
        _, adj = fdr_bh(p_sim[finite], alpha=alpha)
        p_adjusted[finite] = adj
    return GLocalResult(
        Gs=Gs,
        p_sim=p_sim,
        star=star,
        p_adjusted=p_adjusted,
        permutations=permutations,
        n=w.n,
    )
