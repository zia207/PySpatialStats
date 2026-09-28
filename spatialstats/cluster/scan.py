"""
spatialstats.cluster.scan
=========================
Kulldorff's spatial scan statistic (Poisson model) for detecting clusters of *cases*
in excess of what the population at risk would produce.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.special import xlogy

from spatialstats.core.result import SpatialResult


def _llr(c, e, C, E, direction="high"):
    """Poisson log-likelihood ratio of a window with ``c`` cases / ``e`` expected (totals C, E)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inside = xlogy(c, c / e)
        outside = xlogy(C - c, (C - c) / (C - e))
    llr = inside + outside
    if direction == "high":
        ok = c > e
    elif direction == "low":
        ok = c < e
    else:
        ok = c != e
    return np.where(ok & np.isfinite(llr), llr, 0.0)


class ScanResult(SpatialResult):
    """
    Attributes
    ----------
    clusters : DataFrame   one row per reported cluster: ``center``, ``radius``, ``n_locations``,
                           ``observed``, ``expected``, ``rr`` (relative risk inside vs outside),
                           ``llr``, ``p_value``, ``significant``
    members : list of ndarray   location indices of each cluster
    null_llr : ndarray     max-LLR of each Monte Carlo replicate (the null distribution)
    """

    def __init__(self, clusters: pd.DataFrame, members: List[np.ndarray], null_llr: np.ndarray, n: int, **stats_):
        super().__init__(name="Spatial scan statistic (Poisson)", **stats_)
        self.clusters, self.members, self.null_llr, self.n = clusters, members, null_llr, n
        self._stats["n_significant"] = int(clusters["significant"].sum()) if len(clusters) else 0

    def to_frame(self) -> pd.DataFrame:
        """Per-location table: cluster id (0 = none) and the cluster's relative risk, for *significant* clusters."""
        cid = np.zeros(self.n, dtype=int)
        rr = np.full(self.n, np.nan)
        for k, (row, mem) in enumerate(zip(self.clusters.itertuples(), self.members), start=1):
            if row.significant:
                cid[mem] = k
                rr[mem] = row.rr
        return pd.DataFrame({"cluster": cid, "rr": rr})

    def summary(self) -> str:
        head = f"{self.name}\n{'=' * len(self.name)}"
        return head + "\n" + self.clusters.round(4).to_string(index=False)


def spatial_scan(coords, cases, population=None, expected=None, max_pop_fraction: float = 0.5,
                 max_radius: Optional[float] = None, max_neighbors: Optional[int] = None,
                 min_cases: int = 2, direction: str = "high", n_clusters: int = 3,
                 permutations: int = 999, alpha: float = 0.05, seed: Optional[int] = None) -> ScanResult:
    """
    Kulldorff spatial scan statistic with a Poisson model and circular windows.

    Circles are centred on every location and grown one location at a time. For each window
    the Poisson likelihood ratio of "risk inside differs from risk outside" is computed; the
    window with the largest ratio is the **most likely cluster**. Its significance is judged by
    Monte Carlo: cases are re-drawn (multinomially, proportional to the expected counts, with the
    total fixed) and the maximum ratio over *all* windows recorded, so the p-value already
    accounts for having searched everywhere. Further non-overlapping windows are reported as
    **secondary clusters** and tested against the same null distribution.

    Parameters
    ----------
    coords : array (n, 2)          location (e.g. centroid) of each area, projected units
    cases : array (n,)             observed counts
    population, expected : array (n,)
        Give ``expected`` counts directly, or ``population`` (expected is then proportional to it and
        scaled to sum to the total number of cases).
    max_pop_fraction : float
        Largest window, as a share of the total expected count (SaTScan default 0.5; 0.1–0.25 avoids
        very large, uninteresting clusters).
    max_radius : float, optional     largest window radius in coordinate units
    max_neighbors : int, optional    cap the number of locations in a window (speeds up large n)
    min_cases : int                  minimum cases inside a window
    direction : {'high', 'low', 'both'}
    n_clusters : int                 number of clusters (most likely + secondary) to report
    permutations : int               Monte Carlo replicates (999 → smallest p = 0.001)

    Notes
    -----
    Windows whose boundary would split ties in distance are not evaluated. Memory and time scale as
    ``n²`` (or ``n · max_neighbors``); a few thousand locations is comfortable with ``max_neighbors``.
    """
    xy = np.asarray(coords, dtype=float)
    C_ = np.asarray(cases, dtype=float).ravel()
    n = len(C_)
    if xy.shape != (n, 2):
        raise ValueError("coords must have shape (n, 2) matching cases")
    if np.any(C_ < 0):
        raise ValueError("cases must be non-negative")
    if (population is None) == (expected is None):
        raise ValueError("give exactly one of population or expected")
    base = np.asarray(expected if expected is not None else population, dtype=float).ravel()
    if np.any(base <= 0):
        raise ValueError("population / expected must be strictly positive")
    C = float(C_.sum())
    if C <= 0:
        raise ValueError("no cases")
    E_ = base * C / base.sum()
    E = float(E_.sum())

    k = n if max_neighbors is None else min(int(max_neighbors), n)
    dist, order = cKDTree(xy).query(xy, k=k)
    dist = dist.reshape(n, -1)
    order = order.reshape(n, -1)
    e_cum = np.cumsum(E_[order], axis=1)
    # a window is allowed if it respects the size limits and ends where the distance changes
    valid = e_cum <= max_pop_fraction * E
    if max_radius is not None:
        valid &= dist <= max_radius
    valid[:, :-1] &= dist[:, :-1] < dist[:, 1:]

    c_cum = np.cumsum(C_[order], axis=1)
    llr_mat = np.where(valid & (c_cum >= min_cases), _llr(c_cum, e_cum, C, E, direction), 0.0)

    # ---- Monte Carlo null distribution of the maximum LLR -----------------------------------
    rng = np.random.default_rng(seed)
    p_null = E_ / E
    null = np.empty(permutations)
    vmask = valid
    for b in range(permutations):
        sim = rng.multinomial(int(round(C)), p_null).astype(float)
        cs = np.cumsum(sim[order], axis=1)
        L = np.where(vmask & (cs >= min_cases), _llr(cs, e_cum, C, E, direction), 0.0)
        null[b] = L.max()

    # ---- most likely cluster, then non-overlapping secondaries ------------------------------
    flat = llr_mat.ravel()
    top = np.argsort(-flat)[: min(flat.size, 200_000)]
    chosen, members, rows = np.zeros(n, dtype=bool), [], []
    for f in top:
        if flat[f] <= 0 or len(rows) >= n_clusters:
            break
        i, j = divmod(int(f), llr_mat.shape[1])
        mem = order[i, : j + 1]
        if chosen[mem].any():
            continue
        chosen[mem] = True
        c_in, e_in = c_cum[i, j], e_cum[i, j]
        rr = (c_in / e_in) / (((C - c_in) / (E - e_in)) if (E - e_in) > 0 and (C - c_in) > 0 else np.nan)
        p = (1.0 + np.sum(null >= flat[f] - 1e-12)) / (permutations + 1.0)
        rows.append({"center": i, "radius": float(dist[i, j]), "n_locations": j + 1, "observed": float(c_in),
                     "expected": float(e_in), "rr": float(rr), "llr": float(flat[f]), "p_value": float(p),
                     "significant": bool(p <= alpha)})
        members.append(np.sort(mem))
    cols = ["center", "radius", "n_locations", "observed", "expected", "rr", "llr", "p_value", "significant"]
    clusters = pd.DataFrame(rows, columns=cols)
    return ScanResult(clusters, members, null, n, permutations=permutations, direction=direction,
                      max_pop_fraction=max_pop_fraction, alpha=alpha)
