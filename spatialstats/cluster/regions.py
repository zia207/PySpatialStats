"""
spatialstats.cluster.regions
============================
Spatially constrained clustering (regionalisation): group areas into contiguous
regions that are homogeneous in their attributes.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree

from spatialstats.core.result import SpatialResult


class RegionResult(SpatialResult):
    """
    Attributes
    ----------
    labels : ndarray (n,)   region id (0 … k-1)
    ssw, sst : float        within-region and total sum of squares of the (standardised) attributes
    r2 : float              ``1 - ssw / sst`` (share of attribute variation explained by the regions)
    sizes : ndarray         number of areas per region
    """

    def __init__(self, data, labels, method, **stats_):
        labels = np.asarray(labels, dtype=int)
        X = np.asarray(data, dtype=float)
        ssw = float(sum(((X[labels == r] - X[labels == r].mean(axis=0)) ** 2).sum() for r in np.unique(labels)))
        sst = float(((X - X.mean(axis=0)) ** 2).sum())
        super().__init__(name=f"{method} regionalisation", n_regions=int(len(np.unique(labels))),
                         ssw=ssw, sst=sst, r2=1 - ssw / sst if sst > 0 else np.nan, **stats_)
        self.labels, self.X, self.method = labels, X, method
        self.ssw, self.sst, self.r2 = ssw, sst, self._stats["r2"]
        self.sizes = np.bincount(labels)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"region": self.labels})

    def is_contiguous(self, w) -> bool:
        """True if every region forms a single connected block under the neighbour graph ``w``."""
        A = w.sparse.tocsr()
        A = ((A + A.T) > 0)
        for r in np.unique(self.labels):
            idx = np.flatnonzero(self.labels == r)
            ncomp, _ = connected_components(A[idx][:, idx], directed=False)
            if ncomp > 1:
                return False
        return True

    def plot(self, gdf=None, ax=None, **kwargs):
        import matplotlib.pyplot as plt
        if gdf is None:
            raise ValueError("pass the GeoDataFrame (gdf=...)")
        if ax is None:
            _, ax = plt.subplots(figsize=kwargs.pop("figsize", (7, 6)))
        g = gdf.copy()
        g["region"] = self.labels
        g.plot(column="region", categorical=True, cmap=kwargs.pop("cmap", "tab20"), ax=ax,
               edgecolor="k", linewidth=0.2, **kwargs)
        ax.set_axis_off()
        ax.set_title(f"{self.method}: {self._stats['n_regions']} regions (R²={self.r2:.2f})")
        return ax


def _prepare(data, scale):
    X = data.to_numpy(dtype=float) if hasattr(data, "to_numpy") else np.asarray(data, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if not np.isfinite(X).all():
        raise ValueError("data contain NaN or infinite values")
    if scale:
        sd = X.std(axis=0)
        sd[sd == 0] = 1.0
        X = (X - X.mean(axis=0)) / sd
    return X


def _symmetric_adjacency(w) -> sparse.csr_matrix:
    A = w.sparse.tocsr().astype(float)
    A = ((A + A.T) > 0).astype(float).tolil()
    A.setdiag(0.0)
    A = A.tocsr()
    A.eliminate_zeros()
    return A


def agglomerative(data, w, n_clusters: int, linkage: str = "ward", scale: bool = True) -> RegionResult:
    """
    Contiguity-constrained hierarchical clustering: at each step only *adjacent* clusters may merge
    (the neighbour graph ``w`` is the connectivity constraint).

    ``linkage='ward'`` minimises the increase in within-region variance. If ``w`` has islands or several
    disconnected components, scikit-learn completes the graph, so regions are guaranteed contiguous only
    within each component (check :meth:`RegionResult.is_contiguous`).
    """
    from sklearn.cluster import AgglomerativeClustering
    X = _prepare(data, scale)
    if len(X) != w.n:
        raise ValueError(f"data has {len(X)} rows but w has {w.n} observations")
    A = _symmetric_adjacency(w)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        labels = AgglomerativeClustering(n_clusters=n_clusters, connectivity=A, linkage=linkage).fit_predict(X)
    return RegionResult(X, labels, f"Constrained {linkage}", linkage=linkage)


def _ssd(n, s1, s2):
    return s2 - float(s1 @ s1) / n


def skater(data, w, n_clusters: int, min_size: int = 1, floor=None, floor_value: float = 0.0,
           scale: bool = True) -> RegionResult:
    """
    SKATER (Spatial 'K'luster Analysis by Tree Edge Removal; Assunção et al. 2006).

    1. Build the minimum spanning tree of the neighbour graph, with edge cost = squared
       attribute distance between the two areas, so the tree joins similar neighbours.
    2. Repeatedly cut the tree edge whose removal most reduces the total within-region sum of
       squares, until ``n_clusters`` regions remain. Every region stays contiguous by construction.

    Parameters
    ----------
    min_size : int
        Minimum number of areas per region. It constrains every *cut*; a disconnected component of the
        neighbour graph (an island) is always a region of its own and a warning is issued if it is
        smaller than ``min_size`` or below ``floor_value``.
    floor, floor_value : array (n,), float
        Optional extensive attribute (e.g. population) that each region must total at least
        ``floor_value`` — the "minimum population" constraint used to build stable-rate regions
        for disease mapping. A cut is only allowed if both parts satisfy it.
    """
    X = _prepare(data, scale)
    n, p = X.shape
    if n != w.n:
        raise ValueError(f"data has {n} rows but w has {w.n} observations")
    A = _symmetric_adjacency(w)
    if floor is not None:
        fl = np.asarray(floor, dtype=float).ravel()
        if len(fl) != n:
            raise ValueError("floor must have one value per area")
    else:
        fl = np.zeros(n)
    n_comp, comp = connected_components(A, directed=False)
    if n_clusters < n_comp:
        raise ValueError(f"the neighbour graph has {n_comp} disconnected components (islands count), so at "
                         f"least {n_comp} regions are needed")
    if n_clusters > n:
        raise ValueError("n_clusters exceeds the number of areas")

    coo = sparse.triu(A, k=1).tocoo()
    cost = ((X[coo.row] - X[coo.col]) ** 2).sum(axis=1) + 1e-12       # strictly positive: 0 would drop edges
    G = sparse.coo_matrix((cost, (coo.row, coo.col)), shape=(n, n)).tocsr()
    T = minimum_spanning_tree(G).tocoo()
    tree_adj = [[] for _ in range(n)]
    for a, b in zip(T.row, T.col):
        tree_adj[a].append(b)
        tree_adj[b].append(a)

    # forest: one tree per connected component; each tree is a set of nodes
    trees = [np.flatnonzero(comp == c) for c in range(n_comp)]
    small = [t for t in trees if len(t) < min_size or (floor is not None and fl[t].sum() < floor_value)]
    if small:
        import warnings
        warnings.warn(
            f"{len(small)} disconnected component(s) of the neighbour graph (e.g. islands; sizes "
            f"{[len(t) for t in small][:5]}) cannot meet min_size/floor on their own. They cannot be joined to "
            "another region, so each forms its own region and violates the constraint. Link them to a "
            "neighbour in `w` if that is not acceptable.", stacklevel=2)

    def best_cut(nodes):
        """Best edge to cut in the tree induced on ``nodes``: returns (reduction, child_nodes)."""
        if len(nodes) < 2 * max(min_size, 1):
            return None
        inset = np.zeros(n, dtype=bool)
        inset[nodes] = True
        root = nodes[0]
        parent = {root: -1}
        order = [root]
        stack = [root]
        while stack:
            u = stack.pop()
            for v in tree_adj[u]:
                if inset[v] and v not in parent:
                    parent[v] = u
                    order.append(v)
                    stack.append(v)
        cnt = {u: 1.0 for u in order}
        s1 = {u: X[u].copy() for u in order}
        s2 = {u: float(X[u] @ X[u]) for u in order}
        fsum = {u: fl[u] for u in order}
        for u in reversed(order[1:]):
            pu = parent[u]
            cnt[pu] += cnt[u]
            s1[pu] += s1[u]
            s2[pu] += s2[u]
            fsum[pu] += fsum[u]
        N, S1, S2, F = cnt[root], s1[root], s2[root], fsum[root]
        total = _ssd(N, S1, S2)
        best = None
        for u in order[1:]:
            a_n, b_n = cnt[u], N - cnt[u]
            if a_n < min_size or b_n < min_size:
                continue
            if floor is not None and (fsum[u] < floor_value or F - fsum[u] < floor_value):
                continue
            red = total - _ssd(a_n, s1[u], s2[u]) - _ssd(b_n, S1 - s1[u], S2 - s2[u])
            if best is None or red > best[0]:
                best = (red, u)
        if best is None:
            return None
        # nodes of the subtree rooted at best[1]
        sub, stack = [], [best[1]]
        while stack:
            u = stack.pop()
            sub.append(u)
            for v in tree_adj[u]:
                if inset[v] and parent.get(v) == u:
                    stack.append(v)
        return best[0], np.array(sub), best[1], parent[best[1]]

    cuts = [best_cut(t) for t in trees]
    while len(trees) < n_clusters:
        cand = [(c[0], i) for i, c in enumerate(cuts) if c is not None]
        if not cand:
            raise ValueError(f"cannot form {n_clusters} regions satisfying min_size={min_size} / floor "
                             f"(stopped at {len(trees)})")
        _, i = max(cand)
        red, sub, u, pu = cuts[i]
        tree_adj[u].remove(pu)
        tree_adj[pu].remove(u)
        rest = np.setdiff1d(trees[i], sub)
        trees[i], cuts[i] = sub, None
        trees.append(rest)
        cuts.append(None)
        cuts[i] = best_cut(trees[i])
        cuts[-1] = best_cut(trees[-1])
    labels = np.empty(n, dtype=int)
    for r, t in enumerate(trees):
        labels[t] = r
    return RegionResult(X, labels, "SKATER", min_size=min_size)


def evaluate_regions(data, labels, scale: bool = True) -> dict:
    """``ssw`` (within-region SS), ``r2`` (share of variation explained) and the mean ``silhouette``."""
    from sklearn.metrics import silhouette_score
    X = _prepare(data, scale)
    r = RegionResult(X, labels, "regions")
    out = {"ssw": r.ssw, "sst": r.sst, "r2": r.r2, "n_regions": r.stats["n_regions"]}
    if 1 < len(np.unique(labels)) < len(labels):
        out["silhouette"] = float(silhouette_score(X, labels))
    return out
