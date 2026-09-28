"""
spatialstats.cluster.points
===========================
Density-based clustering of point events (DBSCAN / HDBSCAN).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, cKDTree

from spatialstats.core.result import SpatialResult


class PointClusterResult(SpatialResult):
    """
    Attributes
    ----------
    labels : ndarray (n,)   cluster id (0, 1, …) or -1 for noise
    summary_frame : DataFrame   per-cluster size, centroid, hull area and density
    """

    def __init__(self, coords, labels, method, **stats_):
        labels = np.asarray(labels, dtype=int)
        super().__init__(name=f"{method} point clusters", n_clusters=int(labels.max() + 1) if (labels >= 0).any() else 0,
                         n_noise=int((labels < 0).sum()), n=len(labels), **stats_)
        self.coords, self.labels = np.asarray(coords, dtype=float), labels
        rows = []
        for c in range(self._stats["n_clusters"]):
            pts = self.coords[labels == c]
            area = np.nan
            if len(pts) >= 3:
                try:
                    area = float(ConvexHull(pts).volume)
                except Exception:
                    area = np.nan
            rows.append({"cluster": c, "n": len(pts), "x": pts[:, 0].mean(), "y": pts[:, 1].mean(),
                         "hull_area": area, "density": len(pts) / area if area and area > 0 else np.nan})
        self.summary_frame = pd.DataFrame(rows, columns=["cluster", "n", "x", "y", "hull_area", "density"])

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"x": self.coords[:, 0], "y": self.coords[:, 1], "cluster": self.labels})

    def plot(self, ax=None, **kwargs):
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=kwargs.pop("figsize", (6, 6)))
        noise = self.labels < 0
        ax.scatter(*self.coords[noise].T, s=2, c="0.8", label="noise")
        kwargs.setdefault("s", 6)
        sc = ax.scatter(*self.coords[~noise].T, c=self.labels[~noise], cmap="tab20", **kwargs)
        ax.set_aspect("equal")
        ax.set_title(f"{self._stats['n_clusters']} clusters, {self._stats['n_noise']} noise points")
        return ax


def k_distance(coords, k: int = 5) -> np.ndarray:
    """
    Sorted distance of every point to its ``k``-th nearest neighbour (the "k-distance plot").
    The knee of this curve is a good ``eps`` for DBSCAN with ``min_samples = k``.
    """
    xy = np.asarray(coords, dtype=float)
    d, _ = cKDTree(xy).query(xy, k=k + 1)
    return np.sort(d[:, -1])[::-1]


def dbscan_clusters(coords, eps: float, min_samples: int = 5) -> PointClusterResult:
    """
    DBSCAN: a point is *core* if at least ``min_samples`` points (itself included) lie within ``eps``;
    clusters are the connected sets of core points plus their border points; the rest is noise.
    Finds arbitrarily shaped clusters of a single density; ``eps`` is in coordinate units (projected!).
    """
    from sklearn.cluster import DBSCAN
    xy = np.asarray(coords, dtype=float)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(xy)
    return PointClusterResult(xy, labels, "DBSCAN", eps=float(eps), min_samples=int(min_samples))


def hdbscan_clusters(coords, min_cluster_size: int = 20, min_samples: Optional[int] = None,
                     cluster_selection_epsilon: float = 0.0) -> PointClusterResult:
    """
    HDBSCAN: like DBSCAN across *all* ``eps`` at once, keeping the clusters that persist, so it
    finds clusters of differing density without tuning ``eps`` (needs scikit-learn ≥ 1.3).
    """
    try:
        from sklearn.cluster import HDBSCAN
    except ImportError as e:  # pragma: no cover
        raise ImportError("hdbscan_clusters requires scikit-learn >= 1.3") from e
    xy = np.asarray(coords, dtype=float)
    labels = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                     cluster_selection_epsilon=cluster_selection_epsilon).fit_predict(xy)
    return PointClusterResult(xy, labels, "HDBSCAN", min_cluster_size=int(min_cluster_size))
