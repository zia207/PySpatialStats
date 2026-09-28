"""
spatialstats.weights.distance
=============================
Distance-based spatial weights: KNN, distance band, kernel.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
from scipy.spatial import cKDTree

from spatialstats.core.distance import (
    coords_from_geometry,
    pairwise_distances,
)
from spatialstats.weights.w import W


def _resolve_coords(data) -> np.ndarray:
    if hasattr(data, "geometry"):
        return coords_from_geometry(data)
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("coords must be (n, 2) array or GeoDataFrame")
    return arr


def _kernel_fn(name: str) -> Callable[[np.ndarray, float], np.ndarray]:
    name = name.lower()
    try:
        from spatialstats.gwmodel.core.kernels import KERNEL_FUNCTIONS
        if name in KERNEL_FUNCTIONS:
            return KERNEL_FUNCTIONS[name]
    except ImportError:
        pass

    def gaussian(d, h):
        return np.exp(-(np.asarray(d) / h) ** 2)

    def triangular(d, h):
        d = np.asarray(d, dtype=float)
        return np.where(d < h, 1.0 - d / h, 0.0)

    def bisquare(d, h):
        d = np.asarray(d, dtype=float)
        return np.where(d < h, (1.0 - (d / h) ** 2) ** 2, 0.0)

    mapping = {
        "gaussian": gaussian,
        "triangular": triangular,
        "bisquare": bisquare,
    }
    if name not in mapping:
        raise ValueError(f"Unknown kernel '{name}'. Choose from {list(mapping)}")
    return mapping[name]


def KNN(
    data,
    k: int = 8,
    ids: Optional[Sequence[Any]] = None,
    transform: str = "row",
) -> W:
    """
    k-nearest neighbor weights.

    Parameters
    ----------
    data : GeoDataFrame or ndarray (n, 2)
    k : int
        Number of neighbors (excluding self).
    """
    coords = _resolve_coords(data)
    n = coords.shape[0]
    if ids is None:
        ids = list(range(n)) if not hasattr(data, "index") else list(data.index)
    if k >= n:
        raise ValueError(f"k={k} must be < n={n}")

    tree = cKDTree(coords)
    dists, idxs = tree.query(coords, k=k + 1)  # include self
    neighbors: Dict[Any, List[Any]] = {}
    weights: Dict[Any, List[float]] = {}
    for i, id_i in enumerate(ids):
        nbr_idx = [int(j) for j in np.atleast_1d(idxs[i]) if int(j) != i][:k]
        neighbors[id_i] = [ids[j] for j in nbr_idx]
        weights[id_i] = [1.0] * len(neighbors[id_i])
    return W(neighbors, weights, ids=ids, transform=transform)


def DistanceBand(
    data,
    threshold: float,
    ids: Optional[Sequence[Any]] = None,
    binary: bool = True,
    alpha: float = -1.0,
    transform: Optional[str] = None,
) -> W:
    """
    Distance-band weights: neighbors within ``threshold``.

    Parameters
    ----------
    threshold : float
        Distance cutoff (same units as coordinates).
    binary : bool
        If True, weights are 1; else ``d**alpha``.
    alpha : float
        Distance decay exponent when ``binary=False``.
    """
    coords = _resolve_coords(data)
    n = coords.shape[0]
    if ids is None:
        ids = list(range(n)) if not hasattr(data, "index") else list(data.index)

    tree = cKDTree(coords)
    neighbors: Dict[Any, List[Any]] = {}
    weights: Dict[Any, List[float]] = {}
    for i, id_i in enumerate(ids):
        idxs = tree.query_ball_point(coords[i], r=threshold)
        nbrs, ws = [], []
        for j in idxs:
            if j == i:
                continue
            d = float(np.linalg.norm(coords[i] - coords[j]))
            nbrs.append(ids[j])
            if binary:
                ws.append(1.0)
            else:
                ws.append(d ** alpha if d > 0 else 0.0)
        neighbors[id_i] = nbrs
        weights[id_i] = ws

    if transform is None:
        transform = "binary" if binary else "row"
    return W(neighbors, weights, ids=ids, transform=transform)


def Kernel(
    data,
    bandwidth: Optional[float] = None,
    k: Optional[int] = None,
    kernel: str = "bisquare",
    fixed: bool = True,
    ids: Optional[Sequence[Any]] = None,
    transform: str = "row",
) -> W:
    """
    Kernel-based distance weights (Gaussian, triangular, bisquare).

    Parameters
    ----------
    bandwidth : float, optional
        Fixed distance bandwidth (required if ``fixed=True`` and ``k`` is None).
    k : int, optional
        Adaptive bandwidth: use distance to k-th neighbor as bandwidth.
    kernel : str
        Kernel name.
    fixed : bool
        If False, use adaptive k-NN bandwidth (``k`` required).
    """
    coords = _resolve_coords(data)
    n = coords.shape[0]
    if ids is None:
        ids = list(range(n)) if not hasattr(data, "index") else list(data.index)

    fn = _kernel_fn(kernel)
    tree = cKDTree(coords)
    D = pairwise_distances(coords, metric="euclidean")

    if not fixed:
        if k is None:
            raise ValueError("Adaptive kernel requires k")
        if k >= n:
            raise ValueError(f"k={k} must be < n={n}")
        _, idxs = tree.query(coords, k=k + 1)
        bandwidths = np.array([
            float(np.linalg.norm(coords[i] - coords[int(idxs[i, -1])]))
            for i in range(n)
        ])
        bandwidths[bandwidths == 0] = np.finfo(float).eps
    else:
        if bandwidth is None:
            if k is not None:
                _, idxs = tree.query(coords, k=min(k + 1, n))
                bandwidth = float(np.median([
                    np.linalg.norm(coords[i] - coords[int(idxs[i, -1])])
                    for i in range(n)
                ]))
            else:
                raise ValueError("Fixed kernel requires bandwidth or k")
        bandwidths = np.full(n, float(bandwidth))

    neighbors: Dict[Any, List[Any]] = {}
    weights: Dict[Any, List[float]] = {}
    for i, id_i in enumerate(ids):
        h = bandwidths[i]
        wrow = fn(D[i], h)
        wrow[i] = 0.0
        nbr_idx = np.where(wrow > 0)[0]
        neighbors[id_i] = [ids[j] for j in nbr_idx]
        weights[id_i] = [float(wrow[j]) for j in nbr_idx]

    return W(neighbors, weights, ids=ids, transform=transform)
