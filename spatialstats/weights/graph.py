"""
spatialstats.weights.graph
==========================
Graph-based spatial weights: Delaunay, Gabriel, relative neighborhood.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.spatial import Delaunay as SciDelaunay

from spatialstats.core.distance import coords_from_geometry
from spatialstats.weights.w import W


def _resolve_coords(data) -> np.ndarray:
    if hasattr(data, "geometry"):
        return coords_from_geometry(data)
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("coords must be (n, 2) array or GeoDataFrame")
    return arr


def _edges_to_w(
    n: int,
    edges: Set[Tuple[int, int]],
    ids: Sequence[Any],
    transform: str,
) -> W:
    neighbors: Dict[Any, List[Any]] = {i: [] for i in ids}
    for i, j in edges:
        neighbors[ids[i]].append(ids[j])
        neighbors[ids[j]].append(ids[i])
    for i in ids:
        neighbors[i] = sorted(set(neighbors[i]), key=lambda x: list(ids).index(x))
    return W(neighbors, ids=ids, transform=transform)


def Delaunay(
    data,
    ids: Optional[Sequence[Any]] = None,
    transform: str = "row",
) -> W:
    """Delaunay triangulation adjacency weights."""
    coords = _resolve_coords(data)
    n = coords.shape[0]
    if ids is None:
        ids = list(range(n)) if not hasattr(data, "index") else list(data.index)
    if n < 3:
        return W({i: [] for i in ids}, ids=ids, transform=transform)

    tri = SciDelaunay(coords)
    edges: Set[Tuple[int, int]] = set()
    for simplex in tri.simplices:
        for a, b in ((0, 1), (1, 2), (0, 2)):
            i, j = int(simplex[a]), int(simplex[b])
            edges.add((min(i, j), max(i, j)))
    return _edges_to_w(n, edges, ids, transform)


def Gabriel(
    data,
    ids: Optional[Sequence[Any]] = None,
    transform: str = "row",
) -> W:
    """
    Gabriel graph: edge (i,j) if no point lies in the diameter circle of i-j.

    Computed by filtering Delaunay edges (Gabriel ⊆ Delaunay).
    """
    coords = _resolve_coords(data)
    n = coords.shape[0]
    if ids is None:
        ids = list(range(n)) if not hasattr(data, "index") else list(data.index)
    if n < 3:
        return W({i: [] for i in ids}, ids=ids, transform=transform)

    tri = SciDelaunay(coords)
    cand: Set[Tuple[int, int]] = set()
    for simplex in tri.simplices:
        for a, b in ((0, 1), (1, 2), (0, 2)):
            i, j = int(simplex[a]), int(simplex[b])
            cand.add((min(i, j), max(i, j)))

    edges: Set[Tuple[int, int]] = set()
    for i, j in cand:
        mid = 0.5 * (coords[i] + coords[j])
        radius = 0.5 * np.linalg.norm(coords[i] - coords[j])
        # Check other points
        ok = True
        for k in range(n):
            if k == i or k == j:
                continue
            if np.linalg.norm(coords[k] - mid) < radius - 1e-12:
                ok = False
                break
        if ok:
            edges.add((i, j))
    return _edges_to_w(n, edges, ids, transform)


def RelativeNeighborhood(
    data,
    ids: Optional[Sequence[Any]] = None,
    transform: str = "row",
) -> W:
    """
    Relative neighborhood graph (RNG): edge (i,j) if no k is closer to both
    than i and j are to each other. RNG ⊆ Gabriel ⊆ Delaunay.
    """
    coords = _resolve_coords(data)
    n = coords.shape[0]
    if ids is None:
        ids = list(range(n)) if not hasattr(data, "index") else list(data.index)
    if n < 3:
        return W({i: [] for i in ids}, ids=ids, transform=transform)

    # Start from Gabriel (subset) then filter, or from Delaunay
    tri = SciDelaunay(coords)
    cand: Set[Tuple[int, int]] = set()
    for simplex in tri.simplices:
        for a, b in ((0, 1), (1, 2), (0, 2)):
            i, j = int(simplex[a]), int(simplex[b])
            cand.add((min(i, j), max(i, j)))

    edges: Set[Tuple[int, int]] = set()
    for i, j in cand:
        dij = np.linalg.norm(coords[i] - coords[j])
        ok = True
        for k in range(n):
            if k == i or k == j:
                continue
            dik = np.linalg.norm(coords[i] - coords[k])
            djk = np.linalg.norm(coords[j] - coords[k])
            if max(dik, djk) < dij - 1e-12:
                ok = False
                break
        if ok:
            edges.add((i, j))
    return _edges_to_w(n, edges, ids, transform)
