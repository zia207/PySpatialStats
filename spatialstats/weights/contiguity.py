"""
spatialstats.weights.contiguity
===============================
Contiguity-based spatial weights (Queen, Rook, higher-order).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np

from spatialstats.weights.w import W


def _build_contiguity(gdf, rook: bool = False) -> Dict[Any, List[Any]]:
    """Build neighbor dict from polygon contiguity using shapely predicates."""
    try:
        from shapely import STRtree
    except ImportError:
        try:
            from shapely.strtree import STRtree
        except ImportError as e:
            raise ImportError(
                "Queen/Rook contiguity requires shapely. "
                "Install with: pip install shapely geopandas"
            ) from e

    geoms = list(gdf.geometry.values)
    n = len(geoms)
    ids = list(gdf.index) if hasattr(gdf, "index") else list(range(n))
    tree = STRtree(geoms)

    neighbors: Dict[Any, List[Any]] = {i: [] for i in ids}

    for i, geom in enumerate(geoms):
        if geom is None or geom.is_empty:
            continue
        try:
            hits = tree.query(geom, predicate="intersects")
        except TypeError:
            # Older shapely: query returns candidate indices without predicate
            hits = tree.query(geom)
        for j in np.atleast_1d(hits):
            j = int(j)
            if j <= i:
                continue
            other = geoms[j]
            if other is None or other.is_empty:
                continue
            if not geom.intersects(other):
                continue
            if rook:
                inter = geom.intersection(other)
                if inter.is_empty:
                    continue
                gtype = inter.geom_type
                if gtype in ("Point", "MultiPoint"):
                    continue
            # Symmetric add
            neighbors[ids[i]].append(ids[j])
            neighbors[ids[j]].append(ids[i])

    for i in neighbors:
        neighbors[i] = sorted(
            set(neighbors[i]),
            key=lambda x: ids.index(x) if x in ids else str(x),
        )
    return neighbors


def Queen(gdf, ids: Optional[Sequence[Any]] = None, transform: str = "row") -> W:
    """
    Queen contiguity weights (share edge or vertex).

    Parameters
    ----------
    gdf : GeoDataFrame
        Polygon geometries.
    ids : sequence, optional
        Observation ids (default: GeoDataFrame index).
    transform : str
        Weight standardization passed to ``W``.
    """
    neighbors = _build_contiguity(gdf, rook=False)
    if ids is None:
        ids = list(gdf.index)
    # Remap if custom ids provided matching positional order
    if list(ids) != list(gdf.index):
        old_ids = list(gdf.index)
        remap = {old: new for old, new in zip(old_ids, ids)}
        neighbors = {
            remap[i]: [remap[j] for j in nbrs]
            for i, nbrs in neighbors.items()
        }
    return W(neighbors, ids=ids, transform=transform)


def Rook(gdf, ids: Optional[Sequence[Any]] = None, transform: str = "row") -> W:
    """
    Rook contiguity weights (share edge of positive length).
    """
    neighbors = _build_contiguity(gdf, rook=True)
    if ids is None:
        ids = list(gdf.index)
    if list(ids) != list(gdf.index):
        old_ids = list(gdf.index)
        remap = {old: new for old, new in zip(old_ids, ids)}
        neighbors = {
            remap[i]: [remap[j] for j in nbrs]
            for i, nbrs in neighbors.items()
        }
    return W(neighbors, ids=ids, transform=transform)


def higher_order(w: W, k: int = 2, lower_order: bool = False) -> W:
    """
    K-order contiguity from an existing weights object.

    Parameters
    ----------
    w : W
        Base (typically 1st-order) weights.
    k : int
        Order of contiguity.
    lower_order : bool
        If True, include all orders 1..k; else only exact order k.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    # BFS from each node
    neighbors: Dict[Any, List[Any]] = {i: [] for i in w.id_order}
    for start in w.id_order:
        visited: Dict[Any, int] = {start: 0}
        queue: List[Any] = [start]
        while queue:
            node = queue.pop(0)
            d = visited[node]
            if d >= k:
                continue
            for nbr in w.neighbors[node]:
                if nbr not in visited:
                    visited[nbr] = d + 1
                    queue.append(nbr)
        for node, d in visited.items():
            if node == start:
                continue
            if lower_order:
                if 1 <= d <= k:
                    neighbors[start].append(node)
            else:
                if d == k:
                    neighbors[start].append(node)
    return W(neighbors, ids=w.id_order, transform=w.transform_type)
