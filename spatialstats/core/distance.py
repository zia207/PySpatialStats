"""
spatialstats.core.distance
==========================
Coordinate and distance utilities (great-circle, projected, network hooks).
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np
from scipy.spatial.distance import cdist


ArrayLike = Union[np.ndarray, float]
NetworkDistanceFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def coords_from_geometry(geometry) -> np.ndarray:
    """
    Extract (x, y) coordinate array from a GeoSeries or GeoDataFrame.

    Uses representative points for polygons/lines.
    """
    if hasattr(geometry, "geometry"):
        geometry = geometry.geometry
    pts = geometry.representative_point()
    return np.column_stack([pts.x.to_numpy(), pts.y.to_numpy()])


def euclidean_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Compute n×n Euclidean distance matrix."""
    coords = np.asarray(coords, dtype=float)
    return cdist(coords, coords, metric="euclidean")


def euclidean_distances(coords_a: np.ndarray, coords_b: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distances between two coordinate sets."""
    return cdist(
        np.asarray(coords_a, dtype=float),
        np.asarray(coords_b, dtype=float),
        metric="euclidean",
    )


def haversine_distance_matrix(lat_lon: np.ndarray) -> np.ndarray:
    """
    Great-circle distance matrix in kilometres.

    Parameters
    ----------
    lat_lon : ndarray of shape (n, 2)
        Columns are [latitude, longitude] in degrees.
    """
    R = 6371.0
    lat_lon = np.asarray(lat_lon, dtype=float)
    lat = np.radians(lat_lon[:, 0])
    lon = np.radians(lat_lon[:, 1])
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2) ** 2
    )
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def pairwise_distances(
    coords: np.ndarray,
    metric: str = "euclidean",
    network_fn: Optional[NetworkDistanceFn] = None,
) -> np.ndarray:
    """
    Dispatch distance computation.

    Parameters
    ----------
    coords : ndarray (n, 2)
    metric : {'euclidean', 'haversine', 'network'}
    network_fn : callable, optional
        ``network_fn(coords, coords) -> (n, n)`` distance matrix.
        Required when ``metric='network'``.
    """
    coords = np.asarray(coords, dtype=float)
    if metric == "euclidean":
        return euclidean_distance_matrix(coords)
    if metric == "haversine":
        return haversine_distance_matrix(coords)
    if metric == "network":
        if network_fn is None:
            raise ValueError(
                "metric='network' requires network_fn(coords_a, coords_b) -> distances"
            )
        return np.asarray(network_fn(coords, coords), dtype=float)
    return cdist(coords, coords, metric=metric)


def normalize_coords(coords: np.ndarray) -> np.ndarray:
    """Scale coordinates to [0, 1] per axis."""
    coords = np.asarray(coords, dtype=float)
    mn = coords.min(axis=0)
    mx = coords.max(axis=0)
    rng = mx - mn
    rng[rng == 0] = 1.0
    return (coords - mn) / rng
