"""
spatialstats.gwmodel.utils.distance
========================
Distance computation utilities.
"""

import numpy as np
from typing import Optional


def euclidean_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Compute n×n Euclidean distance matrix efficiently."""
    from scipy.spatial.distance import cdist
    return cdist(coords, coords, metric="euclidean")


def haversine_distance_matrix(lat_lon: np.ndarray) -> np.ndarray:
    """
    Compute n×n great-circle distance matrix in kilometres.

    Parameters
    ----------
    lat_lon : ndarray of shape (n, 2)  [latitude, longitude] in degrees.
    """
    R = 6371.0
    lat = np.radians(lat_lon[:, 0])
    lon = np.radians(lat_lon[:, 1])
    n = len(lat)
    D = np.zeros((n, n))
    for i in range(n):
        dlat = lat - lat[i]
        dlon = lon - lon[i]
        a = np.sin(dlat / 2) ** 2 + np.cos(lat[i]) * np.cos(lat) * np.sin(dlon / 2) ** 2
        D[i] = 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return D


def pairwise_distances(coords: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """
    Dispatch to the correct distance function.

    Parameters
    ----------
    coords : ndarray of shape (n, 2)
    metric : 'euclidean' | 'haversine'
    """
    if metric == "haversine":
        return haversine_distance_matrix(coords)
    from scipy.spatial.distance import cdist
    return cdist(coords, coords, metric=metric)


def coords_from_geometry(geometry) -> np.ndarray:
    """
    Extract (x, y) coordinate array from a GeoSeries or GeoDataFrame geometry column.
    """
    import geopandas as gpd
    if hasattr(geometry, "geometry"):
        geometry = geometry.geometry
    pts = geometry.representative_point()
    return np.column_stack([pts.x, pts.y])


def normalize_coords(coords: np.ndarray) -> np.ndarray:
    """Scale coordinates to [0, 1] range (per axis)."""
    mn = coords.min(axis=0)
    mx = coords.max(axis=0)
    rng = mx - mn
    rng[rng == 0] = 1.0
    return (coords - mn) / rng
