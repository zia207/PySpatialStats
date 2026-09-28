"""
spatialstats.core
=================
Core infrastructure: spatial data containers, result classes, CRS & distance.
"""

from spatialstats.core.data import SpatialData
from spatialstats.core.result import SpatialResult
from spatialstats.core.crs import (
    validate_crs,
    to_crs,
    ensure_projected,
    is_geographic,
    is_projected,
)
from spatialstats.core.distance import (
    coords_from_geometry,
    euclidean_distance_matrix,
    euclidean_distances,
    haversine_distance_matrix,
    pairwise_distances,
    normalize_coords,
)

__all__ = [
    "SpatialData",
    "SpatialResult",
    "validate_crs",
    "to_crs",
    "ensure_projected",
    "is_geographic",
    "is_projected",
    "coords_from_geometry",
    "euclidean_distance_matrix",
    "euclidean_distances",
    "haversine_distance_matrix",
    "pairwise_distances",
    "normalize_coords",
]
