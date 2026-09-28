"""
spatialstats.pointpattern.window
================================
Observation windows for point patterns.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import shapely
from shapely.geometry import Polygon, box
from scipy.signal import fftconvolve
from scipy.ndimage import map_coordinates


class Window:
    """
    Observation window (a polygon, or a rectangle in the common case).

    Parameters
    ----------
    polygon : shapely Polygon or MultiPolygon
        The window geometry, in the same (projected) units as the points.
    """

    def __init__(self, polygon):
        if polygon.is_empty:
            raise ValueError("Window geometry is empty")
        self.polygon = polygon
        self.bounds: Tuple[float, float, float, float] = tuple(polygon.bounds)
        self.area: float = float(polygon.area)
        xmin, ymin, xmax, ymax = self.bounds
        self.is_rectangle: bool = bool(
            np.isclose(self.area, (xmax - xmin) * (ymax - ymin), rtol=1e-9)
        )
        self._cov_cache = None

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_bounds(cls, xmin, ymin, xmax, ymax) -> "Window":
        return cls(box(xmin, ymin, xmax, ymax))

    @classmethod
    def from_points(cls, coords, pad: float = 0.0) -> "Window":
        """Bounding-box window of a coordinate array, optionally padded."""
        c = np.asarray(coords, dtype=float)
        return cls.from_bounds(
            c[:, 0].min() - pad, c[:, 1].min() - pad,
            c[:, 0].max() + pad, c[:, 1].max() + pad,
        )

    @classmethod
    def from_gdf(cls, gdf) -> "Window":
        """Union of all geometries of a GeoDataFrame / GeoSeries."""
        geom = gdf.geometry if hasattr(gdf, "geometry") else gdf
        return cls(geom.union_all() if hasattr(geom, "union_all") else geom.unary_union)

    @classmethod
    def convex_hull(cls, coords) -> "Window":
        c = np.asarray(coords, dtype=float)
        hull = shapely.MultiPoint(c).convex_hull
        if not isinstance(hull, Polygon):
            raise ValueError("Points are degenerate (collinear); cannot form a hull")
        return cls(hull)

    # -- geometry -----------------------------------------------------------

    @property
    def short_side(self) -> float:
        xmin, ymin, xmax, ymax = self.bounds
        return min(xmax - xmin, ymax - ymin)

    def contains(self, coords) -> np.ndarray:
        c = np.asarray(coords, dtype=float)
        # ``covers``-like behaviour: boundary points count as inside
        inside = shapely.contains_xy(self.polygon, c[:, 0], c[:, 1])
        on_edge = shapely.intersects_xy(self.polygon, c[:, 0], c[:, 1])
        return np.asarray(inside | on_edge)

    def distance_to_boundary(self, coords) -> np.ndarray:
        """Distance from each point to the window boundary."""
        c = np.asarray(coords, dtype=float)
        return np.asarray(shapely.distance(shapely.points(c), self.polygon.boundary))

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """``n`` uniform random locations in the window (rejection sampling)."""
        xmin, ymin, xmax, ymax = self.bounds
        if self.is_rectangle:
            return np.column_stack([rng.uniform(xmin, xmax, n), rng.uniform(ymin, ymax, n)])
        out = np.empty((0, 2))
        frac = max(self.area / ((xmax - xmin) * (ymax - ymin)), 1e-3)
        while len(out) < n:
            m = int(np.ceil((n - len(out)) / frac * 1.2)) + 10
            cand = np.column_stack([rng.uniform(xmin, xmax, m), rng.uniform(ymin, ymax, m)])
            out = np.vstack([out, cand[self.contains(cand)]])
        return out[:n]

    def raster(self, resolution: int = 512):
        """Boolean mask of the window on a regular grid (rows = y, cols = x)."""
        xmin, ymin, xmax, ymax = self.bounds
        step = max(xmax - xmin, ymax - ymin) / resolution
        nx = max(int(np.ceil((xmax - xmin) / step)), 1)
        ny = max(int(np.ceil((ymax - ymin) / step)), 1)
        xs = xmin + (np.arange(nx) + 0.5) * step
        ys = ymin + (np.arange(ny) + 0.5) * step
        gx, gy = np.meshgrid(xs, ys)
        mask = shapely.contains_xy(self.polygon, gx.ravel(), gy.ravel()).reshape(ny, nx)
        return mask, xs, ys, step

    def set_covariance(self, vectors: np.ndarray, resolution: int = 512) -> np.ndarray:
        """
        Set covariance ``|W ∩ (W + v)|`` for displacement vectors ``v`` (m, 2).

        Exact for rectangles; for general polygons it is computed from a raster
        autocorrelation of the window (accuracy ~ window extent / ``resolution``).
        """
        v = np.asarray(vectors, dtype=float)
        if self.is_rectangle:
            xmin, ymin, xmax, ymax = self.bounds
            return np.clip((xmax - xmin) - np.abs(v[:, 0]), 0, None) * \
                np.clip((ymax - ymin) - np.abs(v[:, 1]), 0, None)
        if self._cov_cache is None or self._cov_cache[0] != resolution:
            mask, _, _, step = self.raster(resolution)
            m = mask.astype(float)
            cov = fftconvolve(m, m[::-1, ::-1], mode="full") * step ** 2
            self._cov_cache = (resolution, cov, step)
        _, cov, step = self._cov_cache
        ny2, nx2 = cov.shape
        rows = (ny2 - 1) / 2.0 + v[:, 1] / step
        cols = (nx2 - 1) / 2.0 + v[:, 0] / step
        out = map_coordinates(cov, [rows, cols], order=1, mode="constant", cval=0.0)
        return np.clip(out, 0, None)

    def __repr__(self) -> str:
        kind = "rectangle" if self.is_rectangle else "polygon"
        return f"Window({kind}, area={self.area:.6g}, bounds={tuple(round(b, 3) for b in self.bounds)})"
