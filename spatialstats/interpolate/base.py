"""
spatialstats.interpolate.base
=============================
Common interface for spatial interpolators and grid helpers.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def as_xy(coords) -> np.ndarray:
    """Coordinates as an (n, 2) float array (ndarray, DataFrame with x/y, or point GeoDataFrame/GeoSeries)."""
    if hasattr(coords, "geometry") or hasattr(coords, "geom_type"):
        from spatialstats.core.distance import coords_from_geometry
        geom = coords.geometry if hasattr(coords, "geometry") else coords
        return coords_from_geometry(geom)
    if isinstance(coords, pd.DataFrame):
        low = {c.lower(): c for c in coords.columns}
        for a, b in (("x", "y"), ("lon", "lat"), ("long", "lat")):
            if a in low and b in low:
                return coords[[low[a], low[b]]].to_numpy(dtype=float)
        raise ValueError("DataFrame coordinates need x/y (or lon/lat) columns")
    c = np.asarray(coords, dtype=float)
    if c.ndim != 2 or c.shape[1] != 2:
        raise ValueError(f"coordinates must have shape (n, 2), got {c.shape}")
    return c


@dataclass
class Grid:
    """A regular prediction grid; ``coords`` lists the cell centres row by row (y ascending)."""
    x: np.ndarray
    y: np.ndarray
    coords: np.ndarray
    inside: np.ndarray           # boolean (ny, nx): cell centre inside the region

    @property
    def shape(self) -> Tuple[int, int]:
        return (len(self.y), len(self.x))

    @property
    def extent(self):
        dx, dy = self.x[1] - self.x[0], self.y[1] - self.y[0]
        return (self.x[0] - dx / 2, self.x[-1] + dx / 2, self.y[0] - dy / 2, self.y[-1] + dy / 2)

    def to_raster(self, values) -> np.ndarray:
        """Reshape predictions for the *inside* cells into an (ny, nx) array (NaN outside)."""
        v = np.asarray(values, dtype=float)
        out = np.full(self.shape, np.nan)
        if len(v) == self.inside.sum():
            out[self.inside] = v
        elif len(v) == out.size:
            out = v.reshape(self.shape)
        else:
            raise ValueError("values must have one entry per grid cell or per inside cell")
        return out

    def to_frame(self, **columns) -> pd.DataFrame:
        df = pd.DataFrame({"x": self.coords[:, 0], "y": self.coords[:, 1]})
        for k, v in columns.items():
            df[k] = np.asarray(v)
        return df


def make_grid(region, resolution: Optional[float] = None, n_cells: int = 100) -> Grid:
    """
    Regular grid over a region.

    Parameters
    ----------
    region : (xmin, ymin, xmax, ymax), shapely polygon, GeoDataFrame, or (n, 2) point array
        For a polygon/GeoDataFrame only cells whose centre lies inside are returned; for a point
        array the bounding box is used.
    resolution : float, optional
        Cell size in coordinate units.
    n_cells : int
        Number of cells along the longer side when ``resolution`` is not given.
    """
    import shapely
    from shapely.geometry import box
    poly = None
    if hasattr(region, "geometry"):
        g = region.geometry
        poly = g.union_all() if hasattr(g, "union_all") else g.unary_union
        bounds = poly.bounds
        if poly.area == 0:                  # a point / line layer: use its bounding box
            poly = None
    elif hasattr(region, "bounds") and hasattr(region, "area"):
        poly, bounds = region, region.bounds
    else:
        arr = np.asarray(region, dtype=float)
        if arr.ndim == 1 and len(arr) == 4:
            bounds = tuple(arr)
        elif arr.ndim == 2 and arr.shape[1] == 2:
            bounds = (arr[:, 0].min(), arr[:, 1].min(), arr[:, 0].max(), arr[:, 1].max())
        else:
            raise ValueError("region must be bounds, a polygon, a GeoDataFrame or an (n, 2) array")
    xmin, ymin, xmax, ymax = bounds
    if resolution is None:
        resolution = max(xmax - xmin, ymax - ymin) / n_cells
    x = np.arange(xmin + resolution / 2, xmax, resolution)
    y = np.arange(ymin + resolution / 2, ymax, resolution)
    gx, gy = np.meshgrid(x, y)
    inside = np.ones(gx.shape, dtype=bool)
    if poly is not None:
        inside = shapely.contains_xy(poly, gx.ravel(), gy.ravel()).reshape(gx.shape)
    coords = np.column_stack([gx[inside], gy[inside]])
    return Grid(x=x, y=y, coords=coords, inside=inside)


class Interpolator:
    """
    Base class: ``fit(coords, values)`` then ``predict(new_coords)``.

    Coordinates must be in projected units for distance-based methods.
    """

    name = "interpolator"

    def fit(self, coords, values) -> "Interpolator":
        xy = as_xy(coords)
        v = np.asarray(values, dtype=float).ravel()
        if len(xy) != len(v):
            raise ValueError(f"{len(xy)} coordinates but {len(v)} values")
        keep = np.isfinite(v) & np.isfinite(xy).all(axis=1)
        if not keep.all():
            xy, v = xy[keep], v[keep]
        if len(v) < 2:
            raise ValueError("need at least 2 valid observations")
        self.coords_, self.values_ = xy, v
        self._fit(xy, v)
        return self

    def _fit(self, xy, v):  # pragma: no cover - interface
        raise NotImplementedError

    def predict(self, coords) -> np.ndarray:
        if not hasattr(self, "coords_"):
            raise RuntimeError("call fit() first")
        return self._predict(as_xy(coords))

    def _predict(self, xy):  # pragma: no cover - interface
        raise NotImplementedError

    def predict_grid(self, region, resolution: Optional[float] = None, n_cells: int = 100):
        """Predict on a regular grid over ``region``; returns ``(grid, values)``."""
        grid = make_grid(region, resolution, n_cells)
        return grid, self.predict(grid.coords)

    def get_params(self) -> dict:
        """Constructor parameters (as in scikit-learn), used by ``clone`` and ``repr``."""
        sig = inspect.signature(type(self).__init__)
        return {p: getattr(self, p) for p in sig.parameters if p != "self" and hasattr(self, p)}

    def clone(self) -> "Interpolator":
        """A fresh, unfitted copy with the same parameters."""
        return type(self)(**self.get_params())

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{self.__class__.__name__}({params})"
