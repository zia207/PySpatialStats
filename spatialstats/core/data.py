"""
spatialstats.core.data
======================
Spatial data containers wrapping GeoDataFrame with metadata.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd


class SpatialData:
    """
    Thin wrapper around a GeoDataFrame with CRS, geometry type, and variable roles.

    Parameters
    ----------
    gdf : GeoDataFrame
        Source spatial table.
    y : str or sequence of str, optional
        Response / attribute column name(s).
    X : sequence of str, optional
        Covariate column names.
    coords : sequence of str, optional
        Coordinate column names (default: derived from geometry).
    time : str, optional
        Time column name for space-time extensions.
    """

    def __init__(
        self,
        gdf,
        y: Optional[Union[str, Sequence[str]]] = None,
        X: Optional[Sequence[str]] = None,
        coords: Optional[Sequence[str]] = None,
        time: Optional[str] = None,
    ):
        try:
            import geopandas as gpd
        except ImportError as e:
            raise ImportError("SpatialData requires geopandas") from e

        if not isinstance(gdf, gpd.GeoDataFrame):
            raise TypeError("gdf must be a GeoDataFrame")
        if gdf.geometry is None:
            raise ValueError("GeoDataFrame must have a geometry column")

        self._gdf = gdf.copy()
        self.roles: Dict[str, Any] = {
            "y": y if isinstance(y, (list, tuple)) or y is None else [y],
            "X": list(X) if X is not None else None,
            "coords": list(coords) if coords is not None else None,
            "time": time,
        }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def gdf(self):
        return self._gdf

    @property
    def crs(self):
        return self._gdf.crs

    @property
    def geometry_type(self) -> Optional[str]:
        types = self._gdf.geometry.geom_type.unique()
        if len(types) == 1:
            return str(types[0])
        return "Mixed"

    @property
    def n(self) -> int:
        return len(self._gdf)

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:
        crs = self.crs if self.crs is not None else "None"
        return (
            f"SpatialData(n={self.n}, geom={self.geometry_type}, "
            f"crs={crs}, roles={ {k: v for k, v in self.roles.items() if v} })"
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def coords_array(self) -> np.ndarray:
        """Return (n, 2) coordinate array from roles or geometry centroids."""
        if self.roles["coords"] is not None:
            cols = self.roles["coords"]
            return self._gdf[list(cols)].to_numpy(dtype=float)
        from spatialstats.core.distance import coords_from_geometry
        return coords_from_geometry(self._gdf)

    def y_array(self) -> np.ndarray:
        """Return response array for role 'y'."""
        cols = self.roles.get("y")
        if not cols:
            raise ValueError("No y role assigned")
        if len(cols) == 1:
            return self._gdf[cols[0]].to_numpy(dtype=float)
        return self._gdf[list(cols)].to_numpy(dtype=float)

    def X_array(self) -> np.ndarray:
        """Return design matrix for role 'X'."""
        cols = self.roles.get("X")
        if not cols:
            raise ValueError("No X role assigned")
        return self._gdf[list(cols)].to_numpy(dtype=float)

    def set_roles(
        self,
        y: Optional[Union[str, Sequence[str]]] = None,
        X: Optional[Sequence[str]] = None,
        coords: Optional[Sequence[str]] = None,
        time: Optional[str] = None,
    ) -> "SpatialData":
        """Update variable roles in place; returns self."""
        if y is not None:
            self.roles["y"] = y if isinstance(y, (list, tuple)) else [y]
        if X is not None:
            self.roles["X"] = list(X)
        if coords is not None:
            self.roles["coords"] = list(coords)
        if time is not None:
            self.roles["time"] = time
        return self

    def to_crs(self, crs) -> "SpatialData":
        """Return a new SpatialData reprojected to ``crs``."""
        from spatialstats.core.crs import to_crs as _to_crs
        return SpatialData(
            _to_crs(self._gdf, crs),
            y=self.roles["y"],
            X=self.roles["X"],
            coords=self.roles["coords"],
            time=self.roles["time"],
        )

    def ensure_projected(self, target_crs: str = "EPSG:3857") -> "SpatialData":
        """Return projected SpatialData (passthrough if already projected)."""
        from spatialstats.core.crs import ensure_projected
        gdf = ensure_projected(self._gdf, target_crs=target_crs)
        return SpatialData(
            gdf,
            y=self.roles["y"],
            X=self.roles["X"],
            coords=self.roles["coords"],
            time=self.roles["time"],
        )

    @classmethod
    def from_frame(
        cls,
        df: pd.DataFrame,
        x_col: str = "x",
        y_col: str = "y",
        crs=None,
        **roles,
    ) -> "SpatialData":
        """Build SpatialData from a DataFrame with x/y columns."""
        import geopandas as gpd
        from shapely.geometry import Point

        geometry = [Point(xy) for xy in zip(df[x_col], df[y_col])]
        gdf = gpd.GeoDataFrame(df.copy(), geometry=geometry, crs=crs)
        return cls(gdf, coords=[x_col, y_col], **roles)
