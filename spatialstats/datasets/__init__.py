"""
spatialstats.datasets
=====================
Bundled example datasets and teaching loaders.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def _make_gdf(data: pd.DataFrame, x_col: str = "x", y_col: str = "y", crs=None):
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        geometry = [Point(xy) for xy in zip(data[x_col], data[y_col])]
        return gpd.GeoDataFrame(data, geometry=geometry, crs=crs)
    except ImportError:
        return data


# Re-export GW teaching sets
def load_georgia(as_gdf: bool = True):
    from spatialstats.gwmodel.datasets import load_georgia as _load
    return _load(as_gdf=as_gdf)


def load_londonhouse(as_gdf: bool = True):
    from spatialstats.gwmodel.datasets import load_londonhouse as _load
    return _load(as_gdf=as_gdf)


def load_beijing_pm25(as_gdf: bool = True):
    from spatialstats.gwmodel.datasets import load_beijing_pm25 as _load
    return _load(as_gdf=as_gdf)


def load_ncovid(as_gdf: bool = True):
    from spatialstats.gwmodel.datasets import load_ncovid as _load
    return _load(as_gdf=as_gdf)


def load_lattice(n: int = 8, seed: int = 0, as_gdf: bool = True):
    """
    Regular square lattice with a spatially autocorrelated attribute.

    Useful for ESDA / weights tutorials.

    Returns
    -------
    gdf or DataFrame, y_col
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        for j in range(n):
            rows.append({"row": i, "col": j, "x": j + 0.5, "y": i + 0.5})

    data = pd.DataFrame(rows)
    data["z"] = (
        np.sin(2 * np.pi * data["x"] / n)
        + np.cos(2 * np.pi * data["y"] / n)
        + rng.normal(0, 0.3, len(data))
    )
    data["binary"] = (data["z"] > data["z"].median()).astype(int)

    if as_gdf:
        try:
            import geopandas as gpd
            from shapely.geometry import box
            geoms = [box(j, i, j + 1, i + 1) for i in range(n) for j in range(n)]
            gdf = gpd.GeoDataFrame(data, geometry=geoms, crs=None)
            return gdf, "z"
        except ImportError:
            return data, "z"
    return data, "z"


def load_points_csr(n: int = 100, seed: int = 1, as_gdf: bool = True):
    """
    Homogeneous Poisson (CSR) point pattern in the unit square.

    Returns
    -------
    gdf or DataFrame
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    y = rng.uniform(0, 1, n)
    data = pd.DataFrame({"x": x, "y": y})
    if as_gdf:
        return _make_gdf(data), None
    return data, None


__all__ = [
    "load_georgia",
    "load_londonhouse",
    "load_beijing_pm25",
    "load_ncovid",
    "load_lattice",
    "load_points_csr",
]
