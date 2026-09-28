"""
spatialstats.core.crs
=====================
CRS validation and reprojection helpers.
"""

from __future__ import annotations

from typing import Any, Optional, Union


def validate_crs(crs) -> Any:
    """
    Validate and normalize a CRS specification.

    Parameters
    ----------
    crs : str, int, dict, or pyproj.CRS-like
        CRS to validate.

    Returns
    -------
    pyproj.CRS
    """
    try:
        from pyproj import CRS
    except ImportError as e:
        raise ImportError("CRS helpers require pyproj (via geopandas)") from e

    if crs is None:
        raise ValueError("CRS is None; specify a valid CRS (e.g. 'EPSG:4326')")
    try:
        return CRS.from_user_input(crs)
    except Exception as e:
        raise ValueError(f"Invalid CRS: {crs!r}") from e


def to_crs(gdf, crs):
    """
    Reproject a GeoDataFrame to ``crs``.

    Parameters
    ----------
    gdf : GeoDataFrame
    crs : CRS-like

    Returns
    -------
    GeoDataFrame
    """
    target = validate_crs(crs)
    if gdf.crs is None:
        raise ValueError(
            "GeoDataFrame has no CRS set. Assign gdf.crs before reprojecting."
        )
    return gdf.to_crs(target)


def is_geographic(crs) -> bool:
    """Return True if CRS is geographic (lat/lon)."""
    if crs is None:
        return False
    c = validate_crs(crs)
    return bool(c.is_geographic)


def is_projected(crs) -> bool:
    """Return True if CRS is projected."""
    if crs is None:
        return False
    c = validate_crs(crs)
    return bool(c.is_projected)


def ensure_projected(gdf, target_crs: Union[str, int] = "EPSG:3857"):
    """
    Return a projected GeoDataFrame.

    If ``gdf`` already has a projected CRS, return a copy unchanged.
    If geographic (or missing CRS after validation of present CRS), reproject
    to ``target_crs``. Missing CRS raises.
    """
    if gdf.crs is None:
        raise ValueError(
            "GeoDataFrame has no CRS. Set gdf.crs or pass a projected dataset."
        )
    if is_projected(gdf.crs):
        return gdf.copy()
    return to_crs(gdf, target_crs)
