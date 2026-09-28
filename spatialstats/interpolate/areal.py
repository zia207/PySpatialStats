"""
spatialstats.interpolate.areal
==============================
Areal interpolation: transferring data between incompatible zone systems.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _as_list(cols):
    if cols is None:
        return []
    return [cols] if isinstance(cols, str) else list(cols)


def _check_crs(source, target):
    if source.crs is not None and target.crs is not None and source.crs != target.crs:
        raise ValueError(
            f"source CRS ({source.crs.to_string()[:40]}) and target CRS "
            f"({target.crs.to_string()[:40]}) differ; reproject one of them first"
        )
    if source.crs is not None and getattr(source.crs, "is_geographic", False):
        raise ValueError("areal interpolation measures areas; reproject to a projected CRS first")


def areal_weighting(source, target, extensive=None, intensive=None) -> pd.DataFrame:
    """
    Area-weighted interpolation from ``source`` polygons to ``target`` polygons.

    The value is assumed to be *uniformly distributed* inside each source zone.

    * **extensive** variables (counts, totals) are split in proportion to the share
      of each source zone's area falling in the target, then summed: ``Σ_s v_s · A(s∩t)/A(s)``.
      Totals are conserved wherever the targets cover the sources.
    * **intensive** variables (rates, densities, means) are averaged with weights
      ``A(s∩t)``: ``Σ_s v_s A(s∩t) / Σ_s A(s∩t)``.

    Parameters
    ----------
    source, target : GeoDataFrame (polygons, same projected CRS)
    extensive, intensive : column name(s) of ``source``

    Returns
    -------
    DataFrame indexed like ``target``.
    """
    import geopandas as gpd
    ext, inte = _as_list(extensive), _as_list(intensive)
    if not ext and not inte:
        raise ValueError("give at least one extensive or intensive column")
    _check_crs(source, target)
    cols = ext + inte
    src = source[cols + ["geometry"]].copy()
    src["_sid"] = np.arange(len(src))
    src["_sarea"] = src.geometry.area
    tgt = target[["geometry"]].copy()
    tgt["_tid"] = np.arange(len(tgt))
    pieces = gpd.overlay(src, tgt, how="intersection", keep_geom_type=True)
    pieces["_a"] = pieces.geometry.area
    out = pd.DataFrame(index=np.arange(len(tgt)))
    for c in ext:
        val = pieces[c] * pieces["_a"] / pieces["_sarea"]
        out[c] = val.groupby(pieces["_tid"]).sum()
    for c in inte:
        num = (pieces[c] * pieces["_a"]).groupby(pieces["_tid"]).sum()
        den = pieces["_a"].groupby(pieces["_tid"]).sum()
        out[c] = num / den
    out = out.reindex(np.arange(len(tgt)))
    out.index = target.index
    return out


def dasymetric(source, target, ancillary, weight: Optional[str] = None, extensive=None,
               intensive=None) -> pd.DataFrame:
    """
    Dasymetric interpolation: like :func:`areal_weighting`, but a *point* layer of ancillary
    information (e.g. populated places, buildings, land-cover cells) replaces the assumption
    of uniform density.

    Each source zone's extensive value is allocated to the target zones in proportion to the
    ancillary weight found in each ``source ∩ target`` piece; intensive variables are averaged
    with the same ancillary weights. Because activity is concentrated where the ancillary
    weight is, the result is usually far more accurate than area weighting for population-like
    quantities.

    Source zones with no ancillary weight (or points falling outside every target) fall back to
    area weighting so no total is lost.

    Parameters
    ----------
    ancillary : GeoDataFrame of points, same CRS
    weight : column of ``ancillary`` with the weight (default: 1 per point)
    """
    import geopandas as gpd
    ext, inte = _as_list(extensive), _as_list(intensive)
    if not ext and not inte:
        raise ValueError("give at least one extensive or intensive column")
    _check_crs(source, target)
    if not (ancillary.geom_type == "Point").all():
        raise ValueError("ancillary must contain point geometries")
    if ancillary.crs is not None and source.crs is not None and ancillary.crs != source.crs:
        raise ValueError("ancillary points must share the source CRS")

    cols = ext + inte
    src = source[cols + ["geometry"]].copy()
    src["_sid"] = np.arange(len(src))
    tgt = target[["geometry"]].copy()
    tgt["_tid"] = np.arange(len(tgt))
    pts = ancillary[["geometry"]].copy()
    pts["_w"] = 1.0 if weight is None else ancillary[weight].to_numpy(dtype=float)

    j1 = gpd.sjoin(pts, src[["_sid", "geometry"]], how="inner", predicate="within")
    j1 = j1.drop(columns="index_right")
    j2 = gpd.sjoin(j1, tgt[["_tid", "geometry"]], how="inner", predicate="within")
    cell = j2.groupby(["_sid", "_tid"])["_w"].sum().rename("w").reset_index()
    zone_w = cell.groupby("_sid")["w"].sum().rename("W")
    cell = cell.join(zone_w, on="_sid")
    cell["share"] = cell["w"] / cell["W"]

    out = pd.DataFrame(index=np.arange(len(tgt)))
    have_w = set(zone_w.index)
    missing = [i for i in range(len(src)) if i not in have_w]
    fallback = None
    if missing:
        fallback = areal_weighting(source.iloc[missing].reset_index(drop=True), target, ext, inte)
        fallback.index = np.arange(len(tgt))

    for c in ext:
        vals = src.set_index("_sid")[c]
        alloc = cell["share"] * cell["_sid"].map(vals)
        r = alloc.groupby(cell["_tid"]).sum().reindex(np.arange(len(tgt))).fillna(0.0)
        if fallback is not None:
            r = r + fallback[c].fillna(0.0)
        out[c] = r
    for c in inte:
        vals = src.set_index("_sid")[c]
        num = (cell["w"] * cell["_sid"].map(vals)).groupby(cell["_tid"]).sum()
        den = cell["w"].groupby(cell["_tid"]).sum()
        r = (num / den).reindex(np.arange(len(tgt)))
        if fallback is not None:
            r = r.fillna(fallback[c])
        out[c] = r
    out.index = target.index
    return out
