"""
spatialstats.interpolate
========================
Spatial interpolation: deterministic surfaces (polynomial trend, Thiessen
polygons, nearest neighbour, IDW, TIN, thin-plate splines), geostatistical
prediction (variogram modelling, ordinary kriging with prediction variance),
areal / dasymetric transfer between zone systems, and cross-validation.

Every interpolator shares one interface::

    model = IDW(power=2).fit(coords, values)
    model.predict(new_coords)
    grid, surface = model.predict_grid(region, resolution=5_000)

Coordinates must be in a *projected* CRS. Assess accuracy with
:func:`cross_validate` (``method='spatial_block'`` for clustered networks).
"""

from spatialstats.interpolate.base import Interpolator, Grid, make_grid, as_xy
from spatialstats.interpolate.deterministic import (
    NearestNeighbor, IDW, TIN, RBF, Spline, TrendSurface, Thiessen, thiessen_polygons,
    ThinPlateSpline,
)
from spatialstats.interpolate.kriging import (
    VariogramModel, empirical_variogram, fit_variogram, fit_variogram_auto, OrdinaryKriging,
)
from spatialstats.interpolate.areal import areal_weighting, dasymetric
from spatialstats.interpolate.validate import CVResult, cross_validate, grid_search, compare_methods

__all__ = [
    "Interpolator", "Grid", "make_grid", "as_xy",
    "NearestNeighbor", "IDW", "TIN", "RBF", "Spline",
    "TrendSurface", "Thiessen", "thiessen_polygons", "ThinPlateSpline",
    "VariogramModel", "empirical_variogram", "fit_variogram", "fit_variogram_auto", "OrdinaryKriging",
    "areal_weighting", "dasymetric",
    "CVResult", "cross_validate", "grid_search", "compare_methods",
]
