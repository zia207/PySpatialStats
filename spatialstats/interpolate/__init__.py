"""
spatialstats.interpolate
========================
Spatial interpolation: deterministic surfaces (polynomial trend, Thiessen
polygons, nearest neighbour, IDW, TIN, thin-plate splines), geostatistical
prediction (variogram modelling; ordinary, simple, universal, co-,
regression and indicator kriging, with E-type estimates), areal / dasymetric transfer
between zone systems, and cross-validation.

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
    VariogramModel, empirical_variogram, fit_variogram, fit_variogram_auto,
    OrdinaryKriging, SimpleKriging, UniversalKriging,
    CrossVariogramModel, empirical_cross_variogram, fit_cross_variogram, fit_lmc,
    check_lmc_validity, Cokriging, MultivariateCokriging, ColocatedCokriging,
    RegressionKriging, GAMRegressor, enforce_quantile_monotonicity, AVAILABLE_REGRESSORS,
    indicator_transform, empirical_indicator_variogram, fit_indicator_variogram,
    IndicatorKriging, etype_estimate,
)
from spatialstats.interpolate.areal import areal_weighting, dasymetric
from spatialstats.interpolate.validate import CVResult, cross_validate, grid_search, compare_methods

__all__ = [
    "Interpolator", "Grid", "make_grid", "as_xy",
    "NearestNeighbor", "IDW", "TIN", "RBF", "Spline",
    "TrendSurface", "Thiessen", "thiessen_polygons", "ThinPlateSpline",
    "VariogramModel", "empirical_variogram", "fit_variogram", "fit_variogram_auto",
    "OrdinaryKriging", "SimpleKriging", "UniversalKriging",
    "CrossVariogramModel", "empirical_cross_variogram", "fit_cross_variogram", "fit_lmc",
    "check_lmc_validity", "Cokriging", "MultivariateCokriging", "ColocatedCokriging",
    "RegressionKriging", "GAMRegressor", "enforce_quantile_monotonicity", "AVAILABLE_REGRESSORS",
    "indicator_transform", "empirical_indicator_variogram", "fit_indicator_variogram",
    "IndicatorKriging", "etype_estimate",
    "areal_weighting", "dasymetric",
    "CVResult", "cross_validate", "grid_search", "compare_methods",
]
