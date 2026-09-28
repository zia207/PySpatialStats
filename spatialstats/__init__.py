"""
spatialstats
============
PySpatialStats — spatial statistics and geographically weighted modeling.

Foundation modules
------------------
core       : SpatialData, SpatialResult, CRS and distance utilities
weights    : Contiguity, distance, and graph-based spatial weights
esda       : Global and local spatial autocorrelation
viz        : Choropleth, Moran scatter, LISA maps
datasets   : Example and teaching datasets
gwmodel    : Geographically weighted models (GWR, MGWR, GW ML, GNNWR, ...)

Spatial statistics modules
--------------------------
pointpattern : Intensity, Ripley K/L/g, CSR envelopes
regression   : OLS diagnostics, LM tests, spatial lag / error / Durbin models
cluster      : Hot spots (Gi*, LISA), scan statistic, DBSCAN, regionalisation
bayes        : Empirical Bayes and BYM disease mapping
interpolate  : trend surfaces, Thiessen polygons, nearest neighbour, IDW, TIN, thin-plate splines, kriging, areal / dasymetric transfer
sampling     : Random, stratified, GRTS, cLHS designs and estimators

Planned modules
---------------
spacetime
"""

__version__ = "0.2.0"
__author__ = "PySpatialStats Contributors"
__license__ = "BSD-3-Clause"

from spatialstats import core, weights, esda, viz, datasets, gwmodel
from spatialstats import (
    pointpattern, interpolate, regression, bayes, spacetime, sampling, cluster,
)

from spatialstats.core import SpatialData, SpatialResult
from spatialstats.weights import W, Queen, Rook, KNN
from spatialstats.esda import Moran, Moran_Local, Geary, GeneralG
from spatialstats.gwmodel import (
    GWR, MGWR, GWSS, GWGLM, GTWR,
    GWRandomForest, GWXGBoost, GWLightGBM,
    GNNWR, GTNNWR, GWPCA, NetworkGWR,
    set_compute_options, get_compute_config, ComputeConfig,
)

__all__ = [
    "__version__",
    # Core
    "SpatialData", "SpatialResult",
    # Weights
    "W", "Queen", "Rook", "KNN",
    # ESDA
    "Moran", "Moran_Local", "Geary", "GeneralG",
    # GW models
    "GWR", "MGWR", "GWSS", "GWGLM", "GTWR",
    "GWRandomForest", "GWXGBoost", "GWLightGBM",
    "GNNWR", "GTNNWR", "GWPCA", "NetworkGWR",
    "set_compute_options", "get_compute_config", "ComputeConfig",
    # Subpackages
    "core", "weights", "esda", "viz", "datasets", "gwmodel",
    "pointpattern", "interpolate", "regression", "bayes", "spacetime", "sampling", "cluster",
]
