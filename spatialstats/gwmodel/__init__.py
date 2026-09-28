"""
spatialstats.gwmodel
====================
Geographically Weighted Modeling (formerly PyGWmodel).

Modules
-------
core          : Kernels, bandwidth selection, base classes
linear        : GWR, MGWR, GWGLM, GWSS
spatiotemporal: GTWR, Multiscale GTWR
ensemble      : GW Random Forest, GW Extra Trees, GW XGBoost, GW LightGBM
deep          : GNNWR, GTNNWR (PyTorch-based neural GW models)
multivariate  : GWPCA, GW Discriminant Analysis
network       : NetworkGWR (network distance-based GWR)
diagnostics   : Collinearity, heterogeneity tests, Moran's I
visualization : Coefficient maps, importance maps, diagnostic plots
datasets      : Bundled example datasets
utils         : Distance, stats, parallel helpers
"""

__version__ = "0.1.0"
__author__ = "PySpatialStats Contributors"
__license__ = "BSD-3-Clause"

from spatialstats.gwmodel import (
    core, linear, spatiotemporal, ensemble, deep,
    multivariate, network, diagnostics, visualization, datasets, utils,
)

from spatialstats.gwmodel.linear import GWR, MGWR, GWSS, GWGLM
from spatialstats.gwmodel.spatiotemporal import GTWR
from spatialstats.gwmodel.ensemble import GWRandomForest, GWXGBoost, GWLightGBM
from spatialstats.gwmodel.deep import GNNWR, GTNNWR
from spatialstats.gwmodel.multivariate import GWPCA
from spatialstats.gwmodel.network import NetworkGWR
from spatialstats.gwmodel.utils.compute import set_compute_options, get_compute_config, ComputeConfig

__all__ = [
    "GWR", "MGWR", "GWSS", "GWGLM",
    "GTWR",
    "GWRandomForest", "GWXGBoost", "GWLightGBM",
    "GNNWR", "GTNNWR",
    "GWPCA",
    "NetworkGWR",
    "set_compute_options", "get_compute_config", "ComputeConfig",
    "core", "linear", "spatiotemporal", "ensemble", "deep",
    "multivariate", "network", "diagnostics", "visualization",
    "datasets", "utils",
]
