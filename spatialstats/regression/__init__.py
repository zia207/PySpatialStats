"""
spatialstats.regression
=======================
Spatial regression and econometrics: OLS with spatial diagnostics, Lagrange
Multiplier tests, and maximum-likelihood spatial lag / error / Durbin models.

Workflow
--------
>>> from spatialstats.regression import OLS, SpatialLag, SpatialError, compare_models
>>> ols = OLS(y, X, w=w)
>>> ols.lm_tests().recommendation()          # which spatial model does the data ask for?
>>> slm = SpatialLag(y, X, w)                # y = rho W y + X b + e
>>> sem = SpatialError(y, X, w)              # y = X b + u,  u = lam W u + e
>>> compare_models({"OLS": ols, "SLM": slm, "SEM": sem})
>>> slm.impacts()                            # direct / indirect / total effects

Notes
-----
Information criteria count the regression coefficients plus the spatial
parameter (σ² is not counted, identically for every model), so AIC/BIC are
comparable across OLS, SLM, SEM and SDM fitted to the same data and W.
"""

from spatialstats.regression.ols import OLS, RegressionResult
from spatialstats.regression.diagnostics import (
    lm_tests, LMResult, moran_residuals, jarque_bera, breusch_pagan, condition_number,
)
from spatialstats.regression.ml import (
    SpatialLag, SpatialError, SpatialDurbin, SLX, lr_test, compare_models,
)

__all__ = [
    "OLS", "RegressionResult",
    "SpatialLag", "SpatialError", "SpatialDurbin", "SLX",
    "lm_tests", "LMResult", "moran_residuals", "jarque_bera", "breusch_pagan", "condition_number",
    "lr_test", "compare_models",
]
