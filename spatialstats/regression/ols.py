"""
spatialstats.regression.ols
===========================
Ordinary least squares with spatial diagnostics (the baseline for Chapter 3).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from spatialstats.core.result import SpatialResult
from spatialstats.regression import diagnostics as diag
from spatialstats.regression._utils import (
    prepare_data, check_w, coef_table, sig_stars,
)


class RegressionResult(SpatialResult):
    """Shared behaviour of fitted regression models (OLS, SLM, SEM, ...)."""

    model_name = "Regression"
    spatial_param_name: Optional[str] = None

    # subclasses set: y, X, names, y_name, n, k, beta, se, stat, pvals, loglik,
    # residuals, predy, w
    def _finalize_stats(self):
        self._stats.update(
            n=self.n, k=self.k, loglik=self.loglik, aic=self.aic, bic=self.bic,
            pseudo_r2=self.pseudo_r2,
        )

    # -- information criteria (β counts + spatial parameter; σ² not counted, for all models alike)

    @property
    def aic(self) -> float:
        return 2.0 * self.n_params - 2.0 * self.loglik

    @property
    def bic(self) -> float:
        return self.n_params * np.log(self.n) - 2.0 * self.loglik

    @property
    def pseudo_r2(self) -> float:
        """Squared correlation between ``y`` and the model prediction ``predy``."""
        return float(np.corrcoef(self.y, self.predy)[0, 1] ** 2)

    def coef_frame(self) -> pd.DataFrame:
        return coef_table(self.names, self.beta, self.se, self.stat, self.pvals, self.stat_name)

    def to_frame(self) -> pd.DataFrame:
        return self.coef_frame()

    def residual_moran(self, w=None) -> dict:
        """Moran's I of the model residuals (see :func:`spatialstats.regression.moran_residuals`)."""
        w = self.w if w is None else w
        if w is None:
            raise ValueError("provide w")
        return self._residual_moran(w)

    def summary(self) -> str:
        head = f"{self.model_name}   (dependent variable: {self.y_name})"
        lines = [head, "=" * len(head)]
        lines.append(f"  Observations {self.n:>8d}    Log likelihood {self.loglik:>12.4f}")
        lines.append(f"  Parameters   {self.n_params:>8d}    AIC            {self.aic:>12.4f}")
        lines.append(f"  Pseudo R²    {self.pseudo_r2:>8.4f}    BIC            {self.bic:>12.4f}")
        if self.spatial_param_name:
            sp = getattr(self, self.spatial_param_name)
            lines.append(f"  {self.spatial_param_name:<12s} {sp:>8.4f}    "
                         f"std err {self.spatial_se:>8.4f}    p = {self.spatial_p:.4g}")
        lines.append("")
        lines.append(f"  {'variable':<18s}{'coef':>11s}{'std err':>11s}{self.stat_name:>9s}{'p':>10s}")
        for nm, b, s, z, p in zip(self.names, self.beta, self.se, self.stat, self.pvals):
            lines.append(f"  {nm:<18s}{b:>11.4f}{s:>11.4f}{z:>9.3f}{p:>10.4f} {sig_stars(p)}")
        return "\n".join(lines)

    def _residual_moran(self, w) -> dict:
        from spatialstats.esda.global_ import _moran_I
        e = self.residuals
        I = _moran_I(e, w)
        return {"I": float(I)}


class OLS(RegressionResult):
    """
    Ordinary least squares regression with spatial diagnostics.

    Parameters
    ----------
    y : array-like (n,) or Series
    X : array-like (n, p) or DataFrame
        Regressors *without* a constant (added automatically).
    w : W, optional
        Spatial weights. Required for :meth:`lm_tests`, :meth:`residual_moran`
        and ``lag_x=True``.
    add_constant : bool
    names : sequence of str, optional
        Column names when ``X`` is a plain array.
    lag_x : bool
        Add spatially lagged regressors ``WX`` (the SLX model), which captures
        spillovers from neighbours' covariates and is still estimable by OLS.

    Attributes
    ----------
    beta, se, stat (t), pvals, residuals, predy, r2, adj_r2, sigma2,
    loglik (Gaussian ML), aic, bic, vm (covariance of beta)
    """

    model_name = "Ordinary Least Squares"
    stat_name = "t"

    def __init__(self, y, X, w=None, add_constant: bool = True,
                 names: Optional[Sequence[str]] = None, y_name: str = "y", lag_x: bool = False):
        super().__init__(name="OLS")
        yv, Xv, xn, y_name = prepare_data(y, X, add_constant, names, y_name)
        n = len(yv)
        self.w = w
        if lag_x:
            Wm = check_w(w, n)
            start = 1 if add_constant else 0
            WX = Wm @ Xv[:, start:]
            Xv = np.column_stack([Xv, WX])
            xn = xn + [f"W_{c}" for c in xn[start:]]
            if np.linalg.matrix_rank(Xv) < Xv.shape[1]:
                raise ValueError("WX is collinear with X (common for row-standardised W and a constant)")
        elif w is not None:
            check_w(w, n)
        self.y, self.X, self.names, self.y_name = yv, Xv, xn, y_name
        self.n, self.k = Xv.shape
        self.n_params = self.k

        Q = np.linalg.inv(Xv.T @ Xv)
        self.beta = Q @ Xv.T @ yv
        self.predy = Xv @ self.beta
        self.residuals = yv - self.predy
        sse = float(self.residuals @ self.residuals)
        self.df_resid = self.n - self.k
        self.sigma2 = sse / self.df_resid
        self.sigma2_ml = sse / self.n
        self.vm = self.sigma2 * Q
        self.se = np.sqrt(np.diag(self.vm))
        self.stat = self.beta / self.se
        self.pvals = 2 * stats.t.sf(np.abs(self.stat), self.df_resid)
        sst = float(((yv - yv.mean()) ** 2).sum())
        self.r2 = 1 - sse / sst
        self.adj_r2 = 1 - (1 - self.r2) * (self.n - 1) / self.df_resid
        self.loglik = float(-0.5 * self.n * (np.log(2 * np.pi) + np.log(self.sigma2_ml) + 1))
        self._finalize_stats()
        self._stats.update(r2=self.r2, adj_r2=self.adj_r2, sigma2=self.sigma2)

    # -- diagnostics --------------------------------------------------------

    def jarque_bera(self) -> dict:
        return diag.jarque_bera(self.residuals)

    def breusch_pagan(self) -> dict:
        return diag.breusch_pagan(self.residuals, self.X)

    @property
    def condition_number(self) -> float:
        return diag.condition_number(self.X)

    def _residual_moran(self, w) -> dict:
        return diag.moran_residuals(self, w)

    def lm_tests(self, w=None, alpha: float = 0.05) -> diag.LMResult:
        """Lagrange Multiplier tests for a missing spatial lag or spatially correlated errors."""
        w = self.w if w is None else w
        return diag.lm_tests(self, w, alpha=alpha)

    def diagnostics(self, w=None) -> pd.DataFrame:
        """One-table overview of the classical and spatial diagnostics."""
        rows = [
            ("Multicollinearity condition number", self.condition_number, np.nan),
            ("Jarque-Bera (normality)", self.jarque_bera()["statistic"], self.jarque_bera()["p_value"]),
            ("Breusch-Pagan (heteroskedasticity)", self.breusch_pagan()["statistic"],
             self.breusch_pagan()["p_value"]),
        ]
        w = self.w if w is None else w
        if w is not None:
            m = self.residual_moran(w)
            rows.append(("Moran's I (residuals)", m["I"], m["p_value"]))
            lm = self.lm_tests(w).table
            rows += [(r.test, r.statistic, r.p_value) for r in lm.itertuples()]
        return pd.DataFrame(rows, columns=["diagnostic", "value", "p_value"])

    def summary(self, w=None) -> str:
        s = super().summary()
        s += f"\n\n  R² = {self.r2:.4f}   adj. R² = {self.adj_r2:.4f}   σ² = {self.sigma2:.4f}"
        s += "\n\nDiagnostics\n-----------\n" + self.diagnostics(w).to_string(index=False)
        return s
