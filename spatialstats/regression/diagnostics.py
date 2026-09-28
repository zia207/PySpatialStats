"""
spatialstats.regression.diagnostics
===================================
Diagnostics for OLS residuals: spatial dependence (Lagrange Multiplier tests,
Moran's I for regression residuals) and classical assumptions.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from spatialstats.core.result import SpatialResult
from spatialstats.regression._utils import check_w


def _trace_terms(Wm):
    """tr(W W') and tr(W W) for a sparse W."""
    tr_wwt = float(Wm.multiply(Wm).sum())
    tr_ww = float(Wm.multiply(Wm.T).sum())
    return tr_wwt, tr_ww


def jarque_bera(residuals, k: int = 0) -> dict:
    """Jarque–Bera test of normality of residuals (χ² with 2 d.f.)."""
    e = np.asarray(residuals, dtype=float)
    n = len(e)
    e = e - e.mean()
    s2 = np.mean(e ** 2)
    skew = np.mean(e ** 3) / s2 ** 1.5
    kurt = np.mean(e ** 4) / s2 ** 2
    jb = n * (skew ** 2 / 6.0 + (kurt - 3.0) ** 2 / 24.0)
    return {"statistic": float(jb), "df": 2, "p_value": float(stats.chi2.sf(jb, 2)),
            "skewness": float(skew), "kurtosis": float(kurt)}


def breusch_pagan(residuals, X) -> dict:
    """
    Koenker (studentised) Breusch–Pagan test for heteroskedasticity: ``n R²`` of
    the regression of squared residuals on the regressors (χ² with k-1 d.f.).
    """
    e = np.asarray(residuals, dtype=float)
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    u = e ** 2
    b, *_ = np.linalg.lstsq(X, u, rcond=None)
    fitted = X @ b
    sst = np.sum((u - u.mean()) ** 2)
    r2 = 1 - np.sum((u - fitted) ** 2) / sst if sst > 0 else 0.0
    stat = n * r2
    df = k - 1
    return {"statistic": float(stat), "df": df, "p_value": float(stats.chi2.sf(stat, df))}


def condition_number(X) -> float:
    """Multicollinearity condition number (columns scaled to unit length)."""
    X = np.asarray(X, dtype=float)
    Xs = X / np.linalg.norm(X, axis=0)
    s = np.linalg.svd(Xs, compute_uv=False)
    return float(s.max() / s.min())


def moran_residuals(ols, w) -> dict:
    """
    Moran's I for OLS residuals with the exact moments of Cliff & Ord (1981).

    Unlike Moran's I of a raw variable, the residual version's expectation and
    variance depend on the regressors, so the analytical z-score differs from
    the one in :func:`spatialstats.esda.Moran`.

    Returns
    -------
    dict with ``I``, ``expected``, ``variance``, ``z_score``, ``p_value`` (two-sided, normal).
    """
    Wm = check_w(w, ols.n)
    X, e = ols.X, ols.residuals
    n, k = X.shape
    S0 = float(Wm.sum())
    Q = np.linalg.inv(X.T @ X)
    a = X.T @ (Wm @ X)
    b = X.T @ (Wm.T @ X)
    tr_wwt, tr_ww = _trace_terms(Wm)
    tr_mw = float(Wm.diagonal().sum() - np.trace(Q @ a))
    WWt_X = X.T @ (Wm @ (Wm.T @ X))
    WW_X = X.T @ (Wm @ (Wm @ X))
    tr_mwmwt = tr_wwt - 2 * np.trace(Q @ WWt_X) + np.trace(Q @ a @ Q @ b)
    tr_mwmw = tr_ww - 2 * np.trace(Q @ WW_X) + np.trace(Q @ a @ Q @ a)
    df = n - k
    Ei = tr_mw / df
    Vi = (tr_mwmwt + tr_mwmw + tr_mw ** 2) / (df * (df + 2)) - Ei ** 2
    I = float((e @ (Wm @ e)) / (e @ e))
    z = (I - Ei) / np.sqrt(Vi) if Vi > 0 else np.nan
    scale = n / S0
    return {
        "I": I * scale, "expected": Ei * scale, "variance": Vi * scale ** 2,
        "z_score": float(z), "p_value": float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan,
    }


class LMResult(SpatialResult):
    """Lagrange Multiplier diagnostics for spatial dependence in OLS residuals."""

    _ORDER = ["LM-Lag", "Robust LM-Lag", "LM-Error", "Robust LM-Error", "LM-SARMA"]

    def __init__(self, table: pd.DataFrame, alpha: float = 0.05):
        super().__init__(name="Lagrange Multiplier tests")
        self.table = table
        self.alpha = alpha
        for _, row in table.iterrows():
            key = row["test"].lower().replace("-", "_").replace(" ", "_")
            self._stats[key] = float(row["statistic"])
            self._stats["p_" + key] = float(row["p_value"])

    def to_frame(self) -> pd.DataFrame:
        return self.table.copy()

    def summary(self) -> str:
        return f"{self.name}\n{'=' * len(self.name)}\n{self.table.to_string(index=False)}\n" \
               f"\nRecommendation (alpha={self.alpha}): {self.recommendation()}"

    def recommendation(self, alpha: Optional[float] = None) -> str:
        """
        Anselin's (2005) decision rule.

        1. Neither LM-Lag nor LM-Error significant → keep OLS.
        2. Exactly one significant → the corresponding model.
        3. Both significant → compare the robust versions: the one significant
           (or, if both, the larger statistic) picks the model.
        """
        a = self.alpha if alpha is None else alpha
        t = self.table.set_index("test")
        sig = lambda name: t.loc[name, "p_value"] <= a  # noqa: E731
        lag, err = sig("LM-Lag"), sig("LM-Error")
        if not lag and not err:
            return "OLS (no evidence of spatial dependence)"
        if lag and not err:
            return "Spatial lag model"
        if err and not lag:
            return "Spatial error model"
        rlag, rerr = sig("Robust LM-Lag"), sig("Robust LM-Error")
        if rlag and not rerr:
            return "Spatial lag model"
        if rerr and not rlag:
            return "Spatial error model"
        if rlag and rerr:
            better = "Spatial lag model" if t.loc["Robust LM-Lag", "statistic"] >= \
                t.loc["Robust LM-Error", "statistic"] else "Spatial error model"
            return f"{better} (both robust tests significant; larger statistic chosen — consider SARMA/Durbin)"
        return "Inconclusive (robust tests not significant); compare lag and error models by AIC"


def lm_tests(ols, w, alpha: float = 0.05) -> LMResult:
    """
    Lagrange Multiplier tests for spatial dependence (Anselin et al. 1996).

    Computed from the OLS residuals only (no spatial model is fitted):

    * **LM-Lag / LM-Error** – test ρ = 0 / λ = 0 assuming the other is absent.
    * **Robust LM-Lag / Robust LM-Error** – test one form of dependence while
      allowing for the other.
    * **LM-SARMA** – joint test of both.

    Each is χ² with 1 d.f. (2 for SARMA). Use :meth:`LMResult.recommendation`
    for the standard decision rule.
    """
    Wm = check_w(w, ols.n)
    y, X, e = ols.y, ols.X, ols.residuals
    n = ols.n
    s2 = float(e @ e) / n
    tr_wwt, tr_ww = _trace_terms(Wm)
    T = tr_wwt + tr_ww

    d_err = float(e @ (Wm @ e)) / s2
    d_lag = float(e @ (Wm @ y)) / s2
    v = Wm @ (X @ ols.beta)
    Mv = v - X @ np.linalg.solve(X.T @ X, X.T @ v)
    D = float(Mv @ Mv) / s2 + T

    lm_err = d_err ** 2 / T
    lm_lag = d_lag ** 2 / D
    rlm_lag = (d_lag - d_err) ** 2 / (D - T)
    rlm_err = (d_err - (T / D) * d_lag) ** 2 / (T * (1 - T / D))
    sarma = rlm_lag + lm_err

    rows = [
        ("LM-Lag", lm_lag, 1), ("Robust LM-Lag", rlm_lag, 1),
        ("LM-Error", lm_err, 1), ("Robust LM-Error", rlm_err, 1),
        ("LM-SARMA", sarma, 2),
    ]
    table = pd.DataFrame(
        [(t, s, df, float(stats.chi2.sf(s, df))) for t, s, df in rows],
        columns=["test", "statistic", "df", "p_value"],
    )
    return LMResult(table, alpha=alpha)
