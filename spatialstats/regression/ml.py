"""
spatialstats.regression.ml
==========================
Maximum-likelihood spatial regression: spatial lag (SAR / SLM), spatial error
(SEM), and the spatial Durbin model (lag + WX).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar

from spatialstats.regression._utils import (
    prepare_data, check_w, logdet_factory, rho_bounds, dense_inverse_ok,
)
from spatialstats.regression.ols import RegressionResult


def _moran_of(e, w) -> dict:
    from spatialstats.esda import Moran
    m = Moran(e, w, permutations=0)
    return {"I": m.I, "expected": m.get("expected_I"), "z_score": m.z_score, "p_value": m.get("p_norm")}


def _maximise(ll, lo, hi):
    """Bounded maximisation of a concentrated log-likelihood, guarded against boundary optima."""
    res = minimize_scalar(lambda r: -ll(r), bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-8})
    return float(res.x)


class SpatialLag(RegressionResult):
    """
    Spatial lag model (SAR / SLM), fitted by maximum likelihood::

        y = ρ W y + X β + ε,   ε ~ N(0, σ² I)

    The lag ``Wy`` is endogenous, so OLS is inconsistent; ML uses the log-Jacobian
    ``ln|I - ρW|``. A significant ``rho`` indicates *substantive* spatial
    dependence (spillovers). **Coefficients are not marginal effects**: because
    ``y = (I - ρW)⁻¹(Xβ + ε)`` a change in ``x_k`` at one place also moves
    neighbours; use :meth:`impacts` for direct / indirect / total effects.

    Parameters
    ----------
    y, X, w, add_constant, names, y_name : see :class:`~spatialstats.regression.OLS`.
    slx : bool
        Also include ``WX`` (spatial Durbin model, SDM).
    vm : bool
        Compute asymptotic standard errors (needs a dense n×n inverse; n ≤ 6000).

    Attributes
    ----------
    rho, spatial_se, spatial_p, beta, se, stat (z), pvals, sigma2, loglik, aic, bic,
    residuals (ε̂ = y - ρWy - Xβ), predy (ρWy + Xβ), predy_reduced (A⁻¹Xβ), vm
    """

    stat_name = "z"
    spatial_param_name = "rho"

    def __init__(self, y, X, w, add_constant: bool = True, names: Optional[Sequence[str]] = None,
                 y_name: str = "y", slx: bool = False, vm: bool = True):
        super().__init__(name="Spatial Lag")
        yv, Xv, xn, y_name = prepare_data(y, X, add_constant, names, y_name)
        n = len(yv)
        Wm = check_w(w, n)
        if slx:
            start = 1 if add_constant else 0
            Xv = np.column_stack([Xv, Wm @ Xv[:, start:]])
            xn = xn + [f"W_{c}" for c in xn[start:]]
            if np.linalg.matrix_rank(Xv) < Xv.shape[1]:
                raise ValueError("WX is collinear with X (common for row-standardised W and a constant)")
        self.model_name = "Spatial Durbin Model (ML)" if slx else "Spatial Lag Model (ML)"
        self.w, self.slx = w, slx
        self.y, self.X, self.names, self.y_name = yv, Xv, xn, y_name
        self.n, self.k = Xv.shape
        self.n_params = self.k + 1

        Wy = Wm @ yv
        Q = np.linalg.inv(Xv.T @ Xv)
        b0, bL = Q @ Xv.T @ yv, Q @ Xv.T @ Wy
        e0, eL = yv - Xv @ b0, Wy - Xv @ bL
        a, b, c = float(e0 @ e0), float(e0 @ eL), float(eL @ eL)
        logdet = logdet_factory(Wm)

        def conc_ll(rho):
            sse = a - 2 * rho * b + rho ** 2 * c
            return -0.5 * n * np.log(2 * np.pi) - 0.5 * n * np.log(sse / n) - 0.5 * n + logdet(rho)

        lo, hi = rho_bounds(w, Wm)
        rho = _maximise(conc_ll, lo, hi)
        self.rho = rho
        self.rho_bounds = (lo, hi)
        self.beta = b0 - rho * bL
        sse = a - 2 * rho * b + rho ** 2 * c
        self.sigma2 = sse / n
        self.loglik = float(conc_ll(rho))
        self.residuals = yv - rho * Wy - Xv @ self.beta
        self.predy = rho * Wy + Xv @ self.beta
        self._Wm = Wm
        self._eig = None

        A = np.eye(n) - rho * Wm.toarray() if n <= 6000 else None
        self.predy_reduced = np.linalg.solve(A, Xv @ self.beta) if A is not None else None

        self.vm = None
        self.se = self.stat = self.pvals = np.full(self.k, np.nan)
        self.spatial_se = self.spatial_p = np.nan
        if vm:
            self._compute_vm(A)
        self._finalize_stats()
        self._stats.update(rho=self.rho, sigma2=self.sigma2)

    def _compute_vm(self, A):
        dense_inverse_ok(self.n)
        n, k, rho, s2 = self.n, self.k, self.rho, self.sigma2
        Ainv = np.linalg.inv(A)
        Wd = self._Wm.toarray()
        Wt = Wd @ Ainv                        # W A^{-1}
        Xb = self.X @ self.beta
        WtXb = Wt @ Xb
        tr_wt = np.trace(Wt)
        tr_wt2 = np.sum(Wt * Wt.T)            # tr(Wt Wt)
        tr_wtt_wt = np.sum(Wt * Wt)           # tr(Wt' Wt)
        info = np.zeros((k + 2, k + 2))       # order: beta, rho, sigma2
        info[:k, :k] = self.X.T @ self.X / s2
        info[:k, k] = info[k, :k] = self.X.T @ WtXb / s2
        info[k, k] = tr_wt2 + tr_wtt_wt + float(WtXb @ WtXb) / s2
        info[k, k + 1] = info[k + 1, k] = tr_wt / s2
        info[k + 1, k + 1] = n / (2 * s2 ** 2)
        cov = np.linalg.inv(info)
        self.vm_full = cov
        self.vm = cov[:k, :k]
        self.se = np.sqrt(np.diag(self.vm))
        self.stat = self.beta / self.se
        self.pvals = 2 * stats.norm.sf(np.abs(self.stat))
        self.spatial_se = float(np.sqrt(cov[k, k]))
        self.spatial_p = float(2 * stats.norm.sf(abs(rho / self.spatial_se)))

    # -- effects ------------------------------------------------------------

    def _eigenvalues(self):
        if self._eig is None:
            self._eig = np.linalg.eigvals(self._Wm.toarray())
        return self._eig

    def _effects(self, beta, rho):
        """Average direct / total effect of each regressor for given (β, ρ)."""
        n = self.n
        Wm = self._Wm
        lam = self._eigenvalues()
        inv = 1.0 / (1.0 - rho * lam)
        tr_ainv = float(np.real(inv.sum()))
        tr_ainv_w = float(np.real((lam * inv).sum()))
        from scipy.sparse import identity
        from scipy.sparse.linalg import spsolve
        A = (identity(n, format="csc") - rho * Wm).tocsc()
        one = np.ones(n)
        s1 = float(one @ spsolve(A, one))         # 1' A^-1 1
        s2 = float(one @ spsolve(A, Wm @ one))    # 1' A^-1 W 1
        k0 = 1 if "const" in self.names else 0
        p = (len(self.names) - k0) // 2 if self.slx else len(self.names) - k0    # original regressors
        names = self.names[k0:k0 + p]
        b = np.asarray(beta)[k0:k0 + p]
        th = np.asarray(beta)[k0 + p:k0 + 2 * p] if self.slx else np.zeros(p)
        direct = (b * tr_ainv + th * tr_ainv_w) / n
        total = (b * s1 + th * s2) / n
        return names, direct, total - direct, total

    def impacts(self, nsim: int = 500, seed: Optional[int] = None, alpha: float = 0.05) -> pd.DataFrame:
        """
        Direct, indirect (spillover) and total effects of each regressor
        (LeSage & Pace 2009), with simulation-based confidence intervals.

        * **direct** – average effect of changing ``x_k`` in a place on its own ``y``
          (including feedback through neighbours);
        * **indirect** – average effect on *other* places' ``y`` from the same change;
        * **total** = direct + indirect.

        Intervals come from ``nsim`` draws of (β, ρ) from their asymptotic normal
        distribution. Requires ``vm=True`` at fit time.
        """
        if self.vm is None:
            raise ValueError("model was fitted with vm=False; standard errors are required")
        names, d0, i0, t0 = self._effects(self.beta, self.rho)
        rng = np.random.default_rng(seed)
        k = self.k
        idx = list(range(k)) + [k]
        cov = self.vm_full[np.ix_(idx, idx)]
        mean = np.append(self.beta, self.rho)
        lo_r, hi_r = self.rho_bounds
        draws = []
        while len(draws) < nsim:
            s = rng.multivariate_normal(mean, cov)
            if lo_r < s[-1] < hi_r:
                draws.append(s)
        sims = [self._effects(s[:k], s[-1]) for s in draws]
        D = np.array([s[1] for s in sims])
        I = np.array([s[2] for s in sims])
        T = np.array([s[3] for s in sims])
        q = lambda M, p: np.quantile(M, p, axis=0)  # noqa: E731
        out = pd.DataFrame({
            "variable": names,
            "direct": d0, "direct_lo": q(D, alpha / 2), "direct_hi": q(D, 1 - alpha / 2),
            "indirect": i0, "indirect_lo": q(I, alpha / 2), "indirect_hi": q(I, 1 - alpha / 2),
            "total": t0, "total_lo": q(T, alpha / 2), "total_hi": q(T, 1 - alpha / 2),
        })
        return out

    def _residual_moran(self, w) -> dict:
        return _moran_of(self.residuals, w)


class SpatialError(RegressionResult):
    """
    Spatial error model (SEM), fitted by maximum likelihood::

        y = X β + u,   u = λ W u + ε,   ε ~ N(0, σ² I)

    Spatial dependence is a *nuisance*: it arises from omitted or mismeasured
    spatially structured variables, and β keeps its ordinary interpretation
    (no spillover). OLS remains unbiased but its standard errors are wrong.

    Attributes
    ----------
    lam, spatial_se, spatial_p, beta, se, stat (z), pvals, sigma2, loglik, aic, bic,
    residuals (filtered ε̂ = (I - λW)(y - Xβ), i.i.d. if the model fits),
    residuals_raw (y - Xβ), predy (Xβ)
    """

    stat_name = "z"
    spatial_param_name = "lam"
    model_name = "Spatial Error Model (ML)"

    def __init__(self, y, X, w, add_constant: bool = True, names: Optional[Sequence[str]] = None,
                 y_name: str = "y", slx: bool = False, vm: bool = True):
        super().__init__(name="Spatial Error")
        yv, Xv, xn, y_name = prepare_data(y, X, add_constant, names, y_name)
        n = len(yv)
        Wm = check_w(w, n)
        if slx:
            start = 1 if add_constant else 0
            Xv = np.column_stack([Xv, Wm @ Xv[:, start:]])
            xn = xn + [f"W_{c}" for c in xn[start:]]
            self.model_name = "Spatial Durbin Error Model (ML)"
        self.w, self.slx = w, slx
        self.y, self.X, self.names, self.y_name = yv, Xv, xn, y_name
        self.n, self.k = Xv.shape
        self.n_params = self.k + 1

        Wy, WX = Wm @ yv, Wm @ Xv
        logdet = logdet_factory(Wm)

        def parts(lam):
            ys = yv - lam * Wy
            Xs = Xv - lam * WX
            beta, *_ = np.linalg.lstsq(Xs, ys, rcond=None)
            e = ys - Xs @ beta
            return beta, e, Xs

        def conc_ll(lam):
            _, e, _ = parts(lam)
            sse = float(e @ e)
            return -0.5 * n * np.log(2 * np.pi) - 0.5 * n * np.log(sse / n) - 0.5 * n + logdet(lam)

        lo, hi = rho_bounds(w, Wm)
        lam = _maximise(conc_ll, lo, hi)
        self.lam = lam
        self.lam_bounds = (lo, hi)
        beta, e, Xs = parts(lam)
        self.beta = beta
        self.residuals = e
        self.residuals_raw = yv - Xv @ beta
        self.predy = Xv @ beta
        self.sigma2 = float(e @ e) / n
        self.loglik = float(conc_ll(lam))
        self._Xs = Xs

        self.vm = None
        self.se = self.stat = self.pvals = np.full(self.k, np.nan)
        self.spatial_se = self.spatial_p = np.nan
        if vm:
            self._compute_vm(Wm)
        self._finalize_stats()
        self._stats.update(lam=self.lam, sigma2=self.sigma2)

    def _compute_vm(self, Wm):
        dense_inverse_ok(self.n)
        n, k, lam, s2 = self.n, self.k, self.lam, self.sigma2
        B = np.eye(n) - lam * Wm.toarray()
        Wt = Wm.toarray() @ np.linalg.inv(B)
        tr_wt = np.trace(Wt)
        info_ll = np.sum(Wt * Wt.T) + np.sum(Wt * Wt)
        # (λ, σ²) block
        blk = np.array([[info_ll, tr_wt / s2], [tr_wt / s2, n / (2 * s2 ** 2)]])
        cov_ls = np.linalg.inv(blk)
        self.vm = s2 * np.linalg.inv(self._Xs.T @ self._Xs)
        self.se = np.sqrt(np.diag(self.vm))
        self.stat = self.beta / self.se
        self.pvals = 2 * stats.norm.sf(np.abs(self.stat))
        self.spatial_se = float(np.sqrt(cov_ls[0, 0]))
        self.spatial_p = float(2 * stats.norm.sf(abs(lam / self.spatial_se)))

    def _residual_moran(self, w) -> dict:
        return _moran_of(self.residuals, w)


def SpatialDurbin(y, X, w, **kwargs) -> SpatialLag:
    """Spatial Durbin model: :class:`SpatialLag` with spatially lagged regressors (``slx=True``)."""
    return SpatialLag(y, X, w, slx=True, **kwargs)


def SLX(y, X, w, **kwargs):
    """SLX model: OLS with spatially lagged regressors (``lag_x=True``)."""
    from spatialstats.regression.ols import OLS
    return OLS(y, X, w=w, lag_x=True, **kwargs)


def lr_test(restricted, full) -> dict:
    """
    Likelihood-ratio test between nested models fitted by ML/OLS on the *same*
    data and weights, e.g. OLS vs :class:`SpatialLag`, or SAR vs SDM.
    """
    df = full.n_params - restricted.n_params
    if df <= 0:
        raise ValueError("`full` must have more parameters than `restricted`")
    stat = 2.0 * (full.loglik - restricted.loglik)
    return {"statistic": float(stat), "df": int(df), "p_value": float(stats.chi2.sf(max(stat, 0.0), df))}


def compare_models(models, w=None) -> pd.DataFrame:
    """
    Side-by-side comparison of fitted models.

    Parameters
    ----------
    models : dict ``{label: model}`` or list of models
    w : W, optional
        Weights used for the residual-Moran column (default: each model's own ``w``).

    Returns
    -------
    DataFrame with log-likelihood, AIC, BIC, pseudo-R², spatial parameter and the
    Moran's I of residuals (with its p-value). Likelihood-based columns are only
    comparable for the same data and the same W.
    """
    if not isinstance(models, dict):
        models = {getattr(m, "model_name", f"m{i}"): m for i, m in enumerate(models)}
    rows = []
    for label, m in models.items():
        ww = w if w is not None else m.w
        row = {"model": label, "n_params": m.n_params, "loglik": m.loglik, "aic": m.aic,
               "bic": m.bic, "pseudo_r2": m.pseudo_r2,
               "rho": getattr(m, "rho", np.nan), "lambda": getattr(m, "lam", np.nan)}
        if ww is not None:
            mr = m.residual_moran(ww)
            row["resid_moran_I"] = mr["I"]
            row["resid_moran_p"] = mr.get("p_value", np.nan)
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")
