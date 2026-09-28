"""
spatialstats.bayes.eb
=====================
Empirical Bayes smoothing of small-area rates: gamma–Poisson (global),
James–Stein / normal shrinkage, and spatial (local neighbourhood) EB.
"""

from __future__ import annotations


import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from spatialstats.bayes.rates import _as_counts
from spatialstats.core.result import SpatialResult


class EBResult(SpatialResult):
    """
    Empirical Bayes estimates.

    Attributes
    ----------
    estimate : ndarray   smoothed relative risks (posterior means)
    raw : ndarray        the unsmoothed SMRs
    shrinkage : ndarray  weight on the *data* in [0, 1] (0 = fully shrunk to the prior mean)
    prior_mean : float or ndarray
    lo, hi : ndarray     credible interval (gamma–Poisson only)
    """

    def __init__(self, name, estimate, raw, shrinkage, prior_mean, lo=None, hi=None, **stats_):
        super().__init__(name=name, **stats_)
        self.estimate = np.asarray(estimate, dtype=float)
        self.raw = np.asarray(raw, dtype=float)
        self.shrinkage = np.asarray(shrinkage, dtype=float)
        self.prior_mean = prior_mean
        if np.ndim(prior_mean) == 0:
            self._stats["prior_mean"] = float(prior_mean)
        self.lo = lo
        self.hi = hi

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame({"raw": self.raw, "eb": self.estimate, "weight_on_data": self.shrinkage})
        if self.lo is not None:
            df["lo"], df["hi"] = self.lo, self.hi
        return df


def _nb_negloglik(params, O, E):
    a, b = np.exp(params)
    return -np.sum(
        special.gammaln(O + a) - special.gammaln(a) - special.gammaln(O + 1)
        + a * np.log(b / (b + E)) + O * np.log(E / (b + E))
    )


def eb_gamma_poisson(observed, expected, method: str = "mle", alpha: float = 0.05) -> EBResult:
    """
    Global gamma–Poisson empirical Bayes.

    Model: ``O_i | θ_i ~ Poisson(E_i θ_i)``, ``θ_i ~ Gamma(shape a, rate b)``.
    The posterior is ``Gamma(a + O_i, b + E_i)`` with mean::

        (a + O_i) / (b + E_i)  =  w_i · SMR_i + (1 - w_i) · (a/b),   w_i = E_i / (E_i + b)

    so areas with a small expected count ``E_i`` are pulled hardest toward the
    overall mean ``a/b``. The prior parameters are estimated from the data.

    Parameters
    ----------
    method : {'mle', 'mom'}
        ``'mle'`` maximises the negative-binomial marginal likelihood;
        ``'mom'`` uses Marshall's (1991) method of moments.
    alpha : float
        Credible-interval level (``1 - alpha``) from the gamma posterior.
    """
    O, E = _as_counts(observed, expected)
    r = O / E
    m = O.sum() / E.sum()
    s2 = float(np.sum(E * (r - m) ** 2) / E.sum() - m / E.mean())   # Marshall's variance of θ

    if method == "mom":
        if s2 <= 0:
            a = b = np.inf
        else:
            a, b = m ** 2 / s2, m / s2
    elif method == "mle":
        if s2 <= 0:
            start = np.log([1e3, 1e3 / m])      # near-degenerate prior
        else:
            start = np.log([m ** 2 / s2, m / s2])
        res = optimize.minimize(_nb_negloglik, start, args=(O, E), method="Nelder-Mead",
                                options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 2000})
        a, b = np.exp(res.x)
    else:
        raise ValueError("method must be 'mle' or 'mom'")

    if np.isinf(b):
        w = np.zeros_like(E)
        est = np.full_like(E, m)
        lo = hi = None
    else:
        w = E / (E + b)
        est = (a + O) / (b + E)
        lo = stats.gamma.ppf(alpha / 2, a + O, scale=1.0 / (b + E))
        hi = stats.gamma.ppf(1 - alpha / 2, a + O, scale=1.0 / (b + E))
    return EBResult("EB gamma-Poisson", est, r, w, a / b if np.isfinite(b) else m, lo, hi,
                    shape=float(a), rate=float(b),
                    prior_var=float(a / b ** 2) if np.isfinite(b) else 0.0, method=method)


def eb_james_stein(estimates, variances, method: str = "normal", positive_part: bool = True) -> EBResult:
    """
    Normal shrinkage of noisy estimates ``x_i ~ N(θ_i, σ_i²)`` toward their common mean.

    Suitable for log-relative-risks or transformed rates (``σ_i² ≈ 1 / O_i`` for
    ``log SMR``). Two variants:

    ``'james-stein'``
        The classical rule with a *common* variance ``σ²`` (the mean of
        ``variances``): ``θ̂_i = x̄ + c (x_i - x̄)``, ``c = 1 - (k-3) σ² / Σ(x_j - x̄)²``
        (Lindley's version, shrinking toward the estimated mean; needs ``k ≥ 4``).
    ``'normal'``
        Parametric EB for unequal variances: ``θ_i ~ N(μ, τ²)`` with ``τ²`` estimated
        by the DerSimonian–Laird moment estimator and ``μ`` the precision-weighted mean;
        ``θ̂_i = μ + τ²/(τ² + σ_i²) (x_i - μ)``.

    Parameters
    ----------
    positive_part : bool
        Truncate negative shrinkage factors at zero (never over-shrink past the mean).
    """
    x = np.asarray(estimates, dtype=float).ravel()
    v = np.asarray(variances, dtype=float).ravel()
    if len(x) != len(v):
        raise ValueError("estimates and variances must have the same length")
    if np.any(v <= 0):
        raise ValueError("variances must be positive")
    k = len(x)
    if method == "james-stein":
        if k < 4:
            raise ValueError("the James-Stein estimator needs at least 4 estimates")
        s2 = float(v.mean())
        xbar = x.mean()
        c = 1.0 - (k - 3) * s2 / np.sum((x - xbar) ** 2)
        if positive_part:
            c = max(c, 0.0)
        est = xbar + c * (x - xbar)
        return EBResult("James-Stein", est, x, np.full(k, c), xbar, method=method, factor=float(c))
    if method == "normal":
        wt = 1.0 / v
        mu0 = np.sum(wt * x) / wt.sum()
        Q = np.sum(wt * (x - mu0) ** 2)
        tau2 = max(0.0, (Q - (k - 1)) / (wt.sum() - np.sum(wt ** 2) / wt.sum()))
        wstar = 1.0 / (tau2 + v)
        mu = np.sum(wstar * x) / wstar.sum()
        B = tau2 / (tau2 + v)
        est = mu + B * (x - mu)
        return EBResult("Normal EB", est, x, B, mu, method=method, tau2=float(tau2), mu=float(mu))
    raise ValueError("method must be 'james-stein' or 'normal'")


def eb_local(observed, expected, w, include_self: bool = True) -> EBResult:
    """
    Spatial (local) empirical Bayes, after Marshall (1991).

    Each area is shrunk toward the mean of its own *neighbourhood* instead of the
    global mean, so the smoothing respects spatial trends and does not pull an
    area in a high-risk region down toward the national average.

    For area *i* with neighbourhood ``N_i`` (its contiguity neighbours, plus *i*
    itself if ``include_self``)::

        m_i  = Σ_{N_i} O / Σ_{N_i} E
        s²_i = Σ_{N_i} E_j (r_j - m_i)² / Σ_{N_i} E_j  -  m_i / mean_{N_i}(E)
        θ̂_i = m_i + C_i (r_i - m_i),      C_i = s²_i / (s²_i + m_i / E_i)   (C_i = 0 if s²_i ≤ 0)

    An island (no neighbours) has no local reference, so it is shrunk toward the
    global mean instead.
    """
    O, E = _as_counts(observed, expected)
    if w.n != len(O):
        raise ValueError(f"w has {w.n} observations but data have {len(O)}")
    r = O / E
    gm = O.sum() / E.sum()
    est = np.empty_like(r)
    weight = np.empty_like(r)
    pm = np.empty_like(r)
    for i, id_i in enumerate(w.id_order):
        idx = [w.id_to_index[j] for j in w.neighbors[id_i]]
        if include_self or not idx:
            idx = idx + [i]
        idx = np.asarray(idx)
        Ej, Oj, rj = E[idx], O[idx], r[idx]
        if len(idx) < 2:
            m_i = gm
            s2 = float(np.sum(E * (r - gm) ** 2) / E.sum() - gm / E.mean())
        else:
            m_i = Oj.sum() / Ej.sum()
            s2 = float(np.sum(Ej * (rj - m_i) ** 2) / Ej.sum() - m_i / Ej.mean())
        C = s2 / (s2 + m_i / E[i]) if s2 > 0 else 0.0
        est[i] = m_i + C * (r[i] - m_i)
        weight[i] = C
        pm[i] = m_i
    return EBResult("Local EB", est, r, weight, pm)
