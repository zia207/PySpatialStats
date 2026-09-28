"""
spatialstats.bayes.bym
======================
Besag–York–Mollié (BYM) Bayesian disease mapping.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from spatialstats.bayes.icar import icar_structure, icar_scaling_factor, rhat, ess
from spatialstats.bayes.pymc_backend import pymc_available, run_pymc
from spatialstats.bayes.rates import _as_counts
from spatialstats.bayes.sampler import run_chain
from spatialstats.core.result import SpatialResult


class BYM(SpatialResult):
    """
    Bayesian spatial smoothing of small-area counts with a BYM model.

    ::

        O_i ~ Poisson(E_i θ_i),    log θ_i = α + x_i β + u_i + v_i
        u ~ ICAR(τ_u)     (spatially structured: an area resembles its neighbours)
        v_i ~ N(0, 1/τ_v) (unstructured: area-specific heterogeneity)

    ``θ_i`` is the *relative risk*. The two random effects borrow strength from
    neighbours (u) and from the whole region (v); how much of each is used is learned
    from the data. Compared with the raw SMR the posterior mean of ``θ_i`` is
    shrunk, most for areas with small ``E_i``.

    Parameters
    ----------
    observed, expected : array (n,)
        Counts and expected counts (see :func:`~spatialstats.bayes.expected_counts`).
    w : W
        Spatial weights; only the neighbour structure is used (binary adjacency).
        Areas with no neighbours receive only the unstructured effect.
    X : array (n, p) or DataFrame, optional
        Covariates (centred and scaled internally; coefficients are reported on the
        original scale).
    model : {'bym', 'icar', 'iid', 'bym2'}
        ``'bym2'`` (Riebler et al. 2016, mixing parameter φ) is available with the
        PyMC backend only.
    backend : {'gibbs', 'pymc', 'auto'}
        ``'gibbs'`` – built-in blocked Gibbs / Metropolis sampler (no extra
        dependencies). ``'pymc'`` – NUTS via PyMC (``pip install pyspatialstats[bayes]``).
        ``'auto'`` – PyMC if importable, otherwise the built-in sampler.
    draws, tune, chains, thin : int
        MCMC settings (per chain; ``tune`` iterations are discarded). The built-in
        sampler updates one area at a time, which mixes slowly for smooth spatial
        fields, so the defaults keep every 10th iteration (``thin=10``); it is fast
        (~10 s for 200 areas). Always check ``max_rhat`` (< 1.05) and ``min_ess``.
        For PyMC, ``thin`` is ignored.
    seed : int, optional
    alpha : float
        Credible-interval level (``1 - alpha``).
    priors : dict, optional
        Gamma(shape, rate) priors on the precisions: ``{'tau_u': (1, 0.01), 'tau_v': (1, 0.01)}``.

    Attributes
    ----------
    rr_mean, rr_median, rr_lo, rr_hi, rr_sd : ndarray (n,)   posterior summaries of θ
    smr : ndarray (n,)                                       raw ``O / E``
    spatial_fraction : ndarray (draws,)                      share of random-effect s.d. that is spatial
    dic, p_eff : float                                       deviance information criterion
    diagnostics : DataFrame                                  R̂ and effective sample size per parameter
    backend : str                                            backend actually used
    """

    def __init__(self, observed, expected, w, X=None, names: Optional[Sequence[str]] = None,
                 model: str = "bym", backend: str = "gibbs", draws: int = 2000, tune: int = 2000,
                 chains: int = 4, thin: int = 10, seed: Optional[int] = None, alpha: float = 0.05,
                 priors: Optional[Dict[str, tuple]] = None, **pymc_kwargs):
        super().__init__(name=f"{model.upper()} disease mapping")
        O, E = _as_counts(observed, expected)
        n = len(O)
        if w.n != n:
            raise ValueError(f"w has {w.n} observations but data have {n}")
        if model not in ("bym", "icar", "iid", "bym2"):
            raise ValueError("model must be one of 'bym', 'icar', 'iid', 'bym2'")
        if backend == "auto":
            backend = "pymc" if pymc_available() else "gibbs"
        if backend not in ("gibbs", "pymc"):
            raise ValueError("backend must be 'gibbs', 'pymc' or 'auto'")
        if model == "bym2" and backend != "pymc":
            raise ValueError("model='bym2' requires backend='pymc'")
        if chains < 2:
            raise ValueError("use at least 2 chains (needed for R-hat)")
        pri = {"tau_u": (1.0, 0.01), "tau_v": (1.0, 0.01)}
        pri.update(priors or {})
        (a_u, b_u), (a_v, b_v) = pri["tau_u"], pri["tau_v"]

        # covariates: centre and scale
        Xs = xmean = xsd = None
        self.covariate_names = []
        if X is not None:
            if isinstance(X, pd.DataFrame):
                self.covariate_names = [str(c) for c in X.columns]
                Xv = X.to_numpy(dtype=float)
            else:
                Xv = np.asarray(X, dtype=float)
                if Xv.ndim == 1:
                    Xv = Xv[:, None]
                self.covariate_names = list(names) if names is not None else \
                    [f"x{i + 1}" for i in range(Xv.shape[1])]
            xmean, xsd = Xv.mean(axis=0), Xv.std(axis=0)
            if np.any(xsd == 0):
                raise ValueError("a covariate is constant")
            Xs = (Xv - xmean) / xsd

        self.observed, self.expected, self.w, self.model = O, E, w, model
        self.backend = backend
        struct = icar_structure(w)
        self.structure = struct
        if struct["island"].any():
            self._stats["n_islands"] = int(struct["island"].sum())

        # ---- run -------------------------------------------------------------
        if backend == "gibbs":
            master = np.random.SeedSequence(seed)
            rngs = [np.random.default_rng(s) for s in master.spawn(chains)]
            runs = [run_chain(O, E, struct, Xs, model, draws, tune, thin, rng, a_u, b_u, a_v, b_v,
                              init_scale=float(sc))
                    for rng, sc in zip(rngs, np.linspace(0.5, 2.0, chains))]
            raw = {k: np.stack([r[k] for r in runs]) for k in ("alpha", "tau_u", "tau_v", "b", "u", "beta")}
            self._stats["accept_omega"] = float(np.mean([r["accept_omega"] for r in runs]))
            if model == "icar":
                raw["tau_v"] = np.full_like(raw["tau_v"], np.nan)
            if model == "iid":
                raw["tau_u"] = np.full_like(raw["tau_u"], np.nan)
        else:
            scaling = icar_scaling_factor(struct) if model == "bym2" else None
            raw = run_pymc(O, E, struct, Xs, model, draws, tune, chains, seed, a_u, b_u, a_v, b_v,
                           scaling=scaling, **pymc_kwargs)
            self._stats["divergences"] = raw.pop("divergences")
            self.idata = raw.pop("idata")
            self.scaling_factor = scaling

        self.chains, self.n_draws = raw["alpha"].shape[:2]
        alpha_d = raw["alpha"]                                   # (chains, draws)
        beta_s = raw["beta"]                                     # (chains, draws, p)
        eta = alpha_d[..., None] + raw["b"]
        if Xs is not None:
            eta = eta + beta_s @ Xs.T
        theta = np.exp(eta)                                      # (chains, draws, n)
        flat = theta.reshape(-1, n)
        self._theta = flat.astype(np.float32)

        # back-transform covariate coefficients to the original scale
        if Xs is not None:
            beta_o = beta_s / xsd
            alpha_o = alpha_d - (beta_s * (xmean / xsd)).sum(axis=-1)
        else:
            beta_o, alpha_o = beta_s, alpha_d

        self.rr_mean = flat.mean(axis=0)
        self.rr_median = np.median(flat, axis=0)
        self.rr_lo = np.quantile(flat, alpha / 2, axis=0)
        self.rr_hi = np.quantile(flat, 1 - alpha / 2, axis=0)
        self.rr_sd = flat.std(axis=0)
        self.smr = O / E
        self.alpha_level = alpha

        # spatial fraction
        if model == "bym2":
            self.spatial_fraction = raw["phi"].ravel()
        elif model == "bym":
            su = raw["u"].std(axis=-1)
            sv = (raw["b"] - raw["u"]).std(axis=-1)
            self.spatial_fraction = (su / (su + sv)).ravel()
        else:
            self.spatial_fraction = np.full(chains * self.n_draws, 1.0 if model == "icar" else 0.0)

        # DIC
        def dev(th):
            return -2.0 * stats.poisson.logpmf(O, E * th).sum(axis=-1)
        d_draws = dev(flat.astype(float)) if n * flat.shape[0] < 5e7 else \
            np.array([dev(t) for t in flat.astype(float)])
        d_bar = float(np.mean(d_draws))
        d_hat = float(dev(flat.mean(axis=0)))
        self.p_eff = d_bar - d_hat
        self.dic = d_bar + self.p_eff

        # diagnostics table
        rows = []

        def add(label, arr):
            a = np.asarray(arr, dtype=float)
            if np.isnan(a).all():
                return
            rows.append({"parameter": label, "mean": a.mean(), "sd": a.std(),
                         "q_lo": np.quantile(a, alpha / 2), "q_hi": np.quantile(a, 1 - alpha / 2),
                         "rhat": rhat(a), "ess": ess(a)})
        add("intercept", alpha_o)
        for j, nm in enumerate(self.covariate_names):
            add(nm, beta_o[..., j])
        add("tau_u", raw["tau_u"])
        add("tau_v", raw["tau_v"])
        if model == "bym2":
            add("sigma", raw["sigma"])
            add("phi", raw["phi"])
        self.diagnostics = pd.DataFrame(rows).set_index("parameter")
        # worst-case over the n relative risks
        rr_r = np.array([rhat(theta[..., i]) for i in range(n)])
        rr_e = np.array([ess(theta[..., i]) for i in range(n)])
        self.max_rhat = float(np.nanmax(rr_r))
        self.min_ess = float(np.nanmin(rr_e))

        self.coef = None
        if Xs is not None:
            self.coef = pd.DataFrame({
                "variable": self.covariate_names,
                "mean": beta_o.reshape(-1, beta_o.shape[-1]).mean(axis=0),
                "sd": beta_o.reshape(-1, beta_o.shape[-1]).std(axis=0),
                "lo": np.quantile(beta_o.reshape(-1, beta_o.shape[-1]), alpha / 2, axis=0),
                "hi": np.quantile(beta_o.reshape(-1, beta_o.shape[-1]), 1 - alpha / 2, axis=0),
            })
        self._raw = raw
        self._stats.update(model=model, backend=backend, n=n, chains=chains, draws=self.n_draws,
                           dic=self.dic, p_eff=self.p_eff, max_rhat=self.max_rhat,
                           min_ess=self.min_ess,
                           spatial_fraction=float(np.mean(self.spatial_fraction)))

    # -- posterior queries ---------------------------------------------------

    def exceedance(self, threshold: float = 1.0) -> np.ndarray:
        """Posterior probability ``P(θ_i > threshold)`` for each area (a probability map for hot spots)."""
        return (self._theta > threshold).mean(axis=0)

    def posterior_draws(self) -> np.ndarray:
        """Relative-risk draws, shape ``(chains * draws, n)`` (float32)."""
        return self._theta

    def shrinkage(self) -> np.ndarray:
        """
        Fraction of the distance between the raw SMR and the overall mean risk that has
        been removed: 0 = no smoothing, 1 = fully shrunk to the overall level.
        """
        g = self.observed.sum() / self.expected.sum()
        denom = self.smr - g
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(np.abs(denom) > 1e-9, (self.smr - self.rr_mean) / denom, np.nan)
        return s

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "observed": self.observed, "expected": self.expected, "smr": self.smr,
            "rr_mean": self.rr_mean, "rr_median": self.rr_median,
            "rr_lo": self.rr_lo, "rr_hi": self.rr_hi,
            "p_exceed_1": self.exceedance(1.0),
        })

    def summary(self) -> str:
        head = f"{self.model.upper()} model ({self.backend})"
        lines = [head, "=" * len(head),
                 f"  areas {len(self.observed)}   chains {self.chains}   draws/chain {self.n_draws}",
                 f"  DIC {self.dic:.2f}   effective parameters {self.p_eff:.1f}",
                 f"  spatial fraction of random-effect s.d.: {np.mean(self.spatial_fraction):.3f}",
                 f"  max R-hat over relative risks {self.max_rhat:.3f}   min ESS {self.min_ess:.0f}", ""]
        lines.append(self.diagnostics.round(4).to_string())
        if self.max_rhat > 1.05:
            lines.append("\n  WARNING: R-hat > 1.05 for at least one relative risk; run longer.")
        return "\n".join(lines)
