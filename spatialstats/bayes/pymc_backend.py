"""
spatialstats.bayes.pymc_backend
===============================
Optional PyMC implementation of the disease-mapping models (NUTS sampling).

PyMC is imported lazily, so the rest of :mod:`spatialstats.bayes` works without it.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

_INSTALL_HINT = (
    "The 'pymc' backend requires PyMC (>=5). Install with: pip install pyspatialstats[bayes]  "
    "(or use backend='gibbs', which has no extra dependencies)."
)


def pymc_available() -> bool:
    try:
        import pymc  # noqa: F401
        return True
    except Exception:
        return False


def run_pymc(O, E, struct: Dict, X: Optional[np.ndarray], model: str, draws: int, tune: int,
             chains: int, seed: Optional[int], a_u: float, b_u: float, a_v: float, b_v: float,
             scaling: Optional[float] = None, target_accept: float = 0.9,
             progressbar: bool = False, **sample_kwargs) -> Dict[str, np.ndarray]:
    """
    Fit ``O_i ~ Poisson(E_i exp(η_i))`` with NUTS.

    ``model`` ∈ {'bym', 'icar', 'iid', 'bym2'}. For ``'bym'`` the priors are
    ``τ_u ~ Gamma(a_u, b_u)``, ``τ_v ~ Gamma(a_v, b_v)`` (identical to the built-in
    sampler); ``'bym2'`` uses the Riebler et al. (2016) parametrisation
    ``η = α + Xβ + σ(√(φ/s) u* + √(1-φ) v*)`` with ``σ ~ HalfNormal(1)``,
    ``φ ~ Beta(1/2, 1/2)`` and ICAR scaling factor ``s``.

    Returns arrays shaped ``(chains, draws, ...)``.
    """
    try:
        import pymc as pm
    except ImportError as e:  # pragma: no cover
        raise ImportError(_INSTALL_HINT) from e

    if struct["island"].any() and model in ("bym", "icar", "bym2"):
        raise ValueError(
            f"{int(struct['island'].sum())} area(s) have no neighbours; PyMC's ICAR needs a "
            "connected adjacency. Link the islands to their nearest area, or use backend='gibbs'."
        )
    n = len(O)
    A = struct["A"].toarray()
    p = 0 if X is None else X.shape[1]
    mu0 = float(np.log(max(O.sum(), 0.5) / E.sum()))

    with pm.Model():
        alpha = pm.Normal("alpha", mu0, 5.0)
        eta = alpha
        if p:
            beta = pm.Normal("beta", 0.0, 10.0, shape=p)
            eta = eta + pm.math.dot(X, beta)
        if model == "bym2":
            sigma = pm.HalfNormal("sigma", 1.0)
            phi = pm.Beta("phi", 0.5, 0.5)
            u_star = pm.ICAR("u_star", W=A, sigma=1.0)
            v_star = pm.Normal("v_star", 0.0, 1.0, shape=n)
            s = 1.0 if scaling is None else float(scaling)
            b = sigma * (pm.math.sqrt(phi / s) * u_star + pm.math.sqrt(1.0 - phi) * v_star)
            pm.Deterministic("b", b)
            eta = eta + b
        else:
            b = 0.0
            if model in ("bym", "icar"):
                tau_u = pm.Gamma("tau_u", a_u, b_u)
                # PyMC's ICAR is unnormalised in ``sigma`` (its density lacks the
                # sigma^-(n-c) term), so a hyperprior on sigma would be improper. The
                # documented remedy is a fixed-scale ICAR multiplied by the scale.
                u_std = pm.ICAR("u_std", W=A, sigma=1.0)
                u = pm.Deterministic("u", u_std / pm.math.sqrt(tau_u))
                b = b + u
            if model in ("bym", "iid"):
                tau_v = pm.Gamma("tau_v", a_v, b_v)
                v = pm.Normal("v", 0.0, 1.0 / pm.math.sqrt(tau_v), shape=n)
                b = b + v
            pm.Deterministic("b", b)
            eta = eta + b
        pm.Poisson("y", mu=E * pm.math.exp(eta), observed=O)
        idata = pm.sample(draws=draws, tune=tune, chains=chains, random_seed=seed,
                          target_accept=target_accept, progressbar=progressbar, **sample_kwargs)

    post = idata.posterior

    def get(name):
        return np.asarray(post[name].values)

    out = {"alpha": get("alpha"), "b": get("b"),
           "beta": get("beta") if p else np.zeros(get("alpha").shape + (0,))}
    shape = out["alpha"].shape
    if model == "bym2":
        out["sigma"] = get("sigma")
        out["phi"] = get("phi")
        out["tau_u"] = np.full(shape, np.nan)
        out["tau_v"] = np.full(shape, np.nan)
        out["u"] = out["sigma"][..., None] * np.sqrt(out["phi"] / (scaling or 1.0))[..., None] * get("u_star")
    else:
        out["tau_u"] = get("tau_u") if "tau_u" in post else np.full(shape, np.nan)
        out["tau_v"] = get("tau_v") if "tau_v" in post else np.full(shape, np.nan)
        out["u"] = get("u") if "u" in post else np.zeros_like(out["b"])
    try:
        out["divergences"] = int(idata.sample_stats["diverging"].values.sum())
    except Exception:  # pragma: no cover
        out["divergences"] = -1
    out["idata"] = idata
    return out
