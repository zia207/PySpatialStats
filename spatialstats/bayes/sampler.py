"""
spatialstats.bayes.sampler
==========================
A dependency-free MCMC sampler for Poisson disease-mapping models with
ICAR / iid / BYM random effects (blocked Gibbs + Metropolis-within-Gibbs).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def _mh_update(rng, idx, cur, prior_mean, prior_prec, scale, eta_base, O, E):
    """
    Random-walk Metropolis for a set of conditionally independent parameters ``cur[idx]``
    with Gaussian prior N(prior_mean, 1/prior_prec) and Poisson likelihood
    ``O ~ Poisson(E exp(eta_base + cur))``. Returns (new_values, accepted_mask).
    """
    x = cur[idx]
    prop = x + rng.normal(0.0, 1.0, len(idx)) * scale[idx]

    def logp(v):
        eta = eta_base[idx] + v
        return O[idx] * eta - E[idx] * np.exp(eta) - 0.5 * prior_prec * (v - prior_mean) ** 2

    accept = np.log(rng.uniform(size=len(idx))) < logp(prop) - logp(x)
    return np.where(accept, prop, x), accept


def run_chain(O, E, struct: Dict, X: Optional[np.ndarray], model: str, draws: int, tune: int,
              thin: int, rng: np.random.Generator, a_u: float, b_u: float, a_v: float, b_v: float,
              init_scale: float = 1.0) -> Dict[str, np.ndarray]:
    """
    One MCMC chain for ``O_i ~ Poisson(E_i exp(η_i))``, ``η = α + Xβ + b``.

    * ``model='bym'``   : ``b = u + v``, u ICAR(τ_u), v ~ N(0, 1/τ_v)
    * ``model='icar'``  : ``b = u``
    * ``model='iid'``   : ``b = v``

    BYM is sampled in the *convolution* parametrisation (ω = u + v, then u | ω is
    an exact Gaussian Gibbs step), which mixes far better than updating u and v
    separately. Areas of the same colour class are conditionally independent and
    are updated together.
    """
    n = len(O)
    A, n_nbr, colors = struct["A"], struct["n_nbr"], struct["colors"]
    labels, island = struct["labels"], struct["island"]
    comps = [np.where(labels == c)[0] for c in np.unique(labels[~island])]
    edges = sparse_edges(A)
    n_edge_dof = int(sum(len(c) - 1 for c in comps))     # rank of the ICAR precision
    color_data = []
    for idx in colors:
        idx = idx[~island[idx]]
        if len(idx):
            color_data.append((idx, A[idx].tocsr()))

    p = 0 if X is None else X.shape[1]
    alpha = float(np.log(max(O.sum(), 0.5) / E.sum())) + rng.normal(0, 0.05 * init_scale)
    beta = np.zeros(p)
    u = np.zeros(n)
    v = np.zeros(n)
    if model != "iid":
        u = rng.normal(0, 0.05 * init_scale, n) * (~island)
        for c in comps:
            u[c] -= u[c].mean()
    omega = u + v if model == "bym" else (v if model == "iid" else u)
    if model == "iid":
        omega = rng.normal(0, 0.05 * init_scale, n)
    tau_u, tau_v = 10.0 * init_scale ** -1, 10.0 * init_scale ** -1

    base_scale = 2.4 / np.sqrt(O + 1.0)
    c_omega, c_alpha, c_beta = 1.0, 0.05, 0.05
    total = tune + draws * thin
    out = {k: [] for k in ("alpha", "tau_u", "tau_v", "b", "u", "beta")}
    acc_w = np.zeros(2)
    acc_a = np.zeros(2)
    Xb = np.zeros(n)

    for it in range(total):
        eta_fix = alpha + Xb

        # --- random effects ---------------------------------------------------
        if model == "iid":
            idx = np.arange(n)
            new, acc = _mh_update(rng, idx, omega, 0.0, tau_v, base_scale * c_omega,
                                  eta_fix * np.ones(n), O, E)
            omega = new
            acc_w += (acc.sum(), n)
            s = alpha + omega.mean()                       # exact Gibbs step for the level shift
            m_new = rng.normal(0.0, 1.0 / np.sqrt(n * tau_v))
            omega = omega - omega.mean() + m_new
            alpha = s - m_new
        elif model == "icar":
            for idx, Ac in color_data:
                pm = np.asarray(Ac @ omega).ravel() / n_nbr[idx]
                new, acc = _mh_update(rng, idx, omega, pm, tau_u * n_nbr[idx],
                                      base_scale * c_omega, eta_fix * np.ones(n), O, E)
                omega[idx] = new
                acc_w += (acc.sum(), len(idx))
            for c in comps:
                omega[c] -= omega[c].mean()
            u = omega
        else:  # bym: omega = u + v
            # ω | u : Poisson likelihood x N(u, 1/τ_v)  (Metropolis, all areas together)
            idx = np.arange(n)
            new, acc = _mh_update(rng, idx, omega - u, 0.0, tau_v, base_scale * c_omega,
                                  (eta_fix + u) * np.ones(n), O, E)
            omega = new + u
            acc_w += (acc.sum(), n)
            # u | ω : Gaussian (exact), by colour
            for idx, Ac in color_data:
                nb_sum = np.asarray(Ac @ u).ravel()
                prec = tau_v + tau_u * n_nbr[idx]
                mean = (tau_v * omega[idx] + tau_u * nb_sum) / prec
                u[idx] = mean + rng.normal(size=len(idx)) / np.sqrt(prec)
            for c in comps:
                u[c] -= u[c].mean()
            v = omega - u
            # exact Gibbs step for the level shift between α and mean(ω)
            s = alpha + omega.mean()
            m_new = rng.normal(u.mean(), 1.0 / np.sqrt(n * tau_v))
            omega = omega - omega.mean() + m_new
            alpha = s - m_new
            v = omega - u

        # --- fixed effects ------------------------------------------------------
        b = omega
        if p:
            prop = beta + rng.normal(size=p) * c_beta
            def lp(bt):
                eta = alpha + X @ bt + b
                return np.sum(O * eta - E * np.exp(eta)) - 0.5 * np.sum(bt ** 2) / 100.0 ** 2
            if np.log(rng.uniform()) < lp(prop) - lp(beta):
                beta = prop
                acc_a[0] += 1
            acc_a[1] += 1
            Xb = X @ beta
        # intercept (flat prior)
        prop_a = alpha + rng.normal() * c_alpha
        def la(a_):
            eta = a_ + Xb + b
            return np.sum(O * eta - E * np.exp(eta))
        if np.log(rng.uniform()) < la(prop_a) - la(alpha):
            alpha = prop_a
            acc_a[0] += 1
        acc_a[1] += 1

        # --- precisions (conjugate gamma) ----------------------------------------
        if model in ("bym", "icar"):
            d = (u[edges[0]] - u[edges[1]]) ** 2
            tau_u = rng.gamma(a_u + 0.5 * n_edge_dof, 1.0 / (b_u + 0.5 * d.sum()))
        if model in ("bym", "iid"):
            vv = v if model == "bym" else omega
            tau_v = rng.gamma(a_v + 0.5 * n, 1.0 / (b_v + 0.5 * np.sum(vv ** 2)))

        # --- adaptation during burn-in ------------------------------------------
        if it < tune and (it + 1) % 50 == 0:
            r_w = acc_w[0] / max(acc_w[1], 1)
            c_omega *= np.exp(0.5 * (r_w - 0.35))
            r_a = acc_a[0] / max(acc_a[1], 1)
            c_alpha *= np.exp(0.5 * (r_a - 0.35))
            c_beta = c_alpha
            acc_w[:] = 0
            acc_a[:] = 0

        if it >= tune and (it - tune) % thin == 0:
            out["alpha"].append(alpha)
            out["tau_u"].append(tau_u)
            out["tau_v"].append(tau_v)
            out["b"].append(omega.copy())
            out["u"].append(u.copy() if model != "iid" else np.zeros(n))
            out["beta"].append(beta.copy())

    res = {k: np.asarray(v_) for k, v_ in out.items()}
    res["accept_omega"] = float(acc_w[0] / max(acc_w[1], 1)) if acc_w[1] else np.nan
    return res


def sparse_edges(A):
    """Undirected edge list (i < j) of a symmetric sparse adjacency matrix."""
    import scipy.sparse as sp
    coo = sp.triu(A, k=1).tocoo()
    return coo.row, coo.col
