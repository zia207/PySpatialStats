"""
spatialstats.bayes.icar
=======================
Graph structure for intrinsic CAR (ICAR) priors and MCMC diagnostics.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def icar_structure(w) -> Dict:
    """
    Binary adjacency structure of a :class:`~spatialstats.weights.W` for an ICAR prior.

    ICAR is defined by the *presence* of a neighbour relation, so any weights
    transformation is discarded. The neighbour relation is symmetrised.

    Returns
    -------
    dict with
        ``A``         symmetric binary adjacency (scipy CSR)
        ``n_nbr``     number of neighbours per area
        ``labels``    connected-component label per area
        ``island``    boolean mask of areas with no neighbours
        ``n_comp``    number of components that contain at least two areas
        ``colors``    list of index arrays (a proper graph colouring, used to update
                      conditionally independent areas simultaneously)
    """
    S = w.sparse.tocsr()
    A = ((S + S.T) > 0).astype(float).tolil()
    A.setdiag(0)
    A = A.tocsr()
    A.eliminate_zeros()
    n = A.shape[0]
    n_nbr = np.asarray(A.sum(axis=1)).ravel().astype(int)
    _, labels = connected_components(A, directed=False)
    island = n_nbr == 0
    sizes = np.bincount(labels)
    n_comp = int(np.sum(sizes >= 2))

    # greedy colouring, largest-degree first
    order = np.argsort(-n_nbr)
    color = -np.ones(n, dtype=int)
    indptr, indices = A.indptr, A.indices
    for i in order:
        used = {color[j] for j in indices[indptr[i]:indptr[i + 1]] if color[j] >= 0}
        c = 0
        while c in used:
            c += 1
        color[i] = c
    colors = [np.where(color == c)[0] for c in range(color.max() + 1)]
    return {"A": A, "n_nbr": n_nbr, "labels": labels, "island": island, "n_comp": n_comp,
            "colors": colors}


def icar_scaling_factor(structure: Dict) -> float:
    """
    Sørbye–Rue (2014) scaling factor ``s``: the geometric mean of the marginal
    variances of the ICAR field with unit precision. Dividing the structured
    effect by ``sqrt(s)`` makes its marginal standard deviation ≈ 1, which is what
    lets BYM2's mixing parameter have a clean interpretation.
    """
    A, labels = structure["A"], structure["labels"]
    n = A.shape[0]
    logs = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) < 2:
            continue
        sub = A[idx][:, idx].toarray()
        Q = np.diag(sub.sum(axis=1)) - sub
        m = len(idx)
        # generalised inverse under the sum-to-zero constraint
        Qinv = np.linalg.inv(Q + np.ones((m, m)) / m) - np.ones((m, m)) / m
        logs.append(np.log(np.diag(Qinv)))
    if not logs:
        return 1.0
    return float(np.exp(np.mean(np.concatenate(logs))))


# ---------------------------------------------------------------------------
# Convergence diagnostics on arrays of shape (chains, draws)
# ---------------------------------------------------------------------------

def _split(x: np.ndarray) -> np.ndarray:
    c, d = x.shape
    h = d // 2
    return np.concatenate([x[:, :h], x[:, d - h:]], axis=0)


def rhat(x: np.ndarray) -> float:
    """Split-R̂ (Gelman–Rubin) for draws of shape (chains, draws); values ≲ 1.01 indicate convergence."""
    x = _split(np.asarray(x, dtype=float))
    m, n = x.shape
    if n < 4:
        return np.nan
    W = x.var(axis=1, ddof=1).mean()
    B = n * x.mean(axis=1).var(ddof=1)
    if W == 0:
        return 1.0 if B == 0 else np.inf
    var_plus = (n - 1) / n * W + B / n
    return float(np.sqrt(var_plus / W))


def ess(x: np.ndarray) -> float:
    """Effective sample size (Geyer's initial monotone sequence) for draws of shape (chains, draws)."""
    x = _split(np.asarray(x, dtype=float))
    m, n = x.shape
    if n < 4:
        return np.nan
    xc = x - x.mean(axis=1, keepdims=True)
    f = np.fft.rfft(xc, n=2 * n, axis=1)
    acov = np.fft.irfft(f * np.conj(f), axis=1)[:, :n] / n
    acov *= n / (n - np.arange(n))          # unbiased
    W = x.var(axis=1, ddof=1).mean()
    var_plus = (n - 1) / n * W + n * x.mean(axis=1).var(ddof=1) / n
    if var_plus == 0:
        return float(m * n)
    rho = 1.0 - (W - acov.mean(axis=0)) / var_plus
    rho[0] = 1.0
    tau = -1.0
    prev = np.inf
    for t in range(0, n - 1, 2):
        pair = rho[t] + rho[t + 1]
        if pair < 0:
            break
        pair = min(pair, prev)
        tau += 2 * pair
        prev = pair
    return float(m * n / max(tau, 1.0 / np.log10(m * n)))
