"""
spatialstats.cluster._perm
==========================
Conditional permutation inference for local statistics of the form
``s_i = f(y_i, Σ_j w_ij y_j)``, and multiple-testing helpers.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import sparse


def _draw_without_replacement(rng, n_other: int, k: int, P: int) -> np.ndarray:
    """P rows of k distinct integers in [0, n_other)."""
    if k > n_other:
        raise ValueError("more neighbours than other observations")
    idx = rng.integers(0, n_other, size=(P, k))
    if k == 1:
        return idx
    while True:
        s = np.sort(idx, axis=1)
        dup = (s[:, 1:] == s[:, :-1]).any(axis=1)
        if not dup.any():
            return idx
        idx[dup] = rng.integers(0, n_other, size=(int(dup.sum()), k))


def conditional_lag_simulations(y: np.ndarray, W: sparse.csr_matrix, permutations: int,
                                rng: np.random.Generator) -> np.ndarray:
    """
    Simulated spatial lags under *conditional* randomisation.

    For each area ``i`` the value ``y_i`` is held fixed and its neighbours' values are
    replaced by ``n_i`` values drawn without replacement from the other ``n - 1`` observations.
    Returns an array (permutations, n) of ``Σ_j w_ij y_{π(j)}``.
    """
    n = len(y)
    W = W.tocsr()
    sims = np.zeros((permutations, n))
    for i in range(n):
        s, e = W.indptr[i], W.indptr[i + 1]
        cols, wts = W.indices[s:e], W.data[s:e]
        keep = cols != i                              # the self-weight (if any) uses the fixed y_i
        w_self = float(wts[~keep].sum())
        cols, wts = cols[keep], wts[keep]
        k = len(cols)
        lag = np.full(permutations, w_self * y[i])
        if k:
            idx = _draw_without_replacement(rng, n - 1, k, permutations)
            idx = idx + (idx >= i)                    # skip position i
            lag = lag + y[idx] @ wts
        sims[:, i] = lag
    return sims


def permutation_pvalues(observed: np.ndarray, sims: np.ndarray, alternative: str = "two-sided") -> np.ndarray:
    """
    Pseudo p-values from simulated values, shape (permutations, n) vs observed (n,).

    ``'two-sided'`` = ``min(1, 2 · min(p_upper, p_lower))``; ``'greater'`` / ``'less'`` are one-sided.
    All use ``(1 + #extreme) / (1 + P)``.
    """
    P = sims.shape[0]
    ge = (sims >= observed[None, :]).sum(axis=0)
    le = (sims <= observed[None, :]).sum(axis=0)
    p_up, p_lo = (1.0 + ge) / (P + 1.0), (1.0 + le) / (P + 1.0)
    if alternative == "greater":
        return p_up
    if alternative == "less":
        return p_lo
    if alternative == "two-sided":
        return np.minimum(1.0, 2.0 * np.minimum(p_up, p_lo))
    raise ValueError("alternative must be 'two-sided', 'greater' or 'less'")


def adjust_pvalues(p: np.ndarray, method: str = "fdr") -> np.ndarray:
    """Multiple-testing adjustment: ``'fdr'`` (Benjamini–Hochberg), ``'bonferroni'`` or ``'none'``."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    if method == "none":
        return p.copy()
    if method == "bonferroni":
        return np.minimum(1.0, p * n)
    if method == "fdr":
        order = np.argsort(p)
        adj = np.empty(n)
        prev = 1.0
        for r in range(n - 1, -1, -1):
            prev = min(prev, p[order[r]] * n / (r + 1))
            adj[order[r]] = prev
        return adj
    raise ValueError("correction must be 'fdr', 'bonferroni' or 'none'")


def check_resolution(permutations: int, n: int, alpha: float, correction: str) -> None:
    """Warn when permutation p-values are too coarse to survive the multiple-testing correction."""
    if correction == "none" or permutations <= 0:
        return
    p_min = 2.0 / (permutations + 1.0)                 # smallest two-sided pseudo p-value
    if correction == "bonferroni" and p_min > alpha / n:
        warnings.warn(
            f"With {permutations} permutations the smallest attainable p-value is {p_min:.4g}, but "
            f"Bonferroni needs p <= {alpha / n:.3g}; nothing can be significant. Use "
            f"permutations >= {int(np.ceil(2 * n / alpha))} or correction='fdr'.", stacklevel=3)
    elif correction == "fdr" and p_min > alpha:
        warnings.warn(f"{permutations} permutations cannot reach p <= {alpha}; increase permutations.", stacklevel=3)
    elif correction == "fdr":
        k_min = int(np.ceil(p_min * n / alpha))
        if k_min > 3:
            warnings.warn(
                f"With {permutations} permutations the smallest attainable p-value is {p_min:.3g}, so at least "
                f"{k_min} of {n} areas would have to sit at that minimum to survive FDR at alpha={alpha}; "
                f"few or no clusters may be detected. Use more permutations "
                f"(>= {int(np.ceil(2 * n / alpha))}) or inference='analytic' (Gi*).", stacklevel=3)
