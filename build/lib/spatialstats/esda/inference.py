"""
spatialstats.esda.inference
===========================
Permutation inference and FDR correction utilities.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np


def conditional_permutation(
    y: np.ndarray,
    statistic_fn: Callable[[np.ndarray], np.ndarray],
    permutations: int = 999,
    seed: Optional[int] = None,
    two_tailed: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Conditional (randomization) permutation test.

    Parameters
    ----------
    y : ndarray
        Observed attribute.
    statistic_fn : callable
        ``statistic_fn(y_perm) -> scalar or array`` matching observed shape.
    permutations : int
        Number of permutations.
    seed : int, optional
        RNG seed.
    two_tailed : bool
        If True, p = (1 + count(|sim| >= |obs|)) / (1 + perms).

    Returns
    -------
    p_value : float or ndarray
    sim : ndarray of shape (permutations, ...) simulated statistics
    """
    y = np.asarray(y, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    observed = np.asarray(statistic_fn(y), dtype=float)
    sims = []
    for _ in range(permutations):
        yp = rng.permutation(y)
        sims.append(statistic_fn(yp))
    sim = np.asarray(sims, dtype=float)

    if observed.ndim == 0:
        if two_tailed:
            extreme = np.abs(sim) >= np.abs(observed)
        else:
            extreme = sim >= observed
        p = (1.0 + float(np.sum(extreme))) / (1.0 + permutations)
        return p, sim

    # Element-wise for local statistics
    if two_tailed:
        extreme = np.abs(sim) >= np.abs(observed)[None, :]
    else:
        extreme = sim >= observed[None, :]
    p = (1.0 + extreme.sum(axis=0).astype(float)) / (1.0 + permutations)
    return p, sim


def fdr_bh(p_values: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Benjamini–Hochberg FDR correction.

    Returns
    -------
    reject : ndarray of bool
    p_adjusted : ndarray
    """
    p = np.asarray(p_values, dtype=float).ravel()
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        val = ranked[i] * n / (i + 1)
        prev = min(prev, val)
        adjusted[i] = prev
    p_adj = np.empty(n, dtype=float)
    p_adj[order] = np.clip(adjusted, 0, 1)
    reject = p_adj <= alpha
    return reject, p_adj
