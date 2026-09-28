"""
spatialstats.bayes.evaluate
===========================
Simulation and comparison tools for judging smoothers against a known truth.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats



def simulate_relative_risk(w, sigma: float = 0.35, spatial_strength: float = 0.95,
                           seed: Optional[int] = None) -> np.ndarray:
    """
    Draw a spatially smooth *true* relative-risk surface for simulation studies.

    ``log θ = σ · standardise((I - c W_row)⁻¹ z)`` with ``z`` white noise; ``c``
    (``spatial_strength`` < 1) controls the range of the spatial correlation. The
    result has mean log-risk 0 and standard deviation ``sigma``.
    """
    from scipy.sparse import identity
    from scipy.sparse.linalg import spsolve
    rng = np.random.default_rng(seed)
    Wm = w.sparse.tocsr()
    rs = np.asarray(Wm.sum(axis=1)).ravel()
    rs[rs == 0] = 1.0
    from scipy.sparse import diags
    Wr = diags(1.0 / rs) @ Wm
    z = rng.normal(size=w.n)
    f = spsolve((identity(w.n, format="csc") - spatial_strength * Wr).tocsc(), z)
    f = (f - f.mean()) / f.std()
    return np.exp(sigma * f)


def simulate_counts(theta, expected, seed: Optional[int] = None) -> np.ndarray:
    """Draw ``O_i ~ Poisson(E_i θ_i)``."""
    rng = np.random.default_rng(seed)
    return rng.poisson(np.asarray(expected, dtype=float) * np.asarray(theta, dtype=float))


def shrinkage_summary(raw, smoothed, expected=None) -> pd.DataFrame:
    """
    Per-area account of what smoothing did.

    Columns: ``raw``, ``smoothed``, ``change``, ``shrinkage`` (share of the distance to the
    overall mean that was removed: 1 = fully shrunk), and ``expected`` if given.
    """
    r = np.asarray(raw, dtype=float)
    s = np.asarray(smoothed, dtype=float)
    g = np.average(r, weights=None if expected is None else np.asarray(expected, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        shr = np.where(np.abs(r - g) > 1e-9, (r - s) / (r - g), np.nan)
    df = pd.DataFrame({"raw": r, "smoothed": s, "change": s - r, "shrinkage": shr})
    if expected is not None:
        df["expected"] = np.asarray(expected, dtype=float)
    return df


def evaluate_smoothing(truth, estimates: Dict[str, np.ndarray], expected=None,
                       intervals: Optional[Dict[str, tuple]] = None) -> pd.DataFrame:
    """
    Score estimators against a known true risk surface.

    Parameters
    ----------
    truth : array (n,)
    estimates : dict ``{label: array (n,)}`` – e.g. raw SMR, EB, BYM posterior means.
    expected : array (n,), optional
        If given, also report the E-weighted RMSE (which is what matters for total
        burden, and is dominated by populous areas).
    intervals : dict ``{label: (lo, hi)}``, optional
        Credible/confidence intervals for the coverage column.

    Returns
    -------
    DataFrame with ``rmse``, ``mae``, ``bias``, ``spearman`` (rank agreement) and
    ``coverage`` / ``width`` where intervals are given.
    """
    t = np.asarray(truth, dtype=float)
    rows = {}
    for label, est in estimates.items():
        e = np.asarray(est, dtype=float)
        d = e - t
        row = {"rmse": float(np.sqrt(np.mean(d ** 2))), "mae": float(np.mean(np.abs(d))),
               "bias": float(np.mean(d)), "spearman": float(stats.spearmanr(e, t)[0])}
        if expected is not None:
            E = np.asarray(expected, dtype=float)
            row["rmse_weighted"] = float(np.sqrt(np.sum(E * d ** 2) / E.sum()))
        if intervals and label in intervals:
            lo, hi = (np.asarray(a, dtype=float) for a in intervals[label])
            row["coverage"] = float(np.mean((t >= lo) & (t <= hi)))
            row["width"] = float(np.mean(hi - lo))
        rows[label] = row
    return pd.DataFrame(rows).T
