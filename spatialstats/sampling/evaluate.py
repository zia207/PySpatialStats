"""
spatialstats.sampling.evaluate
==============================
Spatial balance / coverage measures and Monte Carlo evaluation of designs
against a fully known population.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from spatialstats.sampling.sample import SpatialSample, as_rng
from spatialstats.sampling.estimate import estimate_mean


def spatial_balance(sample: SpatialSample) -> float:
    """
    Stevens–Olsen (2004) spatial balance statistic ``B``.

    Every frame unit is assigned to its nearest sampled unit (a discrete Voronoi
    partition); for sampled unit *i*, ``v_i`` is the sum of inclusion probabilities
    over the units in its cell. For a perfectly balanced design every ``v_i = 1``, so::

        B = mean_i (v_i - 1)²

    Lower is better; a simple random sample has larger ``B`` than a GRTS sample.
    """
    if sample.frame_pi is None:
        pi_frame = np.full(sample.N, sample.n / sample.N)
    else:
        pi_frame = sample.frame_pi
    _, nearest = cKDTree(sample.coords).query(sample.frame_coords)
    v = np.bincount(nearest, weights=pi_frame, minlength=sample.n)
    return float(np.mean((v - 1.0) ** 2))


def coverage_metrics(sample: SpatialSample) -> Dict[str, float]:
    """
    Geometric quality of a sample: ``balance`` (Stevens–Olsen B), ``mean_dist_to_sample`` and
    ``max_dist_to_sample`` (from every frame unit to its nearest sampled unit – the quantity a
    kriging-oriented design tries to minimise) and ``min_spacing`` between sampled units.
    """
    tree = cKDTree(sample.coords)
    d, _ = tree.query(sample.frame_coords)
    dd, _ = tree.query(sample.coords, k=2) if sample.n > 1 else (np.array([[0.0, np.nan]]), None)
    return {
        "balance": spatial_balance(sample),
        "mean_dist_to_sample": float(d.mean()),
        "max_dist_to_sample": float(d.max()),
        "min_spacing": float(dd[:, 1].min()),
    }


def evaluate_design(values, design: Callable[[np.random.Generator], SpatialSample], n_reps: int = 500,
                    seed: Optional[int] = None, variance: str = "auto", conf: float = 0.95,
                    k: int = 4) -> pd.Series:
    """
    Repeatedly draw a sample from a fully known population and score the estimator.

    Because ``values`` covers the *entire* frame, the true mean is known, so the
    performance of a design is measured directly rather than assumed.

    Parameters
    ----------
    values : array (N,)
        The population values (e.g. all 3,107 counties).
    design : callable ``rng -> SpatialSample``
        e.g. ``lambda rng: grts(coords, 60, seed=rng)``.
    n_reps : int
    variance, k : passed to :func:`estimate_mean`.

    Returns
    -------
    Series with ``bias``, ``sd`` (Monte Carlo s.d. of the estimator), ``rmse``,
    ``mean_se`` (average estimated standard error), ``coverage`` (share of confidence
    intervals containing the truth), ``mean_width``, ``balance`` (mean Stevens–Olsen B).
    """
    y = np.asarray(values, dtype=float)
    truth = float(y.mean())
    rng = as_rng(seed)
    est, se, cov, width, bal = [], [], [], [], []
    for _ in range(n_reps):
        s = design(rng)
        r = estimate_mean(y, s, variance=variance, k=k, conf=conf)
        est.append(r.mean)
        se.append(r.se)
        cov.append(r.lo <= truth <= r.hi)
        width.append(r.hi - r.lo)
        bal.append(spatial_balance(s))
    est = np.asarray(est)
    return pd.Series({
        "truth": truth, "bias": float(est.mean() - truth), "sd": float(est.std(ddof=1)),
        "rmse": float(np.sqrt(np.mean((est - truth) ** 2))), "mean_se": float(np.mean(se)),
        "coverage": float(np.mean(cov)), "mean_width": float(np.mean(width)),
        "balance": float(np.mean(bal)),
    })


def compare_designs(values, designs: Dict[str, Callable], n_reps: int = 500,
                    seed: Optional[int] = None, **kwargs) -> pd.DataFrame:
    """
    :func:`evaluate_design` for several designs; adds ``deff`` = (Monte Carlo variance)/(variance of the
    first design), so list simple random sampling first to get design effects relative to SRS.
    """
    rows = {}
    for i, (label, fn) in enumerate(designs.items()):
        rows[label] = evaluate_design(values, fn, n_reps, None if seed is None else seed + i, **kwargs)
    df = pd.DataFrame(rows).T
    base = df["sd"].iloc[0] ** 2
    df["deff"] = df["sd"] ** 2 / base
    return df
