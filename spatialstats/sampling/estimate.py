"""
spatialstats.sampling.estimate
==============================
Design-based estimation of a population mean from a spatial sample, and
sample-size planning tools.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree

from spatialstats.core.result import SpatialResult
from spatialstats.sampling.sample import SpatialSample


class EstimateResult(SpatialResult):
    """Point estimate of a population mean (and total) with its standard error and CI."""

    def __init__(self, **stats_):
        super().__init__(name="Design-based estimate", **stats_)
        for k, v in stats_.items():
            setattr(self, k, v)


def local_mean_variance(z: np.ndarray, coords: np.ndarray, pi: np.ndarray, k: int = 4) -> float:
    """
    Local-mean variance estimator of the Horvitz–Thompson total ``Σ z_i`` (``z_i = y_i / π_i``)
    for spatially balanced samples (Stevens & Olsen 2003; Grafström & Schelin 2014).

    Each sampled unit is compared with the mean of its ``k`` nearest sampled units
    (itself included)::

        V̂ = Σ_i (1 - π_i) · k/(k-1) · (z_i - z̄_{D_i})²

    Because neighbours in a balanced sample have similar values when ``y`` is
    spatially structured, this captures the variance reduction that the
    simple-random-sampling formula ignores (and which would otherwise make
    confidence intervals too wide). For an independent ``y`` it reduces to the SRS variance.
    """
    n = len(z)
    if n < 3:
        raise ValueError("need at least 3 sampled units for the local-mean variance")
    k = int(min(max(k, 2), n))
    _, nbr = cKDTree(coords).query(coords, k=k)
    zbar = z[nbr].mean(axis=1)
    return float(np.sum((1.0 - pi) * (k / (k - 1.0)) * (z - zbar) ** 2))


def estimate_mean(y, sample: SpatialSample, variance: str = "auto", k: int = 4,
                  conf: float = 0.95, hajek: bool = True) -> EstimateResult:
    """
    Estimate the finite-population mean of ``y`` from a probability sample.

    Parameters
    ----------
    y : array (n,) or (N,)
        Values for the sampled units (length ``n``), or for the whole frame (length
        ``N``, in which case the sampled rows are extracted).
    sample : SpatialSample
        Must have defined inclusion probabilities (not ``space_filling`` / ``clhs``).
    variance : {'auto', 'srs', 'stratified', 'local'}
        ``'auto'`` uses the stratified formula for stratified samples, the local-mean
        estimator for spatially balanced / systematic samples, and the SRS formula for
        simple random samples. Two-stage samples use the ultimate-cluster
        (with-replacement) approximation.
    k : int
        Neighbourhood size for the local-mean estimator.
    hajek : bool
        Use the ratio (Hájek) form ``Σ y_i/π_i / Σ 1/π_i``, which is more stable than
        dividing by the known ``N`` when the realised sample size varies.

    Returns
    -------
    EstimateResult with ``mean``, ``se``, ``lo``, ``hi``, ``total``, ``variance_method``.
    """
    y = np.asarray(y, dtype=float).ravel()
    if len(y) == sample.N:
        y = y[sample.indices]
    if len(y) != sample.n:
        raise ValueError(f"y has {len(y)} values but the sample has {sample.n} units (frame N={sample.N})")
    if not np.all(np.isfinite(sample.pi)):
        raise ValueError(f"{sample.design!r} has no defined inclusion probabilities; "
                         "design-based estimation does not apply")
    pi, N, n = sample.pi, sample.N, sample.n
    z = y / pi
    total = float(z.sum())
    mean = total / float(np.sum(1.0 / pi)) if hajek else total / N

    if variance == "auto":
        d = sample.design
        variance = "stratified" if d == "stratified" else \
            "srs" if d == "simple random" else \
            "cluster" if d == "two-stage cluster" else "local"

    if variance == "srs":
        f = n / N
        var_mean = (1 - f) * y.var(ddof=1) / n
    elif variance == "stratified":
        if sample.strata is None or sample.frame_strata is None:
            raise ValueError("stratified variance needs a stratified sample")
        var_mean = 0.0
        for h in np.unique(sample.strata):
            yh = y[sample.strata == h]
            Nh = float((sample.frame_strata == h).sum())
            nh = len(yh)
            if nh < 2:
                raise ValueError(f"stratum {h!r} has a single unit; its variance cannot be estimated")
            var_mean += (Nh / N) ** 2 * (1 - nh / Nh) * yh.var(ddof=1) / nh
    elif variance == "local":
        var_mean = local_mean_variance(z, sample.coords, pi, k=k) / N ** 2
    elif variance == "cluster":
        if sample.clusters is None:
            raise ValueError("cluster variance needs a two-stage sample")
        # ratio-estimator linearisation: cluster totals of the residuals (y_i - mean) / pi_i
        resid = (y - mean) / pi
        cl = pd.Series(resid).groupby(sample.clusters).sum().to_numpy()
        m = len(cl)
        denom = float(np.sum(1.0 / pi)) if hajek else float(N)
        var_mean = m * cl.var(ddof=1) / denom ** 2 if m > 1 else np.nan
    else:
        raise ValueError("variance must be 'auto', 'srs', 'stratified', 'local' or 'cluster'")

    se = float(np.sqrt(max(var_mean, 0.0)))
    crit = stats.norm.ppf(0.5 + conf / 2)
    return EstimateResult(mean=float(mean), se=se, lo=float(mean - crit * se), hi=float(mean + crit * se),
                          total=total, n=n, N=N, variance_method=variance, conf=conf)


def design_effect(var_design: float, var_srs: float) -> float:
    """``deff = Var(design) / Var(SRS of the same size)``: < 1 means the design is more efficient."""
    return float(var_design / var_srs)


def effective_sample_size(n: int, rho: float, kind: str = "ar1") -> float:
    """
    Effective number of independent observations in ``n`` spatially autocorrelated ones.

    ``kind='ar1'``           ``n (1 - ρ) / (1 + ρ)``   (exponentially decaying correlation)
    ``kind='equicorrelated'``  ``n / (1 + (n - 1) ρ)``   (constant correlation ``ρ`` between all pairs)

    ``rho`` is the average correlation between neighbouring (or all) pairs, e.g. Moran's I under a
    row-standardised neighbour matrix. Both are approximations meant to convey the *scale* of the
    loss of information.
    """
    if not -1 < rho < 1:
        raise ValueError("rho must lie in (-1, 1)")
    if kind == "ar1":
        return float(n * (1 - rho) / (1 + rho))
    if kind == "equicorrelated":
        return float(n / (1 + (n - 1) * rho))
    raise ValueError("kind must be 'ar1' or 'equicorrelated'")


def sample_size_mean(sd: float, margin: float, N: Optional[int] = None, conf: float = 0.95,
                     deff: float = 1.0) -> int:
    """
    Sample size to estimate a mean to within ``± margin`` at confidence ``conf``:
    ``n0 = deff · (z σ / margin)²`` with the finite-population correction ``n = n0 / (1 + n0/N)``.
    """
    z = stats.norm.ppf(0.5 + conf / 2)
    n0 = deff * (z * sd / margin) ** 2
    n = n0 / (1 + n0 / N) if N else n0
    return int(np.ceil(n))
