"""
spatialstats.bayes.rates
========================
Raw small-area disease-mapping quantities: expected counts, SMR, exact
confidence intervals, and funnel-plot limits.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats


def _as_counts(observed, expected):
    O = np.asarray(observed, dtype=float).ravel()
    E = np.asarray(expected, dtype=float).ravel()
    if len(O) != len(E):
        raise ValueError("observed and expected must have the same length")
    if np.any(O < 0):
        raise ValueError("observed counts must be non-negative")
    if np.any(E <= 0):
        raise ValueError("expected counts must be strictly positive")
    return O, E


def expected_counts(population, observed=None, reference_rate=None) -> np.ndarray:
    """
    Expected counts by *indirect standardisation*.

    Parameters
    ----------
    population : array (n,) or (n, S)
        Person-years at risk; a second axis holds strata (e.g. age groups).
    observed : array (n,), optional
        Used with 1-D ``population`` to derive the overall reference rate
        ``sum(observed) / sum(population)`` (so that ``sum(E) == sum(O)``).
    reference_rate : float or array (S,), optional
        Reference rate(s) per person-year. Required for stratified populations.

    Notes
    -----
    Without strata (age/sex) the expected counts ignore population structure, so
    a high SMR may partly reflect an older population.
    """
    pop = np.asarray(population, dtype=float)
    if reference_rate is None:
        if pop.ndim != 1 or observed is None:
            raise ValueError("give reference_rate, or 1-D population together with observed")
        reference_rate = float(np.sum(observed) / pop.sum())
    rate = np.asarray(reference_rate, dtype=float)
    return pop * rate if pop.ndim == 1 else pop @ rate


def smr(observed, expected) -> np.ndarray:
    """Standardised morbidity/mortality ratio ``O / E`` (the crude relative risk)."""
    O, E = _as_counts(observed, expected)
    return O / E


def smr_confidence_interval(observed, expected, alpha: float = 0.05) -> pd.DataFrame:
    """
    Exact (Garwood) Poisson confidence interval for the SMR.

    The interval is very wide when ``E`` is small, which is the small-area
    problem in one picture.
    """
    O, E = _as_counts(observed, expected)
    lo = np.where(O > 0, stats.chi2.ppf(alpha / 2, 2 * O) / (2 * E), 0.0)
    hi = stats.chi2.ppf(1 - alpha / 2, 2 * (O + 1)) / (2 * E)
    return pd.DataFrame({"smr": O / E, "lo": lo, "hi": hi, "significant": (lo > 1) | (hi < 1)})


def excess_pvalue(observed, expected) -> np.ndarray:
    """One-sided exact Poisson p-value ``P(X >= O | RR = 1)`` for an excess of cases."""
    O, E = _as_counts(observed, expected)
    return stats.poisson.sf(O - 1, E)


def funnel_limits(expected, levels: Sequence[float] = (0.95, 0.998)) -> pd.DataFrame:
    """
    Funnel-plot control limits for the SMR under RR = 1 (exact Poisson quantiles).

    Plot SMR against ``E`` and overlay these limits: areas outside are unusual
    given their size. Returns one lower/upper column pair per confidence level.
    """
    E = np.sort(np.asarray(expected, dtype=float).ravel())
    out = {"expected": E}
    for lev in levels:
        a = (1 - lev) / 2
        out[f"lo_{lev:g}"] = stats.poisson.ppf(a, E) / E
        out[f"hi_{lev:g}"] = stats.poisson.ppf(1 - a, E) / E
    return pd.DataFrame(out)
