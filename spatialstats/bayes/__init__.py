"""
spatialstats.bayes
==================
Bayesian and empirical-Bayes disease mapping: raw SMRs and their uncertainty,
empirical Bayes smoothing (gamma–Poisson, James–Stein, spatial local EB), and
the BYM model fitted by MCMC.

Workflow
--------
>>> from spatialstats.bayes import expected_counts, smr, eb_gamma_poisson, eb_local, BYM
>>> E = expected_counts(pop, observed=O)
>>> raw = smr(O, E)
>>> eb  = eb_gamma_poisson(O, E)                  # shrinks toward the global mean
>>> loc = eb_local(O, E, w)                       # shrinks toward the neighbourhood mean
>>> fit = BYM(O, E, w, seed=1)                    # full Bayesian smoothing (built-in sampler)
>>> fit.rr_mean, fit.exceedance(1.0)

The default sampler is dependency-free. For NUTS via PyMC (and BYM2) install
``pip install pyspatialstats[bayes]`` and pass ``backend='pymc'``.
"""

from spatialstats.bayes.rates import (
    expected_counts, smr, smr_confidence_interval, excess_pvalue, funnel_limits,
)
from spatialstats.bayes.eb import (
    EBResult, eb_gamma_poisson, eb_james_stein, eb_local,
)
from spatialstats.bayes.bym import BYM
from spatialstats.bayes.icar import icar_structure, icar_scaling_factor, rhat, ess
from spatialstats.bayes.pymc_backend import pymc_available
from spatialstats.bayes.evaluate import (
    simulate_relative_risk, simulate_counts, shrinkage_summary, evaluate_smoothing,
)

__all__ = [
    "expected_counts", "smr", "smr_confidence_interval", "excess_pvalue", "funnel_limits",
    "EBResult", "eb_gamma_poisson", "eb_james_stein", "eb_local",
    "BYM", "icar_structure", "icar_scaling_factor", "rhat", "ess", "pymc_available",
    "simulate_relative_risk", "simulate_counts", "shrinkage_summary", "evaluate_smoothing",
]
