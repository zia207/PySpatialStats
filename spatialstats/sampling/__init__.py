"""
spatialstats.sampling
=====================
Spatial sampling and design: probability designs (simple random, systematic,
stratified, two-stage), spatially balanced GRTS-style sampling, space-filling and
conditioned Latin hypercube designs, design-based estimators, sample-size tools,
and Monte Carlo evaluation of designs against a fully known population.

Quick start
-----------
>>> from spatialstats.sampling import simple_random, grts, estimate_mean, compare_designs
>>> s = grts(coords, n=60, seed=1)                 # spatially balanced sample
>>> r = estimate_mean(y, s)                        # mean, s.e., CI (local-mean variance)
>>> compare_designs(y_all, {"SRS": lambda rng: simple_random(coords, 60, rng),
...                         "GRTS": lambda rng: grts(coords, 60, seed=rng)}, n_reps=500)

Every design accepts ``seed`` as an int or a ``numpy.random.Generator``.
"""

from spatialstats.sampling.sample import SpatialSample, as_coords
from spatialstats.sampling.designs import (
    simple_random, systematic, grid_points, spatial_order, stratified, two_stage,
    neyman_allocation, grts, space_filling, clhs,
)
from spatialstats.sampling.estimate import (
    EstimateResult, estimate_mean, local_mean_variance, design_effect,
    effective_sample_size, sample_size_mean,
)
from spatialstats.sampling.evaluate import (
    spatial_balance, coverage_metrics, evaluate_design, compare_designs,
)

__all__ = [
    "SpatialSample", "as_coords",
    "simple_random", "systematic", "grid_points", "spatial_order", "stratified", "two_stage",
    "neyman_allocation", "grts", "space_filling", "clhs",
    "EstimateResult", "estimate_mean", "local_mean_variance", "design_effect",
    "effective_sample_size", "sample_size_mean",
    "spatial_balance", "coverage_metrics", "evaluate_design", "compare_designs",
]
