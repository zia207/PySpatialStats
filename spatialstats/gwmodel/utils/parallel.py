"""
spatialstats.gwmodel.utils.parallel
========================
Parallel computation helpers.
"""

import numpy as np
from joblib import Parallel, delayed
from typing import Callable, List, Any, Iterable, Optional

from spatialstats.gwmodel.utils.compute import get_compute_config, resolve_n_jobs


def parallel_map(
    tasks: Iterable,
    n_jobs: Optional[int] = None,
    prefer: Optional[str] = None,
) -> list:
    """
    Run delayed tasks in parallel using global compute settings.

    Parameters
    ----------
    tasks : iterable of joblib.delayed(...) objects
    n_jobs : int or None
        Worker count. ``None`` uses :func:`get_compute_config`.
    prefer : str or None
        ``'threads'`` or ``'processes'``. ``None`` uses global config.
    """
    cfg = get_compute_config()
    jobs = resolve_n_jobs(n_jobs if n_jobs is not None else cfg.n_jobs)
    backend = prefer if prefer is not None else cfg.joblib_prefer()
    return Parallel(n_jobs=jobs, prefer=backend)(tasks)


def parallel_calibrate(fn: Callable, indices: List[int],
                       n_jobs: int = -1, prefer: Optional[str] = None,
                       **kwargs) -> List[Any]:
    """
    Run ``fn(i, **kwargs)`` for each index in parallel.

    Parameters
    ----------
    fn      : callable  Function accepting index i as first argument.
    indices : list      Calibration point indices.
    n_jobs  : int       Number of parallel workers (-1 = all CPUs).
    prefer  : str or None
        Joblib backend override.

    Returns
    -------
    list of results in order of ``indices``.
    """
    return parallel_map(
        (delayed(fn)(i, **kwargs) for i in indices),
        n_jobs=n_jobs,
        prefer=prefer,
    )


def chunk_indices(n: int, n_jobs: int) -> List[List[int]]:
    """Split n indices into n_jobs roughly equal chunks."""
    all_idx = list(range(n))
    jobs = resolve_n_jobs(n_jobs)
    if jobs > n:
        jobs = n
    size = max(1, n // jobs)
    chunks = [all_idx[i:i + size] for i in range(0, n, size)]
    return chunks
