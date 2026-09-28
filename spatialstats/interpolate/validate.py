"""
spatialstats.interpolate.validate
=================================
Cross-validation and parameter tuning for interpolators.
"""

from __future__ import annotations

import itertools
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from spatialstats.core.result import SpatialResult
from spatialstats.interpolate.base import Interpolator, as_xy


class CVResult(SpatialResult):
    """
    Cross-validation predictions and error summaries.

    Attributes
    ----------
    observed, predicted : arrays (n,)
    rmse, mae, me (mean error = bias), r2 (1 - SSE/SST), corr, n_failed
    """

    def __init__(self, observed, predicted, method: str, **stats_):
        o = np.asarray(observed, dtype=float)
        p = np.asarray(predicted, dtype=float)
        ok = np.isfinite(p)
        e = p[ok] - o[ok]
        sst = float(np.sum((o[ok] - o[ok].mean()) ** 2))
        super().__init__(
            name=f"Cross-validation ({method})",
            rmse=float(np.sqrt(np.mean(e ** 2))), mae=float(np.mean(np.abs(e))), me=float(np.mean(e)),
            r2=float(1 - np.sum(e ** 2) / sst) if sst > 0 else np.nan,
            corr=float(np.corrcoef(o[ok], p[ok])[0, 1]) if ok.sum() > 2 else np.nan,
            n=int(ok.sum()), n_failed=int((~ok).sum()), **stats_,
        )
        self.observed, self.predicted, self.method = o, p, method
        for k in ("rmse", "mae", "me", "r2", "corr"):
            setattr(self, k, self._stats[k])

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"observed": self.observed, "predicted": self.predicted,
                             "error": self.predicted - self.observed})


def _folds(xy: np.ndarray, method: str, k: int, block_size: Optional[float], rng):
    n = len(xy)
    if method == "loo":
        return [np.array([i]) for i in range(n)]
    if method == "kfold":
        perm = rng.permutation(n)
        return [perm[i::k] for i in range(k)]
    if method == "spatial_block":
        if block_size is None:
            block_size = 0.2 * max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]))
        ix = np.floor((xy[:, 0] - xy[:, 0].min()) / block_size).astype(int)
        iy = np.floor((xy[:, 1] - xy[:, 1].min()) / block_size).astype(int)
        block = ix * (iy.max() + 1) + iy
        ids = rng.permutation(np.unique(block))
        if len(ids) < 2:
            raise ValueError("block_size is too large: fewer than 2 spatial blocks")
        kk = min(k, len(ids))
        groups = [ids[i::kk] for i in range(kk)]
        return [np.flatnonzero(np.isin(block, g)) for g in groups]
    raise ValueError("method must be 'loo', 'kfold' or 'spatial_block'")


def cross_validate(model: Interpolator, coords, values, method: str = "loo", k: int = 10,
                   block_size: Optional[float] = None, seed: Optional[int] = None) -> CVResult:
    """
    Cross-validate an interpolator.

    Parameters
    ----------
    method : {'loo', 'kfold', 'spatial_block'}
        ``'loo'`` leaves one observation out (optimistic when observations are clustered:
        each held-out point has close neighbours). ``'spatial_block'`` holds out whole square
        blocks (``block_size`` in coordinate units), which mimics predicting into unsampled
        areas and gives more honest error estimates for clustered monitoring networks.
    k : int
        Number of folds for ``'kfold'`` / ``'spatial_block'``.
    """
    xy = as_xy(coords)
    z = np.asarray(values, dtype=float).ravel()
    ok = np.isfinite(z)
    xy, z = xy[ok], z[ok]
    rng = np.random.default_rng(seed)
    pred = np.full(len(z), np.nan)
    for test in _folds(xy, method, k, block_size, rng):
        train = np.setdiff1d(np.arange(len(z)), test)
        m = model.clone().fit(xy[train], z[train])
        try:
            pred[test] = m.predict(xy[test])
        except Exception:                      # e.g. singular kriging system in a fold
            pred[test] = np.nan
    return CVResult(z, pred, method)


def grid_search(model_class: Callable[..., Interpolator], param_grid: Dict[str, Sequence], coords, values,
                method: str = "loo", k: int = 10, block_size: Optional[float] = None,
                seed: Optional[int] = None, metric: str = "rmse") -> pd.DataFrame:
    """
    Try every combination of ``param_grid`` and rank by cross-validated ``metric``.

    Example
    -------
    >>> grid_search(IDW, {"power": [1, 2, 3, 4], "k": [8, 16]}, xy, z, method="spatial_block")
    """
    rows = []
    keys = list(param_grid)
    for combo in itertools.product(*(param_grid[k_] for k_ in keys)):
        params = dict(zip(keys, combo))
        cv = cross_validate(model_class(**params), coords, values, method, k, block_size, seed)
        rows.append({**params, "rmse": cv.rmse, "mae": cv.mae, "me": cv.me, "r2": cv.r2})
    df = pd.DataFrame(rows)
    return df.sort_values(metric).reset_index(drop=True)


def compare_methods(models: Dict[str, Interpolator], coords, values, method: str = "loo", k: int = 10,
                    block_size: Optional[float] = None, seed: Optional[int] = None) -> pd.DataFrame:
    """Cross-validated error table for several interpolators (same folds for all)."""
    rows = {}
    for label, m in models.items():
        cv = cross_validate(m, coords, values, method, k, block_size, seed)
        rows[label] = {"rmse": cv.rmse, "mae": cv.mae, "me": cv.me, "r2": cv.r2, "n_failed": cv.stats["n_failed"]}
    return pd.DataFrame(rows).T.sort_values("rmse")
