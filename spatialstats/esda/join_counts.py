"""
spatialstats.esda.join_counts
=============================
Join count statistics for binary / categorical data.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from spatialstats.core.result import SpatialResult
from spatialstats.esda.inference import conditional_permutation
from spatialstats.esda._utils import check_y_w
from spatialstats.weights.w import W


class JoinCountResult(SpatialResult):
    def __init__(self, **stats):
        super().__init__(name="Join Counts", **stats)

    def to_frame(self) -> pd.DataFrame:
        keys = ["BB", "WW", "BW", "p_BB", "p_WW", "p_BW"]
        data = {k: [self._stats.get(k)] for k in keys if k in self._stats}
        return pd.DataFrame(data)


def _join_counts_binary(y: np.ndarray, w: W):
    y = np.asarray(y).ravel()
    # Encode as 1 / 0
    vals = np.unique(y)
    if len(vals) != 2:
        raise ValueError("Binary join counts require exactly 2 unique values")
    # Map larger/"True"/1 to Black (1)
    if set(vals) == {0, 1}:
        yb = y.astype(int)
    else:
        yb = (y == vals.max()).astype(int)

    BB = WW = BW = 0.0
    Wmat = w.sparse.tocoo()
    # Count each undirected join once
    seen = set()
    for i, j, wij in zip(Wmat.row, Wmat.col, Wmat.data):
        if i >= j:
            continue
        key = (i, j)
        if key in seen:
            continue
        seen.add(key)
        a, b = yb[i], yb[j]
        if a == 1 and b == 1:
            BB += 1
        elif a == 0 and b == 0:
            WW += 1
        else:
            BW += 1
    return BB, WW, BW


def Join_Counts(
    y,
    w: W,
    permutations: int = 999,
    seed: Optional[int] = None,
) -> JoinCountResult:
    """
    Binary join count statistics (BB, WW, BW).

    Parameters
    ----------
    y : array-like
        Binary or two-category labels.
    w : W
        Spatial weights (binary recommended).
    """
    y = np.asarray(y).ravel()
    check_y_w(y, w, name="y")
    BB, WW, BW = _join_counts_binary(y, w)

    p_BB = p_WW = p_BW = None
    if permutations and permutations > 0:
        def bb(yp):
            return _join_counts_binary(yp, w)[0]

        def ww(yp):
            return _join_counts_binary(yp, w)[1]

        def bw(yp):
            return _join_counts_binary(yp, w)[2]

        p_BB, _ = conditional_permutation(y, bb, permutations, seed=seed)
        p_WW, _ = conditional_permutation(y, ww, permutations, seed=seed)
        p_BW, _ = conditional_permutation(y, bw, permutations, seed=seed)

    return JoinCountResult(
        BB=float(BB),
        WW=float(WW),
        BW=float(BW),
        p_BB=float(p_BB) if p_BB is not None else np.nan,
        p_WW=float(p_WW) if p_WW is not None else np.nan,
        p_BW=float(p_BW) if p_BW is not None else np.nan,
        permutations=permutations,
        n=w.n,
    )
