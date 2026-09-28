"""
spatialstats.esda._utils
========================
Shared validation helpers for ESDA routines.
"""

from __future__ import annotations

import numpy as np

from spatialstats.weights.w import W


def as_float_vector(y, name: str = "y") -> np.ndarray:
    """Convert to 1-D float array."""
    arr = np.asarray(y, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    return arr


def check_y_w(y: np.ndarray, w: W, name: str = "y") -> None:
    """Ensure attribute length matches weights."""
    if len(y) != w.n:
        raise ValueError(
            f"{name} length ({len(y)}) does not match weights n ({w.n})"
        )


def check_xy_w(x: np.ndarray, y: np.ndarray, w: W) -> None:
    """Ensure x and y lengths match weights."""
    if len(x) != len(y):
        raise ValueError(f"x length ({len(x)}) != y length ({len(y)})")
    if len(x) != w.n:
        raise ValueError(
            f"variable length ({len(x)}) does not match weights n ({w.n})"
        )
