"""
spatialstats.regression._utils
==============================
Shared helpers: design-matrix preparation, log-determinants, coefficient tables.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import splu


def prepare_data(y, X, add_constant: bool = True, names: Optional[Sequence[str]] = None,
                 y_name: str = "y") -> Tuple[np.ndarray, np.ndarray, List[str], str]:
    """Coerce (y, X) to float arrays, returning variable names."""
    if isinstance(y, pd.Series) and y.name is not None:
        y_name = str(y.name)
    yv = np.asarray(y, dtype=float).ravel()

    if isinstance(X, pd.DataFrame):
        xnames = [str(c) for c in X.columns]
        Xv = X.to_numpy(dtype=float)
    else:
        if isinstance(X, pd.Series):
            xnames = [str(X.name) if X.name is not None else "x1"]
        else:
            xnames = None
        Xv = np.asarray(X, dtype=float)
        if Xv.ndim == 1:
            Xv = Xv[:, None]
        if names is not None:
            xnames = list(names)
        elif xnames is None or len(xnames) != Xv.shape[1]:
            xnames = [f"x{i + 1}" for i in range(Xv.shape[1])]
    if len(xnames) != Xv.shape[1]:
        raise ValueError(f"got {len(xnames)} names for {Xv.shape[1]} columns")
    if len(yv) != Xv.shape[0]:
        raise ValueError(f"y has {len(yv)} rows but X has {Xv.shape[0]}")
    if not (np.isfinite(yv).all() and np.isfinite(Xv).all()):
        raise ValueError("y and X must not contain NaN or infinite values")
    if add_constant:
        Xv = np.column_stack([np.ones(len(yv)), Xv])
        xnames = ["const"] + xnames
    if np.linalg.matrix_rank(Xv) < Xv.shape[1]:
        raise ValueError(
            "X is rank deficient (perfect multicollinearity, or a constant column "
            "was included together with add_constant=True)"
        )
    return yv, Xv, xnames, y_name


def check_w(w, n: int):
    """Validate a W object against the sample size and return its sparse matrix."""
    if w is None:
        raise ValueError("a spatial weights object w is required")
    if w.n != n:
        raise ValueError(f"w has {w.n} observations but the data have {n}")
    return w.sparse.tocsr()


def logdet_factory(Wm: sparse.csr_matrix):
    """Return ``f(rho) = ln|I - rho W|`` using a sparse LU factorisation."""
    n = Wm.shape[0]
    eye = sparse.identity(n, format="csc")
    Wc = Wm.tocsc()

    def f(rho: float) -> float:
        lu = splu((eye - rho * Wc).tocsc())
        d = lu.U.diagonal()
        if np.any(d == 0):
            return -np.inf
        return float(np.sum(np.log(np.abs(d))))

    return f


def rho_bounds(w, Wm: sparse.csr_matrix) -> Tuple[float, float]:
    """Feasible interval for the spatial parameter (inverse extreme eigenvalues of W)."""
    eps = 1e-3
    if w.transform_type == "row":
        return -1.0 + eps, 1.0 - eps
    from scipy.sparse.linalg import eigs
    try:
        lmax = float(np.real(eigs(Wm.astype(float), k=1, which="LR", return_eigenvectors=False)[0]))
        lmin = float(np.real(eigs(Wm.astype(float), k=1, which="SR", return_eigenvectors=False)[0]))
    except Exception:  # pragma: no cover - ARPACK convergence issues
        ev = np.real(np.linalg.eigvals(Wm.toarray()))
        lmax, lmin = float(ev.max()), float(ev.min())
    hi = 1.0 / lmax if lmax > 0 else 1.0
    lo = 1.0 / lmin if lmin < 0 else -1.0
    return lo * (1 - eps) if lo < 0 else lo + eps, hi * (1 - eps)


def dense_inverse_ok(n: int, limit: int = 6000) -> None:
    if n > limit:
        raise MemoryError(
            f"n={n} is too large for the dense inverse used for standard errors "
            f"(limit {limit}); pass vm=False to skip them"
        )


def coef_table(names: Sequence[str], est, se, stat, p, stat_name: str = "z") -> pd.DataFrame:
    return pd.DataFrame({
        "variable": list(names), "coef": est, "std_err": se, stat_name: stat, "p_value": p,
    })


def sig_stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.1 else ""
