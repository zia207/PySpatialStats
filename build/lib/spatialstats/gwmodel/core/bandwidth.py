"""
spatialstats.gwmodel.core.bandwidth
========================
Bandwidth selection algorithms for geographically weighted models.
"""

import numpy as np
from typing import Callable, Optional, Tuple, Union
from scipy.optimize import minimize_scalar


class BandwidthSelector:
    """
    Bandwidth selector using golden-section search or grid search.

    Parameters
    ----------
    score_fn : callable
        Function ``score_fn(bandwidth) -> float`` to minimise.
    fixed : bool
        True = fixed distance bandwidth; False = adaptive k-NN.
    bw_min : float or int, optional
        Lower bound on bandwidth search.
    bw_max : float or int, optional
        Upper bound on bandwidth search.
    criterion : str
        Label stored for reference ('AICc', 'CV', etc.).
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum iterations.
    """

    def __init__(self, score_fn: Callable, fixed: bool = False,
                 bw_min=None, bw_max=None, criterion: str = "AICc",
                 tol: float = 1.0, max_iter: int = 200):
        self.score_fn = score_fn
        self.fixed = fixed
        self.bw_min = bw_min
        self.bw_max = bw_max
        self.criterion = criterion
        self.tol = tol
        self.max_iter = max_iter
        self.bw_history_: list = []
        self.score_history_: list = []

    def golden_section(self) -> Tuple[float, float]:
        """
        Golden-section search for optimal bandwidth.

        Returns
        -------
        (optimal_bandwidth, optimal_score)
        """
        a, b = float(self.bw_min), float(self.bw_max)
        phi = (np.sqrt(5) - 1) / 2
        c = b - phi * (b - a)
        d = a + phi * (b - a)
        fc = self.score_fn(c)
        fd = self.score_fn(d)
        self.bw_history_ = [c, d]
        self.score_history_ = [fc, fd]

        for _ in range(self.max_iter):
            if abs(b - a) < self.tol:
                break
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - phi * (b - a)
                fc = self.score_fn(c)
                self.bw_history_.append(c)
                self.score_history_.append(fc)
            else:
                a, c, fc = c, d, fd
                d = a + phi * (b - a)
                fd = self.score_fn(d)
                self.bw_history_.append(d)
                self.score_history_.append(fd)

        opt_bw = (a + b) / 2
        opt_score = self.score_fn(opt_bw)
        return opt_bw, opt_score

    def grid_search(self, n_grid: int = 50) -> Tuple[float, float]:
        """
        Exhaustive grid search over [bw_min, bw_max].

        Returns
        -------
        (optimal_bandwidth, optimal_score)
        """
        if self.fixed:
            grid = np.linspace(self.bw_min, self.bw_max, n_grid)
        else:
            grid = np.arange(int(self.bw_min), int(self.bw_max) + 1, dtype=float)

        scores = []
        for bw in grid:
            s = self.score_fn(bw)
            scores.append(s)
            self.bw_history_.append(bw)
            self.score_history_.append(s)

        best_idx = np.argmin(scores)
        return float(grid[best_idx]), float(scores[best_idx])

    def select(self, method: str = "golden_section") -> Tuple[float, float]:
        """
        Run bandwidth selection.

        Parameters
        ----------
        method : 'golden_section' | 'grid'

        Returns
        -------
        (optimal_bandwidth, optimal_score)
        """
        if method == "golden_section":
            return self.golden_section()
        elif method == "grid":
            return self.grid_search()
        else:
            raise ValueError(f"Unknown method '{method}'.")
