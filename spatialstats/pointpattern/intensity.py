"""
spatialstats.pointpattern.intensity
===================================
First-order properties: quadrat counts and kernel intensity estimation.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import shapely
from scipy import stats
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

from spatialstats.core.result import SpatialResult
from spatialstats.pointpattern.pattern import PointPattern


# ---------------------------------------------------------------------------
# Quadrat test
# ---------------------------------------------------------------------------

class QuadratResult(SpatialResult):
    def __init__(self, counts, expected, cell_area, xedges, yedges, **stats_):
        super().__init__(name="Quadrat test of CSR", **stats_)
        self.counts = counts
        self.expected = expected
        self.cell_area = cell_area
        self.xedges = xedges
        self.yedges = yedges

    def plot(self, ax=None, **kwargs):
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots()
        c = np.where(self.expected > 0, self.counts, np.nan)
        m = ax.pcolormesh(self.xedges, self.yedges, c, **kwargs)
        ax.figure.colorbar(m, ax=ax, label="count")
        ax.set_aspect("equal")
        ax.set_title("Quadrat counts")
        return ax


def quadrat_test(pp: PointPattern, nx: int = 5, ny: int = 5,
                 permutations: int = 0, seed: Optional[int] = None) -> QuadratResult:
    """
    Quadrat-count chi-square test of complete spatial randomness.

    The bounding box of the window is divided into ``nx × ny`` cells. Under CSR
    the count in each cell is Poisson with mean proportional to the part of the
    cell that lies inside the window, so irregular windows are handled properly.

    Parameters
    ----------
    permutations : int
        If > 0, also report a Monte Carlo p-value from that many multinomial
        simulations (recommended when expected counts are small).

    Returns
    -------
    QuadratResult
        ``chi2``, ``df``, ``p_value`` (two-sided), ``p_clustered`` (upper tail),
        ``p_regular`` (lower tail), ``index_of_dispersion`` (variance / mean).
    """
    xmin, ymin, xmax, ymax = pp.window.bounds
    xe = np.linspace(xmin, xmax, nx + 1)
    ye = np.linspace(ymin, ymax, ny + 1)
    counts, _, _ = np.histogram2d(pp.coords[:, 0], pp.coords[:, 1], bins=[xe, ye])
    counts = counts.T  # rows = y

    if pp.window.is_rectangle:
        cell_area = np.full((ny, nx), (xe[1] - xe[0]) * (ye[1] - ye[0]))
    else:
        cell_area = np.zeros((ny, nx))
        for i in range(ny):
            for j in range(nx):
                cell = shapely.box(xe[j], ye[i], xe[j + 1], ye[i + 1])
                cell_area[i, j] = cell.intersection(pp.window.polygon).area
    prob = cell_area / cell_area.sum()
    expected = pp.n * prob
    use = expected > 0
    chi2 = float(np.sum((counts[use] - expected[use]) ** 2 / expected[use]))
    df = int(use.sum() - 1)
    p_up = float(stats.chi2.sf(chi2, df))
    p_lo = float(stats.chi2.cdf(chi2, df))
    p_two = min(1.0, 2 * min(p_up, p_lo))

    extra = {}
    if permutations and permutations > 0:
        rng = np.random.default_rng(seed)
        sims = rng.multinomial(pp.n, prob[use], size=permutations)
        e = expected[use]
        chi_sim = ((sims - e) ** 2 / e).sum(axis=1)
        extra["p_mc"] = float((1 + np.sum(chi_sim >= chi2)) / (1 + permutations))

    ioD = float(counts[use].var(ddof=1) / counts[use].mean()) if counts[use].mean() > 0 else np.nan
    return QuadratResult(
        counts, expected, cell_area, xe, ye,
        chi2=chi2, df=df, p_value=p_two, p_clustered=p_up, p_regular=p_lo,
        index_of_dispersion=ioD, n_cells=int(use.sum()),
        min_expected=float(expected[use].min()), **extra,
    )


# ---------------------------------------------------------------------------
# Kernel intensity
# ---------------------------------------------------------------------------

def bandwidth_scott(pp: PointPattern) -> float:
    """Scott's rule for a 2-D Gaussian kernel: ``σ · n^(-1/6)`` (σ = mean coordinate s.d.)."""
    sd = float(np.mean(pp.coords.std(axis=0, ddof=1)))
    return sd * pp.n ** (-1.0 / 6.0)


def bandwidth_cv(pp: PointPattern, candidates: Optional[np.ndarray] = None,
                 max_points: int = 5000, seed: Optional[int] = 0) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Likelihood cross-validation bandwidth for a Gaussian kernel.

    Maximises ``Σ_i log f_{-i}(x_i)`` where ``f_{-i}`` is the leave-one-out density.
    Large patterns are subsampled to ``max_points`` events for speed.

    Returns
    -------
    (best_h, candidates, cv_scores)
    """
    coords = pp.coords
    if len(coords) > max_points:
        rng = np.random.default_rng(seed)
        coords = coords[rng.choice(len(coords), max_points, replace=False)]
    n = len(coords)
    h0 = bandwidth_scott(pp)
    if candidates is None:
        candidates = h0 * np.geomspace(0.1, 3.0, 25)
    tree = cKDTree(coords)
    scores = []
    for h in candidates:
        neigh = tree.query_ball_point(coords, r=4.0 * h)
        ll = 0.0
        for i, idx in enumerate(neigh):
            idx = [j for j in idx if j != i]
            if idx:
                d2 = ((coords[idx] - coords[i]) ** 2).sum(axis=1)
                dens = np.exp(-d2 / (2 * h * h)).sum() / (2 * np.pi * h * h * (n - 1))
            else:
                dens = 0.0
            ll += np.log(max(dens, 1e-300))
        scores.append(ll)
    scores = np.asarray(scores)
    return float(candidates[int(np.argmax(scores))]), np.asarray(candidates), scores


class KDEResult(SpatialResult):
    """Kernel intensity surface on a regular grid (rows = y ascending, cols = x ascending)."""

    def __init__(self, density, x, y, bandwidth, mask, n_events, **stats_):
        super().__init__(name="Kernel intensity", bandwidth=bandwidth, **stats_)
        self.density = density
        self.x = x
        self.y = y
        self.bandwidth = bandwidth
        self.mask = mask
        self.n_events = n_events

    @property
    def extent(self):
        dx = self.x[1] - self.x[0]
        dy = self.y[1] - self.y[0]
        return (self.x[0] - dx / 2, self.x[-1] + dx / 2, self.y[0] - dy / 2, self.y[-1] + dy / 2)

    @property
    def integral(self) -> float:
        dx = self.x[1] - self.x[0]
        dy = self.y[1] - self.y[0]
        return float(np.nansum(self.density) * dx * dy)

    def at(self, coords) -> np.ndarray:
        """
        Bilinear lookup of the intensity at arbitrary locations.

        Cells outside the window take the value of the nearest cell inside it, and
        locations beyond the grid edge are clamped to it, so events on the window
        boundary get a finite value.
        """
        from scipy.ndimage import distance_transform_edt
        grid = self.density
        bad = ~np.isfinite(grid)
        if bad.any() and not bad.all():
            idx = distance_transform_edt(bad, return_distances=False, return_indices=True)
            grid = grid[tuple(idx)]
        f = RegularGridInterpolator((self.y, self.x), grid, bounds_error=False, fill_value=None)
        c = np.asarray(coords, dtype=float)
        xq = np.clip(c[:, 0], self.x[0], self.x[-1])
        yq = np.clip(c[:, 1], self.y[0], self.y[-1])
        return f(np.column_stack([yq, xq]))

    def to_frame(self):
        import pandas as pd
        gx, gy = np.meshgrid(self.x, self.y)
        return pd.DataFrame({"x": gx.ravel(), "y": gy.ravel(), "density": self.density.ravel()})

    def plot(self, ax=None, points: Optional[PointPattern] = None, **kwargs):
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=kwargs.pop("figsize", (6, 6)))
        kwargs.setdefault("cmap", "viridis")
        im = ax.imshow(self.density, origin="lower", extent=self.extent, **kwargs)
        ax.figure.colorbar(im, ax=ax, label="intensity (events / unit area)")
        if points is not None:
            ax.scatter(points.coords[:, 0], points.coords[:, 1], s=1, c="w", alpha=0.4)
        ax.set_aspect("equal")
        ax.set_title(f"Kernel intensity (h={self.bandwidth:.4g})")
        return ax


def _kernel_1d(u: np.ndarray, kernel: str) -> np.ndarray:
    if kernel == "gaussian":
        return np.exp(-0.5 * u * u) / np.sqrt(2 * np.pi)
    if kernel == "epanechnikov":
        return np.where(np.abs(u) <= 1, 0.75 * (1 - u * u), 0.0)
    raise ValueError("kernel must be 'gaussian' or 'epanechnikov'")


def kde(pp: PointPattern, bandwidth: Union[float, str] = "scott", grid_size: int = 128,
        kernel: str = "gaussian", edge_correction: bool = True,
        weights: Optional[np.ndarray] = None) -> KDEResult:
    """
    Kernel estimate of the first-order intensity λ(u).

    The result is scaled so that it integrates to the number of events (or the
    sum of ``weights``): values are events per unit area.

    Parameters
    ----------
    bandwidth : float or {'scott', 'cv'}
        Kernel scale in coordinate units. ``'cv'`` uses :func:`bandwidth_cv`.
    grid_size : int
        Number of grid cells along the longer side of the window.
    kernel : {'gaussian', 'epanechnikov'}
        Product kernels (the Epanechnikov one is separable, not radial).
    edge_correction : bool
        Uniform (Jones) edge correction: divide by the kernel mass falling
        inside the window. Works for any window shape.
    weights : array (n,), optional
        Per-event weights (e.g. counts).
    """
    if isinstance(bandwidth, str):
        if bandwidth == "scott":
            h = bandwidth_scott(pp)
        elif bandwidth == "cv":
            h = bandwidth_cv(pp)[0]
        else:
            raise ValueError("bandwidth must be a number, 'scott' or 'cv'")
    else:
        h = float(bandwidth)
    if h <= 0:
        raise ValueError("bandwidth must be positive")

    mask, xs, ys, step = pp.window.raster(grid_size)
    w = np.ones(pp.n) if weights is None else np.asarray(weights, dtype=float)

    # separable kernels: density = Ky (ny×n) · diag(w) · Kx.T (n×nx)
    kx = _kernel_1d((xs[None, :] - pp.coords[:, 0][:, None]) / h, kernel) / h   # (n, nx)
    ky = _kernel_1d((ys[None, :] - pp.coords[:, 1][:, None]) / h, kernel) / h   # (n, ny)
    dens = (ky * w[:, None]).T @ kx                                             # (ny, nx)

    if edge_correction:
        gx = _kernel_1d((xs[None, :] - xs[:, None]) / h, kernel) / h            # (nx, nx)
        gy = _kernel_1d((ys[None, :] - ys[:, None]) / h, kernel) / h            # (ny, ny)
        mass = gy @ mask.astype(float) @ gx.T * step * step
        with np.errstate(divide="ignore", invalid="ignore"):
            dens = np.where(mass > 1e-12, dens / mass, np.nan)
    dens = np.where(mask, dens, np.nan)
    return KDEResult(dens, xs, ys, h, mask, pp.n, kernel=kernel,
                     edge_correction=edge_correction)


def intensity_at_points(pp: PointPattern, bandwidth: Optional[float] = None,
                        leave_one_out: bool = True) -> np.ndarray:
    """
    Gaussian-kernel intensity evaluated at each event, optionally leaving the
    event itself out (as required for inhomogeneous K-function estimation).
    """
    h = bandwidth_scott(pp) if bandwidth is None else float(bandwidth)
    tree = cKDTree(pp.coords)
    neigh = tree.query_ball_point(pp.coords, r=4.0 * h)
    out = np.empty(pp.n)
    for i, idx in enumerate(neigh):
        d2 = ((pp.coords[idx] - pp.coords[i]) ** 2).sum(axis=1)
        val = np.exp(-d2 / (2 * h * h)).sum()
        if leave_one_out:
            val -= 1.0
        out[i] = val / (2 * np.pi * h * h)
    return np.maximum(out, 1e-12 / pp.area)


def relative_risk(cases: PointPattern, controls: PointPattern, bandwidth: Union[float, str] = "scott",
                  grid_size: int = 128, log: bool = True) -> KDEResult:
    """
    Kernel relative-risk surface ``f_cases(u) / f_controls(u)`` (spatial densities, not intensities).

    Both patterns must share the same window. Returns a :class:`KDEResult` whose
    ``density`` holds the (log) ratio.
    """
    h = bandwidth_scott(controls) if bandwidth == "scott" else bandwidth
    kc = kde(cases, bandwidth=h, grid_size=grid_size)
    ko = kde(controls, bandwidth=h, grid_size=grid_size)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (kc.density / cases.n) / (ko.density / controls.n)
    out = np.log(ratio) if log else ratio
    return KDEResult(out, kc.x, kc.y, kc.bandwidth, kc.mask, cases.n, log=log)
