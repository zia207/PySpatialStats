"""
spatialstats.interpolate.deterministic
======================================
Deterministic interpolators: polynomial trend surfaces, Thiessen (Voronoi)
polygons, nearest neighbour, inverse-distance weighting, Delaunay
triangulation (TIN), and thin-plate / radial-basis splines.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.interpolate import LinearNDInterpolator, RBFInterpolator
from scipy.spatial import cKDTree

from spatialstats.interpolate.base import Interpolator, as_xy


def _n_terms(degree: int) -> int:
    """Number of monomials in a 2-D polynomial of total degree ``degree``."""
    return (degree + 1) * (degree + 2) // 2


def _monomials(xy: np.ndarray, degree: int) -> np.ndarray:
    """Columns ``x^i y^j`` for ``i + j <= degree``, degree by degree, ``i`` descending."""
    x, y = xy[:, 0], xy[:, 1]
    cols = []
    for total in range(degree + 1):
        for j in range(total + 1):
            i = total - j
            cols.append((x ** i) * (y ** j))
    return np.column_stack(cols)


def _powers(degree: int):
    out = []
    for total in range(degree + 1):
        for j in range(total + 1):
            out.append((total - j, j))
    return out


class TrendSurface(Interpolator):
    """
    Global polynomial trend surface of total degree ``degree``.

    .. math::

        \\hat z(x, y) = \\sum_{i+j \\le d} \\beta_{ij}\\, (x-\\bar x)^i (y-\\bar y)^j

    fitted by least squares. A trend surface is a regression, not an exact
    interpolator: it passes through the observations only when they already lie
    on a polynomial of that degree. It extrapolates without bound, so a high
    degree can swing wildly between and beyond the samples.

    Coordinates are centered at the sample mean before the powers are built
    (``center_``). ``coef_`` multiplies those centered monomials, in the order
    stored in ``powers_`` (``(i, j)`` for :math:`x^i y^j`). ``r2_`` is the
    in-sample coefficient of determination.

    Parameters
    ----------
    degree : int
        Total degree (1 = plane, 2 = quadratic, 3 = cubic). Needs at least
        ``(degree + 1)(degree + 2) / 2`` observations, and the samples must
        not be collinear for ``degree >= 1``.
    """

    name = "trend"

    def __init__(self, degree: int = 1):
        if isinstance(degree, bool) or not isinstance(degree, (int, np.integer)) or degree < 1:
            raise ValueError("degree must be an integer >= 1")
        self.degree = int(degree)

    def _fit(self, xy, v):
        n_terms = _n_terms(self.degree)
        if len(v) < n_terms:
            raise ValueError(
                f"degree {self.degree} needs at least {n_terms} observations, got {len(v)}"
            )
        self.center_ = xy.mean(axis=0)
        X = _monomials(xy - self.center_, self.degree)
        norms = np.linalg.norm(X, axis=0)
        norms[norms == 0] = 1.0
        beta, _, rank, _ = np.linalg.lstsq(X / norms, v, rcond=None)
        if rank < n_terms:
            raise ValueError(
                f"degree {self.degree} trend is not identifiable from these coordinates "
                f"(design rank {rank} < {n_terms} terms); lower the degree or drop collinear samples"
            )
        self.coef_ = beta / norms
        self.powers_ = _powers(self.degree)
        fitted = X @ self.coef_
        ss_res = float(np.sum((v - fitted) ** 2))
        ss_tot = float(np.sum((v - v.mean()) ** 2))
        self.r2_ = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
        self.residuals_ = v - fitted

    def _predict(self, xy):
        return _monomials(xy - self.center_, self.degree) @ self.coef_


def _clip_region(region, xy):
    """Polygon to clip Thiessen cells to: bounds, a polygon, a GeoDataFrame, or the sample box."""
    from shapely.geometry import box

    if region is None:
        xmin, ymin = xy.min(axis=0)
        xmax, ymax = xy.max(axis=0)
        if xmax == xmin:
            xmin, xmax = xmin - 1.0, xmax + 1.0
        if ymax == ymin:
            ymin, ymax = ymin - 1.0, ymax + 1.0
        return box(xmin, ymin, xmax, ymax)
    if hasattr(region, "geometry"):
        g = region.geometry
        return g.union_all() if hasattr(g, "union_all") else g.unary_union
    if hasattr(region, "bounds") and hasattr(region, "area"):
        return region
    arr = np.asarray(region, dtype=float)
    if arr.ndim == 1 and len(arr) == 4:
        return box(*arr)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return box(arr[:, 0].min(), arr[:, 1].min(), arr[:, 0].max(), arr[:, 1].max())
    raise ValueError("region must be bounds, a polygon, a GeoDataFrame, or an (n, 2) array")


def thiessen_polygons(coords, values=None, region=None, crs=None):
    """
    Thiessen (Voronoi) polygons: one cell per sample, containing every location
    closer to that sample than to any other.

    Cells are clipped to ``region`` (a ``(xmin, ymin, xmax, ymax)`` tuple, a
    shapely polygon, a GeoDataFrame, or an ``(n, 2)`` array used as a bounding
    box). With ``region=None`` the bounding box of the samples is used. The
    polygons tile that region.

    Parameters
    ----------
    coords : array-like, shape (n, 2)
        Sample locations. Must be unique.
    values : array-like, shape (n,), optional
        Attribute copied onto each polygon (column ``value``).
    region : optional
        Clipping region.
    crs : optional
        CRS of the returned GeoDataFrame. Defaults to ``coords.crs`` or
        ``region.crs`` when one of those is a GeoDataFrame.

    Returns
    -------
    geopandas.GeoDataFrame
        Columns ``x``, ``y``, and ``value`` when ``values`` is given, plus the
        cell polygon. Row order matches ``coords``.
    """
    import geopandas as gpd
    import shapely
    from shapely.geometry import MultiPoint
    from shapely.ops import voronoi_diagram

    if crs is None and hasattr(coords, "crs"):
        crs = coords.crs
    if crs is None and hasattr(region, "crs"):
        crs = region.crs
    xy = as_xy(coords)
    if len(xy) < 2:
        raise ValueError("need at least 2 samples to build Thiessen polygons")
    if len(np.unique(xy, axis=0)) < len(xy):
        raise ValueError(
            "Thiessen polygons need unique sample locations; drop duplicate coordinates first"
        )
    v = None
    if values is not None:
        v = np.asarray(values, dtype=float).ravel()
        if len(v) != len(xy):
            raise ValueError(f"{len(xy)} coordinates but {len(v)} values")
    clip = _clip_region(region, xy)
    if clip.is_empty or clip.area == 0:
        raise ValueError("region has no area to clip Thiessen polygons to")
    diagram = voronoi_diagram(MultiPoint(xy), envelope=clip)
    raw = np.asarray(list(diagram.geoms), dtype=object)
    if len(raw) != len(xy):
        raise RuntimeError(
            f"Voronoi diagram returned {len(raw)} cells for {len(xy)} samples"
        )
    pts = shapely.points(xy[:, 0], xy[:, 1])
    covered = shapely.contains(raw[:, None], pts[None, :])
    if not covered.any(axis=0).all():
        covered = shapely.covers(raw[:, None], pts[None, :])
    if not covered.any(axis=0).all():
        raise RuntimeError("a sample fell outside every Thiessen cell")
    assign = covered.argmax(axis=0)
    if len(np.unique(assign)) != len(xy):
        raise RuntimeError("Thiessen cells could not be matched one-to-one with the samples")
    cells = shapely.intersection(raw[assign], clip)
    data = {"x": xy[:, 0], "y": xy[:, 1]}
    if v is not None:
        data["value"] = v
    return gpd.GeoDataFrame(data, geometry=list(cells), crs=crs)


class NearestNeighbor(Interpolator):
    """
    Value of the nearest observation. Piecewise constant.

    The partition is a Thiessen (Voronoi) tessellation; :class:`Thiessen`
    predicts the same surface and can return the polygons.
    """

    name = "nearest"

    def _fit(self, xy, v):
        self._tree = cKDTree(xy)

    def _predict(self, xy):
        _, i = self._tree.query(xy)
        return self.values_[i]


class Thiessen(NearestNeighbor):
    """
    Proximity analysis by Thiessen (Voronoi) polygons.

    Prediction at a location is the value of the nearest sample — the same
    piecewise-constant surface as :class:`NearestNeighbor`. Call
    :meth:`polygons` for the polygon layer itself.
    """

    name = "thiessen"

    def polygons(self, region=None, crs=None):
        """Thiessen polygons of the fitted samples, clipped to ``region``."""
        if not hasattr(self, "coords_"):
            raise RuntimeError("call fit() first")
        return thiessen_polygons(self.coords_, self.values_, region=region, crs=crs)


class IDW(Interpolator):
    """
    Inverse-distance weighting: ``ẑ(s) = Σ w_i z_i / Σ w_i`` with ``w_i = d_i^(-power)``.

    An exact interpolator (returns the observation at a data location), smooth
    nowhere at the data points, and never predicts outside the range of the
    observations.

    Parameters
    ----------
    power : float
        Distance-decay exponent (2 is conventional; larger = more local).
    k : int or None
        Use the ``k`` nearest observations (None = all).
    radius : float or None
        Ignore observations farther than this; locations with none get NaN.
    smoothing : float
        Added to distances (``d + smoothing``) to remove the spike at data points
        and give a smoothing (non-exact) interpolator.
    """

    name = "idw"

    def __init__(self, power: float = 2.0, k: Optional[int] = 12, radius: Optional[float] = None,
                 smoothing: float = 0.0):
        if power <= 0:
            raise ValueError("power must be positive")
        self.power, self.k, self.radius, self.smoothing = power, k, radius, smoothing

    def _fit(self, xy, v):
        self._tree = cKDTree(xy)

    def _predict(self, xy):
        n = len(self.values_)
        k = n if self.k is None else min(self.k, n)
        dist, idx = self._tree.query(xy, k=k, distance_upper_bound=self.radius or np.inf)
        dist = dist.reshape(len(xy), -1)
        idx = idx.reshape(len(xy), -1)
        valid = np.isfinite(dist) & (idx < n)
        d = np.where(valid, dist, np.inf) + self.smoothing
        vals = np.where(valid, self.values_[np.minimum(idx, n - 1)], 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):     # d == 0 at data points is fixed up below
            w = np.where(valid, d ** -self.power, 0.0)
            wsum = w.sum(axis=1)
            out = np.where(valid.any(axis=1), (w * vals).sum(axis=1) / np.where(wsum == 0, 1.0, wsum), np.nan)
        exact = valid & (d == 0)                       # a prediction site coincides with a datum
        rows = np.flatnonzero(exact.any(axis=1))
        if len(rows):
            out[rows] = vals[rows, exact[rows].argmax(axis=1)]
        return out


class TIN(Interpolator):
    """
    Piecewise-linear interpolation on the Delaunay triangulation (a TIN surface).

    Exact and continuous but not smooth, and undefined outside the convex hull of the
    data: those cells are filled from the nearest observation (``fill='nearest'``) or
    left NaN (``fill='nan'``). This is the linear cousin of natural-neighbour
    interpolation (which additionally smooths across triangles).
    """

    name = "tin"

    def __init__(self, fill: str = "nearest"):
        if fill not in ("nearest", "nan"):
            raise ValueError("fill must be 'nearest' or 'nan'")
        self.fill = fill

    def _fit(self, xy, v):
        self._lin = LinearNDInterpolator(xy, v)
        self._tree = cKDTree(xy)

    def _predict(self, xy):
        out = np.asarray(self._lin(xy), dtype=float)
        if self.fill == "nearest":
            miss = ~np.isfinite(out)
            if miss.any():
                out[miss] = self.values_[self._tree.query(xy[miss])[1]]
        return out


class RBF(Interpolator):
    """
    Radial-basis-function interpolation; ``kernel='thin_plate_spline'`` gives a smooth
    thin-plate surface.

    Parameters
    ----------
    kernel : str
        Any kernel of :class:`scipy.interpolate.RBFInterpolator` (``'thin_plate_spline'``,
        ``'cubic'``, ``'linear'``, ``'gaussian'``, ...).
    smoothing : float
        0 = exact interpolation; > 0 smooths (recommended for noisy data).
    neighbors : int or None
        Fit each prediction from the nearest ``neighbors`` observations (needed for large n).
    epsilon : float
        Shape parameter for kernels that need it (e.g. gaussian).
    """

    name = "rbf"

    def __init__(self, kernel: str = "thin_plate_spline", smoothing: float = 0.0,
                 neighbors: Optional[int] = None, epsilon: float = 1.0):
        self.kernel, self.smoothing, self.neighbors, self.epsilon = kernel, smoothing, neighbors, epsilon

    def _fit(self, xy, v):
        self._scale_shift = (xy.mean(axis=0), xy.std(axis=0).mean() or 1.0)
        c, s = self._scale_shift
        self._rbf = RBFInterpolator((xy - c) / s, v, kernel=self.kernel, smoothing=self.smoothing,
                                    neighbors=self.neighbors, epsilon=self.epsilon)

    def _predict(self, xy):
        c, s = self._scale_shift
        return self._rbf((xy - c) / s)


class ThinPlateSpline(RBF):
    """
    Thin-plate spline: the smooth surface of minimum bending energy through the samples.

    An exact interpolator when ``smoothing`` is 0. Unlike IDW it is not bounded by
    the data range, so it can overshoot (including to impossible values) in sparse
    areas. ``smoothing > 0`` relaxes the exact fit, which is appropriate for noisy
    observations. ``neighbors`` fits each prediction from a local subset (needed
    for large samples).

    This is :class:`RBF` with ``kernel='thin_plate_spline'``.
    """

    name = "thin_plate_spline"

    def __init__(self, smoothing: float = 0.0, neighbors: Optional[int] = None):
        super().__init__(kernel="thin_plate_spline", smoothing=smoothing, neighbors=neighbors)


def Spline(smoothing: float = 0.0, neighbors: Optional[int] = None) -> ThinPlateSpline:
    """Thin-plate spline (an :class:`RBF` with the thin-plate kernel)."""
    return ThinPlateSpline(smoothing=smoothing, neighbors=neighbors)
