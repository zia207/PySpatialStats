"""
spatialstats.sampling.designs
=============================
Sampling designs for a finite frame of spatial units: simple random, systematic,
stratified, two-stage cluster, spatially balanced (GRTS-style), space-filling
and conditioned Latin hypercube.
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from spatialstats.sampling.sample import SpatialSample, as_coords, as_rng


def _check_n(n: int, N: int):
    if not (1 <= n <= N):
        raise ValueError(f"sample size n={n} must be between 1 and the frame size N={N}")


# ---------------------------------------------------------------------------
# Simple random and systematic
# ---------------------------------------------------------------------------

def simple_random(frame, n: int, seed=None) -> SpatialSample:
    """Simple random sample without replacement: ``π_i = n / N``."""
    xy = as_coords(frame)
    N = len(xy)
    _check_n(n, N)
    idx = np.sort(as_rng(seed).choice(N, n, replace=False))
    return SpatialSample("simple random", idx, np.full(n, n / N), xy, np.full(N, n / N))


def _hilbert_index(x: np.ndarray, y: np.ndarray, order: int = 16) -> np.ndarray:
    """Hilbert-curve position of integer grid cells (vectorised)."""
    x, y = x.astype(np.int64).copy(), y.astype(np.int64).copy()
    d = np.zeros_like(x)
    s = 1 << (order - 1)
    while s > 0:
        rx = ((x & s) > 0).astype(np.int64)
        ry = ((y & s) > 0).astype(np.int64)
        d += s * s * ((3 * rx) ^ ry)
        swap = ry == 0
        flip = swap & (rx == 1)
        x = np.where(flip, s - 1 - x, x)
        y = np.where(flip, s - 1 - y, y)
        x, y = np.where(swap, y, x), np.where(swap, x, y)
        s >>= 1
    return d


def spatial_order(xy: np.ndarray, kind: str = "hilbert") -> np.ndarray:
    """Permutation that orders units along a space-filling path ('hilbert') or an axis ('x', 'y')."""
    if kind == "x":
        return np.lexsort((xy[:, 1], xy[:, 0]))
    if kind == "y":
        return np.lexsort((xy[:, 0], xy[:, 1]))
    if kind == "hilbert":
        lo, hi = xy.min(axis=0), xy.max(axis=0)
        span = np.maximum(hi - lo, 1e-12)
        g = np.floor((xy - lo) / span * (2 ** 16 - 1)).astype(np.int64)
        return np.argsort(_hilbert_index(g[:, 0], g[:, 1]), kind="stable")
    raise ValueError("order must be 'hilbert', 'x' or 'y'")


def systematic(frame, n: int, order: str = "hilbert", seed=None) -> SpatialSample:
    """
    Systematic sample from a frame ordered along a path: pick a random start and
    then every ``k = N/n``-th unit. Ordering by a Hilbert curve (default) makes
    the sample spread evenly over the study area; ordering by ``'x'`` gives
    parallel-transect-like sampling.

    Inclusion probabilities are ``n / N`` (exact when ``n`` divides ``N``).
    """
    xy = as_coords(frame)
    N = len(xy)
    _check_n(n, N)
    ordr = spatial_order(xy, order)
    u = as_rng(seed).uniform()
    pos = np.floor((np.arange(n) + u) * N / n).astype(int)
    idx = np.sort(ordr[pos])
    return SpatialSample(f"systematic ({order})", idx, np.full(n, n / N), xy, np.full(N, n / N))


def grid_points(region, n: Optional[int] = None, spacing: Optional[float] = None, seed=None,
                hexagonal: bool = False) -> np.ndarray:
    """
    Systematic grid of locations in a continuous region (random origin).

    Parameters
    ----------
    region : shapely (Multi)Polygon, (xmin, ymin, xmax, ymax), or GeoDataFrame
    n : int, optional
        Approximate number of points (sets the spacing from the region's area).
    spacing : float, optional
        Grid spacing. Give ``n`` or ``spacing``.
    hexagonal : bool
        Offset alternate rows by half a spacing (triangular lattice).

    Returns
    -------
    ndarray (m, 2) of locations inside the region.
    """
    import shapely
    from shapely.geometry import box
    if hasattr(region, "geometry"):
        poly = region.geometry.union_all() if hasattr(region.geometry, "union_all") \
            else region.geometry.unary_union
    elif isinstance(region, (tuple, list, np.ndarray)) and len(region) == 4:
        poly = box(*region)
    else:
        poly = region
    if (n is None) == (spacing is None):
        raise ValueError("give exactly one of n or spacing")
    rng = as_rng(seed)
    xmin, ymin, xmax, ymax = poly.bounds
    if spacing is None:
        spacing = np.sqrt(poly.area / n) * (1.0 if not hexagonal else np.sqrt(2 / np.sqrt(3)))
    dy = spacing * (np.sqrt(3) / 2 if hexagonal else 1.0)
    x0 = xmin - rng.uniform(0, spacing)
    y0 = ymin - rng.uniform(0, dy)
    xs = np.arange(x0, xmax + spacing, spacing)
    ys = np.arange(y0, ymax + dy, dy)
    gx, gy = np.meshgrid(xs, ys)
    if hexagonal:
        gx = gx + (np.arange(len(ys)) % 2)[:, None] * spacing / 2
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    return pts[shapely.contains_xy(poly, pts[:, 0], pts[:, 1])]


# ---------------------------------------------------------------------------
# Stratified and two-stage
# ---------------------------------------------------------------------------

def neyman_allocation(stratum_sizes, stratum_sd, n: int, min_per_stratum: int = 2) -> np.ndarray:
    """
    Optimal (Neyman) allocation ``n_h ∝ N_h σ_h`` with a floor and largest-remainder
    rounding. Strata are capped at their size.
    """
    Nh = np.asarray(stratum_sizes, dtype=float)
    sd = np.asarray(stratum_sd, dtype=float)
    return _allocate(Nh * sd, Nh, n, min_per_stratum)


def _allocate(score: np.ndarray, Nh: np.ndarray, n: int, min_per: int) -> np.ndarray:
    """
    Split ``n`` units over strata in proportion to ``score``, honouring a per-stratum floor
    (``min_per``) and cap (``N_h``). Strata whose proportional share falls outside [floor, cap]
    are pinned to the bound and the rest is re-split; the remainder is rounded by largest fraction.
    """
    H = len(Nh)
    floor = np.minimum(min_per, Nh).astype(float)
    if floor.sum() > n:
        raise ValueError(f"n={n} is too small for {H} strata with at least {min_per} units each")
    if n > Nh.sum():
        raise ValueError("n exceeds the frame size")
    score = np.where(score > 0, score, 1e-12).astype(float)
    pinned = np.zeros(H, dtype=bool)
    alloc = np.zeros(H)
    for _ in range(H + 1):
        free = ~pinned
        remaining = n - alloc[pinned].sum()
        ideal = np.where(free, remaining * score / score[free].sum(), alloc)
        low = free & (ideal < floor)
        high = free & (ideal > Nh)
        if not (low.any() or high.any()):
            break
        alloc[low], alloc[high] = floor[low], Nh[high]
        pinned |= low | high
    alloc = np.where(pinned, alloc, ideal)
    out = np.floor(alloc).astype(int)
    left = int(n - out.sum())
    if left > 0:
        frac = np.where(~pinned & (out < Nh), alloc - np.floor(alloc), -1.0)
        for j in np.argsort(-frac)[:left]:
            out[j] += 1
    return out


def stratified(frame, strata, n: Union[int, Dict], allocation: str = "proportional",
               sd: Optional[Dict] = None, min_per_stratum: int = 2, within: str = "srs",
               seed=None) -> SpatialSample:
    """
    Stratified random sample.

    Parameters
    ----------
    strata : array-like (N,)
        Stratum label of every frame unit.
    n : int or dict
        Total sample size, or ``{stratum: n_h}`` for a user-specified allocation.
    allocation : {'proportional', 'equal', 'neyman'}
        Used when ``n`` is an int. ``'neyman'`` needs ``sd={stratum: σ_h}`` (e.g. from
        a pilot survey or a proxy variable).
    within : {'srs', 'grts', 'systematic'}
        How to select units inside each stratum. ``'grts'`` gives spatially balanced
        samples within strata.
    """
    xy = as_coords(frame)
    N = len(xy)
    lab = np.asarray(strata)
    if len(lab) != N:
        raise ValueError("strata must have one label per frame unit")
    levels = pd.unique(lab)
    Nh = np.array([(lab == h).sum() for h in levels], dtype=float)
    if isinstance(n, dict):
        nh = np.array([int(n[h]) for h in levels])
    else:
        _check_n(int(n), N)
        if allocation == "proportional":
            nh = _allocate(Nh, Nh, int(n), min_per_stratum)
        elif allocation == "equal":
            nh = _allocate(np.ones_like(Nh), Nh, int(n), min_per_stratum)
        elif allocation == "neyman":
            if sd is None:
                raise ValueError("allocation='neyman' needs sd={stratum: s.d.}")
            nh = neyman_allocation(Nh, [sd[h] for h in levels], int(n), min_per_stratum)
        else:
            raise ValueError("allocation must be 'proportional', 'equal' or 'neyman'")
    rng = as_rng(seed)
    idx, pi, st = [], [], []
    frame_pi = np.zeros(N)
    for h, Nh_, nh_ in zip(levels, Nh, nh):
        members = np.where(lab == h)[0]
        if nh_ > len(members):
            raise ValueError(f"stratum {h!r}: n_h={nh_} exceeds N_h={len(members)}")
        if nh_ == 0:
            continue
        if within == "srs":
            pick = members[rng.choice(len(members), nh_, replace=False)]
        elif within == "grts":
            pick = members[grts(xy[members], nh_, seed=rng).indices]
        elif within == "systematic":
            pick = members[systematic(xy[members], nh_, seed=rng).indices]
        else:
            raise ValueError("within must be 'srs', 'grts' or 'systematic'")
        idx.extend(pick)
        pi.extend([nh_ / Nh_] * len(pick))
        st.extend([h] * len(pick))
        frame_pi[members] = nh_ / Nh_
    idx = np.asarray(idx)
    order = np.argsort(idx)
    return SpatialSample("stratified", idx[order], np.asarray(pi)[order], xy, frame_pi,
                         strata=np.asarray(st)[order], frame_strata=lab,
                         allocation=allocation if not isinstance(n, dict) else "user")


def two_stage(frame, clusters, n_clusters: int, n_per_cluster: int, seed=None) -> SpatialSample:
    """
    Two-stage cluster sample: ``n_clusters`` primary units (e.g. counties) drawn by SRS,
    then ``n_per_cluster`` units by SRS inside each (all units if the cluster is smaller).

    ``π_i = (m/M) · (k_j / N_j)`` for a unit in cluster *j* with ``N_j`` members and ``k_j = min(n_per_cluster, N_j)``.
    """
    xy = as_coords(frame)
    N = len(xy)
    cl = np.asarray(clusters)
    if len(cl) != N:
        raise ValueError("clusters must have one label per frame unit")
    labels = pd.unique(cl)
    M = len(labels)
    if not (1 <= n_clusters <= M):
        raise ValueError(f"n_clusters must be between 1 and the number of clusters ({M})")
    rng = as_rng(seed)
    chosen = labels[rng.choice(M, n_clusters, replace=False)]
    idx, pi, cid = [], [], []
    frame_pi = np.zeros(N)
    for c in labels:
        members = np.where(cl == c)[0]
        k = min(n_per_cluster, len(members))
        p = (n_clusters / M) * (k / len(members))
        frame_pi[members] = p
        if c in chosen:
            pick = members[rng.choice(len(members), k, replace=False)]
            idx.extend(pick)
            pi.extend([p] * k)
            cid.extend([c] * k)
    idx = np.asarray(idx)
    order = np.argsort(idx)
    return SpatialSample("two-stage cluster", idx[order], np.asarray(pi)[order], xy, frame_pi,
                         clusters=np.asarray(cid)[order], n_clusters=n_clusters)


# ---------------------------------------------------------------------------
# Spatially balanced sampling (GRTS-style)
# ---------------------------------------------------------------------------

def _inclusion_probabilities(size: np.ndarray, n: int) -> np.ndarray:
    """``π_i = n · size_i / Σ size`` iteratively capped at 1 (certainty units)."""
    pi = np.zeros(len(size))
    free = np.ones(len(size), dtype=bool)
    remaining = float(n)
    while True:
        cand = remaining * size / size[free].sum()
        over = free & (cand >= 1.0)
        pi[free] = cand[free]
        if not over.any():
            break
        pi[over] = 1.0
        remaining -= over.sum()
        free &= ~over
        if remaining <= 0 or not free.any():
            break
    return np.minimum(pi, 1.0)


def _hierarchical_address(xy: np.ndarray, rng: np.random.Generator, max_depth: int = 24):
    """
    Random hierarchical address of each point in a quadtree of the bounding box.

    At every node the four quadrants are labelled with a fresh random permutation of
    0..3, so the ordering of units along the address is a *random* space-filling curve:
    nearby units get nearby positions, but the curve differs for every draw
    (the idea behind Generalized Random Tessellation Stratified sampling).
    """
    n = len(xy)
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    span = float((hi - lo).max()) or 1.0
    unit = (xy - lo) / span
    digits = np.zeros((n, max_depth), dtype=np.int8)
    node = np.zeros(n, dtype=np.int64)          # cell id at the current level
    for level in range(max_depth):
        bit_x = (np.floor(unit[:, 0] * 2 ** (level + 1)).astype(np.int64)) & 1
        bit_y = (np.floor(unit[:, 1] * 2 ** (level + 1)).astype(np.int64)) & 1
        quad = bit_x * 2 + bit_y
        # a fresh random labelling of the four quadrants for every parent cell
        uniq, inv = np.unique(node, return_inverse=True)
        perms = np.argsort(rng.random((len(uniq), 4)), axis=1)
        digits[:, level] = perms[inv, quad]
        node = node * 4 + quad
        _, node = np.unique(node, return_inverse=True)
        if len(np.unique(node)) == n:
            digits = digits[:, :level + 1]
            break
    return digits


def grts(frame, n: int, size=None, seed=None) -> SpatialSample:
    """
    Spatially balanced sample by random-hierarchical-address ordering and systematic
    PPS selection (the core of GRTS; Stevens & Olsen 2004).

    Units are placed on a line in the order of their random quadtree address, then a
    systematic sample is taken through the cumulative inclusion probabilities. Because
    the order preserves spatial proximity, the sample is spread evenly over the frame,
    which lowers the variance of estimators of spatially structured variables relative
    to simple random sampling. Every unit still has exactly the intended inclusion
    probability, so design-based inference remains valid.

    Parameters
    ----------
    frame : (N, 2) array, DataFrame with x/y, or GeoDataFrame (polygons → centroids)
    n : int
        Sample size.
    size : array (N,), optional
        Auxiliary size measure for unequal-probability sampling (``π ∝ size``,
        e.g. population). Default: equal probabilities.
    """
    xy = as_coords(frame)
    N = len(xy)
    _check_n(n, N)
    rng = as_rng(seed)
    sz = np.ones(N) if size is None else np.asarray(size, dtype=float)
    if np.any(sz <= 0):
        raise ValueError("size must be strictly positive")
    pi_all = _inclusion_probabilities(sz, n)

    digits = _hierarchical_address(xy, rng)
    tie = rng.uniform(size=N)                    # random order among units sharing a finest cell
    # np.lexsort treats the LAST key as primary: first digit primary, random tie-break last
    keys = (tie,) + tuple(digits[:, k] for k in range(digits.shape[1] - 1, -1, -1))
    order = np.lexsort(keys)
    cum = np.cumsum(pi_all[order])
    u = rng.uniform()
    marks = u + np.arange(n)
    pos = np.searchsorted(cum, marks, side="left")
    pos = np.minimum(pos, N - 1)
    idx = order[pos]
    if len(np.unique(idx)) != len(idx):                 # only with numerically extreme π
        idx = np.unique(idx)
    idx = np.sort(idx)
    return SpatialSample("GRTS (spatially balanced)", idx, pi_all[idx], xy, pi_all)


# ---------------------------------------------------------------------------
# Model-based designs: space filling and conditioned Latin hypercube
# ---------------------------------------------------------------------------

def space_filling(frame, n: int, method: str = "kmeans", seed=None) -> SpatialSample:
    """
    Purposive design that covers the study area evenly (for model-based inference,
    e.g. choosing monitoring sites for kriging).

    ``method='kmeans'`` places ``n`` centroids and takes the nearest frame unit to each;
    ``method='maximin'`` adds units greedily, each the farthest from those chosen. These
    designs have no defined inclusion probabilities, so design-based estimators do not
    apply (``pi`` is NaN).
    """
    xy = as_coords(frame)
    N = len(xy)
    _check_n(n, N)
    rng = as_rng(seed)
    if method == "kmeans":
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n, n_init=5, random_state=int(rng.integers(2 ** 31 - 1))).fit(xy)
        _, idx = cKDTree(xy).query(km.cluster_centers_)
        idx = np.unique(idx)
        if len(idx) < n:                            # two centroids snapped to one unit: top up
            rest = np.setdiff1d(np.arange(N), idx)
            d = cKDTree(xy[idx]).query(xy[rest])[0]
            idx = np.concatenate([idx, rest[np.argsort(-d)[:n - len(idx)]]])
    elif method == "maximin":
        idx = [int(rng.integers(N))]
        dist = np.linalg.norm(xy - xy[idx[0]], axis=1)
        for _ in range(n - 1):
            j = int(np.argmax(dist))
            idx.append(j)
            dist = np.minimum(dist, np.linalg.norm(xy - xy[j], axis=1))
        idx = np.asarray(idx)
    else:
        raise ValueError("method must be 'kmeans' or 'maximin'")
    idx = np.sort(idx)
    return SpatialSample(f"space-filling ({method})", idx, np.full(len(idx), np.nan), xy, None)


def clhs(covariates, n: int, iterations: int = 20000, weights=(1.0, 1.0), coords=None,
         seed=None) -> SpatialSample:
    """
    Conditioned Latin hypercube sampling (Minasny & McBratney 2006).

    Chooses ``n`` units whose covariate values fill every marginal quantile stratum
    once (objective O1) while reproducing the correlation among covariates (O2), by
    simulated annealing. It samples the *covariate space* rather than the geographic
    space, which is efficient for regression/model-based mapping but is not a
    probability sample (``pi`` is NaN).

    Parameters
    ----------
    covariates : array (N, p) or DataFrame
    n : int
    iterations : int
        Annealing steps.
    weights : (w1, w2)
        Weights of the quantile-stratum and correlation objectives.
    coords : array (N, 2), optional
        Locations, so the result can be plotted and mapped.
    """
    X = covariates.to_numpy(dtype=float) if hasattr(covariates, "to_numpy") else np.asarray(covariates, float)
    if X.ndim == 1:
        X = X[:, None]
    N, p = X.shape
    _check_n(n, N)
    rng = as_rng(seed)
    xy = np.zeros((N, 2)) if coords is None else as_coords(coords)
    # quantile stratum of every unit for each covariate
    edges = np.quantile(X, np.linspace(0, 1, n + 1)[1:-1], axis=0)        # (n-1, p)
    stratum = np.stack([np.searchsorted(edges[:, j], X[:, j], side="right") for j in range(p)], axis=1)
    R = np.corrcoef(X, rowvar=False) if p > 1 else np.ones((1, 1))
    R = np.nan_to_num(R)

    def objective(sel):
        o1 = 0.0
        for j in range(p):
            o1 += np.abs(np.bincount(stratum[sel, j], minlength=n) - 1).sum()
        if p > 1:
            Rs = np.nan_to_num(np.corrcoef(X[sel], rowvar=False))
            o2 = np.abs(R - Rs).sum()
        else:
            o2 = 0.0
        return weights[0] * o1 + weights[1] * o2

    sel = rng.choice(N, n, replace=False)
    in_sample = np.zeros(N, dtype=bool)
    in_sample[sel] = True
    cur = objective(sel)
    best, best_sel = cur, sel.copy()
    temp, decay = 1.0, 0.9
    for it in range(iterations):
        k = int(rng.integers(n))
        cand = int(rng.choice(np.flatnonzero(~in_sample))) if not in_sample.all() else sel[k]
        new = sel.copy()
        new[k] = cand
        val = objective(new)
        if val < cur or rng.uniform() < np.exp(-(val - cur) / max(temp, 1e-12)):
            in_sample[sel[k]] = False
            in_sample[cand] = True
            sel, cur = new, val
            if cur < best:
                best, best_sel = cur, sel.copy()
        if (it + 1) % 100 == 0:
            temp *= decay
        if best == 0:
            break
    idx = np.sort(best_sel)
    s = SpatialSample("cLHS", idx, np.full(len(idx), np.nan), xy, None, objective=float(best))
    s.covariates = X[idx]
    return s
