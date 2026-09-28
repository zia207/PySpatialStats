"""
spatialstats.pointpattern.simulate
==================================
Point-process simulators and Monte Carlo envelopes.
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np

from spatialstats.core.result import SpatialResult
from spatialstats.pointpattern.pattern import PointPattern
from spatialstats.pointpattern.window import Window
from spatialstats.pointpattern import functions as _f


def _rng(seed):
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


def simulate_csr(n: Optional[int] = None, window: Optional[Window] = None,
                 intensity: Optional[float] = None, seed=None) -> PointPattern:
    """
    Complete spatial randomness in ``window``.

    Give ``n`` for a *binomial* process (fixed count) or ``intensity`` for a
    *Poisson* process (Poisson-distributed count).
    """
    if window is None:
        window = Window.from_bounds(0, 0, 1, 1)
    rng = _rng(seed)
    if (n is None) == (intensity is None):
        raise ValueError("specify exactly one of n or intensity")
    if n is None:
        n = int(rng.poisson(intensity * window.area))
    return PointPattern(window.sample(n, rng), window=window)


def simulate_thomas(kappa: float, sigma: float, mu: float, window: Optional[Window] = None,
                    seed=None) -> PointPattern:
    """
    Thomas cluster process: Poisson parent points (intensity ``kappa``), each with
    Poisson(``mu``) offspring displaced by an isotropic Gaussian (s.d. ``sigma``).
    Offspring falling outside the window are discarded. Overall intensity ≈ ``kappa * mu``.
    """
    if window is None:
        window = Window.from_bounds(0, 0, 1, 1)
    rng = _rng(seed)
    xmin, ymin, xmax, ymax = window.bounds
    pad = 4 * sigma
    area = (xmax - xmin + 2 * pad) * (ymax - ymin + 2 * pad)
    n_par = rng.poisson(kappa * area)
    parents = np.column_stack([
        rng.uniform(xmin - pad, xmax + pad, n_par),
        rng.uniform(ymin - pad, ymax + pad, n_par),
    ])
    counts = rng.poisson(mu, n_par)
    pts = np.repeat(parents, counts, axis=0) + rng.normal(0, sigma, (counts.sum(), 2))
    pts = pts[window.contains(pts)] if len(pts) else np.empty((0, 2))
    return PointPattern(pts, window=window)


def simulate_matern_inhibition(intensity: float, radius: float, window: Optional[Window] = None,
                               seed=None) -> PointPattern:
    """
    Matérn type-II hard-core process: a Poisson pattern in which, of any two events
    closer than ``radius``, the later-marked one is deleted.
    """
    from scipy.spatial import cKDTree
    if window is None:
        window = Window.from_bounds(0, 0, 1, 1)
    rng = _rng(seed)
    base = simulate_csr(intensity=intensity, window=window, seed=rng)
    c = base.coords
    if len(c) < 2:
        return base
    t = rng.uniform(size=len(c))
    tree = cKDTree(c)
    keep = np.ones(len(c), dtype=bool)
    for i, j in tree.query_pairs(radius, output_type="ndarray"):
        keep[j if t[i] < t[j] else i] = False
    return PointPattern(c[keep], window=window)


# ---------------------------------------------------------------------------
# Envelopes
# ---------------------------------------------------------------------------

_FUNCS = {
    "K": _f.ripley_k, "L": _f.ripley_l, "g": _f.pair_correlation,
    "G": _f.g_function, "F": _f.f_function, "J": _f.j_function,
}


class EnvelopeResult(SpatialResult):
    """Observed summary function with simulation envelopes and global CSR tests."""

    def __init__(self, name, r, observed, sims, theoretical, lo, hi, alpha, **stats_):
        super().__init__(name=f"{name} envelope", **stats_)
        self.func_name = name
        self.r = r
        self.observed = observed
        self.sims = sims
        self.theoretical = theoretical
        self.lo = lo
        self.hi = hi
        self.alpha = alpha
        self.mean_sim = np.nanmean(sims, axis=0)

    def test(self, kind: str = "mad", rmin: Optional[float] = None, rmax: Optional[float] = None) -> float:
        """
        Global Monte Carlo p-value that avoids the multiple-testing problem of
        reading a pointwise envelope.

        ``kind='mad'`` : maximum absolute deviation from the mean simulated curve.
        ``kind='dclf'``: integrated squared deviation (Diggle–Cressie–Loosmore–Ford).
        Restrict the range of distances with ``rmin`` / ``rmax``.
        """
        sel = np.ones_like(self.r, dtype=bool)
        if rmin is not None:
            sel &= self.r >= rmin
        if rmax is not None:
            sel &= self.r <= rmax
        sel &= np.isfinite(self.observed) & np.all(np.isfinite(self.sims), axis=0)
        dr = np.gradient(self.r)[sel]

        def stat(curve):
            dev = curve[..., sel] - self.mean_sim[sel]
            if kind == "mad":
                return np.max(np.abs(dev), axis=-1)
            if kind == "dclf":
                return np.sum(dev ** 2 * dr, axis=-1)
            raise ValueError("kind must be 'mad' or 'dclf'")

        t_obs = float(stat(self.observed))
        t_sim = stat(self.sims)
        return float((1 + np.sum(t_sim >= t_obs)) / (1 + len(t_sim)))

    def plot(self, ax=None, detrend: Optional[bool] = None, **kwargs):
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots()
        if detrend is None:
            detrend = self.func_name in ("L",)
        off = self.r if detrend else 0.0
        ax.fill_between(self.r, self.lo - off, self.hi - off, color="0.8", label="CSR envelope")
        ax.plot(self.r, self.theoretical - off, "k--", lw=1, label="CSR")
        ax.plot(self.r, self.observed - off, color="C3", label="observed", **kwargs)
        ax.set_xlabel("r")
        ax.set_ylabel(f"{self.func_name}(r)" + (" - r" if detrend else ""))
        ax.legend()
        return ax


def envelope(pp: PointPattern, statistic: Union[str, Callable] = "L", nsim: int = 99,
             r: Optional[np.ndarray] = None, alpha: float = 0.05, seed=None,
             simulator: Optional[Callable[[np.random.Generator], PointPattern]] = None,
             **kwargs) -> EnvelopeResult:
    """
    Monte Carlo envelope for a summary function under a null model.

    By default the null is CSR conditional on the observed number of events
    (a binomial process in the same window). Provide ``simulator(rng) ->
    PointPattern`` to test other nulls (e.g. an inhomogeneous process).

    Parameters
    ----------
    statistic : {'K', 'L', 'g', 'G', 'F', 'J'} or callable
        A callable ``f(pp, r) -> ndarray`` is also accepted.
    nsim : int
        Number of simulations. The pointwise band is the ``alpha/2`` and
        ``1 - alpha/2`` quantiles; with ``nsim=99`` and ``alpha=0.02`` this is the
        min/max envelope. Use :meth:`EnvelopeResult.test` for a global p-value.
    **kwargs
        Passed to the summary function (e.g. ``correction='border'``).
    """
    rng = np.random.default_rng(seed)
    if callable(statistic):
        fn, name = statistic, getattr(statistic, "__name__", "f")
        obs_res = None
    else:
        if statistic not in _FUNCS:
            raise ValueError(f"statistic must be one of {list(_FUNCS)} or a callable")
        name = statistic
        fn = lambda p, rr: _FUNCS[name](p, rr, **kwargs)  # noqa: E731
        obs_res = fn(pp, r)

    if obs_res is not None:
        r_used = obs_res.r
        observed = obs_res.estimate
        theo = obs_res.theoretical
    else:
        if r is None:
            raise ValueError("a callable statistic requires r")
        r_used = np.asarray(r, dtype=float)
        observed = np.asarray(fn(pp, r_used), dtype=float)
        theo = np.full_like(r_used, np.nan)

    sims = np.empty((nsim, len(r_used)))
    for k in range(nsim):
        sp = simulator(rng) if simulator else simulate_csr(n=pp.n, window=pp.window, seed=rng)
        res = fn(sp, r_used)
        sims[k] = res.estimate if hasattr(res, "estimate") else np.asarray(res, dtype=float)

    lo = np.nanquantile(sims, alpha / 2, axis=0)
    hi = np.nanquantile(sims, 1 - alpha / 2, axis=0)
    out = EnvelopeResult(name, r_used, observed, sims, theo, lo, hi, alpha, nsim=nsim)
    out._stats["p_mad"] = out.test("mad")
    out._stats["p_dclf"] = out.test("dclf")
    return out
