"""
spatialstats.viz.moran
======================
Moran scatterplot helpers.
"""

from __future__ import annotations


def moran_scatter(result=None, y=None, w=None, ax=None, **kwargs):
    """
    Moran scatterplot of standardized y vs spatial lag.

    Parameters
    ----------
    result : MoranResult, optional
        If provided, uses ``result.y`` and ``result.w``.
    y, w : optional
        Attribute and weights if ``result`` is not given.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "moran_scatter requires matplotlib. Install with: pip install pyspatialstats[viz]"
        ) from e

    if result is not None:
        y = getattr(result, "y", y)
        w = getattr(result, "w", w)
    if y is None or w is None:
        raise ValueError("Provide a MoranResult or both y and w")

    import numpy as np
    y = np.asarray(y, dtype=float).ravel()
    z = y - y.mean()
    lag = w.lag(z)

    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(z, lag, **kwargs)
    # Fit line
    if np.std(z) > 0:
        coef = np.polyfit(z, lag, 1)
        xs = np.linspace(z.min(), z.max(), 50)
        ax.plot(xs, np.polyval(coef, xs), color="C1", lw=1.5)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("z (standardized attribute)")
    ax.set_ylabel("Wz (spatial lag)")
    title = "Moran Scatterplot"
    if result is not None and hasattr(result, "I"):
        title = f"Moran Scatterplot (I={result.I:.3f})"
    ax.set_title(title)
    return ax
