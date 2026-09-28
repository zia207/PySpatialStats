"""
spatialstats.viz.choropleth
===========================
Choropleth helpers with simple classification schemes.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def _classify(values: np.ndarray, scheme: str = "quantiles", k: int = 5) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    scheme = scheme.lower()
    if scheme in ("quantiles", "quantile", "q"):
        try:
            import mapclassify
            classifier = mapclassify.Quantiles(values, k=k)
            return classifier.yb
        except ImportError:
            qs = np.linspace(0, 1, k + 1)
            bins = np.unique(np.quantile(values[np.isfinite(values)], qs))
            return np.digitize(values, bins[1:-1], right=True)
    if scheme in ("equal_interval", "equal", "ei"):
        try:
            import mapclassify
            classifier = mapclassify.EqualInterval(values, k=k)
            return classifier.yb
        except ImportError:
            vmin, vmax = np.nanmin(values), np.nanmax(values)
            bins = np.linspace(vmin, vmax, k + 1)
            return np.digitize(values, bins[1:-1], right=True)
    if scheme in ("natural_breaks", "fisherjenks", "jenks", "nb"):
        try:
            import mapclassify
            classifier = mapclassify.FisherJenks(values, k=k)
            return classifier.yb
        except ImportError:
            # Fallback: quantiles
            qs = np.linspace(0, 1, k + 1)
            bins = np.unique(np.quantile(values[np.isfinite(values)], qs))
            return np.digitize(values, bins[1:-1], right=True)
    raise ValueError(f"Unknown scheme '{scheme}'")


def choropleth(
    gdf,
    column: str,
    scheme: str = "quantiles",
    k: int = 5,
    ax=None,
    cmap: str = "viridis",
    **kwargs,
):
    """
    Plot a classified choropleth.

    Parameters
    ----------
    gdf : GeoDataFrame
    column : str
        Attribute column.
    scheme : {'quantiles', 'equal_interval', 'natural_breaks'}
    k : int
        Number of classes.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "choropleth requires matplotlib. Install with: pip install pyspatialstats[viz]"
        ) from e

    values = gdf[column].to_numpy(dtype=float)
    classes = _classify(values, scheme=scheme, k=k)
    plot_gdf = gdf.copy()
    plot_gdf["_class"] = classes

    if ax is None:
        _, ax = plt.subplots(figsize=kwargs.pop("figsize", (7, 6)))

    plot_gdf.plot(
        column="_class",
        cmap=cmap,
        ax=ax,
        edgecolor=kwargs.pop("edgecolor", "k"),
        linewidth=kwargs.pop("linewidth", 0.2),
        legend=kwargs.pop("legend", True),
        **kwargs,
    )
    ax.set_axis_off()
    ax.set_title(f"{column} ({scheme}, k={k})")
    return ax
