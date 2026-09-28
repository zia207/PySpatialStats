"""
spatialstats.viz.lisa
=====================
LISA cluster map helpers.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


_LISA_COLORS = {
    "HH": "#d7191c",
    "LL": "#2c7bb6",
    "LH": "#abd9e9",
    "HL": "#fdae61",
    "NS": "#eeeeee",
}


def lisa_cluster_map(result, gdf=None, ax=None, column: Optional[str] = None, **kwargs):
    """
    Plot LISA cluster labels on a GeoDataFrame.

    Parameters
    ----------
    result : LocalMoranResult
    gdf : GeoDataFrame, optional
        Required for mapping. If None, raises.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError as e:
        raise ImportError(
            "lisa_cluster_map requires matplotlib. Install with: pip install pyspatialstats[viz]"
        ) from e

    if gdf is None:
        raise ValueError("lisa_cluster_map requires a GeoDataFrame (gdf=...)")

    labels = result.labels if result.labels is not None else result._labels_from_q()
    plot_gdf = gdf.copy()
    col = column or "_lisa_label"
    plot_gdf[col] = labels

    if ax is None:
        _, ax = plt.subplots(figsize=kwargs.pop("figsize", (7, 6)))

    for lab, color in _LISA_COLORS.items():
        subset = plot_gdf[plot_gdf[col] == lab]
        if len(subset):
            subset.plot(ax=ax, color=color, edgecolor="k", linewidth=0.2)

    handles = [Patch(facecolor=c, edgecolor="k", label=lab) for lab, c in _LISA_COLORS.items()]
    ax.legend(handles=handles, title="LISA", loc="best")
    ax.set_axis_off()
    ax.set_title(kwargs.pop("title", "LISA Cluster Map"))
    return ax
