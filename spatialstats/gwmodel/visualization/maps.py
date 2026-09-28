"""
spatialstats.gwmodel.visualization.maps
==============================
Map-based visualisations for local coefficients, importances, and diagnostics.
"""

import numpy as np
import warnings
from typing import Optional, List, Union


def plot_local_coefficients(model, gdf, covariate=None, significance=True,
                             alpha=0.05, cmap="RdBu_r", figsize=(14, 5),
                             save_path=None):
    """
    Choropleth maps of local coefficient surfaces.

    Parameters
    ----------
    model       : fitted GWR or MGWR model
    gdf         : GeoDataFrame with geometry
    covariate   : int or None  Column index (None = all).
    significance: bool  Mask non-significant estimates with hatch.
    alpha       : float Significance level.
    cmap        : str   Colourmap.
    figsize     : tuple
    save_path   : str or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        raise ImportError("matplotlib required: pip install matplotlib")

    if not hasattr(model, "coef_") or model.coef_ is None:
        raise RuntimeError("Model not fitted.")

    coefs = model.coef_
    n, p = coefs.shape
    names = ["Intercept"] + [f"X{i}" for i in range(1, p)]

    if covariate is not None:
        cols_to_plot = [covariate]
    else:
        cols_to_plot = list(range(p))

    ncols = min(3, len(cols_to_plot))
    nrows = int(np.ceil(len(cols_to_plot) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if not hasattr(axes, "__len__"):
        axes = np.array([[axes]])
    axes = np.array(axes).ravel()

    for ax_idx, j in enumerate(cols_to_plot):
        ax = axes[ax_idx]
        col_vals = coefs[:, j]
        gdf_plot = gdf.copy()
        gdf_plot["_coef"] = col_vals

        gdf_plot.plot(column="_coef", ax=ax, cmap=cmap, legend=True,
                      vmin=np.percentile(col_vals, 2.5),
                      vmax=np.percentile(col_vals, 97.5))

        if significance and hasattr(model, "p_values_") and model.p_values_ is not None:
            pv = model.p_values_[:, j]
            non_sig = gdf_plot[pv >= alpha]
            if len(non_sig) > 0:
                non_sig.plot(ax=ax, color="none", edgecolor="grey",
                             hatch="///", linewidth=0.5)

        ax.set_title(f"Local: {names[j]}", fontsize=11)
        ax.axis("off")

    for ax in axes[len(cols_to_plot):]:
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_local_importance(model, gdf, feature=None, importance_type="gain",
                           cmap="YlOrRd", figsize=(12, 5)):
    """
    Choropleth maps of spatially varying feature importances (for ensemble models).

    Parameters
    ----------
    model          : fitted GWRandomForest or GWXGBoost
    gdf            : GeoDataFrame
    feature        : int or None  Feature index (None = all).
    importance_type: str  'gain' | 'shap' (shap requires shap package)
    cmap           : str
    figsize        : tuple
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required.")

    if not hasattr(model, "feature_importances_local_"):
        raise RuntimeError("Model has no local feature importances.")

    imp = model.feature_importances_local_  # (n, p)
    n, p = imp.shape
    cols = list(range(p)) if feature is None else [feature]

    ncols = min(3, len(cols))
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if not hasattr(axes, "__len__"):
        axes = np.array([[axes]])
    axes = np.array(axes).ravel()

    for ax_idx, j in enumerate(cols):
        ax = axes[ax_idx]
        gdf_plot = gdf.copy()
        gdf_plot["_imp"] = imp[:, j]
        gdf_plot.plot(column="_imp", ax=ax, cmap=cmap, legend=True)
        ax.set_title(f"Importance: Feature {j}", fontsize=11)
        ax.axis("off")

    for ax in axes[len(cols):]:
        ax.axis("off")

    plt.tight_layout()
    return fig


def plot_bandwidth_search(selector, ax=None):
    """
    Plot AICc (or other criterion) vs bandwidth from a BandwidthSelector trace.

    Parameters
    ----------
    selector : BandwidthSelector (after calling .select())
    ax       : matplotlib.axes.Axes or None
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required.")

    if not selector.bw_history_:
        raise RuntimeError("Run selector.select() first.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.get_figure()

    bws = selector.bw_history_
    scores = selector.score_history_
    order = np.argsort(bws)
    ax.plot(np.array(bws)[order], np.array(scores)[order], "o-", color="steelblue")
    best_idx = np.argmin(scores)
    ax.axvline(bws[best_idx], color="crimson", linestyle="--",
                label=f"Optimal BW = {bws[best_idx]:.2f}")
    ax.set_xlabel("Bandwidth", fontsize=12)
    ax.set_ylabel(selector.criterion, fontsize=12)
    ax.set_title("Bandwidth Selection", fontsize=13)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_residuals(model, gdf, cmap="coolwarm", figsize=(8, 6)):
    """Map GWR residuals."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required.")
    if not hasattr(model, "residuals_") or model.residuals_ is None:
        raise RuntimeError("Model not fitted.")
    fig, ax = plt.subplots(figsize=figsize)
    gdf_plot = gdf.copy()
    gdf_plot["_resid"] = model.residuals_
    lim = np.abs(model.residuals_).max()
    gdf_plot.plot(column="_resid", ax=ax, cmap=cmap, legend=True,
                  vmin=-lim, vmax=lim)
    ax.set_title("GWR Residuals", fontsize=13)
    ax.axis("off")
    plt.tight_layout()
    return fig
