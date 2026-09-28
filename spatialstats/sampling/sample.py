"""
spatialstats.sampling.sample
============================
The :class:`SpatialSample` container returned by every design.
"""

from __future__ import annotations


import numpy as np
import pandas as pd

from spatialstats.core.result import SpatialResult


def as_coords(frame) -> np.ndarray:
    """Coordinates (N, 2) of a sampling frame: ndarray, DataFrame with x/y, or GeoDataFrame."""
    if hasattr(frame, "geometry"):
        from spatialstats.core.distance import coords_from_geometry
        geom = frame.geometry
        if not (geom.geom_type == "Point").all():
            geom = geom.centroid
        return coords_from_geometry(geom)
    if isinstance(frame, pd.DataFrame):
        cols = {c.lower(): c for c in frame.columns}
        if "x" in cols and "y" in cols:
            return frame[[cols["x"], cols["y"]]].to_numpy(dtype=float)
        raise ValueError("DataFrame frame needs 'x' and 'y' columns")
    c = np.asarray(frame, dtype=float)
    if c.ndim != 2 or c.shape[1] != 2:
        raise ValueError(f"frame must have shape (N, 2), got {c.shape}")
    return c


def as_rng(seed):
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


class SpatialSample(SpatialResult):
    """
    A sample drawn from a finite sampling frame.

    Attributes
    ----------
    indices : ndarray (n,)      positions of the selected units in the frame
    pi : ndarray (n,)           first-order inclusion probability of each selected unit
    weights : ndarray (n,)      design weights ``1 / pi``
    frame_coords : ndarray (N, 2)
    frame_pi : ndarray (N,)     inclusion probability of *every* frame unit (for balance measures)
    strata : ndarray (n,) or None   stratum of each selected unit
    design : str
    """

    def __init__(self, design: str, indices, pi, frame_coords, frame_pi=None, strata=None,
                 frame_strata=None, clusters=None, **stats_):
        idx = np.asarray(indices, dtype=int)
        super().__init__(name=f"{design} sample", n=len(idx), N=len(frame_coords), **stats_)
        self.design = design
        self.indices = idx
        self.pi = np.asarray(pi, dtype=float)
        self.frame_coords = np.asarray(frame_coords, dtype=float)
        self.N = len(self.frame_coords)
        self.frame_pi = None if frame_pi is None else np.asarray(frame_pi, dtype=float)
        self.strata = None if strata is None else np.asarray(strata)
        self.frame_strata = None if frame_strata is None else np.asarray(frame_strata)
        self.clusters = None if clusters is None else np.asarray(clusters)

    @property
    def n(self) -> int:
        return len(self.indices)

    @property
    def weights(self) -> np.ndarray:
        with np.errstate(divide="ignore"):
            return 1.0 / self.pi

    @property
    def coords(self) -> np.ndarray:
        return self.frame_coords[self.indices]

    def __len__(self) -> int:
        return self.n

    def take(self, data):
        """Rows of a frame-aligned DataFrame / array / GeoDataFrame that were sampled."""
        if hasattr(data, "iloc"):
            return data.iloc[self.indices]
        return np.asarray(data)[self.indices]

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame({"frame_index": self.indices, "x": self.coords[:, 0],
                           "y": self.coords[:, 1], "pi": self.pi, "weight": self.weights})
        if self.strata is not None:
            df["stratum"] = self.strata
        return df

    def plot(self, ax=None, **kwargs):
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=kwargs.pop("figsize", (6, 6)))
        ax.scatter(self.frame_coords[:, 0], self.frame_coords[:, 1], s=4, c="0.8", label="frame")
        kwargs.setdefault("s", 25)
        kwargs.setdefault("c", "C3")
        ax.scatter(self.coords[:, 0], self.coords[:, 1], label=f"sample (n={self.n})", **kwargs)
        ax.set_aspect("equal")
        ax.legend()
        ax.set_title(f"{self.design} sample")
        return ax
