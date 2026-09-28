"""
spatialstats.pointpattern.pattern
=================================
The :class:`PointPattern` container.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from spatialstats.pointpattern.window import Window


class PointPattern:
    """
    A set of event locations observed in a window, with optional marks.

    Parameters
    ----------
    coords : array-like (n, 2)
        Event coordinates in projected units.
    window : Window, optional
        Observation window. Defaults to the bounding box of ``coords``.
    marks : array-like (n,), optional
        Categorical or numeric mark attached to each event.
    clip : bool
        If True, silently drop events outside ``window``; otherwise raise.
    """

    def __init__(self, coords, window: Optional[Window] = None, marks=None, clip: bool = False):
        c = np.asarray(coords, dtype=float)
        if c.ndim != 2 or c.shape[1] != 2:
            raise ValueError(f"coords must have shape (n, 2), got {c.shape}")
        if not np.isfinite(c).all():
            raise ValueError("coords contain NaN or infinite values")
        m = None if marks is None else np.asarray(marks)
        if m is not None and len(m) != len(c):
            raise ValueError("marks must have one entry per event")

        if window is None:
            window = Window.from_points(c)
        else:
            inside = window.contains(c)
            if not inside.all():
                if clip:
                    c = c[inside]
                    m = None if m is None else m[inside]
                else:
                    raise ValueError(
                        f"{int((~inside).sum())} event(s) lie outside the window; "
                        "pass clip=True to drop them or enlarge the window"
                    )
        self.coords = c
        self.window = window
        self.marks = m

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_gdf(cls, gdf, mark: Optional[str] = None, window: Optional[Window] = None,
                 clip: bool = False) -> "PointPattern":
        """Build from a GeoDataFrame of point geometries."""
        from spatialstats.core.distance import coords_from_geometry
        coords = coords_from_geometry(gdf.geometry)
        marks = None if mark is None else gdf[mark].to_numpy()
        return cls(coords, window=window, marks=marks, clip=clip)

    # -- basic properties ---------------------------------------------------

    @property
    def n(self) -> int:
        return len(self.coords)

    def __len__(self) -> int:
        return self.n

    @property
    def area(self) -> float:
        return self.window.area

    @property
    def intensity(self) -> float:
        """Average intensity (events per unit area)."""
        return self.n / self.area

    @property
    def n_duplicates(self) -> int:
        """Number of events that duplicate an earlier event's location."""
        return int(self.n - len(np.unique(self.coords, axis=0)))

    def subset(self, mask) -> "PointPattern":
        mask = np.asarray(mask)
        return PointPattern(
            self.coords[mask], window=self.window,
            marks=None if self.marks is None else self.marks[mask],
        )

    def by_mark(self, value: Any) -> "PointPattern":
        if self.marks is None:
            raise ValueError("pattern has no marks")
        return self.subset(self.marks == value)

    def deduplicate(self) -> "PointPattern":
        """Drop repeated locations (keeps the first occurrence)."""
        _, first = np.unique(self.coords, axis=0, return_index=True)
        keep = np.sort(first)
        return self.subset(keep)

    def jitter(self, scale: float, seed: Optional[int] = None) -> "PointPattern":
        """Uniform random displacement (± ``scale``); events pushed out of the window are reflected back."""
        rng = np.random.default_rng(seed)
        c = self.coords + rng.uniform(-scale, scale, self.coords.shape)
        outside = ~self.window.contains(c)
        c[outside] = self.coords[outside]
        return PointPattern(c, window=self.window, marks=self.marks)

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame({"x": self.coords[:, 0], "y": self.coords[:, 1]})
        if self.marks is not None:
            df["mark"] = self.marks
        return df

    def plot(self, ax=None, **kwargs):
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "plot requires matplotlib. Install with: pip install pyspatialstats[viz]"
            ) from e
        if ax is None:
            _, ax = plt.subplots(figsize=kwargs.pop("figsize", (6, 6)))
        poly = self.window.polygon
        geoms = getattr(poly, "geoms", [poly])
        for g in geoms:
            x, y = g.exterior.xy
            ax.plot(x, y, color="k", lw=0.8)
        kwargs.setdefault("s", 6)
        ax.scatter(self.coords[:, 0], self.coords[:, 1], **kwargs)
        ax.set_aspect("equal")
        ax.set_title(f"Point pattern (n={self.n})")
        return ax

    def __repr__(self) -> str:
        return f"PointPattern(n={self.n}, intensity={self.intensity:.4g}, window={self.window!r})"
