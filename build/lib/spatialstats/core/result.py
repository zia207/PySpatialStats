"""
spatialstats.core.result
========================
Unified result base class with summary / plot / to_frame.
"""

from __future__ import annotations

from abc import ABC
from typing import Any, Dict, Optional

import pandas as pd


class SpatialResult(ABC):
    """
    Base class for analysis results.

    Subclasses should populate ``_stats`` (scalar summary dict) and optionally
    override ``to_frame``, ``summary``, and ``plot``.
    """

    def __init__(self, name: str = "SpatialResult", **stats: Any):
        self.name = name
        self._stats: Dict[str, Any] = dict(stats)

    def __repr__(self) -> str:
        keys = ", ".join(f"{k}={v!r}" for k, v in list(self._stats.items())[:4])
        more = "..." if len(self._stats) > 4 else ""
        return f"{self.__class__.__name__}({keys}{more})"

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [f"{self.name}", "=" * len(self.name)]
        for key, val in self._stats.items():
            if isinstance(val, float):
                lines.append(f"  {key:<20s} {val:.6g}")
            else:
                lines.append(f"  {key:<20s} {val}")
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        """Return scalar statistics as a one-row DataFrame."""
        return pd.DataFrame([self._stats])

    def plot(self, ax=None, **kwargs):
        """
        Default plot hook. Subclasses should override.

        Raises
        ------
        NotImplementedError
            If no plot method is defined for this result type.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement plot(); "
            "use a specialized result class or spatialstats.viz helpers."
        )

    def get(self, key: str, default: Any = None) -> Any:
        return self._stats.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self._stats[key]

    @property
    def stats(self) -> Dict[str, Any]:
        return dict(self._stats)
