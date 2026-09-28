"""
spatialstats.weights.w
======================
Spatial weights matrix class.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


class W:
    """
    Sparse spatial weights matrix.

    Parameters
    ----------
    neighbors : dict
        Mapping ``{i: [j, ...]}`` of neighbor indices.
    weights : dict, optional
        Mapping ``{i: [w_ij, ...]}``. Defaults to 1.0 for each neighbor.
    ids : sequence, optional
        Observation ids (default: ``range(n)``).
    transform : {'binary', 'row', 'variance', 'O'}, optional
        Initial standardization. ``'O'`` / ``None`` leaves raw weights.
    """

    def __init__(
        self,
        neighbors: Dict[Any, Sequence[Any]],
        weights: Optional[Dict[Any, Sequence[float]]] = None,
        ids: Optional[Sequence[Any]] = None,
        transform: Optional[str] = "binary",
    ):
        if ids is None:
            ids = list(neighbors.keys())
        self.id_order = list(ids)
        self.id_to_index = {i: k for k, i in enumerate(self.id_order)}
        self.n = len(self.id_order)

        self.neighbors: Dict[Any, List[Any]] = {
            i: list(neighbors.get(i, [])) for i in self.id_order
        }
        if weights is None:
            self.weights: Dict[Any, List[float]] = {
                i: [1.0] * len(self.neighbors[i]) for i in self.id_order
            }
        else:
            self.weights = {
                i: list(weights.get(i, [1.0] * len(self.neighbors[i])))
                for i in self.id_order
            }

        self._transform = "O"
        self._sparse: Optional[sparse.csr_matrix] = None
        if transform and transform not in ("O", "o", None):
            self.transform(transform)
        else:
            self._rebuild_sparse()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_sparse(
        cls,
        mat: sparse.spmatrix,
        ids: Optional[Sequence[Any]] = None,
        transform: Optional[str] = None,
    ) -> "W":
        """Build W from a square sparse matrix."""
        mat = mat.tocsr()
        n = mat.shape[0]
        if mat.shape[0] != mat.shape[1]:
            raise ValueError("Weight matrix must be square")
        if ids is None:
            ids = list(range(n))
        neighbors: Dict[Any, List[Any]] = {}
        weights: Dict[Any, List[float]] = {}
        for i, id_i in enumerate(ids):
            start, end = mat.indptr[i], mat.indptr[i + 1]
            cols = mat.indices[start:end]
            vals = mat.data[start:end]
            neighbors[id_i] = [ids[j] for j in cols]
            weights[id_i] = [float(v) for v in vals]
        return cls(neighbors, weights, ids=ids, transform=transform)

    def _rebuild_sparse(self) -> None:
        rows, cols, data = [], [], []
        for i, id_i in enumerate(self.id_order):
            for nbr, w in zip(self.neighbors[id_i], self.weights[id_i]):
                j = self.id_to_index[nbr]
                rows.append(i)
                cols.append(j)
                data.append(float(w))
        self._sparse = sparse.csr_matrix(
            (data, (rows, cols)), shape=(self.n, self.n), dtype=float
        )

    @property
    def sparse(self) -> sparse.csr_matrix:
        if self._sparse is None:
            self._rebuild_sparse()
        return self._sparse

    @property
    def transform_type(self) -> str:
        return self._transform

    # ------------------------------------------------------------------
    # Standardization
    # ------------------------------------------------------------------

    def transform(self, conversion: str = "row") -> "W":
        """
        Standardize weights in place.

        Parameters
        ----------
        conversion : {'binary', 'row', 'variance', 'O'}
            - binary: 1 for neighbors, 0 else
            - row: row-stochastic (``w_ij / sum_j w_ij``)
            - variance: variance-stabilizing (``w_ij / sqrt(sum_j w_ij)``)
            - O: restore from current sparse as-is (no-op label)
        """
        conversion = conversion.lower() if conversion else "o"
        if conversion == "b":
            conversion = "binary"
        if conversion == "r":
            conversion = "row"
        if conversion == "v":
            conversion = "variance"

        if conversion == "binary":
            for i in self.id_order:
                self.weights[i] = [1.0] * len(self.neighbors[i])
        elif conversion == "row":
            for i in self.id_order:
                s = float(sum(self.weights[i]))
                if s > 0:
                    self.weights[i] = [w / s for w in self.weights[i]]
                else:
                    self.weights[i] = []
                    self.neighbors[i] = []
        elif conversion == "variance":
            for i in self.id_order:
                s = float(sum(self.weights[i]))
                if s > 0:
                    denom = np.sqrt(s)
                    self.weights[i] = [w / denom for w in self.weights[i]]
        elif conversion in ("o", "none"):
            pass
        else:
            raise ValueError(
                f"Unknown transform '{conversion}'. "
                "Use 'binary', 'row', 'variance', or 'O'."
            )

        self._transform = conversion if conversion != "none" else "O"
        self._rebuild_sparse()
        return self

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def cardinalities(self) -> np.ndarray:
        """Number of neighbors per observation."""
        return np.array([len(self.neighbors[i]) for i in self.id_order], dtype=int)

    def islands(self) -> List[Any]:
        """Ids with zero neighbors."""
        return [i for i in self.id_order if len(self.neighbors[i]) == 0]

    def n_components(self) -> int:
        """Number of connected components (undirected)."""
        n_comp, _ = connected_components(self.sparse, directed=False)
        return int(n_comp)

    def is_symmetric(self, tol: float = 1e-10) -> bool:
        diff = self.sparse - self.sparse.T
        return float(np.abs(diff.data).max()) < tol if diff.nnz else True

    def summary(self) -> str:
        cards = self.cardinalities()
        islands = self.islands()
        lines = [
            "Spatial Weights (W)",
            "===================",
            f"  Observations        {self.n}",
            f"  Transform           {self._transform}",
            f"  Min neighbors       {int(cards.min()) if self.n else 0}",
            f"  Mean neighbors      {float(cards.mean()) if self.n else 0:.3f}",
            f"  Max neighbors       {int(cards.max()) if self.n else 0}",
            f"  Islands             {len(islands)}",
            f"  Components          {self.n_components()}",
            f"  Nonzero weights     {self.sparse.nnz}",
        ]
        return "\n".join(lines)

    def lag(self, y: np.ndarray) -> np.ndarray:
        """Spatial lag Wy."""
        y = np.asarray(y, dtype=float).ravel()
        if len(y) != self.n:
            raise ValueError(f"y length {len(y)} != W.n {self.n}")
        return self.sparse @ y

    # ------------------------------------------------------------------
    # libpysal interoperability
    # ------------------------------------------------------------------

    def to_libpysal(self):
        """Convert to a ``libpysal.weights.W`` object."""
        try:
            from libpysal.weights import W as LibW
        except ImportError as e:
            raise ImportError(
                "to_libpysal requires libpysal. Install with: pip install pyspatialstats[libpysal]"
            ) from e
        return LibW(self.neighbors, self.weights, id_order=self.id_order)

    @classmethod
    def from_libpysal(cls, w) -> "W":
        """Build from a ``libpysal.weights.W`` object."""
        return cls(
            neighbors=dict(w.neighbors),
            weights=dict(w.weights),
            ids=list(w.id_order),
            transform=None,
        )

    def __repr__(self) -> str:
        return (
            f"W(n={self.n}, transform={self._transform!r}, "
            f"nnz={self.sparse.nnz}, islands={len(self.islands())})"
        )
