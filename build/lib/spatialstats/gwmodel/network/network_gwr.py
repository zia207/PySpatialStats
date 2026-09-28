"""
spatialstats.gwmodel.network.network_gwr
==============================
Network-based Geographically Weighted Regression.

Replaces Euclidean distance with network (shortest-path) distance.
"""

import numpy as np
from spatialstats.gwmodel.linear.gwr import GWR
from spatialstats.gwmodel.core.base import _parse_geometry


class NetworkGWR(GWR):
    """
    GWR using network shortest-path distances.

    Parameters
    ----------
    network : networkx.Graph
        Spatial network (nodes should have 'x', 'y' attributes).
    All other parameters are inherited from GWR.

    Examples
    --------
    >>> import networkx as nx
    >>> from spatialstats.gwmodel.network import NetworkGWR
    >>> G = nx.grid_2d_graph(10, 10)
    >>> # Add weights to edges
    >>> for u, v in G.edges():
    ...     G[u][v]['weight'] = 1.0
    >>> model = NetworkGWR(network=G, bandwidth=3, fixed=False)
    """

    def __init__(self, network=None, bandwidth=None, fixed=False,
                 kernel="bisquare", criterion="AICc", n_jobs=-1):
        super().__init__(bandwidth=bandwidth, fixed=fixed, kernel=kernel,
                         criterion=criterion, n_jobs=n_jobs)
        self.network = network

    def _network_distance_matrix(self, coords: np.ndarray) -> np.ndarray:
        """Compute network distance matrix from node coordinates."""
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("networkx required: pip install networkx")
        n = len(coords)
        D = np.zeros((n, n))
        nodes = list(self.network.nodes())
        # Map coords to nearest nodes
        node_coords = np.array([
            [self.network.nodes[nd].get('x', nd[0]),
             self.network.nodes[nd].get('y', nd[1])]
            for nd in nodes
        ])
        from scipy.spatial.distance import cdist
        dist_to_nodes = cdist(coords, node_coords)
        nearest_nodes = [nodes[np.argmin(dist_to_nodes[i])] for i in range(n)]

        for i in range(n):
            lengths = nx.single_source_shortest_path_length(
                self.network, nearest_nodes[i]
            )
            for j in range(n):
                D[i, j] = lengths.get(nearest_nodes[j], np.inf)
        return D

    def fit(self, X, y, geometry):
        """Fit NetworkGWR using network distances."""
        if self.network is not None:
            X = np.asarray(X, dtype=float)
            y = np.asarray(y, dtype=float).ravel()
            coords = _parse_geometry(geometry)
            # Override with network distances by monkey-patching D
            self._network_D = self._network_distance_matrix(coords)
        return super().fit(X, y, geometry)
