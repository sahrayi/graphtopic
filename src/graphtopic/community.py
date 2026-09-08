"""Weighted Leiden RB-configuration community detection."""

import warnings

import numpy as np
from scipy import sparse

from ._validation import positive_real, seed_value


class LeidenDetector:
    """Partition a fixed undirected graph using RBConfigurationVertexPartition.

    The default of two iterations matches the frozen paper experiments. Pass
    n_iterations=-1 to continue until no improvement. Isolates are always singletons;
    connected components are not separately normalized or optimized.
    """

    def __init__(self, resolution=1.0, *, random_state=42, n_iterations=2):
        self.resolution = positive_real(resolution, "resolution")
        self.random_state = seed_value(random_state)
        if (
            isinstance(n_iterations, bool)
            or not isinstance(n_iterations, int)
            or n_iterations == 0
            or n_iterations < -1
        ):
            raise ValueError("n_iterations must be -1 or a positive integer")
        self.n_iterations = n_iterations

    def with_resolution(self, value):
        return type(self)(
            resolution=value, random_state=self.random_state, n_iterations=self.n_iterations
        )

    def fit_predict(self, graph):
        self.effective_config_ = {
            "resolution": self.resolution,
            "random_state": self.random_state,
            "n_iterations": self.n_iterations,
            "objective": "RBConfigurationVertexPartition",
        }
        n = graph.n_vertices
        if graph.n_edges == 0:
            warnings.warn(
                "Graph has no edges; assigning each document a singleton topic",
                UserWarning,
                stacklevel=2,
            )
            return np.arange(n)
        import igraph as ig
        import leidenalg

        adjacency = graph._adjacency
        upper = sparse.triu(adjacency, k=1, format="coo")
        edges = np.column_stack((upper.row, upper.col))
        backend = ig.Graph(n=n, edges=edges, directed=False)
        partition = leidenalg.find_partition(
            backend,
            leidenalg.RBConfigurationVertexPartition,
            weights=upper.data,
            resolution_parameter=self.resolution,
            n_iterations=self.n_iterations,
            seed=self.random_state,
        )
        labels = np.asarray(partition.membership, dtype=np.int64)
        isolated = np.flatnonzero(np.diff(adjacency.indptr) == 0)
        labels[isolated] = np.arange(len(isolated)) + labels.max() + 1
        return labels
