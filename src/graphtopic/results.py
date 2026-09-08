"""Validated sparse graphs and immutable logical result snapshots."""

from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


class DocumentGraph:
    """An undirected positive-weight CSR graph with stable document identities.

    Public adjacency and diagnostics accessors return defensive copies. Internally,
    snapshots share the validated graph without duplicating its sparse arrays.
    """

    __slots__ = ("_adjacency", "_diagnostics", "_document_ids")

    def __init__(self, adjacency, document_ids, diagnostics=None):
        if not sparse.issparse(adjacency):
            raise ValueError("graph adjacency must be a SciPy sparse matrix")
        ids = tuple(document_ids)
        from ._validation import documents_and_ids

        documents_and_ids([""] * len(ids), ids)
        if adjacency.shape != (len(ids), len(ids)):
            raise ValueError("graph adjacency shape must match document_ids")
        if adjacency.dtype.kind not in "iuf":
            raise ValueError("graph weights must be real numeric values")
        matrix = sparse.csr_matrix(adjacency, dtype=np.float64, copy=True)
        if not matrix.has_canonical_format:
            raise ValueError("graph adjacency must not contain duplicate edges")
        matrix.sort_indices()
        if not np.isfinite(matrix.data).all() or np.any(matrix.data <= 0):
            raise ValueError("stored graph weights must be finite and strictly positive")
        if np.any(matrix.diagonal() != 0):
            raise ValueError("graph must not contain self-loops")
        transpose = matrix.T.tocsr()
        if (
            not np.array_equal(matrix.indptr, transpose.indptr)
            or not np.array_equal(matrix.indices, transpose.indices)
            or not np.allclose(matrix.data, transpose.data, rtol=1e-5, atol=1e-7)
        ):
            raise ValueError("graph must be symmetric, including its sparsity pattern")
        for array in (matrix.data, matrix.indices, matrix.indptr):
            array.flags.writeable = False
        self._adjacency = matrix
        self._document_ids = ids
        degree = np.diff(matrix.indptr)
        details = deepcopy(diagnostics or {})
        details.update(
            n_vertices=len(ids),
            n_edges=matrix.nnz // 2,
            isolated_vertices=int(np.count_nonzero(degree == 0)),
            connected_components=int(
                connected_components(matrix, directed=False, return_labels=False)
            ),
            min_degree=int(degree.min()),
            mean_degree=float(degree.mean()),
            max_degree=int(degree.max()),
        )
        self._diagnostics = details

    @property
    def adjacency(self):
        """Return a writable CSR copy suitable for export or inspection."""
        return self._adjacency.copy()

    @property
    def document_ids(self):
        return self._document_ids

    @property
    def diagnostics(self):
        return deepcopy(self._diagnostics)

    @property
    def n_vertices(self):
        return len(self._document_ids)

    @property
    def n_edges(self):
        return self._adjacency.nnz // 2

    def copy(self):
        return DocumentGraph(self._adjacency, self.document_ids, self._diagnostics)

    def __deepcopy__(self, memo):
        # Validated graph is logically immutable; public arrays are defensive copies.
        return self


@dataclass(frozen=True, slots=True)
class GraphTopicResult:
    """A successful partition and its lexical explanation, sharing a fixed graph."""

    documents: tuple[str, ...]
    document_ids: tuple[str | int, ...]
    topics: tuple[int, ...]
    graph: DocumentGraph
    _representations: dict = field(repr=False)
    _metadata: dict = field(repr=False)

    def __post_init__(self):
        object.__setattr__(self, "_representations", deepcopy(self._representations))
        object.__setattr__(self, "_metadata", deepcopy(self._metadata))

    @property
    def topic_words(self):
        return deepcopy(self._representations)

    @property
    def metadata(self):
        return deepcopy(self._metadata)

    @property
    def topic_sizes(self):
        ids, counts = np.unique(self.topics, return_counts=True)
        return dict(zip(ids.tolist(), counts.tolist(), strict=True))

    def topic_name(self, topic):
        words = self._representations[topic]
        return f"{topic}_" + "_".join(w for w, _ in words[:3]) if words else f"topic_{topic}"

    def get_topic_info(self):
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "Topic": topic,
                    "Count": count,
                    "Name": self.topic_name(topic),
                    "Representation": deepcopy(self._representations[topic]),
                }
                for topic, count in self.topic_sizes.items()
            ]
        )

    def get_document_info(self):
        import pandas as pd

        return pd.DataFrame(
            {
                "Document_ID": self.document_ids,
                "Document": self.documents,
                "Topic": self.topics,
                "Name": [self.topic_name(topic) for topic in self.topics],
            }
        )


@dataclass(frozen=True, slots=True)
class ResolutionPath:
    """Independent partitions of one graph, in the requested resolution order."""

    resolutions: tuple[float, ...]
    results: tuple[GraphTopicResult, ...]

    def get_summary(self):
        import pandas as pd

        rows = []
        for resolution, result in zip(self.resolutions, self.results, strict=True):
            sizes = list(result.topic_sizes.values())
            rows.append(
                {
                    "Resolution": resolution,
                    "Topics": len(sizes),
                    "MinSize": min(sizes),
                    "MedianSize": float(np.median(sizes)),
                    "MaxSize": max(sizes),
                }
            )
        return pd.DataFrame(rows)
