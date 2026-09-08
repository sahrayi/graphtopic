"""Canonical exact-cosine top-k union-max graph construction."""

import warnings

import numpy as np
from scipy import sparse

from ._validation import positive_int
from .results import DocumentGraph


class UnionMaxGraph:
    """Retain positive top-k candidates and symmetrize by maximum, never sum."""

    def __init__(self, n_neighbors=20):
        self.n_neighbors = positive_int(n_neighbors, "n_neighbors")

    def build(self, embeddings, candidates, *, document_ids):
        n = len(embeddings)
        k = min(self.n_neighbors, n - 1)
        if k < self.n_neighbors:
            warnings.warn(
                f"n_neighbors reduced from {self.n_neighbors} to {k} for {n} documents",
                UserWarning,
                stacklevel=2,
            )
        if len(candidates) != n:
            raise ValueError("neighbor_search must return one candidate row per document")
        indptr = np.zeros(n + 1, dtype=np.int64)
        # Allocate numeric arrays, not millions of Python edge tuples.
        indices = np.empty(n * k, dtype=np.int64)
        weights = np.empty(n * k, dtype=np.float32)
        cursor = 0
        shortages = 0
        for i, row in enumerate(candidates):
            candidate = np.asarray(row)
            if candidate.ndim != 1 or (candidate.size and candidate.dtype.kind not in "iu"):
                raise ValueError(f"neighbor_search returned non-integer candidates in row {i}")
            if np.any(candidate < 0) or np.any(candidate >= n):
                raise ValueError(f"neighbor_search returned out-of-range candidates in row {i}")
            candidate = np.unique(candidate.astype(np.int64))
            candidate = candidate[candidate != i]
            scores = np.clip(embeddings[candidate] @ embeddings[i], -1, 1)
            valid = scores > 0
            candidate, scores = candidate[valid], scores[valid]
            order = np.lexsort((candidate, -scores))[:k]
            selected, scores = candidate[order], scores[order]
            count = len(selected)
            shortages += count < k
            indices[cursor : cursor + count] = selected
            weights[cursor : cursor + count] = scores
            cursor += count
            indptr[i + 1] = cursor
        directed = sparse.csr_matrix((weights[:cursor], indices[:cursor], indptr), shape=(n, n))
        graph = directed.maximum(directed.T).tocsr()
        graph.sort_indices()
        self.effective_config_ = {
            "n_neighbors": k,
            "weight": "exact_cosine",
            "symmetrization": "union_max",
        }
        return DocumentGraph(
            graph,
            document_ids,
            {
                **self.effective_config_,
                "directed_arcs": cursor,
                "candidate_shortages": int(shortages),
            },
        )
