"""Candidate generators: exact block search and approximate NNDescent."""

import warnings

import numpy as np

from ._validation import positive_int, seed_value


def _budget(requested, n):
    effective = min(requested, n - 1)
    if effective < requested:
        warnings.warn(
            f"n_candidates reduced from {requested} to {effective} for {n} documents",
            UserWarning,
            stacklevel=3,
        )
    return effective


class ExactNeighborSearch:
    """Exact cosine candidates on normalized embeddings using bounded row blocks.

    Time remains quadratic. batch_size bounds temporary similarity rows; no full
    n-by-n matrix is constructed (except the trivial empty/single-document case).
    """

    def __init__(self, n_candidates=50, *, batch_size=256):
        self.n_candidates = positive_int(n_candidates, "n_candidates")
        self.batch_size = positive_int(batch_size, "batch_size")

    def search(self, embeddings):
        n = len(embeddings)
        k = _budget(self.n_candidates, n)
        self.effective_config_ = {
            "n_candidates": k,
            "batch_size": self.batch_size,
            "backend": "exact",
        }
        if not k:
            return [np.empty(0, dtype=np.int64) for _ in range(n)]
        result = []
        ids = np.arange(n)
        for start in range(0, n, min(self.batch_size, max(1, n - 1))):
            block = embeddings[start : start + min(self.batch_size, max(1, n - 1))]
            scores = np.clip(block @ embeddings.T, -1, 1)
            for offset, row in enumerate(scores):
                i = start + offset
                row[i] = -np.inf
                # Partial selection preserves deterministic ties at the cutoff.
                cutoff = np.partition(row, n - k)[n - k]
                above = ids[row > cutoff]
                tied = ids[row == cutoff][: k - len(above)]
                chosen = np.concatenate((above, tied))
                result.append(chosen[np.lexsort((chosen, -row[chosen]))])
        return result


class NNDescentSearch:
    """Approximate non-self candidates, independently rescored by the graph model.

    n_trees=5 and n_iters=10 are explicit engineering defaults, not a promise to
    reproduce a particular experimental run. n_jobs=1 favors reproducibility.
    """

    def __init__(
        self,
        n_candidates=50,
        *,
        random_state=42,
        n_trees=5,
        n_iters=10,
        n_jobs=1,
    ):
        self.n_candidates = positive_int(n_candidates, "n_candidates")
        self.random_state = seed_value(random_state)
        self.n_trees = positive_int(n_trees, "n_trees")
        self.n_iters = positive_int(n_iters, "n_iters")
        if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs == 0 or n_jobs < -1:
            raise ValueError("n_jobs must be -1 or a positive integer")
        self.n_jobs = n_jobs

    def search(self, embeddings):
        n = len(embeddings)
        k = _budget(self.n_candidates, n)
        raw_k = min(n, k + 1)
        self.effective_config_ = {
            "backend": "nndescent",
            "n_candidates": k,
            "backend_neighbors": raw_k,
            "metric": "cosine",
            "random_state": self.random_state,
            "n_trees": self.n_trees,
            "n_iters": self.n_iters,
            "n_jobs": self.n_jobs,
            "low_memory": True,
            "delta": 0.001,
        }
        if not k:
            return [np.empty(0, dtype=np.int64) for _ in range(n)]
        from pynndescent import NNDescent

        # Numba kernels in PyNNDescent require a writable array signature.
        # Never expose the model's protected embedding storage to the backend.
        backend_data = np.array(embeddings, dtype=np.float32, order="C", copy=True)
        index = NNDescent(
            backend_data,
            n_neighbors=raw_k,
            metric="cosine",
            random_state=self.random_state,
            n_trees=self.n_trees,
            n_iters=self.n_iters,
            n_jobs=self.n_jobs,
            low_memory=True,
            delta=0.001,
        )
        indices, _ = index.neighbor_graph
        result = []
        for i, row in enumerate(indices):
            # Backend padding is not a document index. Deduplicate preserving rank.
            seen = set()
            clean = []
            for value in row:
                j = int(value)
                if j >= 0 and j != i and j not in seen:
                    seen.add(j)
                    clean.append(j)
            result.append(np.asarray(clean[:k], dtype=np.int64))
        return result
