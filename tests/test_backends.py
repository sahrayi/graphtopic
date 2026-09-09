import builtins
import sys
import types

import numpy as np
import pytest
from scipy import sparse

from graphtopic import (
    DocumentGraph,
    ExactNeighborSearch,
    GraphTopic,
    LeidenDetector,
    NNDescentSearch,
    SentenceTransformerEmbedding,
)
from graphtopic._validation import normalize_embeddings


@pytest.mark.integration
def test_real_ann_recall_and_complete_pipeline():
    rng = np.random.default_rng(19)
    x = normalize_embeddings(rng.normal(size=(80, 12)), 80)
    search = NNDescentSearch(15, random_state=19)
    candidates = search.search(x)
    exact = ExactNeighborSearch(5).search(x)
    recall = np.mean([len(set(a) & set(b)) / 5 for a, b in zip(candidates, exact, strict=True)])
    assert recall > 0.85  # fixture fidelity check, not a paper-result assertion
    assert all(i not in row and len(row) <= 15 for i, row in enumerate(candidates))
    model = GraphTopic(embedding_model=None, n_candidates=15, n_neighbors=5)
    model.fit(["apple text" if i % 2 else "pear text" for i in range(80)], x)
    assert len(model.topics_) == 80
    assert model.graph_.n_edges <= 80 * 5
    assert (
        model.result_.metadata["base_fit"]["components"]["neighbor_search"]["effective"][
            "backend_neighbors"
        ]
        == 16
    )


@pytest.mark.integration
@pytest.mark.parametrize("n", [2, 3])
def test_real_ann_small_corpora(n):
    with pytest.warns(UserWarning):
        model = GraphTopic(embedding_model=None).fit([""] * n, np.ones((n, 2)))
    assert len(model.topics_) == n


@pytest.mark.integration
def test_leiden_isolates_and_weighted_two_blocks():
    a = sparse.block_diag(
        [
            sparse.csr_matrix([[0, 1], [1, 0]]),
            sparse.csr_matrix([[0, 1], [1, 0]]),
            sparse.csr_matrix((1, 1)),
        ]
    ).tocsr()
    a.eliminate_zeros()
    labels = LeidenDetector().fit_predict(DocumentGraph(a, tuple(range(5))))
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert len(set(labels)) == 3
    empty = DocumentGraph(sparse.csr_matrix((3, 3)), (0, 1, 2))
    with pytest.warns(UserWarning, match="no edges"):
        assert LeidenDetector().fit_predict(empty).tolist() == [0, 1, 2]


def test_lazy_encoder_adapter(monkeypatch):
    calls = []

    class Fake:
        def __init__(self, *args, **kwargs):
            calls.append(("init", args, kwargs))

        def encode(self, docs, **kwargs):
            calls.append(("encode", docs, kwargs))
            return np.ones((len(docs), 2))

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=Fake)
    )
    encoder = SentenceTransformerEmbedding("test", batch_size=7)
    assert calls == []
    encoder.encode(["a"])
    encoder.encode(["b"])
    assert len([c for c in calls if c[0] == "init"]) == 1
    assert calls[1][2]["batch_size"] == 7


def test_missing_optional_encoder(monkeypatch):
    original = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("missing")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match="graphtopic"):
        SentenceTransformerEmbedding().encode(["a"])


@pytest.mark.parametrize(
    "factory, kwargs",
    [
        (NNDescentSearch, {"n_jobs": 0}),
        (NNDescentSearch, {"n_trees": 0}),
        (NNDescentSearch, {"n_iters": 0}),
        (LeidenDetector, {"n_iterations": 0}),
        (ExactNeighborSearch, {"batch_size": 0}),
    ],
)
def test_backend_parameters(factory, kwargs):
    with pytest.raises(ValueError):
        factory(**kwargs)
