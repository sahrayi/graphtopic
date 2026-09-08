import logging
import subprocess
import sys
from copy import deepcopy

import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import CountVectorizer

from graphtopic import CTFIDFRepresentation, DocumentGraph, GraphTopic
from graphtopic._validation import canonical_labels, normalize_embeddings


class ExplodingEncoder:
    def encode(self, documents):
        raise AssertionError("encoder was called")


class Encoder:
    def encode(self, documents):
        return np.tile([1.0, 0.0], (len(documents), 1))


class FixedDetector:
    def fit_predict(self, graph):
        return [10, 10, 20, 20]


class BadDetector:
    def fit_predict(self, graph):
        return [-1] * graph.n_vertices


class BadRepresentation:
    def fit(self, documents, topics):
        raise RuntimeError("lexical failure")

    def get_topics(self):
        return {}


def test_precomputed_precedence_and_outputs(model, corpus, embeddings):
    model.embedding_model = ExplodingEncoder()
    original = embeddings.copy()
    labels = model.fit_transform(corpus, embeddings, document_ids=["a", "b", "c", "d"])
    assert isinstance(labels, list) and len(labels) == 4
    assert labels[0] == labels[1] and labels[2] == labels[3] and labels[0] != labels[2]
    assert np.array_equal(embeddings, original)
    assert sum(model.topic_sizes_.values()) == 4
    assert model.documents_ == tuple(corpus)
    assert model.document_ids_ == ("a", "b", "c", "d")
    assert model.result_.metadata["base_fit"]["embedding_source"] == "provided"
    assert model.result_.metadata["base_fit"]["encoder"] == "unknown"
    assert list(model.get_topic_info()) == ["Topic", "Count", "Name", "Representation"]
    assert list(model.get_document_info()) == ["Document_ID", "Document", "Topic", "Name"]
    labels[0] = 900
    model.get_topics()[0].clear()
    assert model.topics_[0] != 900 and model.get_topic(0)
    with pytest.raises(KeyError):
        model.get_topic(900)
    with pytest.raises(ValueError):
        model.embeddings_[0] = 0
    model.get_graph().adjacency.data[:] = 90
    assert model.graph_.adjacency.data.max() <= 1
    assert deepcopy(model.graph_) is model.graph_
    assert "ExactNeighborSearch" in repr(model)


def test_encoded_path(model, corpus):
    model.embedding_model = Encoder()
    assert model.fit(corpus) is model
    assert model.result_.metadata["base_fit"]["embedding_source"] == "encoded"


@pytest.mark.parametrize(
    "docs", [[], None, "abc", {1, 2}, iter(["a"]), ["a", 1], np.array([["a"]])]
)
def test_bad_documents(model, docs):
    with pytest.raises(ValueError):
        model.fit(docs, [[1, 0]])


@pytest.mark.parametrize(
    "x",
    [
        [[0, 0]] * 4,
        [[np.nan, 1]] * 4,
        [[np.inf, 1]] * 4,
        [[True, False]] * 4,
        [["1", "2"]] * 4,
        [[1j, 0]] * 4,
        [[1, 0]],
        np.empty((4, 0)),
        [1, 2, 3, 4],
    ],
)
def test_bad_embeddings(model, corpus, x):
    with pytest.raises(ValueError):
        model.fit(corpus, x)


@pytest.mark.parametrize("ids", [[0], [0, 0, 1, 2], [0, "a", 2, 3], [True, 1, 2, 3]])
def test_bad_ids(model, corpus, embeddings, ids):
    with pytest.raises(ValueError):
        model.fit(corpus, embeddings, document_ids=ids)


def test_transactional_refit_and_update(model, corpus, embeddings):
    model.fit(corpus, embeddings)
    previous = model.result_
    model.community_model = BadDetector()
    with pytest.raises(ValueError):
        model.fit(corpus, embeddings)
    assert model.result_ is previous
    with pytest.raises(RuntimeError, match="lexical failure"):
        model.update_topics(BadRepresentation())
    assert model.result_ is previous
    model.update_topics(CTFIDFRepresentation(CountVectorizer(), top_n_words=1))
    assert model.result_ is not previous
    assert model.result_.graph is previous.graph
    assert model.result_.topics == previous.topics
    assert len(model.get_topic(0)) == 1
    assert len(previous.topic_words[0]) > 1


def test_path_reuse_and_independence(model, corpus, embeddings, monkeypatch):
    model.fit(corpus, embeddings)
    previous = model.result_
    model.embedding_model = ExplodingEncoder()
    monkeypatch.setattr(type(model.neighbor_search), "search", lambda *args: pytest.fail("search"))
    path = model.resolution_path([0.1, 1, 2])
    assert path.resolutions == (0.1, 1.0, 2.0)
    assert len(path.results) == 3
    assert model.result_ is previous
    assert all(result.graph is previous.graph for result in path.results)
    assert list(path.get_summary()) == ["Resolution", "Topics", "MinSize", "MedianSize", "MaxSize"]
    assert path.results[0].metadata["run"]["graph_reused"]
    assert path.results[0].metadata["base_fit"]["requested"]["resolution"] == 1.0
    assert path.results[0].metadata["run"]["resolution"] == 0.1
    for values in [[], [1, 1], [0], "1", [float("nan")]]:
        with pytest.raises(ValueError):
            model.resolution_path(values)


def test_custom_detector_and_no_path(model, corpus, embeddings):
    custom = FixedDetector()
    model.community_model = custom
    model.fit(corpus, embeddings)
    assert model.topics_ == [0, 0, 1, 1]
    assert model.result_.metadata["run"]["raw_to_topic"] == {10: 0, 20: 1}
    assert model.result_.metadata["run"]["components"]["community_model"]["effective"] == "unknown"
    with pytest.raises(NotImplementedError):
        model.resolution_path([1])


def test_unfitted_and_missing_embeddings():
    m = GraphTopic(embedding_model=None)
    for operation in [m.get_topics, m.get_graph, m.get_topic_info, m.get_document_info]:
        with pytest.raises(NotFittedError):
            operation()
    with pytest.raises(ValueError, match="Supply embeddings"):
        m.fit(["hello"])


def test_single_and_empty_documents():
    with pytest.warns(UserWarning):
        m = GraphTopic(embedding_model=None).fit([""], [[2.0, 0]])
    assert m.topics_ == [0]
    assert m.graph_.n_vertices == 1 and m.graph_.n_edges == 0
    assert m.get_topic(0) == []


def test_normalization_extremes_and_label_order():
    x = normalize_embeddings([[1e300, 1e300], [1e-300, 0]], 2)
    assert np.isfinite(x).all()
    assert np.allclose(np.linalg.norm(x, axis=1), 1)
    labels, _ = canonical_labels([9, 2, 2, 9, 7], 5)
    assert labels == (0, 1, 1, 0, 2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_candidates": 0},
        {"n_neighbors": True},
        {"n_candidates": 2, "n_neighbors": 3},
        {"resolution": 0},
        {"resolution": np.nan},
        {"random_state": -1},
        {"random_state": True},
        {"verbose": "yes"},
    ],
)
def test_constructor_validation(kwargs):
    with pytest.raises(ValueError):
        GraphTopic(**kwargs)


def test_override_warning():
    with pytest.warns(UserWarning, match="does not configure"):
        GraphTopic(community_model=FixedDetector(), resolution=2)


def test_core_import_does_not_load_encoder():
    code = (
        "import sys; import graphtopic; "
        "assert 'sentence_transformers' not in sys.modules; "
        "assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_logging_is_opt_in(model, corpus, embeddings, caplog):
    model.verbose = True
    with caplog.at_level(logging.INFO, logger="graphtopic"):
        model.fit(corpus, embeddings)
    assert "graph_model completed" in caplog.text


def test_bad_custom_graph(model, corpus, embeddings):
    class WrongGraph:
        def build(self, *args, **kwargs):
            return DocumentGraph(sparse.csr_matrix((4, 4)), ("w", "x", "y", "z"))

    model.graph_model = WrongGraph()
    with pytest.raises(ValueError, match="unchanged"):
        model.fit(corpus, embeddings)
