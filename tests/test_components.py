"""Boundary tests for third-party components and their isolated state."""

from copy import deepcopy

import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError

from graphtopic import DocumentGraph, GraphTopic, protocols


class Search:
    def search(self, embeddings):
        self.called = True
        return [
            np.array([j for j in range(len(embeddings)) if j != i]) for i in range(len(embeddings))
        ]


class Graph:
    def build(self, embeddings, candidates, *, document_ids):
        self.called = True
        n = len(embeddings)
        return DocumentGraph(sparse.csr_matrix(np.ones((n, n)) - np.eye(n)), document_ids)


class Detector:
    def __init__(self, resolution=1.0):
        self.resolution = resolution

    def fit_predict(self, graph):
        self.called = True
        self.effective_config_ = {"resolution": self.resolution}
        return [0] * graph.n_vertices if self.resolution <= 1 else list(range(graph.n_vertices))

    def with_resolution(self, value):
        return Detector(value)


class Words:
    def fit(self, documents, topics):
        self.words = {topic: [("custom", 1.0)] for topic in topics}
        return self

    def get_topics(self):
        return self.words


def test_all_components_replaceable_and_templates_untouched(corpus, embeddings):
    search, graph, detector, words = Search(), Graph(), Detector(), Words()
    model = GraphTopic(
        embedding_model=None,
        neighbor_search=search,
        graph_model=graph,
        community_model=detector,
        representation_model=words,
    ).fit(corpus, embeddings)
    assert model.topics_ == [0, 0, 0, 0]
    assert model.graph_.n_edges == 6
    assert model.get_topics() == {0: [("custom", 1.0)]}
    assert not hasattr(search, "called") and not hasattr(graph, "called")
    assert not hasattr(detector, "called") and not hasattr(words, "words")
    path = model.resolution_path([1, 2])
    assert len(path.results[0].topic_sizes) == 1
    assert len(path.results[1].topic_sizes) == 4
    assert protocols.NeighborSearch and protocols.CommunityModel  # import public typing module


@pytest.mark.parametrize(
    "output",
    [
        None,
        {True: []},
        {88: []},
        {0: ("term", 1)},
        {0: [("bad", float("nan"))]},
        {0: [("bad", True)]},
        {0: [("missing",)]},
        {0: [(1, 1.0)]},
    ],
)
def test_bad_representation_output(corpus, embeddings, output):
    class BadWords(Words):
        def get_topics(self):
            return output

    model = GraphTopic(
        embedding_model=None,
        n_neighbors=3,
        neighbor_search=Search(),
        community_model=Detector(),
        representation_model=BadWords(),
    )
    with pytest.raises(ValueError, match="representation_model"):
        model.fit(corpus, embeddings)
    with pytest.raises(NotFittedError):
        model.get_topics()


@pytest.mark.parametrize(
    "labels",
    [
        [0],
        [0.0, 0.0, 1.0, 1.0],
        [False] * 4,
        [[0]] * 4,
        [None] * 4,
        [-1] * 4,
    ],
)
def test_bad_partition_output(corpus, embeddings, labels):
    class Bad(Detector):
        def fit_predict(self, graph):
            return labels

    with pytest.raises(ValueError, match="community_model"):
        GraphTopic(
            embedding_model=None,
            n_neighbors=3,
            neighbor_search=Search(),
            community_model=Bad(),
        ).fit(corpus, embeddings)


def test_uncopyable_components_fail_before_encoder():
    class Uncopyable:
        def __deepcopy__(self, memo):
            raise RuntimeError("not copyable")

    with pytest.raises(TypeError, match="deepcopy"):
        GraphTopic(neighbor_search=Uncopyable()).fit(["text"])

    class Shared:
        def __deepcopy__(self, memo):
            return self

    with pytest.raises(TypeError, match="independent"):
        GraphTopic(neighbor_search=Shared()).fit(["text"])


def test_late_failure_keeps_previous_snapshot(model, corpus, embeddings):
    model.fit(corpus, embeddings)
    previous = model.result_

    class Fail(Search):
        def search(self, embeddings):
            raise RuntimeError("index failed")

    model.neighbor_search = Fail()
    with pytest.raises(RuntimeError) as error:
        model.fit(corpus, embeddings)
    assert "neighbor_search" in str(error.value.__notes__)
    assert model.result_ is previous


def test_seed_none_and_array_document_inputs(model, corpus, embeddings):
    model.random_state = None
    model.fit(np.array(corpus), embeddings, document_ids=np.arange(4))
    assert model.document_ids_ == (0, 1, 2, 3)
    assert model.result_.metadata["base_fit"]["requested"]["random_state"] is None


def test_snapshot_metadata_and_words_are_defensive(model, corpus, embeddings):
    model.fit(corpus, embeddings)
    result = model.result_
    expected = deepcopy(result.metadata)
    result.metadata["base_fit"]["components"].clear()
    result.topic_words.clear()
    result.topic_sizes.clear()
    assert result.metadata == expected
    assert result.topic_words and result.topic_sizes


def test_embedding_string_adapter_path(monkeypatch, corpus, embeddings):
    import graphtopic.model as module

    calls = []

    class Adapter:
        def __init__(self, name):
            calls.append(name)

        def encode(self, documents):
            return embeddings

    monkeypatch.setattr(module, "SentenceTransformerEmbedding", Adapter)
    with pytest.warns(UserWarning):
        model = GraphTopic(
            embedding_model="custom-model",
            neighbor_search=Search(),
            community_model=Detector(),
            representation_model=Words(),
        ).fit(corpus)
    assert calls == ["custom-model"]
    assert model.result_.metadata["base_fit"]["encoder"] == "custom-model"
    with pytest.warns(UserWarning):
        model.fit(corpus)
    assert calls == ["custom-model"]  # the loaded adapter is reused


@pytest.mark.parametrize(
    "name",
    [
        "neighbor_search",
        "graph_model",
        "community_model",
        "representation_model",
    ],
)
def test_missing_component_methods(name, corpus, embeddings):
    with pytest.raises(TypeError, match=name):
        GraphTopic(**{name: object()}).fit(corpus, embeddings)


def test_invalid_encoder(corpus):
    with pytest.raises(TypeError, match="embedding_model"):
        GraphTopic(embedding_model=object()).fit(corpus)
