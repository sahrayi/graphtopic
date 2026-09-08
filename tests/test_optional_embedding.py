"""Opt-in network/model integration tests for the optional embedding extra."""

import os

import pytest
from sklearn.feature_extraction.text import CountVectorizer

from graphtopic import CTFIDFRepresentation, ExactNeighborSearch, GraphTopic

pytestmark = [
    pytest.mark.optional_embedding,
    pytest.mark.skipif(
        os.environ.get("GRAPHTOPIC_TEST_EMBEDDING") != "1",
        reason="set GRAPHTOPIC_TEST_EMBEDDING=1 to download and test the encoder",
    ),
]


def test_real_default_sentence_transformer():
    documents = [
        "Fresh apples and pears are fruit.",
        "The football team scored a goal.",
    ]
    model = GraphTopic(
        n_neighbors=1,
        neighbor_search=ExactNeighborSearch(1),
        representation_model=CTFIDFRepresentation(CountVectorizer()),
    ).fit(documents)
    assert model.embeddings_.shape == (2, 384)
    assert len(model.topics_) == 2
    assert model.result_.metadata["base_fit"]["embedding_source"] == "encoded"
