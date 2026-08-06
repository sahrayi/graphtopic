import numpy as np
import pytest

from graphtopic.graph_topic import GraphTopic

from graphion.core.models import (
    Graph,
    PartitionSet,
)


# ======================================================
# Invalid graph output
# ======================================================


def test_graph_topic_rejects_invalid_graph_output():

    documents = [
        "اقتصاد",
        "بازار",
    ]

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
        ],
        dtype=float,
    )


    class BadGraphBuilder:

        def build(
            self,
            relations,
            nodes=None,
        ):

            return None


    model = GraphTopic(
        graph_builder=BadGraphBuilder()
    )


    with pytest.raises(
        ValueError,
        match="Invalid graph",
    ):
        model.fit(
            documents,
            embeddings=embeddings,
        )



# ======================================================
# Invalid partition detector output
# ======================================================


def test_graph_topic_rejects_invalid_partition_output():

    documents = [
        "اقتصاد",
        "بازار",
    ]

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
        ],
        dtype=float,
    )


    class BadPartitionDetector:

        def detect(self, graph):

            return None


    model = GraphTopic(
        partition_detector=BadPartitionDetector()
    )


    with pytest.raises(
        ValueError,
        match="Invalid partition",
    ):
        model.fit(
            documents,
            embeddings=embeddings,
        )



# ======================================================
# Representation model failure propagation
# ======================================================


def test_graph_topic_propagates_representation_error():

    documents = [
        "اقتصاد",
        "بازار",
    ]

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
        ],
        dtype=float,
    )


    class BrokenRepresentation:

        def fit(
            self,
            documents,
            topic_ids,
        ):

            raise RuntimeError(
                "representation failed"
            )


        def get_topic(
            self,
            topic_id,
        ):

            return []



    model = GraphTopic(
        representation_model=BrokenRepresentation()
    )


    with pytest.raises(
        RuntimeError,
        match="representation failed",
    ):
        model.fit(
            documents,
            embeddings=embeddings,
        )



# ======================================================
# Embedding model returns list
# ======================================================


def test_graph_topic_accepts_embedding_model_list_output():

    documents = [
        "اقتصاد",
        "بازار",
        "فوتبال",
    ]


    class ListEmbeddingModel:

        def encode(
            self,
            documents,
        ):

            return [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
            ]



    model = GraphTopic(
        embedding_model=ListEmbeddingModel()
    )


    topics = model.fit_transform(
        documents
    )


    assert len(topics) == len(documents)
    assert model.embeddings_ is not None
    assert model.graph_ is not None
    assert model.topic_info_ is not None



# ======================================================
# Single document
# ======================================================


def test_graph_topic_handles_single_document():

    documents = [
        "خبر واحد",
    ]

    embeddings = np.array(
        [
            [1.0, 0.0],
        ],
        dtype=float,
    )


    model = GraphTopic()


    topics = model.fit_transform(
        documents,
        embeddings=embeddings,
    )


    assert len(topics) == 1
    assert model.topic_info_ is not None
    assert len(model.topic_info_) >= 1
    assert model.topic_info_[0].size == 1