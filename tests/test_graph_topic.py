"""
Tests for GraphTopic.

The tests verify the high-level GraphTopic pipeline and its
component injection points.

Architecture under test:

    Documents
        |
        v
    Embedding Model
        |
        v
    FeatureSet
        |
        v
    Feature Reduction
        |
        v
    RelationBuilder
        |
        v
    RelationSet
        |
        v
    GraphBuilder
        |
        v
      Graph
        |
        v
    Graph Refiner
        |
        v
    Partition Detector
        |
        v
    Partition Refiner
        |
        v
    Topic Representation
"""

from __future__ import annotations

import numpy as np
import pytest

from graphion.core.models import (
    Graph,
    PartitionSet,
    RelationSet,
)

from graphtopic.graph_topic import GraphTopic


# ==================================================
# Test data
# ==================================================

DOCS_SHORT = [
    "اقتصاد و بازار",
    "تورم و قیمت",
    "فوتبال و ورزش",
    "مسابقه فوتبال",
]

EMBS_SHORT = np.array(
    [
        [1.0, 0.9, 0.0],
        [0.9, 1.0, 0.0],
        [0.0, 0.1, 1.0],
        [0.1, 0.0, 0.95],
    ],
    dtype=float,
)


# ==================================================
# Dummy components
# ==================================================

class DummyEmbeddingModel:
    """
    Simple deterministic embedding model for tests.
    """

    def __init__(self, embeddings):
        self.embeddings = np.asarray(
            embeddings,
            dtype=float,
        )

        self.called = False
        self.received_documents = None

    def encode(self, documents):
        self.called = True
        self.received_documents = documents

        assert len(documents) == len(
            self.embeddings
        )

        return self.embeddings


class DummyRelationBuilder:
    """
    Minimal RelationBuilder used to isolate GraphTopic.

    The RelationBuilder receives the FeatureSet and is
    responsible for producing a RelationSet.

    GraphTopic must not construct relations itself.
    """

    def __init__(self):
        self.called = False
        self.received_feature_set = None

    def build(self, feature_set):
        self.called = True
        self.received_feature_set = feature_set

        return RelationSet([])


class DummyGraphBuilder:
    """
    Minimal GraphBuilder used to isolate GraphTopic.

    The GraphBuilder receives relations, not FeatureSet.

    GraphTopic must pass the RelationSet produced by the
    RelationBuilder to this component.
    """

    def __init__(self):
        self.called = False
        self.received_relations = None
        self.received_nodes = None

    def build(
        self,
        relations,
        nodes=None,
    ):
        self.called = True
        self.received_relations = relations
        self.received_nodes = nodes

        return Graph(
            nodes=tuple(nodes),
            edges=tuple(),
            directed=False,
        )


class DummyRepresentationModel:
    """
    Simple representation model used for dependency
    injection tests.
    """

    def __init__(self):
        self.fit_called = False
        self.received_documents = None
        self.received_topics = None

    def fit(
        self,
        documents,
        topic_ids,
    ):
        self.fit_called = True

        self.received_documents = documents
        self.received_topics = topic_ids

        return self

    def get_topic(
        self,
        topic_id,
    ):
        return [
            ("custom_word", 1.0),
            ("another_word", 0.5),
        ]


class DummyPartitionDetector:
    """
    Partition detector returning one community.
    """

    def __init__(self):
        self.called = False
        self.received_graph = None

    def detect(self, graph):
        self.called = True
        self.received_graph = graph

        return PartitionSet.from_sets(
            [graph.nodes]
        )


# ==================================================
# Pipeline
# ==================================================

def test_graph_topic_pipeline():
    """
    Test the complete GraphTopic pipeline using explicit
    embeddings and the default GraphTopic components.
    """

    documents = [
        "رئیس جمهور درباره اقتصاد کشور صحبت کرد",
        "دولت برنامه جدید اقتصادی اعلام کرد",
        "وزیر اقتصاد درباره تورم توضیح داد",
        "تیم فوتبال ایران در مسابقه پیروز شد",
        "بازیکنان تیم ملی فوتبال تمرین کردند",
        "مسابقه فوتبال قهرمانی برگزار شد",
    ]

    embeddings = np.array(
        [
            [1.0, 0.9, 0.8, 0.0],
            [0.9, 1.0, 0.85, 0.0],
            [0.95, 0.85, 1.0, 0.0],
            [0.0, 0.0, 0.1, 1.0],
            [0.0, 0.1, 0.0, 0.95],
            [0.1, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    topic_model = GraphTopic()

    topics = topic_model.fit_transform(
        documents,
        embeddings=embeddings,
    )

    # --------------------------------------------------
    # Output
    # --------------------------------------------------

    assert len(topics) == len(documents)

    # --------------------------------------------------
    # Fitted state
    # --------------------------------------------------

    assert topic_model.documents_ == documents

    assert topic_model.document_ids_ == tuple(
        range(len(documents))
    )

    assert topic_model.embeddings_ is not None

    assert topic_model.feature_set_ is not None

    assert topic_model.relations_ is not None

    assert topic_model.graph_ is not None

    assert topic_model.partition_set_ is not None

    assert topic_model.topics_ is not None

    assert topic_model.topic_info_ is not None

    # --------------------------------------------------
    # Graph
    # --------------------------------------------------

    assert len(
        topic_model.graph_.nodes
    ) == len(documents)

    # --------------------------------------------------
    # Relations
    # --------------------------------------------------

    assert len(
        topic_model.relations_
    ) > 0

    # --------------------------------------------------
    # Topics
    # --------------------------------------------------

    assert len(
        topic_model.topic_info_
    ) >= 1

    for topic in topic_model.topic_info_:
        assert topic.topic_id is not None

        assert topic.size > 0

        assert (
            len(topic.document_ids)
            == topic.size
        )

        assert isinstance(
            topic.representation,
            list,
        )


# ==================================================
# RelationBuilder injection
# ==================================================

def test_graph_topic_with_custom_relation_builder():
    """
    GraphTopic should accept a custom RelationBuilder.

    The RelationBuilder must receive the FeatureSet
    produced by GraphTopic.

    GraphTopic must not construct relations itself.
    """

    relation_builder = DummyRelationBuilder()

    topic_model = GraphTopic(
        relation_builder=relation_builder,
    )

    topic_model.fit(
        DOCS_SHORT,
        embeddings=EMBS_SHORT,
    )

    # --------------------------------------------------
    # RelationBuilder was called
    # --------------------------------------------------

    assert relation_builder.called is True

    # --------------------------------------------------
    # RelationBuilder received FeatureSet
    # --------------------------------------------------

    assert (
        relation_builder.received_feature_set
        is not None
    )

    # --------------------------------------------------
    # FeatureSet contains all documents
    # --------------------------------------------------

    assert len(
        relation_builder.received_feature_set
    ) == len(DOCS_SHORT)

    # --------------------------------------------------
    # GraphTopic stores relations
    # --------------------------------------------------

    assert topic_model.relations_ is not None

    assert isinstance(
        topic_model.relations_,
        RelationSet,
    )

    # --------------------------------------------------
    # Graph was still constructed
    # --------------------------------------------------

    assert topic_model.graph_ is not None

    assert (
        topic_model.graph_.node_count
        == len(DOCS_SHORT)
    )


# ==================================================
# GraphBuilder injection
# ==================================================

def test_graph_topic_with_custom_graph_builder():
    """
    GraphTopic should accept a custom GraphBuilder.

    The GraphBuilder must receive the RelationSet
    produced by the RelationBuilder.

    GraphTopic must not pass FeatureSet directly
    to the GraphBuilder.
    """

    graph_builder = DummyGraphBuilder()

    topic_model = GraphTopic(
        graph_builder=graph_builder,
    )

    topic_model.fit(
        DOCS_SHORT,
        embeddings=EMBS_SHORT,
    )

    # --------------------------------------------------
    # GraphBuilder was called
    # --------------------------------------------------

    assert graph_builder.called is True

    # --------------------------------------------------
    # GraphBuilder received relations
    # --------------------------------------------------

    assert (
        graph_builder.received_relations
        is not None
    )

    assert isinstance(
        graph_builder.received_relations,
        RelationSet,
    )

    # --------------------------------------------------
    # GraphBuilder received the same RelationSet
    # stored by GraphTopic
    # --------------------------------------------------

    assert (
        graph_builder.received_relations
        is topic_model.relations_
    )

    # --------------------------------------------------
    # GraphBuilder received document nodes
    # --------------------------------------------------

    assert (
        graph_builder.received_nodes
        == topic_model.document_ids_
    )

    # --------------------------------------------------
    # Graph
    # --------------------------------------------------

    assert topic_model.graph_ is not None

    assert (
        topic_model.graph_.node_count
        == len(DOCS_SHORT)
    )

    assert topic_model.partition_set_ is not None

    assert topic_model.topic_info_ is not None


# ==================================================
# RelationBuilder -> GraphBuilder pipeline
# ==================================================

def test_graph_topic_passes_relations_to_graph_builder():
    """
    Verify the explicit boundary between RelationBuilder
    and GraphBuilder.

    FeatureSet -> RelationBuilder -> RelationSet
    -> GraphBuilder -> Graph
    """

    relation_builder = DummyRelationBuilder()
    graph_builder = DummyGraphBuilder()

    topic_model = GraphTopic(
        relation_builder=relation_builder,
        graph_builder=graph_builder,
    )

    topic_model.fit(
        DOCS_SHORT,
        embeddings=EMBS_SHORT,
    )

    # RelationBuilder receives FeatureSet.
    assert (
        relation_builder.received_feature_set
        is not None
    )

    # RelationBuilder creates RelationSet.
    assert topic_model.relations_ is not None

    assert isinstance(
        topic_model.relations_,
        RelationSet,
    )

    # GraphBuilder receives that exact RelationSet.
    assert (
        graph_builder.received_relations
        is topic_model.relations_
    )

    # GraphBuilder does not receive FeatureSet.
    assert (
        graph_builder.received_relations
        is not relation_builder.received_feature_set
    )


# ==================================================
# Embedding model
# ==================================================

def test_graph_topic_with_custom_embedding_model():
    """
    GraphTopic should use the injected embedding model
    when embeddings are not explicitly provided.
    """

    embedding_model = DummyEmbeddingModel(
        EMBS_SHORT
    )

    topic_model = GraphTopic(
        embedding_model=embedding_model,
    )

    topics = topic_model.fit_transform(
        DOCS_SHORT,
    )

    assert embedding_model.called is True

    assert (
        embedding_model.received_documents
        == DOCS_SHORT
    )

    assert len(topics) == len(DOCS_SHORT)

    assert topic_model.embeddings_ is not None

    assert topic_model.feature_set_ is not None

    assert topic_model.relations_ is not None

    assert topic_model.graph_ is not None

    assert topic_model.partition_set_ is not None

    assert topic_model.topic_info_ is not None


# ==================================================
# Representation model
# ==================================================

def test_graph_topic_with_custom_representation_model():
    """
    GraphTopic should use the injected representation model.
    """

    representation_model = (
        DummyRepresentationModel()
    )

    topic_model = GraphTopic(
        representation_model=representation_model,
    )

    topic_model.fit(
        DOCS_SHORT,
        embeddings=EMBS_SHORT,
    )

    assert representation_model.fit_called is True

    assert (
        representation_model.received_documents
        == DOCS_SHORT
    )

    assert (
        len(
            representation_model.received_topics
        )
        == len(DOCS_SHORT)
    )

    assert topic_model.topic_info_ is not None

    for topic in topic_model.topic_info_:
        assert (
            topic.representation
            == [
                ("custom_word", 1.0),
                ("another_word", 0.5),
            ]
        )


# ==================================================
# Partition detector
# ==================================================

def test_graph_topic_with_custom_partition_detector():
    """
    GraphTopic should use the injected partition detector.
    """

    detector = DummyPartitionDetector()

    topic_model = GraphTopic(
        partition_detector=detector,
    )

    topics = topic_model.fit_transform(
        DOCS_SHORT,
        embeddings=EMBS_SHORT,
    )

    assert detector.called is True

    assert (
        detector.received_graph
        is topic_model.graph_
    )

    assert len(topics) == len(DOCS_SHORT)

    assert len(set(topics)) == 1

    assert topic_model.partition_set_ is not None

    assert len(topic_model.topic_info_) == 1


# ==================================================
# FeatureSet boundary
# ==================================================

def test_graph_topic_relation_builder_receives_feature_set():
    """
    FeatureSet must be passed to RelationBuilder.

    GraphTopic owns the boundary:

        FeatureSet -> RelationBuilder
    """

    relation_builder = DummyRelationBuilder()

    topic_model = GraphTopic(
        relation_builder=relation_builder,
    )

    topic_model.fit(
        DOCS_SHORT,
        embeddings=EMBS_SHORT,
    )

    feature_set = (
        relation_builder.received_feature_set
    )

    assert feature_set is not None

    # FeatureSet must contain all documents.
    assert len(feature_set) == len(
        DOCS_SHORT
    )

    # FeatureSet IDs must match document IDs.
    assert (
        feature_set.ids
        == tuple(range(len(DOCS_SHORT)))
    )

    # GraphTopic must keep the original feature set.
    assert topic_model.feature_set_ is not None

    assert len(
        topic_model.feature_set_
    ) == len(DOCS_SHORT)


# ==================================================
# Embedding validation
# ==================================================

def test_graph_topic_rejects_invalid_embedding_count():
    """
    Number of embeddings must match number of documents.
    """

    topic_model = GraphTopic()

    with pytest.raises(
        ValueError,
        match="Number of embeddings must match",
    ):
        topic_model.fit(
            [
                "خبر اول",
                "خبر دوم",
                "خبر سوم",
            ],
            embeddings=np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            ),
        )


def test_graph_topic_rejects_invalid_embedding_shape():
    """
    Embeddings must be a two-dimensional matrix.
    """

    topic_model = GraphTopic()

    with pytest.raises(
        ValueError,
        match="Invalid embedding shape",
    ):
        topic_model.fit(
            [
                "خبر اول",
                "خبر دوم",
            ],
            embeddings=np.array(
                [1.0, 0.0]
            ),
        )


def test_graph_topic_rejects_zero_embedding_dimension():
    """
    Embedding dimension must not be zero.
    """

    topic_model = GraphTopic()

    with pytest.raises(
        ValueError,
        match="Embedding dimension cannot be zero",
    ):
        topic_model.fit(
            [
                "خبر اول",
                "خبر دوم",
            ],
            embeddings=np.empty(
                (2, 0),
                dtype=float,
            ),
        )


# ==================================================
# Embedding source
# ==================================================

def test_graph_topic_uses_embedding_model():
    """
    Verify that GraphTopic obtains embeddings from the
    configured embedding model.
    """

    embedding_model = DummyEmbeddingModel(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ]
    )

    model = GraphTopic(
        embedding_model=embedding_model,
    )

    topics = model.fit_transform(
        [
            "اقتصاد",
            "بازار",
            "فوتبال",
            "ورزش",
        ]
    )

    assert embedding_model.called is True

    assert len(topics) == 4

    assert model.embeddings_ is not None

    assert model.feature_set_ is not None

    assert model.relations_ is not None

    assert model.graph_ is not None

    assert model.partition_set_ is not None

    assert model.topic_info_ is not None


def test_graph_topic_requires_embedding_source():
    """
    GraphTopic must receive either explicit embeddings
    or an embedding model.
    """

    topic_model = GraphTopic()

    with pytest.raises(
        ValueError,
        match="Either embeddings or embedding_model",
    ):
        topic_model.fit(
            [
                "خبر اول",
                "خبر دوم",
            ]
        )


# ==================================================
# Document validation
# ==================================================

def test_graph_topic_rejects_empty_documents():
    """
    Empty document collections are invalid.
    """

    topic_model = GraphTopic()

    with pytest.raises(
        ValueError,
        match="documents cannot be empty",
    ):
        topic_model.fit(
            [],
            embeddings=np.empty(
                (0, 3),
                dtype=float,
            ),
        )


def test_graph_topic_rejects_non_string_documents():
    """
    All documents must be strings.
    """

    topic_model = GraphTopic()

    with pytest.raises(
        ValueError,
        match="All documents must be strings",
    ):
        topic_model.fit(
            [
                "خبر اول",
                123,
            ],
            embeddings=np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            ),
        )


# ==================================================
# Determinism
# ==================================================

def test_graph_topic_is_deterministic():
    """
    Same documents and embeddings should produce the
    same result.
    """

    documents = [
        "اقتصاد و بازار",
        "تورم و قیمت",
        "فوتبال و ورزش",
        "مسابقه فوتبال",
        "تیم ملی فوتبال",
        "بانک و اقتصاد",
    ]

    embeddings = np.array(
        [
            [1.0, 0.9, 0.0],
            [0.9, 1.0, 0.0],
            [0.0, 0.1, 1.0],
            [0.1, 0.0, 0.95],
            [0.0, 0.0, 0.9],
            [0.95, 0.85, 0.0],
        ],
        dtype=float,
    )

    model1 = GraphTopic()

    topics1 = model1.fit_transform(
        documents,
        embeddings=embeddings,
    )

    model2 = GraphTopic()

    topics2 = model2.fit_transform(
        documents,
        embeddings=embeddings,
    )

    assert topics1 == topics2

    assert (
        model1.topic_info_
        == model2.topic_info_
    )


# ==================================================
# Multiple fits
# ==================================================

def test_graph_topic_can_be_fitted_multiple_times():
    """
    GraphTopic should be reusable.

    A second fit must completely replace the state
    generated by the first fit.
    """

    first_documents = [
        "اقتصاد و بازار",
        "تورم و قیمت",
        "فوتبال و ورزش",
        "مسابقه فوتبال",
    ]

    first_embeddings = np.array(
        [
            [1.0, 0.9, 0.0],
            [0.9, 1.0, 0.0],
            [0.0, 0.1, 1.0],
            [0.1, 0.0, 0.95],
        ],
        dtype=float,
    )

    second_documents = [
        "هواشناسی امروز",
        "پیش بینی باران",
        "فناوری جدید",
    ]

    second_embeddings = np.array(
        [
            [1.0, 0.9, 0.0],
            [0.95, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    model = GraphTopic()

    model.fit(
        first_documents,
        embeddings=first_embeddings,
    )

    first_topic_info = model.topic_info_

    first_relations = model.relations_

    first_graph = model.graph_

    model.fit(
        second_documents,
        embeddings=second_embeddings,
    )

    # --------------------------------------------------
    # Current state
    # --------------------------------------------------

    assert (
        model.documents_
        == second_documents
    )

    assert (
        model.document_ids_
        == tuple(range(len(second_documents)))
    )

    assert (
        len(model.embeddings_)
        == len(second_documents)
    )

    assert (
        len(model.topics_)
        == len(second_documents)
    )

    assert model.feature_set_ is not None

    assert model.relations_ is not None

    assert model.graph_ is not None

    assert model.partition_set_ is not None

    assert model.topic_info_ is not None

    # --------------------------------------------------
    # Previous state must not remain
    # --------------------------------------------------

    assert (
        model.topic_info_
        != first_topic_info
    )

    assert (
        model.relations_
        is not first_relations
    )

    assert (
        model.graph_
        is not first_graph
    )


# ==================================================
# Topic information
# ==================================================

def test_graph_topic_get_topic_info_before_fit():
    """
    get_topic_info must reject access before fitting.
    """

    topic_model = GraphTopic()

    with pytest.raises(
        ValueError,
        match="Model is not fitted",
    ):
        topic_model.get_topic_info()


def test_graph_topic_get_topic_info_after_fit():
    """
    get_topic_info should return the generated topic
    information after fitting.
    """

    topic_model = GraphTopic()

    topic_model.fit(
        DOCS_SHORT,
        embeddings=EMBS_SHORT,
    )

    info = topic_model.get_topic_info()

    assert info is topic_model.topic_info_

    assert isinstance(
        info,
        list,
    )

    assert len(info) >= 1


# ==================================================
# Relation output validation
# ==================================================

def test_graph_topic_rejects_invalid_relation_output():
    """
    GraphTopic must reject an invalid output from
    RelationBuilder.
    """

    class InvalidRelationBuilder:

        def build(self, feature_set):
            return None

    topic_model = GraphTopic(
        relation_builder=InvalidRelationBuilder(),
    )

    with pytest.raises(
        ValueError,
        match="Invalid relation output",
    ):
        topic_model.fit(
            DOCS_SHORT,
            embeddings=EMBS_SHORT,
        )


# ==================================================
# Graph output validation
# ==================================================

def test_graph_topic_rejects_invalid_graph_output():
    """
    GraphTopic must reject an invalid output from
    GraphBuilder.
    """

    class InvalidGraphBuilder:

        def build(
            self,
            relations,
            nodes=None,
        ):
            return None

    topic_model = GraphTopic(
        graph_builder=InvalidGraphBuilder(),
    )

    with pytest.raises(
        ValueError,
        match="Invalid graph returned by graph_builder",
    ):
        topic_model.fit(
            DOCS_SHORT,
            embeddings=EMBS_SHORT,
        )


# ==================================================
# Partition validation
# ==================================================

def test_graph_topic_rejects_invalid_partition_output():
    """
    GraphTopic must reject an invalid partition output.
    """

    class InvalidPartitionDetector:

        def detect(self, graph):
            return None

    topic_model = GraphTopic(
        partition_detector=InvalidPartitionDetector(),
    )

    with pytest.raises(
        ValueError,
        match="Invalid partition output",
    ):
        topic_model.fit(
            DOCS_SHORT,
            embeddings=EMBS_SHORT,
        )