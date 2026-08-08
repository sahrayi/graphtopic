"""
Tests for the GraphTopic feature-processing pipeline.

Architecture under test:

    Documents
        |
        v
    Embedding Model
        |
        v
    Original FeatureSet
        |
        v
      Reducer
        |
        v
    Reduced FeatureSet
        |
        v
  Relation Builder
        |
        v
    RelationSet
        |
        v
   Graph Builder
        |
        v
       Graph
"""

import numpy as np

from graphion.core.models import (
    FeatureSet,
    Graph,
    Relation,
    RelationSet,
)

from graphtopic.graph_topic import GraphTopic


# ============================================================
# Test data
# ============================================================

DOCUMENTS = [
    "اقتصاد",
    "بازار",
    "فوتبال",
    "ورزش",
]

DOCUMENT_IDS = tuple(
    range(len(DOCUMENTS))
)

EMBEDDINGS = np.array(
    [
        [1.0, 0.0, 0.5],
        [0.9, 0.1, 0.4],
        [0.0, 1.0, 0.2],
        [0.1, 0.9, 0.3],
    ],
    dtype=float,
)


# ============================================================
# Helpers
# ============================================================

def make_dummy_relation_set():
    """
    Create a deterministic RelationSet for four documents.
    """

    relations = (
        Relation(
            source=0,
            target=1,
            weight=1.0,
        ),
        Relation(
            source=2,
            target=3,
            weight=1.0,
        ),
    )

    return RelationSet(
        relations=relations,
    )


# ============================================================
# Dummy components
# ============================================================

class DummyReducer:
    """
    Reducer that returns the same FeatureSet.

    This verifies that GraphTopic creates the original
    FeatureSet and passes it to the reducer.
    """

    def __init__(self):
        self.called = False
        self.received_feature_set = None
        self.reduced_feature_set = None

    def reduce(
        self,
        feature_set,
    ):
        self.called = True
        self.received_feature_set = feature_set

        self.reduced_feature_set = feature_set

        return feature_set


class DummyModifiedReducer:
    """
    Reducer that returns a new FeatureSet with a reduced
    feature dimension.

    Original dimension:
        3

    Reduced dimension:
        1
    """

    def __init__(self):
        self.called = False
        self.received_feature_set = None
        self.reduced_feature_set = None

    def reduce(
        self,
        feature_set,
    ):
        self.called = True
        self.received_feature_set = feature_set

        reduced_features = (
            (1.0,),
            (0.9,),
            (0.1,),
            (0.0,),
        )

        self.reduced_feature_set = (
            FeatureSet(
                ids=feature_set.ids,
                features=reduced_features,
            )
        )

        return self.reduced_feature_set


class DummyRelationBuilder:
    """
    Minimal RelationBuilder.

    The RelationBuilder must receive the FeatureSet produced
    by the reducer and return a RelationSet.
    """

    def __init__(
        self,
        relations=None,
    ):
        self.called = False
        self.received_feature_set = None

        self.relations = (
            relations
            if relations is not None
            else make_dummy_relation_set()
        )

    def build(
        self,
        feature_set,
    ):
        self.called = True
        self.received_feature_set = feature_set

        return self.relations


class DummyGraphBuilder:
    """
    Minimal GraphBuilder.

    GraphBuilder receives a RelationSet, not a FeatureSet.
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


# ============================================================
# Reducer
# ============================================================

def test_graph_topic_calls_reducer():
    """
    GraphTopic must invoke the configured reducer.
    """

    reducer = DummyReducer()

    model = GraphTopic(
        reducer=reducer,
        relation_builder=DummyRelationBuilder(),
        graph_builder=DummyGraphBuilder(),
    )

    model.fit(
        DOCUMENTS,
        embeddings=EMBEDDINGS,
    )

    assert reducer.called is True


def test_graph_topic_passes_original_feature_set_to_reducer():
    """
    The FeatureSet created from the original embeddings must
    be passed to the reducer.

    GraphTopic.feature_set_ represents this original FeatureSet.
    """

    reducer = DummyReducer()

    model = GraphTopic(
        reducer=reducer,
        relation_builder=DummyRelationBuilder(),
        graph_builder=DummyGraphBuilder(),
    )

    model.fit(
        DOCUMENTS,
        embeddings=EMBEDDINGS,
    )

    assert model.feature_set_ is not None

    assert (
        reducer.received_feature_set
        is model.feature_set_
    )


def test_graph_topic_reducer_can_return_same_feature_set():
    """
    A reducer is allowed to return the same FeatureSet object.
    """

    reducer = DummyReducer()

    relation_builder = DummyRelationBuilder()
    graph_builder = DummyGraphBuilder()

    model = GraphTopic(
        reducer=reducer,
        relation_builder=relation_builder,
        graph_builder=graph_builder,
    )

    model.fit(
        DOCUMENTS,
        embeddings=EMBEDDINGS,
    )

    assert reducer.called is True

    assert (
        reducer.reduced_feature_set
        is model.feature_set_
    )

    assert (
        relation_builder.received_feature_set
        is model.feature_set_
    )


# ============================================================
# Modified reducer
# ============================================================

def test_graph_topic_reducer_can_change_feature_dimension():
    """
    The reducer may return a new FeatureSet with a different
    feature dimension.

    The original FeatureSet must remain unchanged.
    """

    reducer = DummyModifiedReducer()

    model = GraphTopic(
        reducer=reducer,
        relation_builder=DummyRelationBuilder(),
        graph_builder=DummyGraphBuilder(),
    )

    model.fit(
        DOCUMENTS,
        embeddings=EMBEDDINGS,
    )

    original_feature_set = model.feature_set_

    reduced_feature_set = (
        reducer.reduced_feature_set
    )

    assert original_feature_set is not None

    assert reduced_feature_set is not None

    # Original dimension must remain 3.
    assert (
        len(
            original_feature_set.features[0]
        )
        == 3
    )

    # Reduced dimension must be 1.
    assert (
        len(
            reduced_feature_set.features[0]
        )
        == 1
    )


def test_graph_topic_reducer_preserves_document_ids():
    """
    Feature reduction may change the feature dimension but
    must preserve document IDs.
    """

    reducer = DummyModifiedReducer()

    model = GraphTopic(
        reducer=reducer,
        relation_builder=DummyRelationBuilder(),
        graph_builder=DummyGraphBuilder(),
    )

    model.fit(
        DOCUMENTS,
        embeddings=EMBEDDINGS,
    )

    original_feature_set = model.feature_set_

    reduced_feature_set = (
        reducer.reduced_feature_set
    )

    assert (
        original_feature_set.ids
        == DOCUMENT_IDS
    )

    assert (
        reduced_feature_set.ids
        == DOCUMENT_IDS
    )


def test_graph_topic_feature_reduction_does_not_change_document_count():
    """
    Feature reduction must preserve the number of documents.
    """

    reducer = DummyModifiedReducer()

    model = GraphTopic(
        reducer=reducer,
        relation_builder=DummyRelationBuilder(),
        graph_builder=DummyGraphBuilder(),
    )

    model.fit(
        DOCUMENTS,
        embeddings=EMBEDDINGS,
    )

    reduced_feature_set = (
        reducer.reduced_feature_set
    )

    assert (
        len(reduced_feature_set.features)
        == len(DOCUMENTS)
    )


# ============================================================
# RelationBuilder
# ============================================================

def test_graph_topic_relation_builder_receives_reducer_output():
    """
    RelationBuilder must receive the FeatureSet returned by
    the reducer.

    Expected flow:

        original FeatureSet
                |
                v
             Reducer
                |
                v
        reduced FeatureSet
                |
                v
        RelationBuilder
    """

    reducer = DummyModifiedReducer()

    relation_builder = DummyRelationBuilder()

    model = GraphTopic(
        reducer=reducer,
        relation_builder=relation_builder,
        graph_builder=DummyGraphBuilder(),
    )

    model.fit(
        DOCUMENTS,
        embeddings=EMBEDDINGS,
    )

    assert reducer.reduced_feature_set is not None

    assert (
        relation_builder.received_feature_set
        is reducer.reduced_feature_set
    )


def test_graph_topic_relation_builder_does_not_receive_original_feature_set():
    """
    When the reducer creates a new FeatureSet, the
    RelationBuilder must receive that new FeatureSet.

    It must not receive the original FeatureSet stored in
    GraphTopic.feature_set_.
    """

    reducer = DummyModifiedReducer()

    relation_builder = DummyRelationBuilder()

    model = GraphTopic(
        reducer=reducer,
        relation_builder=relation_builder,
        graph_builder=DummyGraphBuilder(),
    )

    model.fit(
        DOCUMENTS,
        embeddings=EMBEDDINGS,
    )

    assert model.feature_set_ is not None

    assert reducer.reduced_feature_set is not None

    assert (
        relation_builder.received_feature_set
        is not model.feature_set_
    )

    assert (
        relation_builder.received_feature_set
        is reducer.reduced_feature_set
    )


def test_graph_topic_relation_builder_receives_feature_set():
    """
    RelationBuilder must receive a FeatureSet rather than a
    RelationSet or Graph.
    """

    reducer = DummyModifiedReducer()

    relation_builder = DummyRelationBuilder()

    model = GraphTopic(
        reducer=reducer,
        relation_builder=relation_builder,
        graph_builder=DummyGraphBuilder(),
    )

    model.fit(
        DOCUMENTS,
        embeddings=EMBEDDINGS,
    )

    assert isinstance(
        relation_builder.received_feature_set,
        FeatureSet,
    )


def test_graph_topic_stores_relation_builder_output():
    """
    The RelationSet returned by RelationBuilder must be stored
    in GraphTopic.relations_.
    """

    relations = make_dummy_relation_set()

    reducer = DummyModifiedReducer()

    relation_builder = DummyRelationBuilder(
        relations=relations,
    )

    model = GraphTopic(
        reducer=reducer,
        relation_builder=relation_builder,
        graph_builder=DummyGraphBuilder(),
    )

    model.fit(
        DOCUMENTS,
        embeddings=EMBEDDINGS,
    )

    assert model.relations_ is relations

    assert isinstance(
        model.relations_,
        RelationSet,
    )


# ============================================================
# GraphBuilder
# ============================================================

def test_graph_topic_passes_relation_set_to_graph_builder():
    """
    GraphBuilder must receive the RelationSet produced by
    RelationBuilder.
    """

    relations = make_dummy_relation_set()

    reducer = DummyModifiedReducer()

    relation_builder = DummyRelationBuilder(
        relations=relations,
    )

    graph_builder = DummyGraphBuilder()

    model = GraphTopic(
        reducer=reducer,
        relation_builder=relation_builder,
        graph_builder=graph_builder,
    )

    model.fit(
        DOCUMENTS,
        embeddings=EMBEDDINGS,
    )

    assert graph_builder.called is True

    assert (
        graph_builder.received_relations
        is relations
    )


def test_graph_topic_graph_builder_does_not_receive_feature_set():
    """
    GraphBuilder must receive a RelationSet, not a FeatureSet.
    """

    reducer = DummyModifiedReducer()

    relation_builder = DummyRelationBuilder()

    graph_builder = DummyGraphBuilder()

    model = GraphTopic(
        reducer=reducer,
        relation_builder=relation_builder,
        graph_builder=graph_builder,
    )

    model.fit(
        DOCUMENTS,
        embeddings=EMBEDDINGS,
    )

    assert graph_builder.received_relations is not None

    assert isinstance(
        graph_builder.received_relations,
        RelationSet,
    )

    assert not isinstance(
        graph_builder.received_relations,
        FeatureSet,
    )


def test_graph_topic_graph_builder_receives_document_ids():
    """
    GraphBuilder must receive the original document IDs as nodes.
    """

    reducer = DummyModifiedReducer()

    relation_builder = DummyRelationBuilder()

    graph_builder = DummyGraphBuilder()

    model = GraphTopic(
        reducer=reducer,
        relation_builder=relation_builder,
        graph_builder=graph_builder,
    )

    model.fit(
        DOCUMENTS,
        embeddings=EMBEDDINGS,
    )

    assert (
        graph_builder.received_nodes
        == DOCUMENT_IDS
    )


def test_graph_topic_graph_is_built_from_relation_set():
    """
    Graph must be constructed from the RelationSet.
    """

    relations = make_dummy_relation_set()

    reducer = DummyModifiedReducer()

    relation_builder = DummyRelationBuilder(
        relations=relations,
    )

    graph_builder = DummyGraphBuilder()

    model = GraphTopic(
        reducer=reducer,
        relation_builder=relation_builder,
        graph_builder=graph_builder,
    )

    model.fit(
        DOCUMENTS,
        embeddings=EMBEDDINGS,
    )

    assert model.graph_ is not None

    assert (
        graph_builder.received_relations
        is model.relations_
    )


# ============================================================
# Complete pipeline
# ============================================================

def test_graph_topic_feature_to_graph_pipeline():
    """
    Verify the complete feature-to-graph data flow.

    Expected:

        Original FeatureSet
                |
                v
             Reducer
                |
                v
        Reduced FeatureSet
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
    """

    reducer = DummyModifiedReducer()

    relations = make_dummy_relation_set()

    relation_builder = DummyRelationBuilder(
        relations=relations,
    )

    graph_builder = DummyGraphBuilder()

    model = GraphTopic(
        reducer=reducer,
        relation_builder=relation_builder,
        graph_builder=graph_builder,
    )

    model.fit(
        DOCUMENTS,
        embeddings=EMBEDDINGS,
    )

    # --------------------------------------------------------
    # Original FeatureSet
    # --------------------------------------------------------

    assert model.feature_set_ is not None

    assert (
        model.feature_set_.ids
        == DOCUMENT_IDS
    )

    assert (
        len(
            model.feature_set_.features[0]
        )
        == 3
    )

    # --------------------------------------------------------
    # Reducer
    # --------------------------------------------------------

    assert reducer.called is True

    assert (
        reducer.received_feature_set
        is model.feature_set_
    )

    assert (
        reducer.reduced_feature_set
        is not model.feature_set_
    )

    assert (
        len(
            reducer.reduced_feature_set.features[0]
        )
        == 1
    )

    # --------------------------------------------------------
    # RelationBuilder
    # --------------------------------------------------------

    assert relation_builder.called is True

    assert (
        relation_builder.received_feature_set
        is reducer.reduced_feature_set
    )

    assert (
        relation_builder.received_feature_set
        is not model.feature_set_
    )

    # --------------------------------------------------------
    # RelationSet
    # --------------------------------------------------------

    assert model.relations_ is relations

    assert isinstance(
        model.relations_,
        RelationSet,
    )

    # --------------------------------------------------------
    # GraphBuilder
    # --------------------------------------------------------

    assert graph_builder.called is True

    assert (
        graph_builder.received_relations
        is model.relations_
    )

    assert not isinstance(
        graph_builder.received_relations,
        FeatureSet,
    )

    # --------------------------------------------------------
    # Graph
    # --------------------------------------------------------

    assert model.graph_ is not None

    assert (
        model.graph_.node_count
        == len(DOCUMENTS)
    )

    assert (
        graph_builder.received_nodes
        == DOCUMENT_IDS
    )