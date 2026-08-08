import numpy as np
import pytest

from graphion.core.models import PartitionSet

from graphtopic.graph_topic import GraphTopic


# ==================================================
# Pipeline execution order
# ==================================================

def test_graph_topic_pipeline_execution_order():
    """
    GraphTopic must execute pipeline components in the
    expected order.

    Expected pipeline:

        Reducer
            |
            v
        Relation Builder
            |
            v
        Graph Builder
            |
            v
        Graph Refiner
            |
            v
        Partition Detector
            |
            v
        Partition Refiner
    """

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

    execution_order = []

    # --------------------------------------------------
    # Reducer
    # --------------------------------------------------

    class DummyReducer:
        def reduce(self, feature_set):
            execution_order.append(
                "reducer"
            )

            return feature_set

    # --------------------------------------------------
    # Relation Builder
    # --------------------------------------------------

    class DummyRelationBuilder:
        def build(self, feature_set):
            execution_order.append(
                "relation_builder"
            )

            return "relations"

    # --------------------------------------------------
    # Graph Builder
    # --------------------------------------------------

    class DummyGraphBuilder:
        def build(
            self,
            relations,
            nodes=None,
        ):
            execution_order.append(
                "graph_builder"
            )

            class DummyGraph:
                def __init__(self, nodes):
                    self.nodes = tuple(nodes)
                    self.edges = tuple()

            return DummyGraph(nodes)

    # --------------------------------------------------
    # Graph Refiner
    # --------------------------------------------------

    class DummyGraphRefiner:
        def refine(self, graph):
            execution_order.append(
                "graph_refiner"
            )

            return graph

    # --------------------------------------------------
    # Partition Detector
    # --------------------------------------------------

    class DummyPartitionDetector:
        def detect(self, graph):
            execution_order.append(
                "partition_detector"
            )

            return PartitionSet.from_sets(
                [graph.nodes]
            )

    # --------------------------------------------------
    # Partition Refiner
    # --------------------------------------------------

    class DummyPartitionRefiner:
        def refine(self, partition):
            execution_order.append(
                "partition_refiner"
            )

            return partition

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = GraphTopic(
        reducer=DummyReducer(),
        relation_builder=DummyRelationBuilder(),
        graph_builder=DummyGraphBuilder(),
        graph_refiner=DummyGraphRefiner(),
        partition_detector=DummyPartitionDetector(),
        partition_refiner=DummyPartitionRefiner(),
    )

    # --------------------------------------------------
    # Fit
    # --------------------------------------------------

    model.fit(
        documents,
        embeddings=embeddings,
    )

    # --------------------------------------------------
    # Execution order
    # --------------------------------------------------

    assert execution_order == [
        "reducer",
        "relation_builder",
        "graph_builder",
        "graph_refiner",
        "partition_detector",
        "partition_refiner",
    ]


# ==================================================
# Invalid embedding model output
# ==================================================

def test_graph_topic_rejects_invalid_embedding_model_output_shape():
    """
    GraphTopic must reject embedding model outputs
    that are not two-dimensional matrices.
    """

    class BadEmbeddingModel:
        def encode(self, documents):
            return np.array(
                [
                    1.0,
                    0.5,
                    0.2,
                ]
            )

    model = GraphTopic(
        embedding_model=BadEmbeddingModel()
    )

    with pytest.raises(
        ValueError,
        match="Invalid embedding shape",
    ):
        model.fit(
            [
                "اقتصاد",
                "بازار",
                "فوتبال",
            ]
        )


# ==================================================
# Topic info before fit
# ==================================================

def test_graph_topic_get_topic_info_before_fit():
    """
    get_topic_info() must fail when GraphTopic has not
    been fitted yet.
    """

    model = GraphTopic()

    with pytest.raises(
        ValueError,
        match="Model is not fitted",
    ):
        model.get_topic_info()