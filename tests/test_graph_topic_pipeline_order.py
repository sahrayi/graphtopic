import numpy as np
import pytest

from graphtopic.graph_topic import GraphTopic


def test_graph_topic_pipeline_execution_order():

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


    class DummyReducer:

        def reduce(self, feature_set):

            execution_order.append(
                "reducer"
            )

            return feature_set



    class DummyRelationBuilder:

        def build(self, feature_set):

            execution_order.append(
                "relation_builder"
            )

            return "relations"

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
                pass

            graph = DummyGraph()

            graph.nodes = tuple(nodes)

            return graph


    class DummyGraphRefiner:

        def refine(self, graph):

            execution_order.append(
                "graph_refiner"
            )

            return graph



    class DummyPartitionDetector:

        def detect(self, graph):

            execution_order.append(
                "partition_detector"
            )

            from graphion.core.models import PartitionSet

            return PartitionSet.from_sets(
                [graph.nodes]
            )



    class DummyPartitionRefiner:

        def refine(self, partition):

            execution_order.append(
                "partition_refiner"
            )

            return partition



    model = GraphTopic(

        reducer=DummyReducer(),

        relation_builder=DummyRelationBuilder(),

        graph_builder=DummyGraphBuilder(),

        graph_refiner=DummyGraphRefiner(),

        partition_detector=DummyPartitionDetector(),

        partition_refiner=DummyPartitionRefiner(),

    )


    model.fit(
        documents,
        embeddings=embeddings,
    )


    assert execution_order == [
        "reducer",
        "relation_builder",
        "graph_builder",
        "graph_refiner",
        "partition_detector",
        "partition_refiner",
    ]

def test_graph_topic_rejects_invalid_embedding_model_output_shape():

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

def test_graph_topic_get_topic_info_before_fit():

    model = GraphTopic()

    with pytest.raises(
        ValueError,
        match="Model is not fitted",
    ):
        model.get_topic_info()