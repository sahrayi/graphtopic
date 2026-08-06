import numpy as np
from graphtopic.graph_topic import GraphTopic
from graphion import Graph

def test_graph_topic_uses_reduced_feature_set():

    documents = [
        "اقتصاد",
        "بازار",
        "فوتبال",
        "ورزش",
    ]

    embeddings = np.array(
        [
            [1.0, 0.0, 0.5],
            [0.9, 0.1, 0.4],
            [0.0, 1.0, 0.2],
            [0.1, 0.9, 0.3],
        ],
        dtype=float,
    )


    state = {
        "reducer_called": False,
        "received_feature_set": None,
    }


    class DummyReducer:

        def reduce(self, feature_set):

            state["reducer_called"] = True

            state["received_feature_set"] = feature_set

            return feature_set



    class DummyRelationBuilder:

        def build(self, feature_set):

            # باید همان خروجی reducer باشد

            assert (
                feature_set
                is state["received_feature_set"]
            )

            return "relations"



    class DummyGraphBuilder:

        def build(
            self,
            relations,
            nodes=None,
        ):

            from graphion.core.models import Graph

            return Graph(
                nodes=tuple(nodes),
                edges=tuple(),
                directed=False,
            )



    model = GraphTopic(
        reducer=DummyReducer(),
        relation_builder=DummyRelationBuilder(),
        graph_builder=DummyGraphBuilder(),
    )


    model.fit(
        documents,
        embeddings=embeddings,
    )


    assert state["reducer_called"] is True
    assert state["received_feature_set"] is not None

def test_graph_topic_uses_modified_feature_set_from_reducer():

    documents = [
        "اقتصاد",
        "بازار",
        "فوتبال",
        "ورزش",
    ]

    embeddings = np.array(
        [
            [1.0, 0.0, 0.5],
            [0.9, 0.1, 0.4],
            [0.0, 1.0, 0.2],
            [0.1, 0.9, 0.3],
        ],
        dtype=float,
    )


    state = {
        "relation_received": None,
    }


    class DummyReducer:

        def reduce(self, feature_set):

            from graphion.core.models import FeatureSet

            reduced_matrix = np.array(
                [
                    [1.0],
                    [0.9],
                    [0.1],
                    [0.0],
                ],
                dtype=float,
            )

            return FeatureSet.from_numpy(
                ids=feature_set.ids,
                matrix=reduced_matrix,
            )


    class DummyRelationBuilder:

        def build(self, feature_set):

            state["relation_received"] = feature_set

            return "relations"



    class DummyGraphBuilder:

        def build(
            self,
            relations,
            nodes=None,
        ):

            return Graph(
                nodes=tuple(nodes),
                edges=tuple(),
                directed=False,
            )



    model = GraphTopic(
        reducer=DummyReducer(),
        relation_builder=DummyRelationBuilder(),
        graph_builder=DummyGraphBuilder(),
    )


    model.fit(
        documents,
        embeddings=embeddings,
    )


    assert state["relation_received"] is not None

    assert (
        state["relation_received"]
        is not model.feature_set_
    )