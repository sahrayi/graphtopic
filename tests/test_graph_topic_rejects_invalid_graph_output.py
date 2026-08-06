import numpy as np
import pytest
from graphtopic.graph_topic import GraphTopic

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