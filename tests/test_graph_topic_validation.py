import numpy as np
import pytest
from graphion import PartitionSet
from graphtopic.graph_topic import GraphTopic
from graphion.core.errors.models import InvalidPartitionSetError

def test_graph_topic_rejects_invalid_partition_size():

    documents = [
        "اقتصاد",
        "بازار",
        "فوتبال",
    ]

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
        ]
    )


    class BadPartitionDetector:

        def detect(self, graph):

            return PartitionSet.from_sets(
                [
                    (0, 1)
                ]
            )


    model = GraphTopic(
        partition_detector=BadPartitionDetector()
    )


    with pytest.raises(
        InvalidPartitionSetError
    ):
        model.fit(
            documents,
            embeddings=embeddings,
        )
