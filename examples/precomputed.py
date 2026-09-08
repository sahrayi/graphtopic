"""Offline example: run with python examples/precomputed.py after installation."""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from graphtopic import CTFIDFRepresentation, ExactNeighborSearch, GraphTopic


def main():
    documents = [
        "apple fruit red",
        "apple fruit green",
        "football goal team",
        "football goal match",
    ]
    embeddings = np.array([[1.0, 0.05], [1.0, 0.1], [0.05, 1.0], [0.1, 1.0]])
    model = GraphTopic(
        embedding_model=None,
        neighbor_search=ExactNeighborSearch(3),
        n_neighbors=1,
        representation_model=CTFIDFRepresentation(CountVectorizer()),
    )
    topics = model.fit_transform(documents, embeddings, document_ids=["a", "b", "c", "d"])
    assert topics[0] == topics[1] and topics[2] == topics[3] and topics[0] != topics[2]
    print(model.get_topic_info().to_string(index=False))
    print(model.resolution_path([0.5, 1.0, 2.0]).get_summary().to_string(index=False))


if __name__ == "__main__":
    main()
