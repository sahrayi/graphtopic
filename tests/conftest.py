import numpy as np
import pytest
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import CountVectorizer

from graphtopic import CTFIDFRepresentation, ExactNeighborSearch, GraphTopic


class ComponentDetector:
    """Deterministic test double that avoids third-party community backends."""

    def __init__(self, resolution=1.0):
        self.resolution = resolution

    def fit_predict(self, graph):
        return connected_components(graph.adjacency, directed=False)[1]

    def with_resolution(self, value):
        return type(self)(value)


@pytest.fixture
def corpus():
    return ["apple fruit red", "apple fruit green", "football goal team", "football goal match"]


@pytest.fixture
def embeddings():
    return np.array([[1, 0.05], [1, 0.1], [0.05, 1], [0.1, 1]], dtype=np.float64)


@pytest.fixture
def model():
    return GraphTopic(
        embedding_model=None,
        n_neighbors=1,
        neighbor_search=ExactNeighborSearch(3),
        community_model=ComponentDetector(),
        representation_model=CTFIDFRepresentation(CountVectorizer()),
    )
