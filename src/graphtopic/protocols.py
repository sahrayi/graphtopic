"""Structural component contracts; inheritance is not required."""

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .results import DocumentGraph


class EmbeddingModel(Protocol):
    def encode(self, documents: Sequence[str]) -> NDArray[np.floating]: ...


class NeighborSearch(Protocol):
    def search(self, embeddings: NDArray[np.floating]) -> Sequence[NDArray[np.integer]]: ...


class GraphModel(Protocol):
    def build(
        self,
        embeddings: NDArray[np.floating],
        candidates: Sequence[NDArray[np.integer]],
        *,
        document_ids: Sequence[str | int],
    ) -> DocumentGraph: ...


class CommunityModel(Protocol):
    def fit_predict(self, graph: DocumentGraph) -> Sequence[int]: ...


class RepresentationModel(Protocol):
    def fit(self, documents: Sequence[str], topics: Sequence[int]) -> "RepresentationModel": ...
    def get_topics(self) -> dict[int, list[tuple[str, float]]]: ...
