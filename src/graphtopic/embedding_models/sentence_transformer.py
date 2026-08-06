from sentence_transformers import SentenceTransformer
import numpy as np

from graphtopic.core.interfaces.embedding_model import (
    BaseEmbeddingModel
)


class SentenceTransformerEmbedding(
    BaseEmbeddingModel
):

    def __init__(
        self,
        model_name: str,
        normalize_embeddings: bool = True,
    ):
        self.model = SentenceTransformer(
            model_name
        )

        self.normalize_embeddings = (
            normalize_embeddings
        )


    def encode(
        self,
        documents: list[str]
    ) -> np.ndarray:

        embeddings = self.model.encode(
            documents,
            normalize_embeddings=self.normalize_embeddings,
        )

        return np.asarray(
            embeddings
        )