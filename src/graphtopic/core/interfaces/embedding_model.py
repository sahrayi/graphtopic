from abc import ABC, abstractmethod
import numpy as np


class BaseEmbeddingModel(ABC):

    @abstractmethod
    def encode(
        self,
        documents: list[str]
    ) -> np.ndarray:
        """
        Convert documents into embeddings.

        Returns:
            np.ndarray:
                Shape: (n_documents, embedding_dimension)
        """
        pass