"""Optional, lazily loaded Sentence Transformers adapter."""


class SentenceTransformerEmbedding:
    """Encode documents without loading the model until encode is called.

    Parameters are forwarded to SentenceTransformer and its encode method.
    Precomputed embeddings bypass this adapter entirely.
    """

    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        *,
        batch_size=32,
        device=None,
        model_kwargs=None,
        encode_kwargs=None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.model_kwargs = dict(model_kwargs or {})
        self.encode_kwargs = dict(encode_kwargs or {})
        self._model = None

    def encode(self, documents):
        """Return embeddings in input order; GraphTopic performs normalization."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    'Text encoding requires pip install "graphtopic[embedding]". '
                    "Alternatively, pass precomputed embeddings."
                ) from exc
            self._model = SentenceTransformer(
                self.model_name, device=self.device, **self.model_kwargs
            )
        kwargs = {"batch_size": self.batch_size, "show_progress_bar": False}
        kwargs.update(self.encode_kwargs)
        return self._model.encode(list(documents), **kwargs)
