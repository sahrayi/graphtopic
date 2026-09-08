"""Sparse class-based TF-IDF, separate from semantic partitioning."""

import warnings
from copy import deepcopy

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils.validation import check_is_fitted

from ._validation import positive_int


class ClassTfidfTransformer(TransformerMixin, BaseEstimator):
    """Apply the class-based TF-IDF formula used by GraphTopic.

    The input contains raw term counts with one row per class. ``fit`` learns
    the inverse-frequency vector and ``transform`` L1-normalizes each class row
    before weighting it. The implementation relies on SciPy and scikit-learn
    primitives while keeping the paper's exact floating-point mean class length.
    """

    def fit(self, X, y=None):
        counts = self._validate_counts(X)
        lengths = np.asarray(counts.sum(axis=1)).ravel()
        frequency = np.asarray(counts.sum(axis=0)).ravel()
        average = float(lengths.mean()) if len(lengths) else 0.0
        self.idf_ = np.log1p(average / np.maximum(frequency, 1.0))
        self.n_features_in_ = counts.shape[1]
        self.average_class_length_ = average
        return self

    def transform(self, X):
        check_is_fitted(self, ("idf_", "n_features_in_"))
        counts = self._validate_counts(X)
        if counts.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {counts.shape[1]} features, expected {self.n_features_in_}")
        lengths = np.asarray(counts.sum(axis=1)).ravel()
        inverse = np.divide(1.0, lengths, out=np.zeros_like(lengths), where=lengths > 0)
        return (sparse.diags(inverse) @ counts).multiply(self.idf_).tocsr()

    @staticmethod
    def _validate_counts(X):
        if not sparse.issparse(X):
            raise ValueError("ClassTfidfTransformer requires a SciPy sparse count matrix")
        counts = sparse.csr_matrix(X, dtype=np.float64, copy=True)
        if counts.ndim != 2 or not np.isfinite(counts.data).all() or np.any(counts.data < 0):
            raise ValueError("class counts must be a finite non-negative two-dimensional matrix")
        return counts


class CTFIDFRepresentation:
    """Explain communities using L1 class counts times class-based IDF.

    The English paper vocabulary defaults are intentionally conservative. For small
    examples or other languages, supply a configured CountVectorizer explicitly.
    """

    def __init__(self, vectorizer_model=None, ctfidf_model=None, *, top_n_words=10):
        self.vectorizer_model = vectorizer_model
        self.ctfidf_model = ctfidf_model
        self.top_n_words = positive_int(top_n_words, "top_n_words")

    def fit(self, documents, topics):
        docs = tuple(documents)
        labels = np.asarray(topics)
        if (
            not docs
            or any(not isinstance(d, str) for d in docs)
            or labels.shape != (len(docs),)
            or labels.dtype.kind not in "iu"
            or np.any(labels < 0)
        ):
            raise ValueError(
                "representation requires strings and one non-negative topic per document"
            )
        order, inverse = np.unique(labels, return_inverse=True)
        vectorizer = (
            CountVectorizer(stop_words="english", min_df=5, max_df=0.95, max_features=20000)
            if self.vectorizer_model is None
            else deepcopy(self.vectorizer_model)
        )
        empty = False
        try:
            counts = vectorizer.fit_transform(docs)
        except ValueError as exc:
            # Only recognized empty-vocabulary conditions may preserve the partition.
            messages = ("empty vocabulary", "After pruning, no terms remain")
            default_small = (
                self.vectorizer_model is None
                and "max_df corresponds to < documents than min_df" in str(exc)
            )
            if not default_small and not any(message in str(exc) for message in messages):
                raise
            empty = True
            counts = sparse.csr_matrix((len(docs), 0), dtype=np.float64)
        if not sparse.issparse(counts) or counts.shape[0] != len(docs):
            raise ValueError("vectorizer_model must return sparse document-term counts")
        counts = sparse.csr_matrix(counts, dtype=np.float64)
        if not np.isfinite(counts.data).all() or np.any(counts.data < 0):
            raise ValueError("vectorizer counts must be finite and non-negative")
        words = np.asarray([] if empty else vectorizer.get_feature_names_out())
        if len(words) != counts.shape[1]:
            raise ValueError("vectorizer vocabulary and count matrix do not match")
        grouping = sparse.csr_matrix(
            (np.ones(len(docs)), (inverse, np.arange(len(docs)))),
            shape=(len(order), len(docs)),
        )
        class_counts = grouping @ counts
        transformer = (
            ClassTfidfTransformer() if self.ctfidf_model is None else deepcopy(self.ctfidf_model)
        )
        if not callable(getattr(transformer, "fit_transform", None)):
            raise TypeError("ctfidf_model must implement fit_transform(class_counts)")
        scores = transformer.fit_transform(class_counts)
        if not sparse.issparse(scores) or scores.shape != class_counts.shape:
            raise ValueError("ctfidf_model must return a sparse matrix with unchanged shape")
        scores = sparse.csr_matrix(scores, dtype=np.float64)
        if not np.isfinite(scores.data).all() or np.any(scores.data < 0):
            raise ValueError("ctfidf_model scores must be finite and non-negative")
        representations = {}
        for row, topic in enumerate(order):
            begin, end = scores.indptr[row : row + 2]
            indices, values = scores.indices[begin:end], scores.data[begin:end]
            valid = values > 0
            indices, values = indices[valid], values[valid]
            ranking = np.lexsort((words[indices], -values))[: self.top_n_words]
            representations[int(topic)] = [
                (str(words[indices[j]]), float(values[j])) for j in ranking
            ]
        if counts.shape[1] == 0:
            warnings.warn(
                "Vocabulary is empty; topics are preserved with empty word lists",
                UserWarning,
                stacklevel=2,
            )
        self.vectorizer_model_ = vectorizer
        self.ctfidf_model_ = transformer
        self.topic_representations_ = representations
        self.effective_config_ = {
            "formula": "class_tfidf",
            "transformer": f"{type(transformer).__module__}.{type(transformer).__qualname__}",
            "top_n_words": self.top_n_words,
            "vocabulary_size": len(words),
            "vectorizer": (
                vectorizer.get_params(deep=False)
                if hasattr(vectorizer, "get_params")
                else "unknown"
            ),
        }
        return self

    def get_topics(self):
        from sklearn.exceptions import NotFittedError

        if not hasattr(self, "topic_representations_"):
            raise NotFittedError("CTFIDFRepresentation has not been fitted")
        return deepcopy(self.topic_representations_)

    def get_topic(self, topic_id):
        return self.get_topics()[topic_id]
