"""Shared input validation without implicit document deletion."""

from collections.abc import Sequence
from numbers import Integral, Real

import numpy as np


def positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def positive_real(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def seed_value(value):
    if value is None:
        return None
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or not 0 <= value <= 2**32 - 1
    ):
        raise ValueError("random_state must be None or an integer in [0, 2**32 - 1]")
    return int(value)


def ordered(value, name):
    if isinstance(value, (str, bytes)) or not isinstance(value, (Sequence, np.ndarray)):
        raise ValueError(f"{name} must be an ordered sequence, not a scalar or iterator")
    if isinstance(value, np.ndarray) and value.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return tuple(value)


def documents_and_ids(documents, document_ids=None):
    docs = ordered(documents, "documents")
    if not docs or any(not isinstance(d, str) for d in docs):
        raise ValueError("documents must contain at least one string and only strings")
    ids = tuple(range(len(docs))) if document_ids is None else ordered(document_ids, "document_ids")
    if len(ids) != len(docs):
        raise ValueError("document_ids must match the number of documents")
    strings = all(isinstance(i, str) for i in ids)
    integers = all(isinstance(i, Integral) and not isinstance(i, (bool, np.bool_)) for i in ids)
    if not (strings or integers) or len(set(ids)) != len(ids):
        raise ValueError("document_ids must be unique, uniformly strings or integers (not bool)")
    return docs, tuple(int(i) for i in ids) if integers else ids


def normalize_embeddings(embeddings, n):
    matrix = np.asarray(embeddings)
    if matrix.ndim != 2 or matrix.shape[0] != n or matrix.shape[1] < 1:
        raise ValueError(f"embeddings must have shape ({n}, d) with d >= 1")
    if matrix.dtype.kind not in "iuf":
        raise ValueError("embeddings must contain real numeric values, not bool/complex/strings")
    matrix = matrix.astype(np.float64, copy=True)
    invalid = np.flatnonzero(~np.isfinite(matrix).all(axis=1))
    if invalid.size:
        raise ValueError(f"embeddings contain non-finite values at rows {invalid[:10].tolist()}")
    scale = np.max(np.abs(matrix), axis=1)
    zero = np.flatnonzero(scale == 0)
    if zero.size:
        raise ValueError(f"embeddings have zero norm at rows {zero[:10].tolist()}")
    matrix /= scale[:, None]
    matrix /= np.linalg.norm(matrix, axis=1)[:, None]
    result = np.array(matrix, dtype=np.float32, order="C")
    result.flags.writeable = False
    return result


def canonical_labels(labels, n):
    array = np.asarray(labels)
    if array.shape != (n,) or array.dtype.kind not in "iu" or np.any(array < 0):
        raise ValueError("community_model must return one non-negative integer label per document")
    unique, first, counts = np.unique(array, return_index=True, return_counts=True)
    order = np.lexsort((first, -counts))
    mapping = {int(unique[index]): i for i, index in enumerate(order)}
    return tuple(mapping[int(label)] for label in array), mapping
