"""Metrics fixed by the final paper protocol."""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)


def external_scores(reference, predicted) -> dict[str, float | int]:
    return {
        "topics": int(np.unique(predicted).size),
        "ari": float(adjusted_rand_score(reference, predicted)),
        "nmi": float(normalized_mutual_info_score(reference, predicted)),
        "homogeneity": float(homogeneity_score(reference, predicted)),
        "completeness": float(completeness_score(reference, predicted)),
        "v_measure": float(v_measure_score(reference, predicted)),
    }


def shared_topic_words(documents, labels, *, top_n=10, vectorizer=None):
    """Apply the paper's common smoothed document-IDF topic-word extractor."""
    vectorizer = vectorizer or CountVectorizer(
        stop_words="english", min_df=5, max_df=0.95, max_features=20_000
    )
    counts = vectorizer.fit_transform(documents).tocsr()
    return (
        counts,
        topic_words_from_counts(counts, labels, vectorizer.get_feature_names_out(), top_n=top_n),
        vectorizer,
    )


def topic_words_from_counts(counts, labels, vocabulary, *, top_n=10):
    """Rank topic words from one shared document-term matrix."""
    topic_ids, inverse = np.unique(labels, return_inverse=True)
    grouping = sparse.csr_matrix(
        (np.ones(len(labels)), (inverse, np.arange(len(labels)))),
        shape=(len(topic_ids), len(labels)),
    )
    class_counts = grouping @ counts
    lengths = np.asarray(class_counts.sum(axis=1)).ravel()
    tf = sparse.diags(1.0 / np.maximum(lengths, 1)) @ class_counts
    document_frequency = np.asarray((counts > 0).sum(axis=0)).ravel()
    idf = np.log((1 + counts.shape[0]) / (1 + document_frequency)) + 1
    scores = tf.multiply(idf).tocsr()
    vocabulary = np.asarray(vocabulary)
    words = {}
    for row, topic_id in enumerate(topic_ids):
        values = scores.getrow(row).toarray().ravel()
        order = np.lexsort((vocabulary, -values))[:top_n]
        words[int(topic_id)] = vocabulary[order].tolist()
    return words


def lexical_scores(counts, labels, topic_words, vocabulary) -> dict[str, float]:
    """Compute binary document-level NPMI and topic diversity."""
    binary = (counts > 0).astype(np.uint8).tocsc()
    term_to_id = {term: index for index, term in enumerate(vocabulary)}
    topic_values, topic_sizes = [], []
    for topic_id, words in topic_words.items():
        scores = []
        for left, right in combinations(words, 2):
            a, b = term_to_id[left], term_to_id[right]
            both = binary[:, a].multiply(binary[:, b]).sum()
            if both == 0:
                scores.append(-1.0)
                continue
            pa = binary[:, a].sum() / binary.shape[0]
            pb = binary[:, b].sum() / binary.shape[0]
            pab = both / binary.shape[0]
            scores.append(float(np.log(pab / (pa * pb)) / -np.log(pab)))
        topic_values.append(float(np.mean(scores)))
        topic_sizes.append(int(np.sum(np.asarray(labels) == topic_id)))
    all_words = [word for words in topic_words.values() for word in words]
    return {
        "npmi_macro": float(np.mean(topic_values)),
        "npmi_document_weighted": float(np.average(topic_values, weights=topic_sizes)),
        "topic_diversity": len(set(all_words)) / len(all_words),
    }
