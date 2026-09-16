"""Small synthetic API check; no corpora, encoders, or benchmark results are downloaded."""

import importlib.metadata

import numpy as np


def main():
    if importlib.metadata.version("tritopic") != "2.3.0":
        raise RuntimeError("Install tritopic==2.3.0 in the separate environment")
    from tritopic import TriTopic, TriTopicConfig
    from tritopic.core.embeddings import EmbeddingEngine

    def forbid_encoding(*args, **kwargs):
        raise AssertionError("TriTopic attempted to encode despite supplied embeddings")

    EmbeddingEngine.encode = forbid_encoding
    rng = np.random.default_rng(1729)
    centers = rng.normal(size=(3, 32))
    embeddings = np.vstack([center + rng.normal(scale=0.03, size=(20, 32)) for center in centers])
    phrases = [
        "football stadium league score team",
        "computer processor hardware memory chip",
        "market finance profit bank stocks",
    ]
    documents = [f"{phrase} report item {i}" for phrase in phrases for i in range(20)]
    config = TriTopicConfig(random_state=11, use_metadata_view=False, n_consensus_runs=10)
    labels = np.asarray(
        TriTopic(config=config, n_topics=3).fit_transform(documents, embeddings=embeddings)
    )
    if labels.shape != (60,):
        raise AssertionError("Unexpected TriTopic output shape")
    print("TriTopic supplied-embedding API passed on 60 synthetic documents.")


if __name__ == "__main__":
    main()
