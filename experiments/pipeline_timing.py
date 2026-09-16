"""Explicit full-pipeline timing; never replace or relabel cached embedding artifacts."""

import argparse
import os
import platform
import random
import time

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from graphtopic import CTFIDFRepresentation, LeidenDetector, NNDescentSearch, UnionMaxGraph
from graphtopic._validation import normalize_embeddings

from .bootstrap import prepare_text
from .common import artifact_root, atomic_json, read_json
from .revision import CONFIG, SETTINGS, freeze_protocol, revision_root
from .scaling import PeakRSS
from .text_cleaning import LEXICAL_CLEANING, clean_lexical_text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("agnews", "dbpedia14"), required=True)
    args = parser.parse_args(argv)
    from .environment import validate_reference_environment

    environment = validate_reference_environment()
    config, settings = read_json(CONFIG), read_json(SETTINGS)
    output = revision_root(settings)
    freeze_protocol(output, settings)
    # prepare_text downloads documents when absent; timing does not first encode
    # the whole corpus as a prerequisite. Existing canonical embeddings are untouched.
    documents, _, metadata = prepare_text(args.dataset, config, artifact_root() / args.dataset)
    import torch
    from sentence_transformers import SentenceTransformer

    seed = settings["timing"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    spec = config["models"]["primary"]
    for repeat in range(settings["timing"]["repeats"]):
        path = output / "timing" / args.dataset / f"repeat-{repeat}.json"
        if path.exists():
            continue
        durations = {}
        with PeakRSS() as memory:
            total = time.perf_counter()
            started = time.perf_counter()
            model = SentenceTransformer(spec["id"], revision=spec["revision"], device="cpu")
            model.max_seq_length = spec["max_seq_length"]
            durations["model_load_seconds"] = time.perf_counter() - started
            started = time.perf_counter()
            raw = model.encode(
                documents,
                batch_size=config["embedding"]["batch_size"],
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
            durations["encoding_seconds"] = time.perf_counter() - started
            del model
            started = time.perf_counter()
            embeddings = normalize_embeddings(raw, len(documents))
            durations["normalization_seconds"] = time.perf_counter() - started
            started = time.perf_counter()
            candidates = NNDescentSearch(
                config["candidate_neighbors"], **config["scaling_ann"]
            ).search(embeddings)
            durations["ann_seconds"] = time.perf_counter() - started
            started = time.perf_counter()
            graph = UnionMaxGraph(config["retained_neighbors"]).build(
                embeddings, candidates, document_ids=range(len(documents))
            )
            durations["graph_seconds"] = time.perf_counter() - started
            started = time.perf_counter()
            labels = LeidenDetector(
                settings["timing"]["resolution"][args.dataset],
                random_state=seed,
                n_iterations=config["leiden_iterations"],
            ).fit_predict(graph)
            durations["leiden_seconds"] = time.perf_counter() - started
            started = time.perf_counter()
            representation = CTFIDFRepresentation(CountVectorizer(**config["vectorizer"]))
            representation.fit([clean_lexical_text(doc) for doc in documents], labels)
            durations["representation_seconds"] = time.perf_counter() - started
            durations["total_seconds"] = time.perf_counter() - total
        atomic_json(
            path,
            {
                "dataset": args.dataset,
                "documents_sha256": metadata["documents_sha256"],
                "documents": len(documents),
                "repeat": repeat,
                "seed": seed,
                "seconds": durations,
                "cache_policy": "no embedding, candidate, or graph reuse inside timed interval",
                "excluded": "source download/read, imports, output serialization",
                "model_download": "loading includes download if absent; inspect before comparing",
                "warmup": "none; repeat 0 includes first-use initialization",
                "lexical_cleaning": LEXICAL_CLEANING,
                "rss_peak_mib": memory.peak / 2**20,
                "platform": platform.platform(),
                "torch_threads": torch.get_num_threads(),
                "ann_workers": config["scaling_ann"]["n_jobs"],
                "thread_environment": {
                    name: value
                    for name, value in os.environ.items()
                    if name.endswith("NUM_THREADS")
                },
                "environment": environment,
            },
        )
        del raw, embeddings, candidates, graph, labels, representation


if __name__ == "__main__":
    main()
