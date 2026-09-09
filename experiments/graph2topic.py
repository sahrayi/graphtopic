"""Run the official Graph2Topic 2.0 implementation in its isolated environment."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)

ROOT = Path(__file__).resolve().parent

LOCKED = {
    "graph2topictm": "2.0",
    "networkx": "3.1",
    "numpy": "1.23.5",
    "pandas": "2.0.0",
    "scikit-learn": "1.2.2",
    "scipy": "1.10.1",
    "sentence-transformers": "2.2.2",
    "umap-learn": "0.5.3",
}
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _environment():
    if tuple(__import__("sys").version_info[:2]) != (3, 9):
        raise RuntimeError("the official Graph2Topic baseline requires CPython 3.9")
    installed = {name: importlib.metadata.version(name) for name in LOCKED}
    mismatched = [
        f"{name}=={expected} (found {installed[name]})"
        for name, expected in LOCKED.items()
        if installed[name] != expected
    ]
    if mismatched:
        raise RuntimeError("Graph2Topic environment mismatch: " + ", ".join(mismatched))
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("set PYTHONHASHSEED=0 before running Graph2Topic")
    invalid_threads = [name for name in THREAD_VARIABLES if os.environ.get(name) != "1"]
    if invalid_threads:
        raise RuntimeError("set these thread variables to 1: " + ", ".join(invalid_threads))
    return {
        "python": platform.python_version(),
        "packages": installed,
        "python_hash_seed": 0,
        "threads": {name: 1 for name in THREAD_VARIABLES},
    }


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _load_artifact(path, config):
    metadata = _read_json(path / "metadata.json")
    expected = {
        "dataset": "20newsgroups",
        "source": config["sources"]["20newsgroups"],
        "model": config["models"]["primary"],
    }
    mismatched = [key for key, value in expected.items() if metadata.get(key) != value]
    if mismatched or not metadata.get("embedding_complete"):
        raise RuntimeError("incompatible artifact: " + ", ".join(mismatched))
    files = {
        "documents": path / "documents.jsonl.gz",
        "labels": path / "labels.npy",
        "embeddings": path / "embeddings.npy",
    }
    for name, file_path in files.items():
        if _sha256(file_path) != metadata[f"{name}_sha256"]:
            raise RuntimeError(f"{name} checksum mismatch")
    with gzip.open(files["documents"], "rt", encoding="utf-8") as stream:
        documents = [json.loads(line) for line in stream]
    labels = np.load(files["labels"], allow_pickle=False)
    embeddings = np.load(files["embeddings"], mmap_mode="r", allow_pickle=False)
    return documents, embeddings, labels, metadata


def _scores(reference, labels):
    return {
        "topics": int(np.unique(labels).size),
        "ari": float(adjusted_rand_score(reference, labels)),
        "nmi": float(normalized_mutual_info_score(reference, labels)),
        "homogeneity": float(homogeneity_score(reference, labels)),
        "completeness": float(completeness_score(reference, labels)),
        "v_measure": float(v_measure_score(reference, labels)),
    }


def _assign_outliers(labels, embeddings):
    complete = labels.copy()
    missing = complete == -1
    if not np.any(missing):
        return complete
    topics = np.unique(complete[~missing])
    if not len(topics):
        raise RuntimeError("Graph2Topic assigned no documents")
    matrix = np.asarray(embeddings, dtype=np.float32)
    centroids = np.vstack([matrix[complete == topic].mean(axis=0) for topic in topics])
    centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
    queries = matrix[missing].copy()
    queries /= np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12)
    complete[missing] = topics[np.argmax(queries @ centroids.T, axis=1)]
    return complete


def _targets(results, resolutions):
    output = []
    for resolution in resolutions:
        paths = sorted(
            (results / "20newsgroups" / "runs").glob(f"gamma-{resolution:g}-seed-*.json")
        )
        if len(paths) != 5:
            raise RuntimeError("complete the five GraphTopic seeds before Graph2Topic")
        values = sorted(_read_json(path)["topics"] for path in paths)
        output.append(int(np.floor(float(np.median(values)) + 0.5)))
    return output


def _official_partition(documents, embeddings, target, seed, settings):
    print("  importing official Graph2Topic graph/community implementation", flush=True)
    # The upstream package imports encoder and evaluation dependencies eagerly even
    # when callers provide embeddings and request no upstream lexical evaluation.
    # Avoid loading those unused legacy stacks; none of these stubs is invoked by
    # the official UMAP/graph/community path exercised below.
    sentence_transformers = types.ModuleType("sentence_transformers")

    class UnusedSentenceTransformer:
        pass

    sentence_transformers.SentenceTransformer = UnusedSentenceTransformer
    sys.modules["sentence_transformers"] = sentence_transformers
    gensim = types.ModuleType("gensim")
    gensim_corpora = types.ModuleType("gensim.corpora")
    gensim_models = types.ModuleType("gensim.models")
    gensim_coherence = types.ModuleType("gensim.models.coherencemodel")

    class UnusedCoherenceModel:
        pass

    gensim_coherence.CoherenceModel = UnusedCoherenceModel
    gensim.corpora = gensim_corpora
    gensim.models = gensim_models
    sys.modules["gensim"] = gensim
    sys.modules["gensim.corpora"] = gensim_corpora
    sys.modules["gensim.models"] = gensim_models
    sys.modules["gensim.models.coherencemodel"] = gensim_coherence
    flair = types.ModuleType("flair")
    flair_embeddings = types.ModuleType("flair.embeddings")

    class UnusedTransformerDocumentEmbeddings:
        pass

    flair_embeddings.TransformerDocumentEmbeddings = UnusedTransformerDocumentEmbeddings
    flair.embeddings = flair_embeddings
    sys.modules["flair"] = flair
    sys.modules["flair.embeddings"] = flair_embeddings
    try:
        from g2t.graph2topic import Graph2Topic
    except ImportError as error:
        raise RuntimeError(
            "run this module in the requirements/graph2topic-reference.txt environment"
        ) from error
    random.seed(seed)
    np.random.seed(seed)
    print("  initializing the official model", flush=True)
    model = Graph2Topic(
        nr_topics=target,
        embedding="shared-all-MiniLM-L6-v2",
        dim_size=settings["dim_size"],
        graph_method=settings["graph_method"],
        seed=seed,
    )
    print("  fitting the official five-dimensional UMAP projection", flush=True)
    reduced = model._reduce_dimensionality(np.asarray(embeddings))
    print("  constructing the official graph and greedy-modularity partition", flush=True)
    frame = pd.DataFrame({"Document": documents, "ID": range(len(documents)), "Topic": None})
    assigned = model._detect_graph_topic(reduced, frame)
    print("  official partition complete", flush=True)
    return np.asarray(
        [-1 if pd.isna(value) else int(value) for value in assigned["Topic"]],
        dtype=np.int32,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "paper.json")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only-target", type=int)
    parser.add_argument("--only-seed", type=int)
    args = parser.parse_args(argv)
    environment = _environment()
    config = _read_json(args.config)
    documents, embeddings, reference, metadata = _load_artifact(args.artifact, config)
    settings = config["graph2topic"]
    targets = _targets(args.results, config["datasets"]["20newsgroups"]["resolutions"])
    if args.only_target is not None:
        if args.only_target not in targets:
            parser.error(f"--only-target must be one of {targets}")
        targets = [args.only_target]
    seeds = config["leiden_seeds"]
    if args.only_seed is not None:
        if args.only_seed not in seeds:
            parser.error(f"--only-seed must be one of {seeds}")
        seeds = [args.only_seed]
    output = args.results / "20newsgroups" / "graph2topic"
    total = len(targets) * len(seeds)
    completed = 0
    for target in targets:
        for seed in seeds:
            stem = f"topics-{target}-seed-{seed}"
            record_path = output / "runs" / (stem + ".json")
            assignment_path = output / "assignments" / (stem + ".npz")
            if record_path.exists() and assignment_path.exists() and not args.force:
                completed += 1
                print(f"[{completed}/{total}] cached {stem}", flush=True)
                continue
            print(f"[{completed + 1}/{total}] fitting official Graph2Topic {stem}")
            native = _official_partition(documents, embeddings, target, seed, settings)
            covered = native != -1
            complete = _assign_outliers(native, embeddings)
            assignment_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(assignment_path, labels=native, reassigned=complete)
            _atomic_json(
                record_path,
                {
                    "method": "Graph2Topic 2.0 (official package)",
                    "target_topics": target,
                    "seed": seed,
                    "coverage": float(covered.mean()),
                    "native": _scores(reference, native),
                    "covered": _scores(reference[covered], native[covered]),
                    "reassigned": _scores(reference, complete),
                    "artifact": {
                        "documents_sha256": metadata["documents_sha256"],
                        "labels_sha256": metadata["labels_sha256"],
                        "embeddings_sha256": metadata["embeddings_sha256"],
                    },
                    "settings": settings,
                    "environment": environment,
                },
            )
            completed += 1
            print(f"[{completed}/{total}] complete {stem}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
