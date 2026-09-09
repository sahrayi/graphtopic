"""Measure exact candidate search on prespecified AG News scaling samples."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from graphtopic import ExactNeighborSearch
from graphtopic._validation import normalize_embeddings

from .common import (
    artifact_root,
    atomic_json,
    load_artifact,
    read_json,
    result_root,
    validate_protocol_artifact,
)
from .progress import Progress, status
from .scaling import PeakRSS

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "paper.json")
    parser.add_argument("--results", type=Path, default=result_root() / "scaling-exact")
    parser.add_argument("--cache", type=Path, default=artifact_root() / "cache")
    args = parser.parse_args(argv)
    config = read_json(args.config)
    documents, embeddings, _, audit = load_artifact(
        args.artifact,
        cache=args.cache / "agnews" / "artifact.npz",
        require_documents=False,
    )
    validate_protocol_artifact(audit, config, "agnews")
    if len(documents) != config["datasets"]["agnews"]["documents"]:
        raise ValueError("artifact does not match frozen AG News size")
    permutation_path = args.cache / "agnews" / "nested-permutation.npy"
    if not permutation_path.exists():
        raise FileNotFoundError("run the AG News scaling stage to freeze its nested sample")
    permutation = np.load(permutation_path, allow_pickle=False)
    sizes = config["scaling"]["exact_comparison_sizes"]
    repeats = config["scaling"]["repeats"]
    progress = Progress("Exact-search scaling runs", len(sizes) * repeats)
    completed = 0
    for size in sizes:
        normalized = normalize_embeddings(embeddings[permutation[:size]], size)
        for repeat in range(1, repeats + 1):
            output = args.results / "agnews" / f"n-{size}-repeat-{repeat}.json"
            if output.exists():
                completed += 1
                progress.update(completed, "cached")
                continue
            status(f"Exact candidate search: n={size:,}, repeat={repeat}/{repeats}")
            with PeakRSS() as memory:
                started = time.perf_counter()
                candidates = ExactNeighborSearch(config["candidate_neighbors"]).search(normalized)
                seconds = time.perf_counter() - started
            if len(candidates) != size:
                raise RuntimeError("exact search returned an incomplete candidate collection")
            atomic_json(
                output,
                {
                    "dataset": "agnews",
                    "method": "exact-block-cosine",
                    "documents": size,
                    "repeat": repeat,
                    "candidate_neighbors": config["candidate_neighbors"],
                    "seconds": seconds,
                    "rss_baseline_mb": memory.baseline / 2**20,
                    "rss_peak_mb": memory.peak / 2**20,
                    "rss_increment_mb": (memory.peak - memory.baseline) / 2**20,
                    "artifact": audit,
                },
            )
            completed += 1
            progress.update(completed, f"{seconds:.2f}s")
    print(json.dumps({"complete": True, "sizes": sizes, "repeats": repeats}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
