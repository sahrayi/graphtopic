"""Measure the frozen post-embedding scaling protocol."""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path

import numpy as np

from graphtopic import LeidenDetector, NNDescentSearch, UnionMaxGraph
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

ROOT = Path(__file__).resolve().parent


class PeakRSS:
    """Sample process RSS at the 10 ms interval used in the paper."""

    def __enter__(self):
        try:
            import psutil
        except ImportError as error:
            raise RuntimeError("install the 'experiments' extra to measure RSS") from error
        self._process = psutil.Process(os.getpid())
        self.baseline = self._process.memory_info().rss
        self.peak = self.baseline
        self._stop = threading.Event()

        def sample():
            while not self._stop.wait(0.01):
                self.peak = max(self.peak, self._process.memory_info().rss)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self._process.memory_info().rss)


def measure(embeddings, config, *, resolution, seed):
    status(f"Normalizing {len(embeddings):,} embeddings for scaling measurement")
    normalized = normalize_embeddings(embeddings, len(embeddings))
    ann = config["scaling_ann"]
    with PeakRSS() as memory:
        total_started = time.perf_counter()
        started = time.perf_counter()
        status("Scaling measurement: NNDescent candidate search")
        candidates = NNDescentSearch(
            config["candidate_neighbors"],
            random_state=ann["random_state"],
            n_trees=ann["n_trees"],
            n_iters=ann["n_iters"],
            n_jobs=ann["n_jobs"],
        ).search(normalized)
        ann_seconds = time.perf_counter() - started
        status(f"Scaling measurement: ANN complete in {ann_seconds:.2f}s")
        started = time.perf_counter()
        status("Scaling measurement: union-max graph construction")
        graph = UnionMaxGraph(config["retained_neighbors"]).build(
            normalized, candidates, document_ids=range(len(normalized))
        )
        graph_seconds = time.perf_counter() - started
        status(f"Scaling measurement: graph complete in {graph_seconds:.2f}s")
        started = time.perf_counter()
        status(f"Scaling measurement: Leiden gamma={resolution:g}, seed={seed}")
        labels = LeidenDetector(
            resolution,
            random_state=seed,
            n_iterations=config["leiden_iterations"],
        ).fit_predict(graph)
        leiden_seconds = time.perf_counter() - started
        status(f"Scaling measurement: Leiden complete in {leiden_seconds:.2f}s")
        total_seconds = time.perf_counter() - total_started
    return {
        "documents": len(normalized),
        "edges": graph.n_edges,
        "topics": int(np.unique(labels).size),
        "isolated_vertices": graph.diagnostics["isolated_vertices"],
        "connected_components": graph.diagnostics["connected_components"],
        "ann_seconds": ann_seconds,
        "graph_seconds": graph_seconds,
        "leiden_seconds": leiden_seconds,
        "total_seconds": total_seconds,
        "rss_baseline_mb": memory.baseline / 2**20,
        "rss_peak_mb": memory.peak / 2**20,
        "rss_increment_mb": (memory.peak - memory.baseline) / 2**20,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("agnews", "dbpedia14"), required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "paper.json")
    parser.add_argument("--results", type=Path, default=result_root() / "scaling")
    parser.add_argument("--cache", type=Path, default=artifact_root() / "cache")
    parser.add_argument("--warmup", action="store_true")
    args = parser.parse_args(argv)
    config = read_json(args.config)
    status(f"Loading {args.dataset} scaling artifact")
    documents, embeddings, _, audit = load_artifact(
        args.artifact,
        cache=args.cache / args.dataset / "artifact.npz",
        require_documents=False,
    )
    validate_protocol_artifact(audit, config, args.dataset)
    definition = config["datasets"][args.dataset]
    if len(documents) != definition["documents"]:
        raise ValueError("artifact does not match the frozen dataset size")
    if args.dataset == "agnews":
        sizes = config["scaling"]["agnews_sizes"]
        permutation_path = args.cache / args.dataset / "nested-permutation.npy"
        if permutation_path.exists():
            permutation = np.load(permutation_path)
        else:
            permutation = np.random.default_rng(config["ann"]["random_state"]).permutation(
                len(embeddings)
            )
            permutation_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(permutation_path, permutation)
        resolution = 0.1
    else:
        sizes = [len(embeddings)]
        permutation = np.arange(len(embeddings))
        resolution = 0.2
    repeats = config["scaling"]["repeats"]
    if args.warmup:
        status("Starting unreported 10,000-document warm-up")
        measure(
            embeddings[permutation[: min(10_000, len(embeddings))]],
            config,
            resolution=resolution,
            seed=11,
        )
        status("Warm-up complete")
    run_progress = Progress(f"Scaling runs for {args.dataset}", len(sizes) * repeats)
    completed = 0
    for size in sizes:
        sample = embeddings[permutation[:size]]
        for repeat in range(1, repeats + 1):
            output = args.results / args.dataset / f"n-{size}-repeat-{repeat}.json"
            if output.exists():
                completed += 1
                run_progress.update(completed, "cached")
                continue
            status(f"Starting measured run: n={size:,}, repeat={repeat}/{repeats}")
            atomic_json(
                output,
                {
                    "dataset": args.dataset,
                    "repeat": repeat,
                    "artifact": audit,
                    **measure(sample, config, resolution=resolution, seed=11),
                },
            )
            completed += 1
            run_progress.update(completed)
    print(json.dumps({"completed": args.dataset, "sizes": sizes, "repeats": repeats}))


if __name__ == "__main__":
    main()
