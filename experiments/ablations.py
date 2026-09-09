"""Frozen neighborhood, resolution, graph-design, and encoder ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from graphtopic import LeidenDetector, NNDescentSearch, UnionMaxGraph
from graphtopic._validation import normalize_embeddings
from graphtopic.results import DocumentGraph

from .common import (
    artifact_root,
    atomic_json,
    load_artifact,
    read_json,
    result_root,
    validate_protocol_artifact,
)
from .metrics import external_scores
from .progress import Progress, status

ROOT = Path(__file__).resolve().parent


def directed_matrix(embeddings, candidates, k):
    n = len(embeddings)
    row_progress = Progress("Directed graph rows", n)
    indptr = np.arange(0, (n + 1) * k, k, dtype=np.int64)
    indices = np.empty(n * k, dtype=np.int32)
    weights = np.empty(n * k, dtype=np.float32)
    for row in range(n):
        candidate = np.unique(np.asarray(candidates[row], dtype=np.int64))
        candidate = candidate[candidate != row]
        scores = embeddings[candidate] @ embeddings[row]
        order = np.lexsort((candidate, -scores))
        selected = candidate[order][scores[order] > 0][:k]
        if len(selected) != k:
            raise RuntimeError(f"row {row} has fewer than {k} positive neighbors")
        start = row * k
        indices[start : start + k] = selected
        weights[start : start + k] = embeddings[selected] @ embeddings[row]
        row_progress.update(row + 1)
    return sparse.csr_matrix((weights, indices, indptr), shape=(n, n))


def variant_graph(directed, name):
    if name == "union_max":
        return directed.maximum(directed.T).tocsr()
    if name == "mutual":
        pattern = directed.sign()
        mutual = directed.multiply(pattern.T)
        return mutual.maximum(mutual.T).tocsr()
    if name == "directed":
        return directed
    raise ValueError(f"unknown graph variant: {name}")


def partition(matrix, *, resolution, seed, directed):
    if not directed:
        graph = DocumentGraph(matrix, range(matrix.shape[0]))
        return LeidenDetector(resolution, random_state=seed, n_iterations=2).fit_predict(graph)
    import igraph as ig
    import leidenalg

    coo = matrix.tocoo()
    backend = ig.Graph(
        n=matrix.shape[0],
        edges=list(zip(coo.row.tolist(), coo.col.tolist(), strict=True)),
        directed=True,
    )
    result = leidenalg.find_partition(
        backend,
        leidenalg.RBConfigurationVertexPartition,
        weights=coo.data,
        resolution_parameter=resolution,
        n_iterations=2,
        seed=seed,
    )
    return np.asarray(result.membership, dtype=np.int32)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("k-resolution", "graph-design", "encoder"))
    parser.add_argument("--dataset", choices=("20newsgroups", "agnews"), required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--encoder-label", default="all-MiniLM-L6-v2")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "paper.json")
    parser.add_argument("--results", type=Path, default=result_root() / "ablations")
    parser.add_argument("--cache", type=Path, default=artifact_root() / "cache")
    args = parser.parse_args(argv)
    config = read_json(args.config)
    status(f"Loading artifact for {args.experiment} on {args.dataset}")
    documents, raw, reference, audit = load_artifact(
        args.artifact,
        cache=args.cache / args.dataset / f"{args.encoder_label}-artifact.npz",
        require_documents=False,
    )
    model_key = "alternate" if args.experiment == "encoder" else "primary"
    validate_protocol_artifact(audit, config, args.dataset, model_key=model_key)
    embeddings = normalize_embeddings(raw, len(documents))
    ann = config["ann"]
    status("Building shared NNDescent candidates")
    candidates = NNDescentSearch(
        config["candidate_neighbors"],
        random_state=ann["random_state"],
        n_trees=ann["n_trees"],
        n_iters=ann["n_iters"],
        n_jobs=ann["n_jobs"],
    ).search(embeddings)
    status("Shared ANN candidates complete")
    output = args.results / args.experiment / args.dataset / args.encoder_label
    seeds = config["leiden_seeds"]
    if args.experiment in {"k-resolution", "encoder"}:
        k_values = (
            config["ablations"]["k_values"]
            if args.experiment == "k-resolution"
            else [config["retained_neighbors"]]
        )
        resolutions = (
            config["ablations"]["resolutions"]
            if args.experiment == "k-resolution"
            else config["datasets"][args.dataset]["resolutions"]
        )
        total = len(k_values) * len(resolutions) * len(seeds)
        run_progress = Progress(f"{args.experiment} runs", total)
        completed = 0
        for k in k_values:
            status(f"Building union-max graph with k={k}")
            graph = UnionMaxGraph(k).build(
                embeddings, candidates, document_ids=range(len(documents))
            )
            for resolution in resolutions:
                for seed in seeds:
                    path = output / f"k-{k}-gamma-{resolution:g}-seed-{seed}.json"
                    if path.exists():
                        completed += 1
                        run_progress.update(completed, "cached")
                        continue
                    status(f"Partitioning k={k}, gamma={resolution:g}, seed={seed}")
                    labels = LeidenDetector(
                        resolution, random_state=seed, n_iterations=config["leiden_iterations"]
                    ).fit_predict(graph)
                    atomic_json(
                        path,
                        {
                            "k": k,
                            "resolution": resolution,
                            "seed": seed,
                            "graph": graph.diagnostics,
                            **external_scores(reference, labels),
                        },
                    )
                    completed += 1
                    run_progress.update(completed)
    else:
        k = config["ablations"]["graph_design_k"][args.dataset]
        status(f"Building directed k-nearest-neighbor matrix with k={k}")
        directed = directed_matrix(embeddings, candidates, k)
        total = (
            len(config["ablations"]["graph_variants"])
            * len(config["datasets"][args.dataset]["resolutions"])
            * len(seeds)
        )
        run_progress = Progress("graph-design runs", total)
        completed = 0
        for variant in config["ablations"]["graph_variants"]:
            status(f"Preparing graph variant: {variant}")
            matrix = variant_graph(directed, variant)
            for resolution in config["datasets"][args.dataset]["resolutions"]:
                for seed in seeds:
                    path = output / f"{variant}-gamma-{resolution:g}-seed-{seed}.json"
                    if path.exists():
                        completed += 1
                        run_progress.update(completed, "cached")
                        continue
                    status(f"Partitioning {variant}, gamma={resolution:g}, seed={seed}")
                    labels = partition(
                        matrix,
                        resolution=resolution,
                        seed=seed,
                        directed=variant == "directed",
                    )
                    atomic_json(
                        path,
                        {
                            "variant": variant,
                            "k": k,
                            "resolution": resolution,
                            "seed": seed,
                            "edges": int(matrix.nnz if variant == "directed" else matrix.nnz // 2),
                            **external_scores(reference, labels),
                        },
                    )
                    completed += 1
                    run_progress.update(completed)
    atomic_json(output / "artifact-audit.json", audit)
    status(f"Completed {args.experiment} on {args.dataset}")
    print(json.dumps({"completed": args.experiment, "dataset": args.dataset}))


if __name__ == "__main__":
    main()
