"""Run the frozen GraphTopic experiments from the final paper protocol.

Example
-------
python -m experiments.run core --dataset 20newsgroups --artifact PATH
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
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
DEFAULT_CONFIG = ROOT / "configs" / "paper.json"
DEFAULT_RESULTS = result_root()
DEFAULT_CACHE = artifact_root() / "cache"


def _recall(scores, selected, exact, cutoff, tie_slots, k):
    chosen = set(selected.tolist())
    strict = len(exact & chosen) / k
    chosen_scores = scores[selected]
    above_hits = int(np.sum(chosen_scores > cutoff))
    tied_hits = int(np.sum(chosen_scores == cutoff))
    return (above_hits + min(tied_hits, tie_slots)) / k, strict


def exact_recall(embeddings, candidates, *, k, query_count, seed, batch_size=128):
    if not query_count:
        return None
    rng = np.random.default_rng(seed)
    query_ids = (
        np.arange(len(embeddings))
        if query_count == "all"
        else np.sort(rng.choice(len(embeddings), min(query_count, len(embeddings)), replace=False))
    )
    candidate_recalls = []
    candidate_strict_recalls = []
    retained_recalls = []
    retained_strict_recalls = []
    tied_queries = 0
    ids = np.arange(len(embeddings))
    recall_progress = Progress("Exact-neighbor recall audit", len(query_ids))
    for start in range(0, len(query_ids), batch_size):
        rows = query_ids[start : start + batch_size]
        scores = np.clip(embeddings[rows] @ embeddings.T, -1, 1)
        scores[np.arange(len(rows)), rows] = -np.inf
        for offset, row_id in enumerate(rows):
            row = scores[offset]
            cutoff = np.partition(row, len(row) - k)[len(row) - k]
            above = ids[row > cutoff]
            all_tied = ids[row == cutoff]
            tie_slots = k - len(above)
            tied = all_tied[:tie_slots]
            exact = set(np.concatenate((above, tied)).tolist())
            candidate_ids = np.unique(np.asarray(candidates[row_id], dtype=np.int64))
            candidate_ids = candidate_ids[candidate_ids != row_id]
            candidate_recall, candidate_strict = _recall(
                row, candidate_ids, exact, cutoff, tie_slots, k
            )
            candidate_scores = np.clip(row[candidate_ids], -1, 1)
            positive = candidate_scores > 0
            retained_ids = candidate_ids[positive]
            retained_scores = candidate_scores[positive]
            order = np.lexsort((retained_ids, -retained_scores))[:k]
            retained_ids = retained_ids[order]
            retained_recall, retained_strict = _recall(
                row, retained_ids, exact, cutoff, tie_slots, k
            )
            candidate_recalls.append(candidate_recall)
            candidate_strict_recalls.append(candidate_strict)
            retained_recalls.append(retained_recall)
            retained_strict_recalls.append(retained_strict)
            tied_queries += int(len(all_tied) > tie_slots)
        recall_progress.update(min(start + len(rows), len(query_ids)))
    candidate_values = np.asarray(candidate_recalls)
    candidate_strict_values = np.asarray(candidate_strict_recalls)
    retained_values = np.asarray(retained_recalls)
    retained_strict_values = np.asarray(retained_strict_recalls)

    def summary(values, strict_values):
        return {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "minimum": float(values.min()),
            "strict_mean": float(strict_values.mean()),
            "strict_minimum": float(strict_values.min()),
        }

    return {
        "k": k,
        "candidate_count": max((len(row) for row in candidates), default=0),
        "queries": len(query_ids),
        "query_seed": seed,
        "tie_policy": "any item tied at the exact kth score is relevant",
        "tied_queries": tied_queries,
        "candidate_recall": summary(candidate_values, candidate_strict_values),
        "retained_recall": summary(retained_values, retained_strict_values),
    }


def _cache_key(audit, config):
    ann = config["ann"]
    return (
        f"{audit['sha256'][:12]}-c{config['candidate_neighbors']}-"
        f"t{ann['n_trees']}-i{ann['n_iters']}-s{ann['random_state']}"
    )


def run_core(args) -> dict:
    status(f"Loading and validating {args.dataset} artifact")
    config = read_json(args.config)
    definition = config["datasets"][args.dataset]
    result_dir = args.results / args.dataset
    cache_dir = args.cache / args.dataset
    migrated = cache_dir / "artifact.npz"
    documents, raw, reference, audit = load_artifact(
        args.artifact, cache=migrated, require_documents=False
    )
    validate_protocol_artifact(audit, config, args.dataset)
    expected_shape = (definition["documents"], definition["dimensions"])
    if raw.shape != expected_shape or np.unique(reference).size != definition["classes"]:
        raise ValueError(
            f"artifact audit differs from frozen protocol: got {raw.shape} and "
            f"{np.unique(reference).size} classes; expected {expected_shape} and "
            f"{definition['classes']} classes"
        )
    atomic_json(result_dir / "artifact-audit.json", audit)
    embeddings = normalize_embeddings(raw, len(documents))
    status(f"Normalized {len(documents):,} embeddings")
    key = _cache_key(audit, config)
    candidate_path = cache_dir / f"candidates-{key}.npz"
    ann = config["ann"]
    search = NNDescentSearch(
        config["candidate_neighbors"],
        random_state=ann["random_state"],
        n_trees=ann["n_trees"],
        n_iters=ann["n_iters"],
        n_jobs=ann["n_jobs"],
    )
    if candidate_path.exists():
        with np.load(candidate_path, allow_pickle=False) as saved:
            candidates = saved["indices"]
        ann_seconds = None
        candidate_source = "cache"
        status("Reused cached ANN candidates")
    else:
        status("Building NNDescent candidate neighborhoods")
        started = time.perf_counter()
        candidates = np.vstack(search.search(embeddings)).astype(np.int32)
        ann_seconds = time.perf_counter() - started
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(candidate_path, indices=candidates)
        candidate_source = "computed"
        status(f"NNDescent candidates complete in {ann_seconds:.2f}s")
    recall = exact_recall(
        embeddings,
        candidates,
        k=config["retained_neighbors"],
        query_count=definition["recall_queries"],
        seed=ann["random_state"],
    )
    atomic_json(
        result_dir / "ann.json",
        {"seconds": ann_seconds, "source": candidate_source, "fidelity_at_20": recall},
    )

    graph_path = cache_dir / f"graph-{key}-k{config['retained_neighbors']}.npz"
    graph_meta_path = graph_path.with_suffix(".json")
    if graph_path.exists() and graph_meta_path.exists():
        graph = DocumentGraph(
            sparse.load_npz(graph_path), range(len(documents)), read_json(graph_meta_path)
        )
        graph_seconds = None
        graph_source = "cache"
        status("Reused cached sparse document graph")
    else:
        status("Re-scoring candidates and building union-max graph")
        builder = UnionMaxGraph(config["retained_neighbors"])
        started = time.perf_counter()
        graph = builder.build(embeddings, candidates, document_ids=range(len(documents)))
        graph_seconds = time.perf_counter() - started
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(graph_path, graph.adjacency)
        atomic_json(graph_meta_path, graph.diagnostics)
        graph_source = "computed"
        status(f"Sparse graph complete in {graph_seconds:.2f}s ({graph.n_edges:,} edges)")
    atomic_json(
        result_dir / "graph.json",
        {**graph.diagnostics, "seconds": graph_seconds, "source": graph_source},
    )

    total_partitions = len(definition["resolutions"]) * len(config["leiden_seeds"])
    partition_progress = Progress(f"Leiden partitions for {args.dataset}", total_partitions)
    completed_partitions = 0
    for resolution in definition["resolutions"]:
        for seed in config["leiden_seeds"]:
            path = result_dir / "runs" / f"gamma-{resolution:g}-seed-{seed}.json"
            assignment = result_dir / "assignments" / f"gamma-{resolution:g}-seed-{seed}.npz"
            if path.exists() and assignment.exists() and not args.force:
                completed_partitions += 1
                partition_progress.update(completed_partitions, "cached")
                continue
            status(f"Running Leiden at gamma={resolution:g}, seed={seed}")
            started = time.perf_counter()
            labels = LeidenDetector(
                resolution,
                random_state=seed,
                n_iterations=config["leiden_iterations"],
            ).fit_predict(graph)
            elapsed = time.perf_counter() - started
            assignment.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(assignment, labels=labels)
            atomic_json(
                path,
                {
                    "dataset": args.dataset,
                    "resolution": resolution,
                    "seed": seed,
                    "seconds": elapsed,
                    **external_scores(reference, labels),
                },
            )
            completed_partitions += 1
            partition_progress.update(completed_partitions)
    summary = aggregate_dataset(result_dir, definition["resolutions"])
    summary["environment"] = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    atomic_json(result_dir / "summary.json", summary)
    status(f"Core experiment complete for {args.dataset}")
    return summary


def aggregate_dataset(result_dir, resolutions):
    output = {"resolutions": {}}
    for resolution in resolutions:
        rows = [
            read_json(path)
            for path in sorted((result_dir / "runs").glob(f"gamma-{resolution:g}-seed-*.json"))
        ]
        if not rows:
            continue
        metrics = ("topics", "ari", "nmi", "homogeneity", "completeness", "v_measure")
        output["resolutions"][str(resolution)] = {
            metric: {
                "mean": float(np.mean([row[metric] for row in rows])),
                "std": float(np.std([row[metric] for row in rows], ddof=1)),
            }
            for metric in metrics
        }
        output["resolutions"][str(resolution)]["runs"] = len(rows)
    return output


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    core = commands.add_parser("core", help="run canonical GraphTopic and ANN validation")
    core.add_argument("--dataset", choices=("20newsgroups", "agnews", "dbpedia14"), required=True)
    core.add_argument("--artifact", type=Path, required=True)
    core.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    core.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    core.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    core.add_argument("--force", action="store_true")
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    print(json.dumps(run_core(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
