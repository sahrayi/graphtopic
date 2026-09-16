"""Reusable graph and topic-count-only resolution selection for supplementary runs."""

import numpy as np
from scipy import sparse

from graphtopic import LeidenDetector, NNDescentSearch, UnionMaxGraph
from graphtopic.results import DocumentGraph

from .common import atomic_json, read_json
from .metrics import external_scores
from .run import _cache_key


def get_graph(embeddings, audit, config, cache):
    cache.mkdir(parents=True, exist_ok=True)
    key = _cache_key(audit, config)
    graph_path = cache / f"graph-{key}-k{config['retained_neighbors']}.npz"
    if graph_path.exists() and graph_path.with_suffix(".json").exists():
        return DocumentGraph(
            sparse.load_npz(graph_path),
            range(len(embeddings)),
            read_json(graph_path.with_suffix(".json")),
        )
    path = cache / f"candidates-{key}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            candidates = saved["indices"]
    else:
        ann = config["ann"]
        candidates = np.asarray(
            NNDescentSearch(config["candidate_neighbors"], **ann).search(embeddings), dtype=np.int32
        )
        np.savez_compressed(path, indices=candidates)
    graph = UnionMaxGraph(config["retained_neighbors"]).build(
        embeddings, candidates, document_ids=range(len(embeddings))
    )
    sparse.save_npz(graph_path, graph.adjacency)
    atomic_json(graph_path.with_suffix(".json"), graph.diagnostics)
    return graph


def select_resolution(graph, target, settings, iterations, output):
    # Reference labels are deliberately absent from this function's interface.
    rows = []
    for gamma in settings["resolution_grid"]:
        path = output / "pilot" / f"gamma-{gamma:g}.json"
        if path.exists():
            row = read_json(path)
        else:
            labels = LeidenDetector(
                gamma, random_state=settings["selection_seed"], n_iterations=iterations
            ).fit_predict(graph)
            row = {"resolution": gamma, "topics": int(np.unique(labels).size)}
            atomic_json(path, row)
        rows.append(row)
    best = min(rows, key=lambda row: (abs(row["topics"] - target), row["resolution"]))
    result = {
        "target_topics": target,
        "selected_resolution": best["resolution"],
        "pilot_topics": best["topics"],
        "pilot_seed": settings["selection_seed"],
        "rule": settings["selection_rule"],
        "candidates": rows,
    }
    atomic_json(output / "selection.json", result)
    return best["resolution"]


def evaluate_graph(graph, reference, target, settings, config, output):
    gamma = select_resolution(graph, target, settings, config["leiden_iterations"], output)
    for seed in config["leiden_seeds"]:
        stem = f"topics-{target}-seed-{seed}"
        path = output / "runs" / f"{stem}.json"
        assignment = output / "assignments" / f"{stem}.npz"
        if path.exists() and assignment.exists():
            continue
        labels = LeidenDetector(
            gamma, random_state=seed, n_iterations=config["leiden_iterations"]
        ).fit_predict(graph)
        assignment.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(assignment, labels=labels)
        atomic_json(
            path,
            {
                "method": "GraphTopic",
                "target_topics": target,
                "resolution": gamma,
                "seed": seed,
                **external_scores(reference, labels),
            },
        )
