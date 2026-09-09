"""Matched-granularity baselines from the final paper protocol."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from .common import (
    artifact_root,
    atomic_json,
    load_artifact,
    load_texts,
    read_json,
    result_root,
    validate_protocol_artifact,
)
from .metrics import external_scores
from .progress import Progress, status

ROOT = Path(__file__).resolve().parent


def reference_topics(results, dataset, resolutions):
    """Derive matched baseline granularities from completed GraphTopic seeds."""
    topics = []
    for resolution in resolutions:
        paths = sorted((results / dataset / "runs").glob(f"gamma-{resolution:g}-seed-*.json"))
        if len(paths) != 5:
            raise RuntimeError(
                f"run all five core GraphTopic seeds for {dataset} at "
                f"resolution {resolution:g} before its baselines"
            )
        values = [read_json(path)["topics"] for path in paths]
        topics.append(int(np.floor(np.median(values) + 0.5)))
    return topics


def assign_remaining_outliers(labels, embeddings):
    """Deterministically assign residual -1 labels to nearest cosine centroid."""
    labels = np.asarray(labels, dtype=np.int32).copy()
    missing = labels == -1
    if not np.any(missing):
        return labels
    topic_ids = np.unique(labels[~missing])
    if topic_ids.size == 0:
        raise RuntimeError("cannot assign outliers without at least one non-outlier topic")
    matrix = np.asarray(embeddings, dtype=np.float32)
    centroids = np.vstack([matrix[labels == topic].mean(axis=0) for topic in topic_ids])
    centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
    queries = matrix[missing]
    queries /= np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12)
    labels[missing] = topic_ids[np.argmax(queries @ centroids.T, axis=1)]
    return labels


def repair_saved_bertopic(output, results, dataset, expected, embeddings, reference):
    """Repair completed files without repeating UMAP/HDBSCAN fitting."""
    repaired = 0
    for assignment in expected[::2]:
        with np.load(assignment, allow_pickle=False) as saved:
            labels = saved["labels"]
            reassigned = saved["reassigned"]
        if not np.any(reassigned == -1):
            continue
        reassigned = assign_remaining_outliers(reassigned, embeddings)
        np.savez_compressed(assignment, labels=labels, reassigned=reassigned)
        record_path = output / "runs" / f"{assignment.stem}.json"
        record = read_json(record_path)
        record["reassigned"] = external_scores(reference, reassigned)
        record["residual_outliers_after_reassignment"] = 0
        atomic_json(record_path, record)
        lexical_root = results / "lexical" / dataset / "bertopic" / "assignments"
        lexical = lexical_root / f"{assignment.stem}.json"
        lexical.unlink(missing_ok=True)
        repaired += 1
    return repaired


def run_classical(args):
    status(f"Loading {args.dataset} for classical baselines")
    config = read_json(args.config)
    definition = config["datasets"][args.dataset]
    documents, embeddings, reference, audit = load_artifact(
        args.artifact,
        cache=args.cache / args.dataset / "artifact.npz",
        require_documents=False,
    )
    validate_protocol_artifact(audit, config, args.dataset)
    if not audit["documents_available"]:
        if args.text_csv is None:
            raise ValueError("--text-csv is required when the artifact has no documents")
        documents, text_audit = load_texts(args.text_csv)
        audit["text_source"] = text_audit
    if len(documents) != len(embeddings):
        raise ValueError("text rows do not match embedding rows")
    if len(documents) != definition["documents"]:
        raise ValueError("artifact does not match frozen dataset size")
    vectorizer = CountVectorizer(**config["vectorizer"])
    status("Building shared count matrix")
    counts = vectorizer.fit_transform(documents)
    status("Building shared TF-IDF matrix")
    tfidf = TfidfTransformer(sublinear_tf=True).fit_transform(counts)
    output = args.results / args.dataset / "baselines"
    matched_topics = reference_topics(args.results, args.dataset, definition["resolutions"])
    total = len(matched_topics) * len(config["leiden_seeds"]) * 3
    run_progress = Progress(f"Classical baselines for {args.dataset}", total)
    completed = 0
    for resolution, topics in zip(definition["resolutions"], matched_topics, strict=True):
        for seed in config["leiden_seeds"]:
            for method in ("MiniBatchKMeans", "NMF", "LDA"):
                path = output / "runs" / f"{method}-topics-{topics}-seed-{seed}.json"
                assignment = output / "assignments" / f"{method}-topics-{topics}-seed-{seed}.npz"
                if path.exists() and assignment.exists() and not args.force:
                    completed += 1
                    run_progress.update(completed, "cached")
                    continue
                status(f"Fitting {method}: topics={topics}, seed={seed}")
                started = time.perf_counter()
                if method == "MiniBatchKMeans":
                    labels = MiniBatchKMeans(
                        n_clusters=topics,
                        random_state=seed,
                        batch_size=2048,
                        n_init=10,
                        max_iter=200,
                        reassignment_ratio=0.01,
                    ).fit_predict(embeddings)
                elif method == "NMF":
                    labels = (
                        NMF(
                            n_components=topics,
                            init="nndsvda",
                            random_state=seed,
                            max_iter=200,
                            solver="cd",
                            beta_loss="frobenius",
                        )
                        .fit_transform(tfidf)
                        .argmax(axis=1)
                    )
                else:
                    labels = (
                        LatentDirichletAllocation(
                            n_components=topics,
                            random_state=seed,
                            max_iter=10,
                            learning_method="online",
                            batch_size=2048,
                            learning_offset=50.0,
                            evaluate_every=-1,
                            n_jobs=1,
                        )
                        .fit_transform(counts)
                        .argmax(axis=1)
                    )
                elapsed = time.perf_counter() - started
                assignment.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(assignment, labels=np.asarray(labels, dtype=np.int32))
                atomic_json(
                    path,
                    {
                        "dataset": args.dataset,
                        "method": method,
                        "resolution_reference": resolution,
                        "target_topics": topics,
                        "seed": seed,
                        "seconds": elapsed,
                        **external_scores(reference, labels),
                    },
                )
                completed += 1
                run_progress.update(completed, f"{method} took {elapsed:.2f}s")
    atomic_json(output / "artifact-audit.json", audit)


def run_bertopic(args):
    status(f"Loading {args.dataset} for BERTopic baselines")
    try:
        from bertopic import BERTopic
        from hdbscan import HDBSCAN
        from umap import UMAP
    except ImportError as error:
        raise RuntimeError("install the 'experiments' extra to run BERTopic") from error
    config = read_json(args.config)
    definition = config["datasets"][args.dataset]
    documents, embeddings, reference, audit = load_artifact(
        args.artifact,
        cache=args.cache / args.dataset / "artifact.npz",
        require_documents=False,
    )
    validate_protocol_artifact(audit, config, args.dataset)
    if not any(documents):
        if args.text_csv is None:
            raise ValueError("--text-csv is required when the artifact has no documents")
        documents, _ = load_texts(args.text_csv)
    if len(documents) != len(embeddings):
        raise ValueError("text rows do not match embedding rows")
    targets = sorted(
        reference_topics(args.results, args.dataset, definition["resolutions"]), reverse=True
    )
    output = args.results / args.dataset / "bertopic"
    total = len(config["leiden_seeds"]) * (1 + len(targets))
    run_progress = Progress(f"BERTopic outputs for {args.dataset}", total)
    completed = 0

    def fresh_model(seed):
        return BERTopic(
            embedding_model=None,
            umap_model=UMAP(
                n_neighbors=15,
                n_components=5,
                min_dist=0.0,
                metric="cosine",
                low_memory=True,
                random_state=seed,
            ),
            hdbscan_model=HDBSCAN(
                min_cluster_size=10,
                metric="euclidean",
                cluster_selection_method="eom",
                prediction_data=True,
                core_dist_n_jobs=1,
            ),
            vectorizer_model=CountVectorizer(stop_words="english", min_df=1),
            calculate_probabilities=False,
            verbose=False,
        )

    for seed in config["leiden_seeds"]:
        names = ["natural", *(f"topics-{target}" for target in targets)]
        expected = [
            path
            for name in names
            for path in (
                output / "assignments" / f"{name}-seed-{seed}.npz",
                output / "runs" / f"{name}-seed-{seed}.json",
            )
        ]
        if not args.force and all(path.exists() for path in expected):
            repaired = repair_saved_bertopic(
                output, args.results, args.dataset, expected, embeddings, reference
            )
            if repaired:
                status(f"Repaired {repaired} saved BERTopic assignments for seed={seed}")
            completed += 1 + len(targets)
            run_progress.update(completed, "repaired" if repaired else "cached")
            continue
        status(f"Fitting natural BERTopic partition with seed={seed}")
        model = fresh_model(seed)
        natural, _ = model.fit_transform(documents, embeddings=embeddings)
        _record_bertopic(
            output,
            model,
            documents,
            embeddings,
            reference,
            seed,
            None,
            natural,
            reduction_path="natural",
        )
        completed += 1
        run_progress.update(completed, "natural partition")
        for target in targets:
            status(
                f"Fitting independent natural BERTopic partition for seed={seed}, target={target}"
            )
            target_model = fresh_model(seed)
            target_natural, _ = target_model.fit_transform(documents, embeddings=embeddings)
            if not np.array_equal(np.asarray(natural), np.asarray(target_natural)):
                raise RuntimeError(
                    "BERTopic natural partition changed across identical seeded fits"
                )
            status(f"Reducing BERTopic seed={seed} directly to {target} non-outlier topics")
            target_model.reduce_topics(documents, nr_topics=target + 1)
            _record_bertopic(
                output,
                target_model,
                documents,
                embeddings,
                reference,
                seed,
                target,
                target_model.topics_,
                reduction_path=f"natural_to_{target}",
            )
            completed += 1
            run_progress.update(completed, f"target={target}")


def _record_bertopic(
    output,
    model,
    documents,
    embeddings,
    reference,
    seed,
    target,
    labels,
    *,
    reduction_path,
):
    labels = np.asarray(labels, dtype=np.int32)
    covered = labels != -1
    reassigned = (
        np.asarray(
            model.reduce_outliers(
                documents, labels.tolist(), strategy="embeddings", embeddings=embeddings
            ),
            dtype=np.int32,
        )
        if np.any(~covered)
        else labels.copy()
    )
    reassigned = assign_remaining_outliers(reassigned, embeddings)
    name = "natural" if target is None else f"topics-{target}"
    assignment = output / "assignments" / f"{name}-seed-{seed}.npz"
    assignment.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(assignment, labels=labels, reassigned=reassigned)
    atomic_json(
        output / "runs" / f"{name}-seed-{seed}.json",
        {
            "method": "BERTopic",
            "seed": seed,
            "target_topics": target,
            "reduction_path": reduction_path,
            "coverage": float(covered.mean()),
            "native": external_scores(reference, labels),
            "covered": external_scores(reference[covered], labels[covered]),
            "reassigned": external_scores(reference, reassigned),
            "residual_outliers_after_reassignment": int(np.sum(reassigned == -1)),
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("classical", "bertopic"))
    parser.add_argument("--dataset", choices=("20newsgroups", "agnews"), required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--text-csv", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "paper.json")
    parser.add_argument("--results", type=Path, default=result_root())
    parser.add_argument("--cache", type=Path, default=artifact_root() / "cache")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    (run_classical if args.method == "classical" else run_bertopic)(args)
    print(json.dumps({"completed": args.method, "dataset": args.dataset}))


if __name__ == "__main__":
    main()
