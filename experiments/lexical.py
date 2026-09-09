"""Run the paper's method-independent lexical audit on saved assignments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from .common import (
    artifact_root,
    atomic_json,
    load_artifact,
    load_texts,
    read_json,
    validate_protocol_artifact,
)
from .metrics import lexical_scores, topic_words_from_counts
from .progress import Progress, status

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--text-csv", type=Path)
    assignments = parser.add_mutually_exclusive_group(required=True)
    assignments.add_argument("--assignment", type=Path)
    assignments.add_argument("--assignment-root", type=Path)
    parser.add_argument("--labels-key", default="labels")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "paper.json")
    parser.add_argument("--cache", type=Path, default=artifact_root() / "lexical-artifact.npz")
    args = parser.parse_args(argv)
    status("Loading documents for lexical audit")
    documents, _, _, audit = load_artifact(args.artifact, cache=args.cache, require_documents=False)
    config = read_json(args.config)
    dataset = audit.get("dataset")
    if dataset not in config["datasets"]:
        raise ValueError("artifact dataset is missing or unknown to the active paper protocol")
    validate_protocol_artifact(audit, config, dataset)
    if not audit["documents_available"]:
        if args.text_csv is None:
            raise ValueError("--text-csv is required when the artifact has no documents")
        documents, text_audit = load_texts(args.text_csv)
        audit["text_source"] = text_audit
    vectorizer_config = config["vectorizer"]
    vectorizer = CountVectorizer(**vectorizer_config)
    status(f"Vectorizing {len(documents):,} documents for the shared lexical audit")
    counts = vectorizer.fit_transform(documents).tocsr()
    vocabulary = vectorizer.get_feature_names_out()
    status(f"Lexical matrix ready with {len(vocabulary):,} terms")
    if args.assignment is not None:
        if args.output is None:
            parser.error("--output is required with --assignment")
        jobs = [(args.assignment, args.labels_key, args.output)]
    else:
        if args.output_root is None:
            parser.error("--output-root is required with --assignment-root")
        jobs = []
        patterns = (
            ("assignments/*.npz", "labels"),
            ("baselines/assignments/*.npz", "labels"),
            ("bertopic/assignments/topics-*.npz", "reassigned"),
            ("graph2topic/assignments/topics-*.npz", "reassigned"),
        )
        for pattern, key in patterns:
            for assignment in sorted(args.assignment_root.glob(pattern)):
                relative = assignment.relative_to(args.assignment_root).with_suffix(".json")
                jobs.append((assignment, key, args.output_root / relative))
    completed = 0
    job_progress = Progress("Lexical assignment audits", len(jobs))
    for position, (assignment, key, output) in enumerate(jobs, start=1):
        if output.exists():
            job_progress.update(position, "cached")
            continue
        status(f"Auditing topic words and NPMI: {assignment.name}")
        with np.load(assignment, allow_pickle=False) as saved:
            labels = saved[key]
        if labels.shape != (len(documents),):
            raise ValueError(f"assignment length does not match the artifact: {assignment}")
        words = topic_words_from_counts(counts, labels, vocabulary)
        scores = lexical_scores(counts, labels, words, vocabulary)
        atomic_json(
            output,
            {
                "artifact": audit,
                "assignment": str(assignment.resolve()),
                "labels_key": key,
                "top_n_words": 10,
                **scores,
                "topic_words": words,
            },
        )
        completed += 1
        job_progress.update(position)
    print(json.dumps({"completed": completed, "total": len(jobs)}, indent=2))


if __name__ == "__main__":
    main()
