"""Select qualitative examples by a fixed, non-cherry-picked rule."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from graphtopic._validation import normalize_embeddings

from .common import (
    atomic_json,
    load_artifact,
    read_json,
    result_root,
    validate_protocol_artifact,
)

ROOT = Path(__file__).resolve().parent


def _excerpt(document, limit):
    cleaned = re.sub(r"\s+", " ", document).strip()
    return cleaned[:limit].rstrip()


def build_examples(dataset, artifact, config_path, results):
    config = read_json(config_path)
    settings = config["qualitative"]
    choice = settings[dataset]
    documents, raw, _, audit = load_artifact(artifact, require_documents=True)
    validate_protocol_artifact(audit, config, dataset)
    embeddings = normalize_embeddings(raw, len(documents))
    stem = f"gamma-{choice['resolution']:g}-seed-{choice['seed']}"
    assignment_path = results / dataset / "assignments" / f"{stem}.npz"
    lexical_path = results / "lexical" / dataset / "assignments" / f"{stem}.json"
    if not assignment_path.exists() or not lexical_path.exists():
        raise FileNotFoundError("run the selected core and lexical stages first")
    with np.load(assignment_path, allow_pickle=False) as saved:
        labels = np.asarray(saved["labels"])
    if labels.shape != (len(documents),):
        raise ValueError("qualitative assignment length does not match the artifact")
    lexical = read_json(lexical_path)
    words = lexical["topic_words"]
    topics, sizes = np.unique(labels, return_counts=True)
    ranking = np.lexsort((topics, -sizes))[: settings["topics_per_dataset"]]
    examples = []
    for position in ranking:
        topic = int(topics[position])
        members = np.flatnonzero(labels == topic)
        centroid = np.asarray(embeddings[members].mean(axis=0), dtype=np.float64)
        norm = np.linalg.norm(centroid)
        if norm == 0:
            representative_order = members
            similarities = np.zeros(len(members))
        else:
            similarities = embeddings[members] @ (centroid / norm)
            representative_order = members[np.lexsort((members, -similarities))]
        representatives = []
        similarity_by_document = dict(zip(members.tolist(), similarities.tolist(), strict=True))
        for document_id in representative_order[: settings["representative_documents"]]:
            representatives.append(
                {
                    "document_id": int(document_id),
                    "centroid_cosine": float(similarity_by_document[int(document_id)]),
                    "excerpt": _excerpt(documents[document_id], settings["excerpt_characters"]),
                }
            )
        examples.append(
            {
                "topic": topic,
                "size": int(sizes[position]),
                "top_words": words[str(topic)][:10],
                "representatives": representatives,
            }
        )
    return {
        "dataset": dataset,
        "selection_rule": "largest topics; ties by topic id",
        "representative_rule": "maximum cosine to normalized topic centroid; ties by row id",
        "resolution": choice["resolution"],
        "seed": choice["seed"],
        "examples": examples,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("20newsgroups", "agnews"), required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "paper.json")
    parser.add_argument("--results", type=Path, default=result_root())
    args = parser.parse_args(argv)
    output = args.results / "qualitative" / f"{args.dataset}.json"
    atomic_json(
        output,
        build_examples(args.dataset, args.artifact, args.config, args.results),
    )
    print(json.dumps({"complete": True, "output": str(output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
