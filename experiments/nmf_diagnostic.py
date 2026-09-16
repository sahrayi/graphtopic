"""Audit saved NMF words and topic sizes without fitting a model or loading embeddings."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from .common import atomic_json, read_json
from .metrics import lexical_scores


def audit(documents_path, assignment_root, lexical_root):
    records = sorted(lexical_root.glob("NMF-topics-*.json"))
    if not records:
        raise FileNotFoundError("No saved NMF lexical records")
    vocabulary = sorted(
        {
            word
            for path in records
            for words in read_json(path)["topic_words"].values()
            for word in words
        }
    )
    vectorizer = CountVectorizer(vocabulary=vocabulary, stop_words="english", binary=True)
    # Only the union of saved top words is needed for exactly the same document
    # frequency/co-occurrence calculation; no full 20,000-term matrix is built.
    with gzip.open(documents_path, "rt", encoding="utf-8") as stream:
        counts = vectorizer.transform(json.loads(line) for line in stream)
    counts = sparse.csr_matrix(counts)
    outputs = []
    for path in records:
        original = read_json(path)
        with np.load(assignment_root / f"{path.stem}.npz", allow_pickle=False) as saved:
            labels = saved["labels"]
        if len(labels) != counts.shape[0]:
            raise ValueError("Assignment/document length mismatch")
        details = lexical_scores(counts, labels, original["topic_words"], vocabulary, details=True)
        for metric in ("npmi_macro", "npmi_document_weighted", "topic_diversity"):
            if not np.isclose(details[metric], original[metric], atol=1e-12, rtol=0):
                raise ValueError(f"Saved metric mismatch: {path.name}: {metric}")
        outputs.append({"run": path.stem, **details, "topic_words": original["topic_words"]})
    return {
        "verified": True,
        "documents": counts.shape[0],
        "audited_terms": len(vocabulary),
        "runs": outputs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--lexical", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.documents, args.assignments, args.lexical)
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
