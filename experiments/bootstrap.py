"""Download frozen corpora and create resumable, auditable embedding artifacts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np

from .common import artifact_root, atomic_json, read_json
from .progress import Progress, status

ROOT = Path(__file__).resolve().parent


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_documents(path, documents):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="\n") as stream:
        for document in documents:
            stream.write(json.dumps(document, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _download(dataset, source):
    status(f"Preparing source data for {dataset}")
    if dataset == "20newsgroups":
        from sklearn.datasets import fetch_20newsgroups

        bunch = fetch_20newsgroups(
            subset=source["subset"], remove=tuple(source["remove"]), shuffle=False
        )
        return list(bunch.data), np.asarray(bunch.target, dtype=np.int32), list(bunch.target_names)
    try:
        import pyarrow.parquet as parquet
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError("install requirements/experiments-reference.txt") from error
    documents, labels = [], []
    for filename in source["files"]:
        status(f"Resolving pinned dataset file: {source['id']}/{filename}")
        local_path = hf_hub_download(
            repo_id=source["id"],
            filename=filename,
            repo_type="dataset",
            revision=source["revision"],
        )
        columns = ["text", "label"] if dataset == "agnews" else ["title", "content", "label"]
        parquet_file = parquet.ParquetFile(local_path)
        file_progress = Progress(
            f"Reading {dataset} {Path(filename).name}", parquet_file.metadata.num_rows
        )
        read_rows = 0
        for batch in parquet_file.iter_batches(columns=columns, batch_size=8192):
            for row in batch.to_pylist():
                if dataset == "agnews":
                    text = row["text"]
                else:
                    text = f"{row['title']} {row['content']}"
                documents.append(text)
                labels.append(row["label"])
            read_rows += len(batch)
            file_progress.update(read_rows)
    names = [str(value) for value in sorted(set(labels))]
    return documents, np.asarray(labels, dtype=np.int32), names


def prepare_text(dataset, config, artifact_dir):
    documents_path = artifact_dir / "documents.jsonl.gz"
    labels_path = artifact_dir / "labels.npy"
    metadata_path = artifact_dir / "metadata.json"
    if documents_path.exists() and labels_path.exists() and metadata_path.exists():
        status(f"Validating cached documents and labels for {dataset}")
        metadata = read_json(metadata_path)
        if metadata.get("protocol_id") != config["protocol_id"]:
            raise RuntimeError(f"cached artifact belongs to another protocol: {artifact_dir}")
        if metadata.get("source") != config["sources"][dataset]:
            raise RuntimeError(f"cached artifact uses another dataset source: {artifact_dir}")
        if _sha256(documents_path) != metadata.get("documents_sha256"):
            raise RuntimeError(f"cached document checksum mismatch: {artifact_dir}")
        if _sha256(labels_path) != metadata.get("labels_sha256"):
            raise RuntimeError(f"cached label checksum mismatch: {artifact_dir}")
        with gzip.open(documents_path, "rt", encoding="utf-8") as stream:
            documents = [json.loads(line) for line in stream]
        labels = np.load(labels_path, allow_pickle=False)
        return documents, labels, metadata
    documents, labels, names = _download(dataset, config["sources"][dataset])
    expected = config["datasets"][dataset]
    if len(documents) != expected["documents"] or np.unique(labels).size != expected["classes"]:
        raise RuntimeError(
            f"frozen source mismatch for {dataset}: {len(documents)} rows and "
            f"{np.unique(labels).size} classes"
        )
    _write_documents(documents_path, documents)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    np.save(labels_path, labels, allow_pickle=False)
    metadata = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "dataset": dataset,
        "source": config["sources"][dataset],
        "documents": len(documents),
        "classes": int(np.unique(labels).size),
        "class_names": names,
        "documents_sha256": _sha256(documents_path),
        "labels_sha256": _sha256(labels_path),
        "embedding_complete": False,
        "completed_rows": 0,
    }
    atomic_json(metadata_path, metadata)
    return documents, labels, metadata


def encode(dataset, config, artifact_dir, model_key):
    documents, _, metadata = prepare_text(dataset, config, artifact_dir)
    embedding_path = artifact_dir / "embeddings.npy"
    metadata_path = artifact_dir / "metadata.json"
    dimensions = config["datasets"][dataset]["dimensions"]
    model_spec = config["models"][model_key]
    shape = (len(documents), dimensions)
    if metadata.get("embedding_complete") and embedding_path.exists():
        if metadata.get("model") != model_spec:
            raise RuntimeError(f"cached embeddings use another encoder: {artifact_dir}")
        if np.load(embedding_path, mmap_mode="r").shape == shape:
            status(f"Reusing complete {dataset} embeddings ({len(documents):,} rows)")
            return
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError("install requirements/experiments-reference.txt") from error
    seed = config["ann"]["random_state"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    status(f"Loading pinned encoder for {dataset}: {model_spec['id']}")
    model = SentenceTransformer(
        model_spec["id"], revision=model_spec["revision"], device=config["embedding"]["device"]
    )
    model.max_seq_length = model_spec["max_seq_length"]
    start = int(metadata.get("completed_rows", 0))
    resumable = embedding_path.exists()
    if resumable:
        try:
            resumable = np.load(embedding_path, mmap_mode="r", allow_pickle=False).shape == shape
        except (OSError, ValueError):
            resumable = False
    if not resumable:
        start = 0
        metadata.update(completed_rows=0, embedding_complete=False)
        atomic_json(metadata_path, metadata)
    mode = "r+" if resumable else "w+"
    output = np.lib.format.open_memmap(embedding_path, mode=mode, dtype=np.float32, shape=shape)
    batch_size = config["embedding"]["batch_size"]
    encoding_progress = Progress(f"Embedding {dataset}", len(documents))
    encoding_progress.update(start, "resuming from checkpoint" if start else None)
    for first in range(start, len(documents), batch_size):
        last = min(first + batch_size, len(documents))
        output[first:last] = model.encode(
            documents[first:last],
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype(np.float32, copy=False)
        output.flush()
        metadata.update(completed_rows=last, embedding_complete=False)
        atomic_json(metadata_path, metadata)
        encoding_progress.update(last)
    status(f"Computing SHA-256 for completed {dataset} embeddings")
    metadata.update(
        embedding_complete=True,
        completed_rows=len(documents),
        dimensions=dimensions,
        model=model_spec,
        embeddings_sha256=_sha256(embedding_path),
    )
    atomic_json(metadata_path, metadata)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "paper.json")
    parser.add_argument("--artifacts", type=Path, default=artifact_root())
    parser.add_argument(
        "--only",
        action="append",
        choices=("20newsgroups", "agnews", "dbpedia14", "20newsgroups-alternate"),
        help="prepare only this artifact; repeat to select more than one",
    )
    args = parser.parse_args(argv)
    config = read_json(args.config)
    status("Bootstrap started: datasets, models, and embeddings")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    jobs = {
        "20newsgroups": ("20newsgroups", "primary"),
        "agnews": ("agnews", "primary"),
        "dbpedia14": ("dbpedia14", "primary"),
        "20newsgroups-alternate": ("20newsgroups", "alternate"),
    }
    selected = set(args.only or jobs)
    for artifact_name, (dataset, model_key) in jobs.items():
        if artifact_name in selected:
            encode(dataset, config, args.artifacts / artifact_name, model_key)
    status("Bootstrap complete")
    print(json.dumps({"complete": True, "artifacts": str(args.artifacts.resolve())}))


if __name__ == "__main__":
    raise SystemExit(main())
