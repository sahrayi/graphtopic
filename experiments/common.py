"""Artifact loading, provenance, and durable output helpers."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .progress import Progress

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "paper.json"


def protocol_id(config_path: Path = DEFAULT_CONFIG) -> str:
    """Return a filesystem-safe identifier for the active frozen protocol."""
    value = json.loads(config_path.read_text(encoding="utf-8"))["protocol_id"]
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789-_"
    if not value or any(character not in allowed for character in value):
        raise ValueError("protocol_id must contain only lowercase letters, digits, '-' and '_'")
    return value


def artifact_root(config_path: Path = DEFAULT_CONFIG) -> Path:
    return ROOT / "artifacts" / protocol_id(config_path)


def result_root(config_path: Path = DEFAULT_CONFIG) -> Path:
    return ROOT / "results" / protocol_id(config_path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    size = path.stat().st_size
    progress = Progress(f"Checksumming {path.name}", size) if size >= 10 * 1024 * 1024 else None
    completed = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
            completed += len(block)
            if progress is not None:
                progress.update(completed)
    return digest.hexdigest()


def parse_embedding(value: str) -> np.ndarray:
    cleaned = value.strip().strip("[]")
    vector = np.fromstring(cleaned.replace(",", " "), dtype=np.float32, sep=" ")
    if vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError("invalid embedding value")
    return vector


def load_texts(path: Path) -> tuple[list[str], dict]:
    """Load text from a bootstrap JSONL gzip or an archived CSV."""
    path = path.resolve()
    if path.name.endswith(".jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            texts = [json.loads(line) for line in stream]
        return texts, {"path": str(path), "sha256": file_sha256(path), "documents": len(texts)}
    texts = (
        pd.read_csv(path, usecols=["text"], dtype={"text": str})["text"]
        .fillna("")
        .astype(str)
        .tolist()
    )
    return texts, {"path": str(path), "sha256": file_sha256(path), "documents": len(texts)}


def load_artifact(path: Path, *, cache: Path | None = None, require_documents: bool = True):
    """Load the frozen NPZ format or migrate the archived paper CSV format.

    The returned tuple is ``documents, embeddings, labels, audit``. A CSV migration
    is chunked and may write a local compressed cache outside version control.
    """
    path = path.resolve()
    if path.is_dir():
        return _from_directory(path)
    source_hash = file_sha256(path)
    text_cache = cache.with_suffix(".documents.jsonl.gz") if cache is not None else None
    if cache is not None and cache.exists() and text_cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            if str(data["source_sha256"]) == source_hash:
                with gzip.open(text_cache, "rt", encoding="utf-8") as stream:
                    documents = [json.loads(line) for line in stream]
                return _validate(
                    documents,
                    data["embeddings"],
                    data["labels"],
                    path,
                    source_hash,
                    "cache",
                )
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            return _from_npz(data, path, source_hash, "npz", require_documents)
    if path.suffix.lower() != ".csv":
        raise ValueError("artifact must be an .npz or archived paper .csv file")

    documents, embeddings, labels = [], [], []
    required = {"text", "target", "embedding"}
    for chunk in pd.read_csv(path, usecols=lambda name: name in required, chunksize=4096):
        if not required.issubset(chunk.columns):
            raise ValueError(f"CSV artifact requires columns {sorted(required)}")
        documents.extend(chunk["text"].fillna("").astype(str).tolist())
        labels.append(chunk["target"].to_numpy())
        embeddings.append(np.vstack([parse_embedding(value) for value in chunk["embedding"]]))
    matrix = np.vstack(embeddings).astype(np.float32, copy=False)
    targets = np.concatenate(labels)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache,
            embeddings=matrix,
            labels=targets,
            source_sha256=np.asarray(source_hash),
        )
        with gzip.open(text_cache, "wt", encoding="utf-8") as stream:
            for document in documents:
                stream.write(json.dumps(document, ensure_ascii=False) + "\n")
    return _validate(documents, matrix, targets, path, source_hash, "csv")


def _from_directory(path):
    required = {
        "embeddings": path / "embeddings.npy",
        "labels": path / "labels.npy",
        "documents": path / "documents.jsonl.gz",
        "metadata": path / "metadata.json",
    }
    missing = [name for name, item in required.items() if not item.exists()]
    if missing:
        raise ValueError(f"incomplete artifact directory {path}: missing {', '.join(missing)}")
    metadata = read_json(required["metadata"])
    if not metadata.get("embedding_complete"):
        raise ValueError(f"embedding artifact is not complete: {path}")
    if file_sha256(required["embeddings"]) != metadata["embeddings_sha256"]:
        raise ValueError(f"embedding checksum mismatch: {path}")
    if file_sha256(required["documents"]) != metadata["documents_sha256"]:
        raise ValueError(f"document checksum mismatch: {path}")
    if file_sha256(required["labels"]) != metadata["labels_sha256"]:
        raise ValueError(f"label checksum mismatch: {path}")
    with gzip.open(required["documents"], "rt", encoding="utf-8") as stream:
        documents = [json.loads(line) for line in stream]
    embeddings = np.load(required["embeddings"], mmap_mode="r", allow_pickle=False)
    labels = np.load(required["labels"], allow_pickle=False)
    result = _validate(
        documents,
        embeddings,
        labels,
        path,
        metadata["embeddings_sha256"],
        "directory",
    )
    result[3].update(
        protocol_id=metadata.get("protocol_id"),
        dataset=metadata.get("dataset"),
        source=metadata.get("source"),
        model=metadata.get("model"),
    )
    return result


def validate_protocol_artifact(audit, config, dataset, *, model_key="primary"):
    """Reject artifacts with incompatible data preparation or embeddings.

    A protocol revision may leave a dataset unchanged. Such artifacts remain valid
    when their pinned source/preprocessing and encoder records match exactly.
    """
    if not all(key in config for key in ("protocol_id", "sources", "models")):
        return
    expected = {
        "dataset": dataset,
        "source": config["sources"][dataset],
        "model": config["models"][model_key],
    }
    missing = [key for key in expected if audit.get(key) is None]
    if missing:
        raise ValueError(
            "paper artifacts require generated provenance metadata; missing " + ", ".join(missing)
        )
    mismatched = [key for key, value in expected.items() if audit.get(key) != value]
    if mismatched:
        raise ValueError(
            "artifact does not match the active paper protocol: " + ", ".join(mismatched)
        )


def _from_npz(data, path, source_hash, artifact_format, require_documents=True):
    if "embeddings" not in data.files or not ({"labels", "targets"} & set(data.files)):
        raise ValueError("NPZ artifact requires embeddings and labels (or archived targets)")
    if "documents" in data.files:
        documents = data["documents"].tolist()
    elif require_documents:
        raise ValueError("this experiment requires documents in the NPZ artifact")
    else:
        documents = [""] * len(data["embeddings"])
    label_key = "labels" if "labels" in data.files else "targets"
    return _validate(
        documents,
        data["embeddings"],
        data[label_key],
        path,
        source_hash,
        artifact_format,
    )


def _validate(documents, embeddings, labels, path, source_hash, artifact_format):
    matrix = np.asarray(embeddings, dtype=np.float32)
    targets = np.asarray(labels)
    if matrix.ndim != 2 or matrix.shape[0] != len(documents):
        raise ValueError("embedding rows must match documents")
    if targets.shape != (len(documents),):
        raise ValueError("labels must contain one value per document")
    if not np.isfinite(matrix).all() or np.any(np.linalg.norm(matrix, axis=1) == 0):
        raise ValueError("embeddings must be finite and nonzero")
    audit = {
        "path": str(path),
        "sha256": source_hash,
        "format": artifact_format,
        "documents": len(documents),
        "dimensions": matrix.shape[1],
        "classes": int(np.unique(targets).size),
        "documents_available": any(bool(text) for text in documents),
        "empty_documents": sum(not text.strip() for text in documents) if any(documents) else None,
    }
    return documents, matrix, targets, audit


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))
