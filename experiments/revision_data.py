"""Prepare local artifacts on demand without embedding personal filesystem paths."""

from .bootstrap import encode
from .common import artifact_root, file_sha256, read_json, validate_protocol_artifact


def ensure_artifact(dataset, config_path):
    config = read_json(config_path)
    path = artifact_root(config_path) / dataset
    files = ("metadata.json", "documents.jsonl.gz", "labels.npy", "embeddings.npy")
    if not all((path / name).exists() for name in files):
        encode(dataset, config, path, "primary")
    metadata = read_json(path / "metadata.json")
    if not metadata.get("embedding_complete"):
        encode(dataset, config, path, "primary")
        metadata = read_json(path / "metadata.json")
    validate_protocol_artifact(metadata, config, dataset)
    for name, key in (
        ("documents.jsonl.gz", "documents_sha256"),
        ("labels.npy", "labels_sha256"),
        ("embeddings.npy", "embeddings_sha256"),
    ):
        if file_sha256(path / name) != metadata[key]:
            raise ValueError(f"Corrupt artifact: {dataset}/{name}")
    return path
