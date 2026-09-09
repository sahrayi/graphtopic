"""Carry validated outputs or unchanged artifacts into a revised paper protocol."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

from .common import artifact_root, atomic_json, file_sha256, read_json, result_root

ROOT = Path(__file__).resolve().parent


def _link_or_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size or file_sha256(
            destination
        ) != file_sha256(source):
            raise RuntimeError(f"existing destination differs from source: {destination}")
        return "existing"
    try:
        os.link(source, destination)
        return "linked"
    except OSError:
        shutil.copy2(source, destination)
        return "copied"


def _tree(source, destination, counters):
    if not source.is_dir():
        raise FileNotFoundError(source)
    for path in source.rglob("*"):
        if path.is_file():
            action = _link_or_copy(path, destination / path.relative_to(source))
            counters[action] = counters.get(action, 0) + 1


def _compatible_metadata(
    path, config, dataset, *, allow_model_extension=False, verify_hashes=False
):
    metadata = read_json(path / "metadata.json")
    expected_model = config["models"]["primary"]
    observed_model = metadata.get("model", {})
    model_matches = observed_model == expected_model
    if allow_model_extension:
        model_matches = all(
            expected_model.get(key) == value for key, value in observed_model.items()
        ) and {
            "id",
            "revision",
        }.issubset(observed_model)
    expected = {"dataset": dataset, "source": config["sources"][dataset]}
    mismatched = [key for key, value in expected.items() if metadata.get(key) != value]
    if not model_matches:
        mismatched.append("model")
    if not metadata.get("embedding_complete"):
        mismatched.append("embedding_complete")
    if verify_hashes:
        for filename, key in (
            ("documents.jsonl.gz", "documents_sha256"),
            ("labels.npy", "labels_sha256"),
            ("embeddings.npy", "embeddings_sha256"),
        ):
            candidate = path / filename
            if not candidate.is_file() or file_sha256(candidate) != metadata.get(key):
                mismatched.append(key)
    if mismatched:
        raise RuntimeError(f"cannot carry incompatible {dataset} artifact: {', '.join(mismatched)}")


def carry_artifacts(previous_artifacts, config_path, datasets):
    """Reuse immutable embeddings when a protocol changes only downstream analysis."""
    config = read_json(config_path)
    destination = artifact_root(config_path)
    counters = {}
    for dataset in datasets:
        source = previous_artifacts / dataset
        _compatible_metadata(
            source, config, dataset, allow_model_extension=True, verify_hashes=True
        )
        for path in source.rglob("*"):
            if path.is_file() and path.name != "metadata.json":
                action = _link_or_copy(path, destination / dataset / path.relative_to(source))
                counters[action] = counters.get(action, 0) + 1
        metadata = read_json(source / "metadata.json")
        metadata.update(protocol_id=config["protocol_id"], model=config["models"]["primary"])
        atomic_json(destination / dataset / "metadata.json", metadata)
        cache = previous_artifacts / "cache" / dataset
        if cache.exists():
            _tree(cache, destination / "cache" / dataset, counters)
    return {
        "schema_version": 1,
        "active_protocol": config["protocol_id"],
        "carried_datasets": list(datasets),
        "artifacts_only": True,
        "previous_artifacts": str(previous_artifacts.resolve()),
        "created_at": datetime.now(UTC).isoformat(),
        "files": counters,
    }


def carry_forward(previous_artifacts, previous_results, config_path):
    config = read_json(config_path)
    comparison = read_json(previous_results / "comparison.json")
    if comparison.get("passed") is not True:
        raise RuntimeError("previous results do not have a passing comparison record")
    destination_artifacts = artifact_root(config_path)
    destination_results = result_root(config_path)
    counters = {}
    for dataset in ("agnews", "dbpedia14"):
        _compatible_metadata(previous_artifacts / dataset, config, dataset)
        _tree(
            previous_artifacts / dataset,
            destination_artifacts / dataset,
            counters,
        )
        _tree(previous_results / dataset, destination_results / dataset, counters)
        cache = previous_artifacts / "cache" / dataset
        if cache.exists():
            _tree(cache, destination_artifacts / "cache" / dataset, counters)
    for relative in (
        Path("ablations/k-resolution/agnews"),
        Path("ablations/graph-design/agnews"),
        Path("lexical/agnews"),
        Path("scaling/agnews"),
        Path("scaling/dbpedia14"),
    ):
        _tree(previous_results / relative, destination_results / relative, counters)
    manifest = {
        "schema_version": 1,
        "active_protocol": config["protocol_id"],
        "carried_datasets": ["agnews", "dbpedia14"],
        "excluded_dataset": "20newsgroups",
        "previous_artifacts": str(previous_artifacts.resolve()),
        "previous_results": str(previous_results.resolve()),
        "created_at": datetime.now(UTC).isoformat(),
        "files": counters,
    }
    atomic_json(destination_results / "carry-forward.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-artifacts", type=Path, required=True)
    parser.add_argument("--previous-results", type=Path)
    parser.add_argument(
        "--artifacts-only",
        action="store_true",
        help="reuse validated embeddings without carrying stale downstream results",
    )
    parser.add_argument(
        "--include",
        action="append",
        choices=("20newsgroups", "agnews", "dbpedia14"),
        help="dataset to reuse with --artifacts-only; repeat as needed",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "paper.json")
    args = parser.parse_args(argv)
    if args.artifacts_only:
        selected = args.include or ["20newsgroups", "agnews"]
        result = carry_artifacts(args.previous_artifacts, args.config, selected)
    else:
        if args.previous_results is None:
            parser.error("--previous-results is required unless --artifacts-only is used")
        result = carry_forward(args.previous_artifacts, args.previous_results, args.config)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
