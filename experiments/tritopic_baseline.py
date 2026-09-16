"""Adapter for the unmodified TriTopic 2.3.0 package on a shared small corpus."""

import argparse
import importlib.metadata
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .baselines import assign_remaining_outliers
from .common import atomic_json, load_artifact, read_json, validate_protocol_artifact
from .metrics import external_scores


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--dataset", required=True, choices=("20newsgroups", "agnews"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--target-topics", type=int, required=True)
    parser.add_argument("--only-seed", type=int)
    args = parser.parse_args(argv)
    version = importlib.metadata.version("tritopic")
    if version != "2.3.0":
        raise RuntimeError("This adapter was inspected for tritopic==2.3.0")
    from tritopic import TriTopic, TriTopicConfig

    config = read_json(args.config)
    documents, embeddings, reference, audit = load_artifact(args.artifact)
    validate_protocol_artifact(audit, config, args.dataset)
    if len(documents) > 5000:
        raise ValueError("Use the shared <=5,000-document subset: upstream consensus is quadratic")
    output = args.results / args.dataset / "tritopic"
    seeds = [args.only_seed] if args.only_seed is not None else config["leiden_seeds"]
    for seed in seeds:
        stem = f"topics-{args.target_topics}-seed-{seed}"
        record = output / "runs" / f"{stem}.json"
        assignment = output / "assignments" / f"{stem}.npz"
        if record.exists() and assignment.exists():
            continue
        settings = TriTopicConfig(
            random_state=seed, use_metadata_view=False, n_consensus_runs=10, verbose=True
        )
        model = TriTopic(config=settings, n_topics=args.target_topics)
        started = time.perf_counter()
        # Copy protects cached inputs from iterative refinement. Do not pass labels
        # as metadata, disable consensus, or invoke LLM labeling.
        native = np.asarray(
            model.fit_transform(documents, embeddings=np.array(embeddings, copy=True)),
            dtype=np.int32,
        )
        seconds = time.perf_counter() - started
        complete = assign_remaining_outliers(native, embeddings)
        covered = native != -1
        assignment.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(assignment, labels=native, reassigned=complete)
        atomic_json(
            record,
            {
                "method": "TriTopic",
                "version": version,
                "seed": seed,
                "target_topics": args.target_topics,
                "seconds": seconds,
                "coverage": float(covered.mean()),
                "native": external_scores(reference, native),
                "reassigned": external_scores(reference, complete),
                "settings": asdict(settings),
                "python": platform.python_version(),
                "packages": {
                    d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
                },
                "input_embeddings_sha256": audit["sha256"],
            },
        )


if __name__ == "__main__":
    main()
