"""Self-bootstrapping supplementary experiments; execute heavy stages explicitly."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

from .common import artifact_root, atomic_json, read_json, result_root

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "configs/paper.json"
SETTINGS = ROOT / "configs/revision.json"
STAGES = ("prepare", "robustness", "graph-design", "lexical", "subset", "timing")
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def command(module, *arguments, python=None, parallel=False):
    env = os.environ.copy()
    env.update(PYTHONHASHSEED="0", TOKENIZERS_PARALLELISM="false")
    for name in THREAD_VARIABLES:
        if parallel:
            env.pop(name, None)
        else:
            env[name] = "1"
    subprocess.run(
        [str(python or sys.executable), "-m", module, *map(str, arguments)],
        cwd=ROOT.parent,
        env=env,
        check=True,
    )


def revision_root(settings):
    name = settings["revision_id"]
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in name):
        raise ValueError("Unsafe revision_id")
    return ROOT / "results" / name


def freeze_protocol(output, settings):
    identity = {
        "settings": settings,
        "base_config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
    }
    path = output / "protocol.json"
    if path.exists() and read_json(path) != identity:
        raise ValueError("Settings changed; use a new revision_id to avoid stale outputs")
    atomic_json(path, identity)


def graph_comparison(dataset, artifact, output, config_path, settings, cache):
    from graphtopic._validation import normalize_embeddings

    from .common import load_artifact
    from .revision_graph import evaluate_graph, get_graph

    config = read_json(config_path)
    source = read_json(artifact / "metadata.json")
    identity = {
        key: source[key] for key in ("documents_sha256", "embeddings_sha256", "labels_sha256")
    }
    identity_path = output / dataset / "input.json"
    if identity_path.exists() and read_json(identity_path) != identity:
        raise ValueError("Input changed; use a new revision namespace")
    atomic_json(identity_path, identity)
    documents, raw, labels, audit = load_artifact(artifact)
    graph = get_graph(normalize_embeddings(raw, len(documents)), audit, config, cache)
    evaluate_graph(
        graph,
        labels,
        settings["target_topics"][dataset],
        settings,
        config,
        output / dataset / "graphtopic",
    )
    for method in ("classical", "bertopic"):
        command(
            "experiments.baselines",
            method,
            "--dataset",
            dataset,
            "--artifact",
            artifact,
            "--config",
            config_path,
            "--results",
            output,
            "--target-topics",
            settings["target_topics"][dataset],
        )


def prepare_subset(dataset, settings, output):
    import numpy as np

    from .bootstrap import _write_documents
    from .common import file_sha256, load_artifact
    from .revision_data import ensure_artifact

    source = ensure_artifact(dataset, CONFIG)
    documents, raw, labels, audit = load_artifact(source)
    settings_subset = settings["subset"]
    ids = np.sort(
        np.random.default_rng(settings_subset["seed"]).choice(
            len(documents), min(settings_subset["documents"], len(documents)), replace=False
        )
    )
    folder = ROOT / "artifacts" / settings["revision_id"] / "subset" / dataset
    folder.mkdir(parents=True, exist_ok=True)
    metadata_path = folder / "metadata.json"
    config = deepcopy(read_json(CONFIG))
    config["protocol_id"] = settings["revision_id"]
    config["datasets"][dataset]["documents"] = len(ids)
    config["datasets"][dataset]["classes"] = int(np.unique(labels[ids]).size)
    config_path = output / f"{dataset}-config.json"
    atomic_json(config_path, config)
    expected_subset = {**settings_subset, "source_embeddings_sha256": audit["sha256"]}
    if metadata_path.exists():
        if read_json(metadata_path).get("subset") != expected_subset:
            raise ValueError("Subset source changed; use a new revision namespace")
    else:
        _write_documents(folder / "documents.jsonl.gz", [documents[i] for i in ids])
        np.save(folder / "embeddings.npy", raw[ids], allow_pickle=False)
        np.save(folder / "labels.npy", labels[ids], allow_pickle=False)
        np.save(folder / "row-ids.npy", ids, allow_pickle=False)
        metadata = {
            "dataset": dataset,
            "protocol_id": config["protocol_id"],
            "source": config["sources"][dataset],
            "model": config["models"]["primary"],
            "embedding_complete": True,
            "documents": len(ids),
            "subset": expected_subset,
        }
        for name, key in (
            ("documents.jsonl.gz", "documents_sha256"),
            ("labels.npy", "labels_sha256"),
            ("embeddings.npy", "embeddings_sha256"),
        ):
            metadata[key] = file_sha256(folder / name)
        atomic_json(metadata_path, metadata)
    return folder, config_path


def run_stage(stage, dataset, settings, output, tritopic_python):
    from .revision_data import ensure_artifact

    if stage == "prepare":
        ensure_artifact(dataset, CONFIG)
    elif stage == "robustness":
        artifact = ensure_artifact(dataset, CONFIG)
        graph_comparison(
            dataset,
            artifact,
            output / "independent",
            CONFIG,
            settings,
            artifact_root() / "cache" / dataset,
        )
    elif stage == "graph-design":
        artifact = ensure_artifact(dataset, CONFIG)
        config = deepcopy(read_json(CONFIG))
        config["ablations"]["graph_design_k"] = {"20newsgroups": 20, "agnews": 20}
        path = output / "graph-design-config.json"
        atomic_json(path, config)
        command(
            "experiments.ablations",
            "graph-design",
            "--dataset",
            dataset,
            "--artifact",
            artifact,
            "--config",
            path,
            "--results",
            output / "ablations",
            "--cache",
            artifact_root() / "cache",
        )
    elif stage == "lexical":
        artifact = ensure_artifact(dataset, CONFIG)
        lexical_root = output / "lexical-html-v2"
        # The public route builds missing assignments before lexical evaluation.
        # Existing stages checkpoint themselves; no private archived results needed.
        command(
            "experiments.paper",
            "--only",
            f"core-{dataset}",
            "--only",
            f"classical-{dataset}",
            "--only",
            f"bertopic-{dataset}",
        )
        command(
            "experiments.lexical",
            "--artifact",
            artifact,
            "--assignment-root",
            result_root() / dataset,
            "--output-root",
            lexical_root / dataset,
            "--clean-html",
            "--details",
        )
        command(
            "experiments.qualitative",
            "--dataset",
            dataset,
            "--artifact",
            artifact,
            "--lexical-root",
            lexical_root,
            "--output-root",
            output / "qualitative-html-v2",
        )
    elif stage == "subset":
        if tritopic_python is None:
            raise ValueError("Pass --tritopic-python for the isolated TriTopic environment")
        command("experiments.tritopic_smoke", python=tritopic_python)
        folder, config_path = prepare_subset(dataset, settings, output / "subset")
        # Run the contemporary comparator first: package/runtime failures should
        # be discovered before paying for all other methods on this subset.
        command(
            "experiments.tritopic_baseline",
            "--artifact",
            folder,
            "--dataset",
            dataset,
            "--config",
            config_path,
            "--results",
            output / "subset",
            "--target-topics",
            settings["target_topics"][dataset],
            python=tritopic_python,
        )
        graph_comparison(
            dataset, folder, output / "subset", config_path, settings, folder / "cache"
        )
    elif stage == "timing":
        command("experiments.pipeline_timing", "--dataset", dataset, parallel=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--dataset", choices=("20newsgroups", "agnews", "dbpedia14"))
    parser.add_argument("--tritopic-python", type=Path)
    args = parser.parse_args(argv)
    if args.list or not args.stage:
        print("\n".join(STAGES))
        return
    if args.dataset is None:
        parser.error("--dataset is required with --stage")
    if args.dataset == "dbpedia14" and args.stage not in ("prepare", "timing"):
        parser.error("DBpedia14 is reserved for preparation and timing")
    if args.stage == "timing" and args.dataset == "20newsgroups":
        parser.error("Timing is prespecified for AG News and DBpedia14")
    if (
        os.environ.get("PYTHONHASHSEED") != "0"
        or any(os.environ.get(name) != "1" for name in THREAD_VARIABLES)
        or os.environ.get("TOKENIZERS_PARALLELISM") != "false"
    ):
        # Apply thread limits before scientific libraries initialize in the child.
        command("experiments.revision", *(sys.argv[1:] if argv is None else argv))
        return
    from .environment import validate_reference_environment

    validate_reference_environment()
    settings = read_json(SETTINGS)
    output = revision_root(settings)
    freeze_protocol(output, settings)
    python = args.tritopic_python.resolve() if args.tritopic_python else None
    run_stage(args.stage, args.dataset, settings, output, python)


if __name__ == "__main__":
    main()
