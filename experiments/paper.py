"""Run every experiment reported in the GraphTopic paper, with checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .common import artifact_root, result_root
from .environment import validate_reference_environment

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent
RESULTS = result_root()
ARTIFACTS = artifact_root()
DEFAULT_PATHS = ROOT / "local-paths.json"
CHECKPOINT = RESULTS / "paper-checkpoint.json"


def now():
    return datetime.now(UTC).isoformat()


def atomic_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def count(pattern):
    return len(list(RESULTS.glob(pattern)))


@dataclass(frozen=True)
class Stage:
    name: str
    description: str
    command: tuple[str, ...]
    complete: Callable[[], bool]
    parallel: bool = False


def _core_complete(dataset, resolutions):
    expected = len(resolutions) * 5
    base = RESULTS / dataset
    return (
        (base / "summary.json").exists()
        and count(f"{dataset}/runs/gamma-*-seed-*.json") == expected
        and count(f"{dataset}/assignments/gamma-*-seed-*.npz") == expected
    )


def _classical_complete(dataset):
    return (
        count(f"{dataset}/baselines/runs/*.json") == 30
        and count(f"{dataset}/baselines/assignments/*.npz") == 30
    )


def _bertopic_complete(dataset):
    assignments = list((RESULTS / dataset / "bertopic" / "assignments").glob("*.npz"))
    if count(f"{dataset}/bertopic/runs/*.json") != 15 or len(assignments) != 15:
        return False
    for path in assignments:
        with np.load(path, allow_pickle=False) as saved:
            if "reassigned" not in saved or np.any(saved["reassigned"] == -1):
                return False
    return True


def _ablation_complete(experiment, dataset, encoder, expected):
    return count(f"ablations/{experiment}/{dataset}/{encoder}/*.json") >= expected


def _scaling_complete(dataset, expected):
    return count(f"scaling/{dataset}/n-*-repeat-*.json") == expected


def _exact_scaling_complete():
    return count("scaling-exact/agnews/n-*-repeat-*.json") == 6


def _lexical_complete(dataset):
    expected = 50
    if dataset == "20newsgroups" and count(f"{dataset}/graph2topic/assignments/*.npz") == 10:
        expected += 10
    return count(f"lexical/{dataset}/**/*.json") == expected


def _qualitative_complete(dataset):
    return (RESULTS / "qualitative" / f"{dataset}.json").exists()


def _bootstrap_complete(name):
    metadata = ARTIFACTS / name / "metadata.json"
    return metadata.exists() and load_json(metadata).get("embedding_complete") is True


def generated_paths():
    artifacts = ARTIFACTS
    return {
        "newsgroups_artifact": str(artifacts / "20newsgroups"),
        "newsgroups_text": str(artifacts / "20newsgroups" / "documents.jsonl.gz"),
        "agnews_artifact": str(artifacts / "agnews"),
        "agnews_text": str(artifacts / "agnews" / "documents.jsonl.gz"),
        "dbpedia_artifact": str(artifacts / "dbpedia14"),
        "alternate_encoder_artifact": str(artifacts / "20newsgroups-alternate"),
    }


def stages(paths, *, bootstrap):
    py = sys.executable
    common = {
        "20newsgroups": (paths["newsgroups_artifact"], paths["newsgroups_text"]),
        "agnews": (paths["agnews_artifact"], paths["agnews_text"]),
    }
    output = []
    if bootstrap:
        descriptions = {
            "20newsgroups": "download cleaned 20 Newsgroups and create primary embeddings",
            "agnews": "download AG News and create primary embeddings",
            "dbpedia14": "download DBpedia14 and create primary embeddings",
            "20newsgroups-alternate": "create cleaned 20 Newsgroups alternate embeddings",
        }
        for name, description in descriptions.items():
            output.append(
                Stage(
                    f"bootstrap-{name}",
                    description,
                    (py, "-m", "experiments.bootstrap", "--only", name),
                    lambda artifact=name: _bootstrap_complete(artifact),
                )
            )
    for dataset, resolutions, artifact_key in (
        ("20newsgroups", (1, 2), "newsgroups_artifact"),
        ("agnews", (0.1, 0.2), "agnews_artifact"),
        ("dbpedia14", (0.05, 0.1, 0.2, 0.5, 1), "dbpedia_artifact"),
    ):
        output.append(
            Stage(
                f"core-{dataset}",
                f"canonical GraphTopic runs on {dataset}",
                (
                    py,
                    "-m",
                    "experiments.run",
                    "core",
                    "--dataset",
                    dataset,
                    "--artifact",
                    paths[artifact_key],
                ),
                lambda d=dataset, r=resolutions: _core_complete(d, r),
            )
        )
    for dataset, (artifact, text) in common.items():
        output.append(
            Stage(
                f"classical-{dataset}",
                f"KMeans, NMF, and LDA on {dataset}",
                (
                    py,
                    "-m",
                    "experiments.baselines",
                    "classical",
                    "--dataset",
                    dataset,
                    "--artifact",
                    artifact,
                    "--text-csv",
                    text,
                ),
                lambda d=dataset: _classical_complete(d),
            )
        )
    for dataset, (artifact, text) in common.items():
        output.append(
            Stage(
                f"bertopic-{dataset}",
                f"BERTopic controlled comparison on {dataset}",
                (
                    py,
                    "-m",
                    "experiments.baselines",
                    "bertopic",
                    "--dataset",
                    dataset,
                    "--artifact",
                    artifact,
                    "--text-csv",
                    text,
                ),
                lambda d=dataset: _bertopic_complete(d),
            )
        )
    for dataset, (artifact, _) in common.items():
        output.append(
            Stage(
                f"k-resolution-{dataset}",
                f"neighborhood/resolution sensitivity on {dataset}",
                (
                    py,
                    "-m",
                    "experiments.ablations",
                    "k-resolution",
                    "--dataset",
                    dataset,
                    "--artifact",
                    artifact,
                ),
                lambda d=dataset: _ablation_complete("k-resolution", d, "all-MiniLM-L6-v2", 100),
            )
        )
        output.append(
            Stage(
                f"graph-design-{dataset}",
                f"directed, mutual, and union-max graphs on {dataset}",
                (
                    py,
                    "-m",
                    "experiments.ablations",
                    "graph-design",
                    "--dataset",
                    dataset,
                    "--artifact",
                    artifact,
                ),
                lambda d=dataset: _ablation_complete("graph-design", d, "all-MiniLM-L6-v2", 30),
            )
        )
    output.append(
        Stage(
            "alternate-encoder-20newsgroups",
            "alternate MiniLM encoder sensitivity",
            (
                py,
                "-m",
                "experiments.ablations",
                "encoder",
                "--dataset",
                "20newsgroups",
                "--artifact",
                paths["alternate_encoder_artifact"],
                "--encoder-label",
                "paraphrase-MiniLM-L3-v2",
            ),
            lambda: _ablation_complete("encoder", "20newsgroups", "paraphrase-MiniLM-L3-v2", 10),
        )
    )
    output.extend(
        (
            Stage(
                "scaling-agnews",
                "three-repeat nested AG News scaling curve",
                (
                    py,
                    "-m",
                    "experiments.scaling",
                    "--dataset",
                    "agnews",
                    "--artifact",
                    paths["agnews_artifact"],
                    "--warmup",
                ),
                lambda: _scaling_complete("agnews", 15),
                True,
            ),
            Stage(
                "scaling-dbpedia14",
                "three-repeat full DBpedia14 resource measurement",
                (
                    py,
                    "-m",
                    "experiments.scaling",
                    "--dataset",
                    "dbpedia14",
                    "--artifact",
                    paths["dbpedia_artifact"],
                    "--warmup",
                ),
                lambda: _scaling_complete("dbpedia14", 3),
                True,
            ),
        )
    )
    output.append(
        Stage(
            "scaling-exact-agnews",
            "exact-search comparison on 10k and 25k AG News samples",
            (
                py,
                "-m",
                "experiments.exact_scaling",
                "--artifact",
                paths["agnews_artifact"],
            ),
            _exact_scaling_complete,
            True,
        )
    )
    for dataset, (artifact, text) in common.items():
        output.append(
            Stage(
                f"lexical-{dataset}",
                f"shared lexical audit on {dataset}",
                (
                    py,
                    "-m",
                    "experiments.lexical",
                    "--artifact",
                    artifact,
                    "--text-csv",
                    text,
                    "--assignment-root",
                    str(RESULTS / dataset),
                    "--output-root",
                    str(RESULTS / "lexical" / dataset),
                ),
                lambda d=dataset: _lexical_complete(d),
            )
        )
        output.append(
            Stage(
                f"qualitative-{dataset}",
                f"deterministic qualitative topic examples on {dataset}",
                (
                    py,
                    "-m",
                    "experiments.qualitative",
                    "--dataset",
                    dataset,
                    "--artifact",
                    artifact,
                ),
                lambda d=dataset: _qualitative_complete(d),
            )
        )
    return output


def run(stage, state, *, position, total):
    progress = f"stage {position}/{total}"
    record = state["stages"].setdefault(stage.name, {})
    if stage.complete():
        record.update(status="complete", verified_at=now())
        atomic_write(CHECKPOINT, state)
        print(f"[{progress}] [skip] {stage.name}: complete")
        return
    record.update(status="running", started_at=now(), command=list(stage.command))
    atomic_write(CHECKPOINT, state)
    log = RESULTS / "logs" / f"{stage.name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        str(REPOSITORY / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    )
    environment["PYTHONHASHSEED"] = "0"
    thread_variables = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    )
    if stage.parallel:
        for name in thread_variables:
            environment.pop(name, None)
    else:
        for name in thread_variables:
            environment[name] = "1"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    print(f"[{progress}] [run] {stage.name}: {stage.description}")
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"\n--- {now()} ---\n")
        process = subprocess.Popen(
            stage.command,
            cwd=REPOSITORY,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        try:
            for line in process.stdout:
                print(line, end="")
                stream.write(line)
            code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            record.update(status="interrupted", interrupted_at=now())
            atomic_write(CHECKPOINT, state)
            raise
    if code or not stage.complete():
        record.update(status="failed", returncode=code, failed_at=now())
        atomic_write(CHECKPOINT, state)
        raise RuntimeError(f"stage failed or produced incomplete outputs: {stage.name}")
    record.update(status="complete", completed_at=now(), returncode=0)
    atomic_write(CHECKPOINT, state)
    print(f"[{progress}] [done] {stage.name}: complete", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths",
        type=Path,
        default=DEFAULT_PATHS,
        help="optional artifact override; generated frozen artifacts are the default",
    )
    parser.add_argument("--only", action="append", help="run only this stage; repeat as needed")
    parser.add_argument("--list", action="store_true", help="list stages and detected status")
    args = parser.parse_args(argv)
    custom_paths = args.paths.exists()
    paths = load_json(args.paths) if custom_paths else generated_paths()
    required = {
        "newsgroups_artifact",
        "newsgroups_text",
        "agnews_artifact",
        "agnews_text",
        "dbpedia_artifact",
        "alternate_encoder_artifact",
    }
    missing = sorted(required - paths.keys())
    if missing:
        parser.error(f"missing path keys: {', '.join(missing)}")
    if custom_paths:
        for key in required:
            if not Path(paths[key]).exists():
                parser.error(f"{key} does not exist: {paths[key]}")
    selected = stages(paths, bootstrap=not custom_paths)
    if args.only:
        wanted = set(args.only)
        unknown = wanted - {stage.name for stage in selected}
        if unknown:
            parser.error(f"unknown stages: {', '.join(sorted(unknown))}")
        selected = [stage for stage in selected if stage.name in wanted]
    if args.list:
        total = len(selected)
        for position, stage in enumerate(selected, start=1):
            status = "complete" if stage.complete() else "pending"
            print(f"[stage {position}/{total}] {status:8} {stage.name}")
        return 0
    environment = validate_reference_environment()
    state = load_json(CHECKPOINT) if CHECKPOINT.exists() else {"schema_version": 1, "stages": {}}
    state["reference_environment"] = environment
    state["updated_at"] = now()
    atomic_write(CHECKPOINT, state)
    try:
        total = len(selected)
        for position, stage in enumerate(selected, start=1):
            run(stage, state, position=position, total=total)
    except KeyboardInterrupt:
        print("\nInterrupted safely. Run the same command to resume.")
        return 130
    state["completed_at"] = now()
    atomic_write(CHECKPOINT, state)
    print("All selected paper experiments are complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
