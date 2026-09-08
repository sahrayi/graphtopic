"""Build, promote, and compare compact reports from completed paper experiments."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

from .common import atomic_json, file_sha256, read_json

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
ARTIFACTS = ROOT / "artifacts"
CONFIG = ROOT / "configs" / "paper.json"
CANDIDATE = RESULTS / "reference-candidate.json"
REFERENCE = ROOT / "expected" / "reference-results.json"

SCORE_NAMES = {
    "ari",
    "nmi",
    "homogeneity",
    "completeness",
    "v_measure",
    "coverage",
    "npmi_macro",
    "npmi_document_weighted",
    "topic_diversity",
}


def _mean_std(rows, names):
    output = {}
    for name in names:
        values = [float(row[name]) for row in rows if name in row]
        if values:
            output[name] = {
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
    return output


def _json_rows(pattern):
    return [read_json(path) for path in sorted(RESULTS.glob(pattern))]


def _seed_groups(pattern):
    groups = defaultdict(list)
    for path in sorted(RESULTS.glob(pattern)):
        key = re.sub(r"-seed-\d+$", "", path.stem)
        groups[key].append(read_json(path))
    return groups


def _core(dataset):
    summary = read_json(RESULTS / dataset / "summary.json")
    ann = read_json(RESULTS / dataset / "ann.json")
    graph = read_json(RESULTS / dataset / "graph.json")
    return {
        "resolutions": summary["resolutions"],
        "recall_at_20": ann["recall_at_20"],
        "graph": {
            key: graph[key]
            for key in (
                "n_vertices",
                "n_edges",
                "directed_arcs",
                "isolated_vertices",
                "connected_components",
                "min_degree",
                "mean_degree",
                "max_degree",
                "candidate_shortages",
            )
        },
    }


def _baselines(dataset):
    output = {}
    for key, rows in _seed_groups(f"{dataset}/baselines/runs/*.json").items():
        output[key] = _mean_std(rows, ("topics", *sorted(SCORE_NAMES)))
    return output


def _bertopic(dataset):
    output = {}
    for key, rows in _seed_groups(f"{dataset}/bertopic/runs/*.json").items():
        flattened = []
        for row in rows:
            flattened.append({"coverage": row["coverage"], **row["reassigned"]})
            if row.get("residual_outliers_after_reassignment", 0) != 0:
                raise RuntimeError(f"residual BERTopic outliers remain in {dataset}/{key}")
        output[key] = _mean_std(flattened, ("topics", *sorted(SCORE_NAMES)))
    return output


def _lexical(dataset):
    output = {}
    base = RESULTS / "lexical" / dataset
    for family in ("assignments", "baselines/assignments", "bertopic/assignments"):
        groups = defaultdict(list)
        for path in sorted((base / family).glob("*.json")):
            key = re.sub(r"-seed-\d+$", "", path.stem)
            groups[key].append(read_json(path))
        output[family] = {
            key: _mean_std(rows, ("npmi_macro", "npmi_document_weighted", "topic_diversity"))
            for key, rows in groups.items()
        }
    return output


def _ablations():
    output = {}
    base = RESULTS / "ablations"
    for experiment in ("k-resolution", "graph-design", "encoder"):
        output[experiment] = {}
        for dataset in ("20newsgroups", "agnews"):
            directory = base / experiment / dataset
            if not directory.exists():
                continue
            rows_by_key = defaultdict(list)
            for path in sorted(directory.glob("*/*.json")):
                if path.name == "artifact-audit.json":
                    continue
                key = re.sub(r"-seed-\d+$", "", path.stem)
                rows_by_key[key].append(read_json(path))
            output[experiment][dataset] = {
                key: _mean_std(rows, ("topics", "ari", "nmi", "edges"))
                for key, rows in rows_by_key.items()
            }
    return output


def _scaling():
    output = {}
    measured = (
        "edges",
        "topics",
        "isolated_vertices",
        "connected_components",
        "ann_seconds",
        "graph_seconds",
        "leiden_seconds",
        "total_seconds",
        "rss_baseline_mb",
        "rss_peak_mb",
        "rss_increment_mb",
    )
    for dataset in ("agnews", "dbpedia14"):
        groups = defaultdict(list)
        for row in _json_rows(f"scaling/{dataset}/n-*-repeat-*.json"):
            groups[str(row["documents"])].append(row)
        output[dataset] = {
            size: {name: statistics.median(float(row[name]) for row in rows) for name in measured}
            for size, rows in groups.items()
        }
    return output


def build_report():
    checkpoint = read_json(RESULTS / "paper-checkpoint.json")
    incomplete = [
        name for name, state in checkpoint["stages"].items() if state.get("status") != "complete"
    ]
    if incomplete:
        raise RuntimeError(f"incomplete stages: {', '.join(incomplete)}")
    artifacts = {}
    for name in ("20newsgroups", "agnews", "dbpedia14", "20newsgroups-alternate"):
        metadata = read_json(ARTIFACTS / name / "metadata.json")
        artifacts[name] = {
            key: metadata[key]
            for key in (
                "dataset",
                "documents",
                "classes",
                "dimensions",
                "source",
                "model",
                "documents_sha256",
                "labels_sha256",
                "embeddings_sha256",
            )
        }
    return {
        "schema_version": 1,
        "protocol_id": read_json(CONFIG)["protocol_id"],
        "config_sha256": file_sha256(CONFIG),
        "environment": checkpoint["reference_environment"],
        "artifacts": artifacts,
        "core": {name: _core(name) for name in ("20newsgroups", "agnews", "dbpedia14")},
        "baselines": {name: _baselines(name) for name in ("20newsgroups", "agnews")},
        "bertopic": {name: _bertopic(name) for name in ("20newsgroups", "agnews")},
        "lexical": {name: _lexical(name) for name in ("20newsgroups", "agnews")},
        "ablations": _ablations(),
        "scaling_observational": _scaling(),
    }


def _compare(reference, candidate, path=""):
    failures = []
    if isinstance(reference, dict):
        for key, value in reference.items():
            child = f"{path}.{key}" if path else key
            if key not in candidate:
                failures.append(f"{child}: missing")
            elif (
                child.startswith("scaling_observational")
                or child.startswith("environment")
                or child.endswith("embeddings_sha256")
            ):
                continue
            else:
                failures.extend(_compare(value, candidate[key], child))
    elif isinstance(reference, (int, float)) and not isinstance(reference, bool):
        actual = float(candidate)
        tolerance = 0.005 if any(f".{name}." in f".{path}." for name in SCORE_NAMES) else 0.0
        if ".topics." in f".{path}.":
            tolerance = 0.5
        if path.startswith("core.") and ".recall_at_20." in path:
            tolerance = 0.0005
        if ".mean_degree" in path:
            tolerance = 0.01
        if path.endswith("n_edges") or path.endswith("directed_arcs"):
            tolerance = max(1.0, abs(float(reference)) * 0.001)
        if abs(actual - float(reference)) > tolerance:
            failures.append(f"{path}: expected {reference}, found {candidate} (tol={tolerance})")
    elif reference != candidate:
        failures.append(f"{path}: expected {reference!r}, found {candidate!r}")
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "promote", "compare"))
    args = parser.parse_args(argv)
    if args.command == "build":
        atomic_json(CANDIDATE, build_report())
        print(f"Wrote {CANDIDATE}")
        return 0
    if args.command == "promote":
        if not CANDIDATE.exists():
            raise FileNotFoundError("build and review the candidate report first")
        REFERENCE.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(CANDIDATE, REFERENCE)
        print(f"Promoted reviewed report to {REFERENCE}")
        return 0
    if not REFERENCE.exists():
        raise FileNotFoundError("no promoted reference report exists")
    candidate = build_report()
    failures = _compare(read_json(REFERENCE), candidate)
    atomic_json(RESULTS / "comparison.json", {"passed": not failures, "failures": failures})
    print(json.dumps({"passed": not failures, "failures": failures}, indent=2))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
