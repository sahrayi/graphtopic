"""Summarize supplementary results without promoting incomplete runs to the reference."""

import json
import statistics

from .common import atomic_json, read_json
from .revision import CONFIG, SETTINGS, revision_root


def summarize(root, config, settings):
    rows, pending = [], []
    seeds = config["leiden_seeds"]
    for scope in ("independent", "subset"):
        for dataset in ("20newsgroups", "agnews"):
            target = settings["target_topics"][dataset]
            methods = ["GraphTopic", "MiniBatchKMeans", "NMF", "LDA", "BERTopic"]
            if dataset == "20newsgroups" and scope == "independent":
                methods.append("Graph2Topic")
            if scope == "subset":
                methods.append("TriTopic")
            for method in methods:
                folder = {
                    "GraphTopic": "graphtopic",
                    "BERTopic": "bertopic",
                    "TriTopic": "tritopic",
                    "Graph2Topic": "graph2topic",
                }.get(method, "baselines")
                prefix = f"{method}-" if folder == "baselines" else ""
                paths = [
                    root
                    / scope
                    / dataset
                    / folder
                    / "runs"
                    / f"{prefix}topics-{target}-seed-{seed}.json"
                    for seed in seeds
                ]
                if not all(path.exists() for path in paths):
                    pending.append(f"{scope}/{dataset}/{method}")
                    continue
                records = [read_json(path) for path in paths]
                scores = [row.get("reassigned", row) for row in records]
                row = {
                    "scope": scope,
                    "dataset": dataset,
                    "method": method,
                    "target_topics": target,
                    "seeds": seeds,
                }
                for name in ("topics", "ari", "nmi"):
                    values = [record[name] for record in scores]
                    row[name] = {"mean": statistics.mean(values), "std": statistics.stdev(values)}
                if all("coverage" in record for record in records):
                    row["native_coverage"] = statistics.mean(
                        record["coverage"] for record in records
                    )
                rows.append(row)
    return {
        "comparison_rows": rows,
        "pending_comparisons": pending,
        "note": "No automatic promotion; lexical, ablation and timing need separate review.",
    }


def main():
    settings = read_json(SETTINGS)
    root = revision_root(settings)
    result = summarize(root, read_json(CONFIG), settings)
    atomic_json(root / "comparison-summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
