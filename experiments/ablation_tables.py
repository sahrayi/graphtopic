"""Generate manuscript tables only from the checked-in frozen reference report."""

import argparse
from pathlib import Path

from .common import read_json

ROOT = Path(__file__).resolve().parent


def metric(row, name):
    value = row[name]
    decimals = 1 if name == "topics" else 3
    return f"{value['mean']:.{decimals}f}$\\pm${value['std']:.{decimals}f}"


def tables(report, supplement, independent):
    graph_rows = []
    for dataset, label, k, resolutions in (
        ("20newsgroups", "20NG", 20, (1, 2)),
        ("agnews", "AG News", 20, (0.1, 0.2)),
    ):
        for gamma in resolutions:
            for variant in ("directed", "mutual", "union_max"):
                key = f"{variant}-gamma-{gamma:g}"
                row = (
                    supplement["runs"][key]
                    if dataset == "agnews"
                    else report["ablations"]["graph-design"][dataset][key]
                )
                graph_rows.append(
                    " & ".join(
                        (
                            label,
                            str(k),
                            str(gamma),
                            variant.replace("_", "-"),
                            *(metric(row, m) for m in ("topics", "ari", "nmi")),
                        )
                    )
                )
    robustness_rows = []
    names = {"20newsgroups": "20NG", "agnews": "AG News", "MiniBatchKMeans": "KMeans"}
    for row in independent["rows"]:
        robustness_rows.append(
            " & ".join(
                (
                    names[row["dataset"]],
                    str(row["target_topics"]),
                    names.get(row["method"], row["method"]),
                    *(metric(row, m) for m in ("topics", "ari", "nmi")),
                )
            )
        )

    def table(caption, label, columns, heading, rows):
        return (
            "\\begin{table}[htbp]\n\\centering\n\\small\n\\caption{" + caption + "}\n"
            "\\label{"
            + label
            + "}\n\\begin{tabular}{"
            + columns
            + "}\n\\toprule\n"
            + heading
            + "\\\\\n\\midrule\n"
            + "\\\\\n".join(rows)
            + "\\\\\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
        )

    return (
        table(
            "Graph-design ablations (mean $\\pm$ sample SD over five seeds). "
            "Both corpora use the final neighborhood size $k=20$. "
            "Resolution is held fixed within each block; realized topic counts are not matched.",
            "tab:graph-ablation",
            "lrrlrrr",
            "Corpus & $k$ & $\\gamma$ & Graph & Topics & ARI & NMI",
            graph_rows,
        )
        + "\n"
        + table(
            "Robustness at independently anchored class-count targets "
            "(mean $\\pm$ sample SD over five seeds). GraphTopic resolution is selected by "
            "pilot topic count only; its realized topic count is reported explicitly.",
            "tab:independent-counts",
            "lrlrrr",
            "Corpus & Target & Method & Realized topics & ARI & NMI",
            robustness_rows,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "expected/reference-results.json")
    parser.add_argument(
        "--supplement", type=Path, default=ROOT / "expected/graph-design-k20-agnews.json"
    )
    parser.add_argument(
        "--independent", type=Path, default=ROOT / "expected/independent-counts.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        tables(read_json(args.report), read_json(args.supplement), read_json(args.independent)),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
