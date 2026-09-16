import json
import statistics

import numpy as np
import pytest
from scipy import sparse

from experiments import revision_graph
from experiments.metrics import lexical_scores
from experiments.revision import freeze_protocol
from experiments.text_cleaning import clean_lexical_text


def test_html_cleanup_keeps_content_and_normal_numbers():
    text = (
        "&amp;lt;b&amp;gt;Market&amp;lt;/b&amp;gt; &#39;news&#39; #39;s quot;report quot; 39 2026"
    )
    cleaned = clean_lexical_text(text)
    assert "Market" in cleaned and "'news'" in cleaned
    assert "39 2026" in cleaned
    assert "'s" in cleaned and '"report "' in cleaned
    assert "#39;" not in cleaned and "quot;" not in cleaned
    assert "&" not in cleaned and "<b>" not in cleaned


def test_npmi_weighting_and_json_topic_ids():
    # Topic 0 has perfectly co-occurring words. Topic 1 has disjoint words.
    counts = sparse.csr_matrix([[1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    result = lexical_scores(
        counts,
        np.array([0, 1, 1, 1]),
        {"0": ["a", "b"], "1": ["c", "d"]},
        ["a", "b", "c", "d"],
        details=True,
    )
    assert result["npmi_macro"] == pytest.approx(0)
    assert result["npmi_document_weighted"] == pytest.approx(-0.5)
    assert [row["documents"] for row in result["per_topic"]] == [1, 3]


def test_resolution_selection_uses_count_only_and_lower_gamma_tie(tmp_path, monkeypatch):
    class Detector:
        def __init__(self, resolution, **kwargs):
            self.gamma = resolution

        def fit_predict(self, graph):
            return np.arange({0.1: 3, 0.2: 5, 0.3: 8}[self.gamma])

    monkeypatch.setattr(revision_graph, "LeidenDetector", Detector)
    settings = {
        "resolution_grid": [0.3, 0.2, 0.1],
        "selection_seed": 1729,
        "selection_rule": "count distance, lower resolution tie",
    }
    assert revision_graph.select_resolution(None, 4, settings, 2, tmp_path) == 0.1
    record = json.loads((tmp_path / "selection.json").read_text())
    assert record["pilot_topics"] == 3
    assert not any("ari" in row or "nmi" in row for row in record["candidates"])


def test_revision_settings_cannot_silently_reuse_outputs(tmp_path):
    freeze_protocol(tmp_path, {"revision_id": "example", "target": 4})
    freeze_protocol(tmp_path, {"revision_id": "example", "target": 4})
    with pytest.raises(ValueError, match="Settings changed"):
        freeze_protocol(tmp_path, {"revision_id": "example", "target": 5})


def test_missing_graph_cache_is_built_then_reused(tmp_path, monkeypatch):
    calls = []

    class Search:
        def __init__(self, *args, **kwargs):
            pass

        def search(self, embeddings):
            calls.append(len(embeddings))
            return np.array([[1, 2], [0, 2], [1, 0]])

    monkeypatch.setattr(revision_graph, "NNDescentSearch", Search)
    embeddings = np.array([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]], dtype=np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    config = {
        "candidate_neighbors": 2,
        "retained_neighbors": 1,
        "ann": {"n_trees": 5, "n_iters": 5, "random_state": 1729, "n_jobs": 1},
    }
    audit = {"sha256": "abcdef0123456789"}
    first = revision_graph.get_graph(embeddings, audit, config, tmp_path)
    second = revision_graph.get_graph(embeddings, audit, config, tmp_path)
    assert calls == [3]
    assert (first.adjacency != second.adjacency).nnz == 0


def test_command_sets_quality_threads_and_releases_them_for_timing(monkeypatch):
    from experiments import revision

    calls = []
    monkeypatch.setattr(revision.subprocess, "run", lambda *args, **kwargs: calls.append(kwargs))
    revision.command("example")
    assert all(calls[0]["env"][name] == "1" for name in revision.THREAD_VARIABLES)
    assert calls[0]["env"]["PYTHONHASHSEED"] == "0"
    revision.command("example", parallel=True)
    assert all(name not in calls[1]["env"] for name in revision.THREAD_VARIABLES)


def test_missing_artifact_bootstraps_without_private_inputs(tmp_path, monkeypatch):
    from experiments import revision_data
    from experiments.common import atomic_json, file_sha256

    config_path = tmp_path / "config.json"
    config = {
        "protocol_id": "test",
        "sources": {"agnews": {"id": "public"}},
        "models": {"primary": {"id": "encoder"}},
    }
    atomic_json(config_path, config)
    monkeypatch.setattr(revision_data, "artifact_root", lambda _: tmp_path / "artifacts")
    calls = []

    def encode(dataset, config, folder, model_key):
        calls.append(dataset)
        folder.mkdir(parents=True)
        metadata = {
            "dataset": dataset,
            "source": config["sources"][dataset],
            "model": config["models"][model_key],
            "embedding_complete": True,
        }
        for name, key in (
            ("documents.jsonl.gz", "documents_sha256"),
            ("labels.npy", "labels_sha256"),
            ("embeddings.npy", "embeddings_sha256"),
        ):
            (folder / name).write_bytes(b"synthetic artifact")
            metadata[key] = file_sha256(folder / name)
        atomic_json(folder / "metadata.json", metadata)

    monkeypatch.setattr(revision_data, "encode", encode)
    folder = revision_data.ensure_artifact("agnews", config_path)
    revision_data.ensure_artifact("agnews", config_path)
    assert calls == ["agnews"]
    (folder / "embeddings.npy").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Corrupt artifact"):
        revision_data.ensure_artifact("agnews", config_path)


def test_frozen_k20_ablation_matches_completed_outputs_when_present():
    from pathlib import Path

    from experiments.common import read_json

    root = Path("experiments/results/paper-supplement-v1/ablations/graph-design/agnews")
    folders = list(root.glob("*/"))
    if not folders:
        pytest.skip("local supplementary outputs are intentionally not distributed")
    expected = read_json(Path("experiments/expected/graph-design-k20-agnews.json"))
    for key, summary in expected["runs"].items():
        variant, gamma = key.rsplit("-gamma-", 1)
        rows = [read_json(path) for path in folders[0].glob(f"{variant}-gamma-{gamma}-seed-*.json")]
        assert [row["seed"] for row in rows] == expected["seeds"]
        assert all(row["k"] == 20 for row in rows)
        for metric in ("topics", "ari", "nmi"):
            values = [row[metric] for row in rows]
            assert statistics.mean(values) == pytest.approx(summary[metric]["mean"])
            assert statistics.stdev(values) == pytest.approx(summary[metric]["std"])


def test_frozen_independent_counts_match_completed_outputs_when_present():
    from pathlib import Path

    from experiments.common import read_json

    root = Path("experiments/results/paper-supplement-v1/independent")
    if not root.exists():
        pytest.skip("local supplementary outputs are intentionally not distributed")
    expected = read_json(Path("experiments/expected/independent-counts.json"))
    folders = {"GraphTopic": "graphtopic", "BERTopic": "bertopic"}
    for summary in expected["rows"]:
        dataset, method, target = (summary["dataset"], summary["method"], summary["target_topics"])
        folder = folders.get(method, "baselines")
        prefix = "" if method in folders else f"{method}-"
        paths = sorted(
            (root / dataset / folder / "runs").glob(f"{prefix}topics-{target}-seed-*.json")
        )
        records = [read_json(path) for path in paths]
        assert len(records) == 5
        assert sorted(record["seed"] for record in records) == [11, 23, 37, 51, 71]
        scores = [record["reassigned"] for record in records] if method == "BERTopic" else records
        for metric in ("topics", "ari", "nmi"):
            values = [record[metric] for record in scores]
            assert statistics.mean(values) == pytest.approx(summary[metric]["mean"])
            assert statistics.stdev(values) == pytest.approx(summary[metric]["std"])


def test_frozen_clean_lexical_means_match_completed_outputs_when_present():
    from pathlib import Path

    from experiments.common import read_json

    root = Path("experiments/results/paper-supplement-v1/lexical-html-v2")
    if not root.exists():
        pytest.skip("local supplementary outputs are intentionally not distributed")
    expected = read_json(Path("experiments/expected/lexical-html-v2.json"))
    names = ("npmi_macro", "npmi_document_weighted", "topic_diversity")
    forbidden = {"39", "gt", "lt", "quot", "href", "fullquote"}
    for dataset, groups in expected["metrics"].items():
        for key, means in groups.items():
            family, stem = key.split("|", 1)
            folder = (
                root
                / dataset
                / ("assignments" if family == "graphtopic" else f"{family}/assignments")
            )
            records = [read_json(path) for path in sorted(folder.glob(f"{stem}-seed-*.json"))]
            assert len(records) == 5
            assert all(row["lexical_cleaning"] == "html-entities-v2" for row in records)
            for name, mean in zip(names, means, strict=True):
                assert statistics.mean(row[name] for row in records) == pytest.approx(
                    mean, abs=5e-7
                )
            words = {
                word for row in records for topic in row["topic_words"].values() for word in topic
            }
            assert not words.intersection(forbidden)
