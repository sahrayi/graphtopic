import json
from argparse import Namespace

import numpy as np
import pytest
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from experiments import environment as experiment_environment
from experiments import paper
from experiments.ablations import variant_graph
from experiments.baselines import assign_remaining_outliers, reference_topics
from experiments.common import atomic_json, load_artifact, parse_embedding
from experiments.metrics import external_scores, lexical_scores, shared_topic_words
from experiments.run import exact_recall, run_core


def test_parse_embedding_and_npz_audit(tmp_path):
    assert np.allclose(parse_embedding("[1, 2, 3]"), [1, 2, 3])
    artifact = tmp_path / "artifact.npz"
    np.savez_compressed(
        artifact,
        documents=np.asarray(["alpha", "beta"]),
        embeddings=np.asarray([[1, 0], [0, 1]], dtype=np.float32),
        labels=np.asarray([0, 1]),
    )
    documents, embeddings, labels, audit = load_artifact(artifact)
    assert documents == ["alpha", "beta"]
    assert embeddings.shape == (2, 2)
    assert labels.tolist() == [0, 1]
    assert audit["sha256"] and audit["documents"] == 2


def test_atomic_json_replaces_output(tmp_path):
    path = tmp_path / "report.json"
    atomic_json(path, {"value": 1})
    atomic_json(path, {"value": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 2}


def test_external_and_lexical_metrics():
    documents = [
        "apple fruit orchard",
        "apple fruit tree",
        "football goal team",
        "football match team",
    ] * 2
    labels = np.asarray([0, 0, 1, 1] * 2)
    assert external_scores(labels, labels)["ari"] == 1.0
    counts, words, vectorizer = shared_topic_words(
        documents,
        labels,
        vectorizer=CountVectorizer(min_df=1, stop_words="english"),
    )
    result = lexical_scores(counts, labels, words, vectorizer.get_feature_names_out())
    assert set(result) == {"npmi_macro", "npmi_document_weighted", "topic_diversity"}
    assert 0 < result["topic_diversity"] <= 1


def test_exact_recall_is_one_for_exact_candidates():
    embeddings = np.eye(4, dtype=np.float32)
    candidates = np.asarray([[1, 2], [0, 2], [0, 1], [0, 1]])
    result = exact_recall(embeddings, candidates, k=2, query_count="all", seed=1)
    assert result["mean"] == 1.0


def test_graph_ablation_variants_have_expected_edges():
    directed = sparse.csr_matrix(np.asarray([[0.0, 0.8, 0.0], [0.6, 0.0, 0.7], [0.5, 0.0, 0.0]]))
    union = variant_graph(directed, "union_max")
    mutual = variant_graph(directed, "mutual")
    assert union.nnz // 2 == 3
    assert mutual.nnz // 2 == 1
    assert union[0, 1] == 0.8


@pytest.mark.integration
def test_resumable_core_runner_on_tiny_artifact(tmp_path):
    documents = np.asarray(
        ["red apple", "green apple", "fresh fruit", "football goal", "soccer team", "match goal"]
    )
    embeddings = np.asarray(
        [[1, 0], [0.98, 0.02], [0.9, 0.1], [0, 1], [0.02, 0.98], [0.1, 0.9]],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    artifact = tmp_path / "tiny.npz"
    np.savez_compressed(artifact, documents=documents, embeddings=embeddings, labels=labels)
    config = {
        "candidate_neighbors": 3,
        "retained_neighbors": 2,
        "ann": {"random_state": 7, "n_trees": 5, "n_iters": 5, "n_jobs": 1},
        "leiden_seeds": [11, 23],
        "leiden_iterations": 2,
        "datasets": {
            "tiny": {
                "documents": 6,
                "classes": 2,
                "dimensions": 2,
                "resolutions": [0.5],
                "recall_queries": "all",
            }
        },
    }
    config_path = tmp_path / "config.json"
    atomic_json(config_path, config)
    args = Namespace(
        config=config_path,
        dataset="tiny",
        artifact=artifact,
        results=tmp_path / "results",
        cache=tmp_path / "cache",
        force=False,
    )
    first = run_core(args)
    second = run_core(args)
    assert first["resolutions"]["0.5"]["runs"] == 2
    assert second["resolutions"]["0.5"]["runs"] == 2
    assert (tmp_path / "results" / "tiny" / "summary.json").exists()


def test_paper_runner_reconciles_checkpoint_with_outputs(tmp_path, monkeypatch):
    artifact = tmp_path / "artifact.npz"
    artifact.touch()
    paths = tmp_path / "paths.json"
    atomic_json(
        paths,
        {
            "newsgroups_artifact": str(artifact),
            "newsgroups_text": str(artifact),
            "agnews_artifact": str(artifact),
            "agnews_text": str(artifact),
            "dbpedia_artifact": str(artifact),
            "alternate_encoder_artifact": str(artifact),
        },
    )
    checkpoint = tmp_path / "checkpoint.json"
    stage = paper.Stage("finished", "fixture", ("unused",), lambda: True)
    monkeypatch.setattr(paper, "CHECKPOINT", checkpoint)
    monkeypatch.setattr(paper, "stages", lambda _, bootstrap: [stage])
    monkeypatch.setattr(paper, "validate_reference_environment", lambda: {"locked": True})

    assert paper.main(["--paths", str(paths)]) == 0
    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert saved["stages"]["finished"]["status"] == "complete"
    assert saved["reference_environment"] == {"locked": True}


def test_reference_environment_rejects_version_drift(tmp_path, monkeypatch):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text(
        "--only-binary=example-package\nexample-package==1.2.3\n",
        encoding="utf-8",
    )
    lock = tmp_path / "lock.json"
    lock.write_text(
        json.dumps({"python": "0.0.0", "requirements": "requirements.txt"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(experiment_environment, "LOCK", lock)
    monkeypatch.setattr(experiment_environment, "REPOSITORY", tmp_path)
    with pytest.raises(RuntimeError, match="locked reference environment"):
        experiment_environment.validate_reference_environment()


def test_baseline_topic_counts_are_rounded_core_medians(tmp_path):
    runs = tmp_path / "sample" / "runs"
    for seed, topics in zip((11, 23, 37, 51, 71), (4, 5, 5, 6, 20), strict=True):
        atomic_json(runs / f"gamma-0.1-seed-{seed}.json", {"topics": topics})
    assert reference_topics(tmp_path, "sample", [0.1]) == [5]


def test_residual_bertopic_outliers_receive_nearest_centroid():
    embeddings = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    labels = np.asarray([4, -1, 9, -1])
    repaired = assign_remaining_outliers(labels, embeddings)
    assert repaired.tolist() == [4, 4, 9, 9]
    assert labels.tolist() == [4, -1, 9, -1]
