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
from experiments.carry_forward import carry_forward
from experiments.common import atomic_json, load_artifact, parse_embedding
from experiments.graph2topic import _assign_outliers
from experiments.metrics import external_scores, lexical_scores, shared_topic_words
from experiments.qualitative import build_examples
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


def test_exact_recall_does_not_penalize_equivalent_tied_neighbors():
    embeddings = np.ones((3, 2), dtype=np.float32)
    candidates = np.asarray([[2], [2], [1]])
    result = exact_recall(embeddings, candidates, k=1, query_count="all", seed=1)
    assert result["mean"] == 1.0
    assert result["strict_mean"] == 0.0
    assert result["tied_queries"] == 3


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


def test_residual_graph2topic_documents_receive_nearest_centroid():
    embeddings = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    labels = np.asarray([4, -1, 9, -1])
    repaired = _assign_outliers(labels, embeddings)
    assert repaired.tolist() == [4, 4, 9, 9]
    assert labels.tolist() == [4, -1, 9, -1]


def test_qualitative_examples_follow_fixed_size_and_centroid_rules(tmp_path):
    documents = np.asarray(["apple red", "apple green", "goal team", "match goal"])
    embeddings = np.asarray([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]], dtype=np.float32)
    labels = np.asarray([0, 0, 1, 1])
    artifact = tmp_path / "artifact.npz"
    np.savez_compressed(
        artifact,
        documents=documents,
        embeddings=embeddings,
        labels=labels,
    )
    config = {
        "qualitative": {
            "topics_per_dataset": 1,
            "representative_documents": 1,
            "excerpt_characters": 20,
            "sample": {"resolution": 1.0, "seed": 11},
        }
    }
    config_path = tmp_path / "config.json"
    atomic_json(config_path, config)
    assignment = tmp_path / "results" / "sample" / "assignments" / "gamma-1-seed-11.npz"
    assignment.parent.mkdir(parents=True)
    np.savez_compressed(assignment, labels=labels)
    atomic_json(
        tmp_path / "results" / "lexical" / "sample" / "assignments" / "gamma-1-seed-11.json",
        {"topic_words": {"0": ["apple"], "1": ["goal"]}},
    )

    result = build_examples("sample", artifact, config_path, tmp_path / "results")

    assert result["examples"][0]["topic"] == 0
    assert result["examples"][0]["size"] == 2
    assert result["examples"][0]["representatives"][0]["document_id"] == 0


def test_carry_forward_excludes_changed_newsgroups(tmp_path, monkeypatch):
    previous_artifacts = tmp_path / "old-artifacts"
    previous_results = tmp_path / "old-results"
    destination_artifacts = tmp_path / "new-artifacts"
    destination_results = tmp_path / "new-results"
    config = {
        "protocol_id": "paper-v2",
        "sources": {"agnews": {"source": "a"}, "dbpedia14": {"source": "d"}},
        "models": {"primary": {"model": "m"}},
    }
    config_path = tmp_path / "config.json"
    atomic_json(config_path, config)
    atomic_json(previous_results / "comparison.json", {"passed": True})
    for dataset in ("agnews", "dbpedia14"):
        artifact = previous_artifacts / dataset
        atomic_json(
            artifact / "metadata.json",
            {
                "dataset": dataset,
                "source": config["sources"][dataset],
                "model": config["models"]["primary"],
                "embedding_complete": True,
            },
        )
        (artifact / "payload.bin").write_bytes(dataset.encode())
        atomic_json(previous_results / dataset / "result.json", {"dataset": dataset})
    for relative in (
        "ablations/k-resolution/agnews",
        "ablations/graph-design/agnews",
        "lexical/agnews",
        "scaling/agnews",
        "scaling/dbpedia14",
    ):
        atomic_json(previous_results / relative / "result.json", {"complete": True})
    monkeypatch.setattr("experiments.carry_forward.artifact_root", lambda _: destination_artifacts)
    monkeypatch.setattr("experiments.carry_forward.result_root", lambda _: destination_results)

    manifest = carry_forward(previous_artifacts, previous_results, config_path)

    assert manifest["excluded_dataset"] == "20newsgroups"
    assert (destination_artifacts / "agnews" / "payload.bin").exists()
    assert not (destination_artifacts / "20newsgroups").exists()
