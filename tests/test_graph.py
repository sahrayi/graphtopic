import numpy as np
import pytest
from scipy import sparse

from graphtopic import DocumentGraph, ExactNeighborSearch, UnionMaxGraph
from graphtopic._validation import normalize_embeddings


def test_rescoring_cleaning_union_max():
    x = normalize_embeddings([[1, 0], [1, 1], [0, 1], [-1, 0]], 4)
    rows = [[0, 2, 1, 1], [1, 0], [2, 1], [3, 0]]
    graph = UnionMaxGraph(1).build(x, rows, document_ids=("a", "b", "c", "d"))
    a = graph.adjacency.toarray()
    assert a[0, 1] == pytest.approx(1 / np.sqrt(2))
    assert a[1, 2] == pytest.approx(1 / np.sqrt(2))
    assert a[0, 2] == a[0, 3] == 0
    assert graph.n_edges == 2
    assert graph.diagnostics["max_degree"] == 2  # final degree can exceed k
    assert graph.diagnostics["isolated_vertices"] == 1
    assert graph.diagnostics["candidate_shortages"] == 1
    assert np.array_equal(a, a.T)


@pytest.mark.parametrize(
    "rows",
    [
        [[1]],
        [[1.0], [0], [0]],
        [[-1], [0], [0]],
        [[3], [0], [0]],
        [[[1]], [0], [0]],
    ],
)
def test_bad_candidates(rows):
    with pytest.raises(ValueError, match="neighbor_search"):
        UnionMaxGraph(1).build(np.eye(3), rows, document_ids=(0, 1, 2))


@pytest.mark.parametrize(
    "a",
    [
        [[0, 1], [0, 0]],
        [[1, 0], [0, 0]],
        [[0, -1], [-1, 0]],
        [[0, float("nan")], [float("nan"), 0]],
        [[0, 1j], [1j, 0]],
    ],
)
def test_invalid_graph(a):
    with pytest.raises(ValueError):
        DocumentGraph(sparse.csr_matrix(a), (0, 1))


def test_graph_shape_dense_zero_and_duplicates():
    with pytest.raises(ValueError):
        DocumentGraph(np.eye(2), (0, 1))
    with pytest.raises(ValueError):
        DocumentGraph(sparse.eye(3), (0, 1))
    zero = sparse.csr_matrix(([0.0], ([0], [1])), shape=(2, 2))
    with pytest.raises(ValueError, match="positive"):
        DocumentGraph(zero, (0, 1))
    duplicate = sparse.csr_matrix(([1.0, 1.0], [1, 1], [0, 2, 2]), shape=(2, 2))
    with pytest.raises(ValueError, match="duplicate"):
        DocumentGraph(duplicate, (0, 1))


def test_graph_defensive_copy():
    original = sparse.csr_matrix([[0.0, 2], [2, 0]])
    graph = DocumentGraph(original, ("x", "y"))
    original.data[:] = 99
    a = graph.adjacency
    a.data[:] = 100
    graph.diagnostics["n_edges"] = 99
    assert graph.adjacency[0, 1] == 2
    assert graph.diagnostics["n_edges"] == 1
    assert graph.copy().document_ids == ("x", "y")


def test_exact_matches_brute_force_and_ties():
    rng = np.random.default_rng(4)
    x = normalize_embeddings(rng.normal(size=(30, 7)), 30)
    actual = ExactNeighborSearch(5, batch_size=3).search(x)
    for i, row in enumerate(actual):
        scores = np.clip(x @ x[i], -1, 1)
        scores[i] = -np.inf
        expected = np.lexsort((np.arange(30), -scores))[:5]
        assert np.array_equal(row, expected)
    duplicate = np.ones((5, 2), dtype=np.float32) / np.sqrt(2)
    rows = ExactNeighborSearch(2).search(duplicate)
    assert rows[0].tolist() == [1, 2]
    assert rows[4].tolist() == [0, 1]


def test_shortage_and_empty_candidates():
    graph = UnionMaxGraph(1).build(np.eye(2), [[], []], document_ids=(0, 1))
    assert graph.n_edges == 0
    assert graph.diagnostics["candidate_shortages"] == 2
