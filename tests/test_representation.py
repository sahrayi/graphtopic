import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import CountVectorizer

from graphtopic import ClassTfidfTransformer, CTFIDFRepresentation


def test_hand_computed_formula_includes_empty_class():
    docs = ["apple apple pear", "pear", "banana", ""]
    rep = CTFIDFRepresentation(CountVectorizer()).fit(docs, [0, 0, 1, 2])
    words = rep.get_topics()
    # Class lengths 4, 1, 0; A=5/3. Corpus counts apple=2, pear=2, banana=1.
    assert dict(words[0])["apple"] == pytest.approx(0.5 * np.log1p((5 / 3) / 2))
    assert words[0][0][0] == "apple"  # lexical tie breaker
    assert dict(words[1])["banana"] == pytest.approx(np.log1p(5 / 3))
    assert words[2] == []
    assert len(words[1]) == 1  # no zero-weight padding
    words[0].clear()
    assert rep.get_topic(0)


def test_class_tfidf_transformer_matches_hand_formula():
    counts = sparse.csr_matrix([[2.0, 2.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    transformer = ClassTfidfTransformer()
    actual = transformer.fit_transform(counts).toarray()
    average = 5.0 / 3.0
    expected_idf = np.log1p(average / np.array([2.0, 2.0, 1.0]))
    expected = np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    expected *= expected_idf
    assert np.allclose(actual, expected)
    assert transformer.average_class_length_ == pytest.approx(average)


def test_custom_ctfidf_transformer_is_used():
    class DoubleCounts:
        def fit_transform(self, class_counts):
            self.called = True
            return class_counts * 2

    rep = CTFIDFRepresentation(CountVectorizer(), DoubleCounts()).fit(["apple", "banana"], [0, 1])
    assert rep.ctfidf_model_.called
    assert rep.get_topic(0) == [("apple", 2.0)]


@pytest.mark.parametrize("matrix", [np.ones((2, 2)), sparse.csr_matrix([[1, -1]])])
def test_class_tfidf_rejects_invalid_counts(matrix):
    with pytest.raises(ValueError):
        ClassTfidfTransformer().fit(matrix)


def test_class_tfidf_feature_mismatch():
    transformer = ClassTfidfTransformer().fit(sparse.csr_matrix([[1.0, 2.0]]))
    with pytest.raises(ValueError, match="expected"):
        transformer.transform(sparse.csr_matrix([[1.0]]))


def test_default_empty_small_and_refit_reset():
    rep = CTFIDFRepresentation()
    with pytest.warns(UserWarning, match="Vocabulary"):
        rep.fit(["apple"], [0])
    assert rep.get_topics() == {0: []}
    rep = CTFIDFRepresentation(CountVectorizer()).fit(["apple", "pear"], [8, 9])
    rep.fit(["banana"], [2])
    assert set(rep.get_topics()) == {2}


def test_empty_and_pruned_vocabulary():
    for vectorizer, docs in [
        (CountVectorizer(), ["", "a"]),
        (CountVectorizer(min_df=2), ["apple", "pear"]),
    ]:
        with pytest.warns(UserWarning, match="Vocabulary"):
            rep = CTFIDFRepresentation(vectorizer).fit(docs, [0, 1])
        assert rep.get_topics() == {0: [], 1: []}


@pytest.mark.parametrize("parameter", [0, -1, True, 1.2])
def test_bad_top_n(parameter):
    with pytest.raises(ValueError):
        CTFIDFRepresentation(top_n_words=parameter)


def test_invalid_vectorizer_not_swallowed():
    with pytest.raises(ValueError):
        CTFIDFRepresentation(CountVectorizer(min_df=-1)).fit(["apple"], [0])
    with pytest.raises(ValueError):
        CTFIDFRepresentation().fit(["apple"], [-1])
    with pytest.raises(NotFittedError):
        CTFIDFRepresentation().get_topics()
