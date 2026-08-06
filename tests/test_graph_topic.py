import numpy as np
import pytest
from graphion.core.models import Graph, PartitionSet
from graphtopic.graph_topic import GraphTopic

DOCS_SHORT = ["اقتصاد و بازار", "تورم و قیمت", "فوتبال و ورزش", "مسابقه فوتبال"]
EMBS_SHORT = np.array(
    [[1.0, 0.9, 0.0], [0.9, 1.0, 0.0], [0.0, 0.1, 1.0], [0.1, 0.0, 0.95]], dtype=float)

def test_graph_topic_pipeline():
    documents = [
        "رئیس جمهور درباره اقتصاد کشور صحبت کرد",
        "دولت برنامه جدید اقتصادی اعلام کرد",
        "وزیر اقتصاد درباره تورم توضیح داد",
        "تیم فوتبال ایران در مسابقه پیروز شد",
        "بازیکنان تیم ملی فوتبال تمرین کردند",
        "مسابقه فوتبال قهرمانی برگزار شد",
    ]
    embeddings = np.array(
        [
            [1.0, 0.9, 0.8, 0.0],
            [0.9, 1.0, 0.85, 0.0],
            [0.95, 0.85, 1.0, 0.0],
            [0.0, 0.0, 0.1, 1.0],
            [0.0, 0.1, 0.0, 0.95],
            [0.1, 0.0, 0.0, 1.0],
        ]
    )
    topic_model = GraphTopic(graph_builder=None)
    topics = topic_model.fit_transform(documents, embeddings=embeddings)
    assert len(topics) == len(documents)
    assert topic_model.graph_ is not None
    assert topic_model.partition_set_ is not None
    assert topic_model.topic_info_ is not None
    assert len(topic_model.topic_info_) >= 1
    for topic in topic_model.topic_info_:
        assert topic.topic_id is not None
        assert topic.size > 0
        assert len(topic.document_ids) == topic.size
        assert isinstance(topic.representation, list)

def test_graph_topic_with_custom_components():
    topic_model = GraphTopic()
    topic_model.fit(DOCS_SHORT, EMBS_SHORT[:, :3])
    info = topic_model.get_topic_info()
    assert isinstance(info, list)
    assert all(hasattr(topic, "topic_id") for topic in info)

def test_graph_topic_with_custom_embedding_model():
    class DummyEmbeddingModel:
        def __init__(self):
            self.called = False
        def encode(self, documents):
            self.called = True
            return EMBS_SHORT[:, :3]
    embedding_model = DummyEmbeddingModel()
    topic_model = GraphTopic(embedding_model=embedding_model)
    topics = topic_model.fit_transform(DOCS_SHORT)
    assert embedding_model.called is True
    assert len(topics) == len(DOCS_SHORT)
    assert all(
        getattr(topic_model, attr) is not None
        for attr in ["embeddings_", "graph_", "partition_set_", "topic_info_"]
    )

def test_graph_topic_with_custom_representation_model():
    class DummyRepresentationModel:
        def __init__(self):
            self.fit_called = False
            self.received_documents = None
            self.received_topics = None
        def fit(self, documents, topic_ids):
            self.fit_called = True
            self.received_documents = documents
            self.received_topics = topic_ids
            return self
        def get_topic(self, topic_id):
            return [("custom_word", 1.0), ("another_word", 0.5)]
    representation_model = DummyRepresentationModel()
    topic_model = GraphTopic(representation_model=representation_model)
    topic_model.fit(DOCS_SHORT, embeddings=EMBS_SHORT[:, :3])
    assert representation_model.fit_called is True
    assert representation_model.received_documents == DOCS_SHORT
    assert len(representation_model.received_topics) == len(DOCS_SHORT)
    assert topic_model.topic_info_ is not None
    assert all(
        topic.representation == [("custom_word", 1.0), ("another_word", 0.5)]
        for topic in topic_model.topic_info_
    )

def test_graph_topic_with_custom_partition_detector():
    class DummyPartitionDetector:
        def __init__(self):
            self.called = False
        def detect(self, graph):
            self.called = True
            return PartitionSet.from_sets([graph.nodes])
    detector = DummyPartitionDetector()
    topic_model = GraphTopic(partition_detector=detector)
    topics = topic_model.fit_transform(DOCS_SHORT, embeddings=EMBS_SHORT[:, :3])
    assert detector.called is True
    assert len(set(topics)) == 1
    assert topic_model.partition_set_ is not None
    assert len(topic_model.topic_info_) == 1

def test_graph_topic_with_custom_graph_builder():
    class DummyGraphBuilder:
        def __init__(self):
            self.called = False
        def build(self, relations, nodes=None):
            self.called = True
            return Graph(nodes=tuple(nodes), edges=tuple(), directed=False)
    graph_builder = DummyGraphBuilder()
    topic_model = GraphTopic(graph_builder=graph_builder)
    topic_model.fit(DOCS_SHORT, embeddings=EMBS_SHORT[:, :3])
    assert graph_builder.called is True
    assert topic_model.graph_ is not None
    assert topic_model.graph_.node_count == len(DOCS_SHORT)
    assert topic_model.partition_set_ is not None
    assert topic_model.topic_info_ is not None

def test_graph_topic_rejects_invalid_embedding_count():
    topic_model = GraphTopic()
    with pytest.raises(ValueError, match="Number of embeddings must match"):
        topic_model.fit(
            ["خبر اول", "خبر دوم", "خبر سوم"],
            embeddings=np.array([[1.0, 0.0], [0.0, 1.0]]),
        )

def test_graph_topic_uses_embedding_model():
    class FakeEmbeddingModel:
        def encode(self, documents):
            assert len(documents) == 4
            return np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    model = GraphTopic(embedding_model=FakeEmbeddingModel())
    topics = model.fit_transform(["اقتصاد", "بازار", "فوتبال", "ورزش"])
    assert len(topics) == 4
    assert all(
        getattr(model, attr) is not None
        for attr in ["embeddings_", "graph_", "topic_info_"]
    )

def test_graph_topic_requires_embedding_source():
    topic_model = GraphTopic()
    with pytest.raises(ValueError, match="Either embeddings or embedding_model"):
        topic_model.fit(["خبر اول", "خبر دوم"])

def test_graph_topic_rejects_empty_documents():
    topic_model = GraphTopic()
    with pytest.raises(
        ValueError,
        match="documents cannot be empty",
    ):
        topic_model.fit([], embeddings=np.array([]))

def test_graph_topic_is_deterministic():
    documents = [
        "اقتصاد و بازار",
        "تورم و قیمت",
        "فوتبال و ورزش",
        "مسابقه فوتبال",
        "تیم ملی فوتبال",
        "بانک و اقتصاد",
    ]
    embeddings = np.array(
        [
            [1.0, 0.9, 0.0],
            [0.9, 1.0, 0.0],
            [0.0, 0.1, 1.0],
            [0.1, 0.0, 0.95],
            [0.0, 0.0, 0.9],
            [0.95, 0.85, 0.0],
        ],
        dtype=float,
    )
    model1 = GraphTopic()
    topics1 = model1.fit_transform(
        documents,
        embeddings=embeddings,
    )
    model2 = GraphTopic()
    topics2 = model2.fit_transform(
        documents,
        embeddings=embeddings,
    )
    assert topics1 == topics2
    assert model1.topic_info_ == model2.topic_info_

def test_graph_topic_can_be_fitted_multiple_times():
    first_documents = [
        "اقتصاد و بازار",
        "تورم و قیمت",
        "فوتبال و ورزش",
        "مسابقه فوتبال",
    ]
    first_embeddings = np.array(
        [
            [1.0, 0.9, 0.0],
            [0.9, 1.0, 0.0],
            [0.0, 0.1, 1.0],
            [0.1, 0.0, 0.95],
        ],
        dtype=float,
    )
    second_documents = [
        "هواشناسی امروز",
        "پیش بینی باران",
        "فناوری جدید",
    ]
    second_embeddings = np.array(
        [
            [1.0, 0.9, 0.0],
            [0.95, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    model = GraphTopic()
    model.fit(
        first_documents,
        embeddings=first_embeddings,
    )
    first_topic_info = model.topic_info_
    model.fit(
        second_documents,
        embeddings=second_embeddings,
    )
    assert model.documents_ == second_documents
    assert len(model.embeddings_) == len(second_documents)
    assert len(model.topics_) == len(second_documents)
    assert model.graph_ is not None
    assert model.partition_set_ is not None
    assert model.topic_info_ is not None
    assert model.topic_info_ != first_topic_info

def test_graph_topic_rejects_empty_documents_duplicate():
    topic_model = GraphTopic()
    with pytest.raises(ValueError, match="documents cannot be empty"):
        topic_model.fit([], embeddings=np.empty((0, 3)))

def test_graph_topic_rejects_invalid_embedding_dimensions():
    documents = [
        "خبر اول",
        "خبر دوم",
        "خبر سوم",
    ]
    embeddings = [
        [1.0, 0.0, 0.5],
        [0.9, 0.1],
        [0.2, 0.8, 0.4],
    ]
    topic_model = GraphTopic()
    with pytest.raises(ValueError, match="Invalid embedding shape"):
        topic_model.fit(
            documents,
            embeddings=embeddings,
        )

def test_graph_topic_rejects_invalid_embedding_model_output():
    class BadEmbeddingModel:
        def encode(self, documents):
            return np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            )
    topic_model = GraphTopic(embedding_model=BadEmbeddingModel())
    with pytest.raises(ValueError, match="Number of embeddings must match"):
        topic_model.fit(
            [
                "اقتصاد",
                "بازار",
                "فوتبال",
                "ورزش",
            ]
        )

def test_graph_topic_handles_graph_without_edges():
    documents = [
        "موضوع اول",
        "موضوع دوم",
        "موضوع سوم",
    ]
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )
    class EmptyGraphBuilder:
        def build(
            self,
            relations,
            nodes=None,
        ):
            return Graph(
                nodes=tuple(nodes),
                edges=tuple(),
                directed=False,
            )
    topic_model = GraphTopic(graph_builder=EmptyGraphBuilder())
    topics = topic_model.fit_transform(
        documents,
        embeddings=embeddings,
    )
    assert len(topics) == len(documents)
    assert topic_model.graph_.edge_count == 0
    assert topic_model.topic_info_ is not None
    assert len(topic_model.topic_info_) == len(documents)

def test_graph_topic_handles_duplicate_documents():
    documents = [
        "فوتبال ایران",
        "فوتبال ایران",
        "اقتصاد کشور",
    ]
    embeddings = np.array(
        [
            [1.0, 0.9],
            [1.0, 0.9],
            [0.0, 1.0],
        ]
    )
    topic_model = GraphTopic()
    topics = topic_model.fit_transform(
        documents,
        embeddings=embeddings,
    )
    assert len(topics) == 3
    assert topic_model.topic_info_ is not None
    assert sum(topic.size for topic in topic_model.topic_info_) == 3

def test_graph_topic_refit_resets_state():
    model = GraphTopic()
    documents_1 = [
        "اقتصاد",
        "بازار",
    ]
    embeddings_1 = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
        ]
    )
    model.fit(
        documents_1,
        embeddings=embeddings_1,
    )
    first_topics = model.topics_
    documents_2 = [
        "فوتبال",
        "ورزش",
        "مسابقه",
    ]
    embeddings_2 = np.array(
        [
            [0.0, 1.0],
            [0.1, 0.9],
            [0.2, 0.8],
        ]
    )
    model.fit(
        documents_2,
        embeddings=embeddings_2,
    )
    assert model.documents_ == documents_2
    assert len(model.topics_) == 3
    assert len(model.topic_info_) >= 1
    assert model.topics_ != first_topics

def test_graph_topic_with_custom_reducer():

    class DummyReducer:

        def __init__(self):
            self.called = False


        def reduce(self, feature_set):
            self.called = True
            return feature_set


    reducer = DummyReducer()

    model = GraphTopic(
        reducer=reducer
    )


    model.fit(
        DOCS_SHORT,
        embeddings=EMBS_SHORT[:, :3],
    )


    assert reducer.called is True

def test_graph_topic_with_custom_relation_builder():

    class DummyRelationBuilder:

        def __init__(self):
            self.called = False


        def build(self, feature_set):
            self.called = True

            from graphion.core.models import RelationSet

            return RelationSet([])


    relation_builder = DummyRelationBuilder()


    class DummyGraphBuilder:

        def build(self, relations, nodes=None):
            return Graph(
                nodes=tuple(nodes),
                edges=tuple(),
                directed=False,
            )


    model = GraphTopic(
        relation_builder=relation_builder,
        graph_builder=DummyGraphBuilder(),
    )


    model.fit(
        DOCS_SHORT,
        embeddings=EMBS_SHORT[:, :3],
    )


    assert relation_builder.called is True

def test_graph_topic_with_custom_graph_refiner():

    class DummyGraphRefiner:

        def __init__(self):
            self.called = False


        def refine(self, graph):
            self.called = True
            return graph


    refiner = DummyGraphRefiner()


    model = GraphTopic(
        graph_refiner=refiner
    )


    model.fit(
        DOCS_SHORT,
        embeddings=EMBS_SHORT[:, :3],
    )


    assert refiner.called is True

def test_graph_topic_with_custom_partition_refiner():

    class DummyPartitionRefiner:

        def __init__(self):
            self.called = False


        def refine(self, partition):
            self.called = True
            return partition


    refiner = DummyPartitionRefiner()


    model = GraphTopic(
        partition_refiner=refiner
    )


    model.fit(
        DOCS_SHORT,
        embeddings=EMBS_SHORT[:, :3],
    )


    assert refiner.called is True

def test_graph_topic_stores_feature_set():

    model = GraphTopic()

    model.fit(
        DOCS_SHORT,
        embeddings=EMBS_SHORT[:, :3],
    )


    assert model.feature_set_ is not None

def test_graph_topic_document_ids_are_created():

    model = GraphTopic()

    model.fit(
        DOCS_SHORT,
        embeddings=EMBS_SHORT[:, :3],
    )


    assert model.document_ids_ == (
        0,
        1,
        2,
        3,
    )

def test_graph_topic_fit_transform_returns_topic_labels():

    model = GraphTopic()

    topics = model.fit_transform(
        DOCS_SHORT,
        embeddings=EMBS_SHORT[:, :3],
    )


    assert isinstance(topics, list)

    assert all(
        isinstance(topic, int)
        for topic in topics
    )

def test_graph_topic_get_topic_info_before_fit():

    model = GraphTopic()

    with pytest.raises(
        ValueError,
        match="Model is not fitted",
    ):
        model.get_topic_info()