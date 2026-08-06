import numpy as np

from graphtopic.graph_topic import GraphTopic

DOCS_SHORT = [
    "اقتصاد",
    "بازار",
    "فوتبال",
    "ورزش",
]


EMBS_SHORT = np.array(
    [
        [1.0, 0.9],
        [0.9, 1.0],
        [0.0, 1.0],
        [0.1, 0.95],
    ],
    dtype=float,
)

def test_graph_topic_separates_clear_topics():

    documents = [
        "اقتصاد و بازار",
        "تورم و بورس",
        "بانک و سرمایه",

        "فوتبال و ورزش",
        "مسابقه فوتبال",
        "تیم ملی فوتبال",
    ]


    embeddings = np.array(
        [
            [1.0, 0.9, 0.8],
            [0.95, 0.85, 0.9],
            [0.9, 1.0, 0.85],

            [0.0, 0.1, 1.0],
            [0.1, 0.0, 0.95],
            [0.0, 0.0, 0.9],
        ],
        dtype=float,
    )


    model = GraphTopic()


    model.fit(
        documents,
        embeddings=embeddings,
    )


    assert len(model.topic_info_) >= 2


    sizes = sorted(
        [
            topic.size
            for topic in model.topic_info_
        ],
        reverse=True,
    )


    assert sizes[0] >= 2
    assert sizes[1] >= 2

def test_graph_topic_groups_similar_documents():

    documents = [
        "اقتصاد",
        "بازار",
        "بورس",
        "سرمایه",
    ]


    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.98, 0.02],
            [0.97, 0.03],
        ],
        dtype=float,
    )


    model = GraphTopic()


    model.fit(
        documents,
        embeddings=embeddings,
    )


    assert len(model.topic_info_) == 1


    assert model.topic_info_[0].size == len(documents)

def test_graph_topic_never_creates_empty_topics():

    documents = [
        "اقتصاد",
        "بازار",
        "فوتبال",
        "ورزش",
    ]


    embeddings = np.array(
        [
            [1.0, 0.9],
            [0.9, 1.0],
            [0.0, 1.0],
            [0.1, 0.95],
        ],
        dtype=float,
    )


    model = GraphTopic()


    model.fit(
        documents,
        embeddings=embeddings,
    )


    for topic in model.topic_info_:

        assert topic.size > 0

        assert len(topic.document_ids) > 0

def test_graph_topic_preserves_all_documents():

    documents = [
        "اقتصاد",
        "بازار",
        "فوتبال",
        "ورزش",
        "فناوری",
    ]


    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.95, 0.05],
            [0.0, 1.0],
            [0.05, 0.95],
            [0.5, 0.5],
        ],
        dtype=float,
    )


    model = GraphTopic()


    model.fit(
        documents,
        embeddings=embeddings,
    )


    assigned_documents = []


    for topic in model.topic_info_:

        assigned_documents.extend(
            topic.document_ids
        )


    assert sorted(assigned_documents) == list(
        range(len(documents))
    )

from graphion.core.models import Graph



def test_graph_topic_handles_sparse_graph():

    documents = [
        "خبر اول",
        "خبر دوم",
        "خبر سوم",
        "خبر چهارم",
    ]


    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ],
        dtype=float,
    )


    class SparseGraphBuilder:

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


    model = GraphTopic(
        graph_builder=SparseGraphBuilder()
    )


    topics = model.fit_transform(
        documents,
        embeddings=embeddings,
    )


    assert len(topics) == len(documents)

    assert model.topic_info_ is not None

    assert sum(
        topic.size
        for topic in model.topic_info_
    ) == len(documents)

def test_graph_topic_ids_are_valid():

    model = GraphTopic()


    model.fit(
        DOCS_SHORT,
        embeddings=EMBS_SHORT,
    )


    for topic_id in model.topics_:

        assert isinstance(
            topic_id,
            int,
        )


    assert min(model.topics_) >= 0

