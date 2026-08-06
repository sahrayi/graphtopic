from graphtopic.core.models.topic_info import TopicInfo
from graphtopic.graph_topic import GraphTopic
import numpy as np

def test_topic_info_creation_with_required_fields():

    topic = TopicInfo(
        topic_id=1,
    )


    assert topic.topic_id == 1

    assert topic.size == 0

    assert topic.representation == []

    assert topic.document_ids == []



def test_topic_info_creation_with_all_fields():

    topic = TopicInfo(
        topic_id=5,

        size=3,

        representation=[
            ("اقتصاد", 0.9),
            ("بازار", 0.7),
        ],

        document_ids=[
            10,
            11,
            12,
        ],
    )


    assert topic.topic_id == 5

    assert topic.size == 3

    assert topic.representation == [
        ("اقتصاد", 0.9),
        ("بازار", 0.7),
    ]

    assert topic.document_ids == [
        10,
        11,
        12,
    ]



def test_topic_info_default_lists_are_independent():

    topic1 = TopicInfo(
        topic_id=1,
    )

    topic2 = TopicInfo(
        topic_id=2,
    )


    topic1.document_ids.append(
        100
    )

    topic1.representation.append(
        ("test", 1.0)
    )


    assert topic2.document_ids == []

    assert topic2.representation == []



def test_topic_info_equality():

    topic1 = TopicInfo(
        topic_id=1,

        size=2,

        representation=[
            ("اقتصاد", 1.0),
        ],

        document_ids=[
            0,
            1,
        ],
    )


    topic2 = TopicInfo(
        topic_id=1,

        size=2,

        representation=[
            ("اقتصاد", 1.0),
        ],

        document_ids=[
            0,
            1,
        ],
    )


    assert topic1 == topic2



def test_topic_info_allows_empty_representation():

    topic = TopicInfo(
        topic_id=3,
        size=5,
        document_ids=[
            1,
            2,
            3,
            4,
            5,
        ],
    )


    assert topic.representation == []

    assert topic.size == 5

    assert len(topic.document_ids) == 5

def test_graph_topic_topic_info_consistency():

    documents = [
        "اقتصاد و بازار",
        "تورم و قیمت",
        "فوتبال و ورزش",
        "مسابقه فوتبال",
        "تیم ملی فوتبال",
    ]

    embeddings = np.array(
        [
            [1.0, 0.9, 0.0],
            [0.9, 1.0, 0.0],

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


    topic_info = model.get_topic_info()


    assert topic_info is not None

    assert len(topic_info) > 0


    total_documents = sum(
        topic.size
        for topic in topic_info
    )


    assert total_documents == len(documents)


    for topic in topic_info:

        assert isinstance(
            topic.topic_id,
            int,
        )

        assert topic.size == len(
            topic.document_ids
        )

        assert topic.size > 0

        assert isinstance(
            topic.representation,
            list,
        )