
import numpy as np
import pytest
from graphtopic.graph_topic import GraphTopic

def test_graph_topic_document_ids_are_sequential():

    documents = [
        "خبر اول",
        "خبر دوم",
        "خبر سوم",
    ]


    embeddings = np.array(
        [
            [1,0],
            [0,1],
            [0.5,0.5],
        ],
        dtype=float,
    )


    model = GraphTopic()

    model.fit(
        documents,
        embeddings=embeddings,
    )


    assert model.document_ids_ == (
        0,
        1,
        2,
    )

def test_graph_topic_preserves_input_embeddings():

    embeddings = np.array(
        [
            [1,0],
            [0,1],
        ],
        dtype=float,
    )


    model = GraphTopic()


    model.fit(
        [
            "الف",
            "ب",
        ],
        embeddings=embeddings,
    )


    assert np.array_equal(
        model.embeddings_,
        embeddings,
    )

def test_graph_topic_fit_returns_self():

    model = GraphTopic()


    result = model.fit(
        [
            "اقتصاد",
            "بازار",
        ],
        embeddings=np.array(
            [
                [1,0],
                [0.9,0.1],
            ]
        ),
    )


    assert result is model

def test_fit_transform_returns_topics():

    model = GraphTopic()


    topics = model.fit_transform(
        [
            "اقتصاد",
            "بازار",
        ],
        embeddings=np.array(
            [
                [1,0],
                [0.9,0.1],
            ]
        ),
    )


    assert topics == model.topics_

def test_graph_topic_requires_representation_get_topic():

    class BadRepresentation:

        def fit(
            self,
            documents,
            topics,
        ):
            return self


    model = GraphTopic(
        representation_model=BadRepresentation()
    )


    with pytest.raises(AttributeError):

        model.fit(
            [
                "اقتصاد",
                "بازار",
            ],
            embeddings=np.array(
                [
                    [1,0],
                    [0.9,0.1],
                ]
            ),
        )

def test_topic_info_is_sorted_by_topic_id():

    model = GraphTopic()


    model.fit(
        [
            "اقتصاد",
            "بازار",
            "فوتبال",
        ],
        embeddings=np.array(
            [
                [1,0],
                [0.9,0.1],
                [0,1],
            ]
        ),
    )


    ids = [
        topic.topic_id
        for topic in model.topic_info_
    ]


    assert ids == sorted(ids)

def test_topic_info_covers_all_documents():

    documents = [
        "اقتصاد",
        "بازار",
        "فوتبال",
        "ورزش",
    ]


    model = GraphTopic()


    model.fit(
        documents,
        embeddings=np.array(
            [
                [1,0],
                [0.9,0.1],
                [0,1],
                [0.1,0.9],
            ]
        ),
    )


    total = sum(
        topic.size
        for topic in model.topic_info_
    )


    assert total == len(documents)

