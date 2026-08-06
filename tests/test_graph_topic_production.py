import pickle

import numpy as np
import pytest

from graphtopic.graph_topic import GraphTopic


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def create_model():

    documents = [
        "اقتصاد و بازار",
        "تورم و قیمت",
        "فوتبال و ورزش",
        "مسابقه فوتبال",
    ]

    embeddings = np.array(
        [
            [1.0, 0.9, 0.0],
            [0.9, 1.0, 0.0],
            [0.0, 0.1, 1.0],
            [0.1, 0.0, 0.95],
        ],
        dtype=float,
    )

    model = GraphTopic()

    model.fit(
        documents,
        embeddings=embeddings,
    )

    return model, documents, embeddings


# --------------------------------------------------
# 1. Serialization
# --------------------------------------------------

def test_graph_topic_can_be_serialized():

    model, _, _ = create_model()

    data = pickle.dumps(model)

    loaded_model = pickle.loads(
        data
    )

    assert loaded_model.topic_info_ is not None

    assert (
        loaded_model.get_topic_info()
        ==
        model.get_topic_info()
    )

    assert loaded_model.topics_ == model.topics_



# --------------------------------------------------
# 2. Input type validation
# --------------------------------------------------

def test_graph_topic_rejects_non_list_documents():

    model = GraphTopic()

    with pytest.raises(
        ValueError
    ):
        model.fit(
            "اقتصاد",
            embeddings=np.array(
                [
                    [1.0, 0.0]
                ]
            ),
        )



def test_graph_topic_rejects_invalid_document_items():

    model = GraphTopic()

    with pytest.raises(
        ValueError
    ):
        model.fit(
            [
                "اقتصاد",
                123,
            ],
            embeddings=np.array(
                [
                    [1.0, 0.0],
                    [0.9, 0.1],
                ]
            ),
        )



# --------------------------------------------------
# 3. Empty embedding validation
# --------------------------------------------------

def test_graph_topic_rejects_empty_embeddings():

    model = GraphTopic()

    with pytest.raises(
        ValueError
    ):
        model.fit(
            [
                "اقتصاد",
                "بازار",
            ],
            embeddings=np.array([]),
        )



# --------------------------------------------------
# 4. Large input smoke test
# --------------------------------------------------

def test_graph_topic_large_input_smoke():

    documents = [
        f"خبر شماره {i}"
        for i in range(1000)
    ]


    embeddings = np.random.rand(
        1000,
        10,
    )


    model = GraphTopic()


    topics = model.fit_transform(
        documents,
        embeddings=embeddings,
    )


    assert len(topics) == 1000

    assert model.graph_ is not None

    assert model.partition_set_ is not None

    assert model.topic_info_ is not None



# --------------------------------------------------
# 5. Topic labels consistency
# --------------------------------------------------

def test_graph_topic_labels_are_consistent_with_topic_info():

    model, documents, _ = create_model()


    assert len(model.topics_) == len(
        documents
    )


    topic_ids_from_info = {
        topic.topic_id
        for topic in model.topic_info_
    }


    topic_ids_from_labels = set(
        model.topics_
    )


    assert (
        topic_ids_from_labels
        ==
        topic_ids_from_info
    )


    for topic in model.topic_info_:

        assert topic.size == (
            model.topics_.count(
                topic.topic_id
            )
        )


        assert len(
            topic.document_ids
        ) == topic.size