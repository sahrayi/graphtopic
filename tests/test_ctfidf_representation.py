from graphtopic.representation_models.ctfidf import (
    CTFIDFRepresentation,
)


def test_ctfidf_fit_creates_topic_representations():

    documents = [
        "اقتصاد بازار بورس",
        "تورم اقتصاد کشور",

        "فوتبال مسابقه تیم",
        "بازیکنان فوتبال تمرین",
    ]

    topic_ids = [
        0,
        0,
        1,
        1,
    ]


    model = CTFIDFRepresentation(
        top_n_words=5
    )


    result = model.fit(
        documents,
        topic_ids,
    )


    assert result is model


    topic_0 = model.get_topic(0)

    topic_1 = model.get_topic(1)


    assert isinstance(
        topic_0,
        list,
    )

    assert isinstance(
        topic_1,
        list,
    )


    assert len(topic_0) > 0

    assert len(topic_1) > 0


def test_ctfidf_respects_top_n_words_limit():

    documents = [
        "اقتصاد بازار بورس سهام سرمایه",
        "تورم اقتصاد قیمت بازار",

        "فوتبال تیم مسابقه بازیکن",
        "ورزش فوتبال لیگ قهرمانی",
    ]

    topic_ids = [
        0,
        0,
        1,
        1,
    ]


    model = CTFIDFRepresentation(
        top_n_words=3
    )


    model.fit(
        documents,
        topic_ids,
    )


    assert len(
        model.get_topic(0)
    ) <= 3


    assert len(
        model.get_topic(1)
    ) <= 3



def test_ctfidf_get_topic_returns_empty_for_unknown_topic():

    documents = [
        "اقتصاد بازار",
        "فوتبال مسابقه",
    ]

    topic_ids = [
        0,
        1,
    ]


    model = CTFIDFRepresentation()


    model.fit(
        documents,
        topic_ids,
    )


    result = model.get_topic(
        999
    )


    assert result == []



def test_ctfidf_representation_contains_word_and_score():

    documents = [
        "اقتصاد اقتصاد بازار",
        "فوتبال فوتبال تیم",
    ]

    topic_ids = [
        0,
        1,
    ]


    model = CTFIDFRepresentation()


    model.fit(
        documents,
        topic_ids,
    )


    topic = model.get_topic(
        0
    )


    assert len(topic) > 0


    word, score = topic[0]


    assert isinstance(
        word,
        str,
    )


    assert isinstance(
        score,
        float,
    )


def test_ctfidf_handles_single_document_topic():

    documents = [
        "اقتصاد بازار",
        "فوتبال مسابقه",
        "هواشناسی باران",
    ]

    topic_ids = [
        0,
        1,
        2,
    ]


    model = CTFIDFRepresentation()


    model.fit(
        documents,
        topic_ids,
    )


    assert len(
        model.get_topic(0)
    ) > 0

    assert len(
        model.get_topic(1)
    ) > 0

    assert len(
        model.get_topic(2)
    ) > 0