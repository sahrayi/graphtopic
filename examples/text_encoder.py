"""Requires the embedding extra and downloads a model on first use."""

from sklearn.feature_extraction.text import CountVectorizer

from graphtopic import CTFIDFRepresentation, GraphTopic, SentenceTransformerEmbedding


def main():
    documents = [
        "Apples and pears grow on trees.",
        "Fresh fruit contains vitamins.",
        "The striker scored a goal.",
        "The football team won the match.",
    ]
    model = GraphTopic(
        embedding_model=SentenceTransformerEmbedding(batch_size=16),
        n_candidates=3,
        n_neighbors=1,
        representation_model=CTFIDFRepresentation(CountVectorizer(stop_words="english")),
    )
    model.fit(documents)
    print(model.get_topic_info().to_string(index=False))


if __name__ == "__main__":
    main()
