from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer

from graphtopic.core.interfaces.representation_model import (
    BaseRepresentationModel
)


class CTFIDFRepresentation(
    BaseRepresentationModel
):

    def __init__(
        self,
        top_n_words: int = 10,
    ):

        self.top_n_words = top_n_words

        self.topic_representations_ = {}



    def fit(
        self,
        documents: list[str],
        topic_ids: list[int],
    ):

        topic_documents = defaultdict(list)


        for document, topic_id in zip(
            documents,
            topic_ids,
        ):

            topic_documents[topic_id].append(
                document
            )


        combined_documents = []
        topic_order = []


        for topic_id, docs in topic_documents.items():

            combined_documents.append(
                " ".join(docs)
            )

            topic_order.append(
                topic_id
            )


        vectorizer = TfidfVectorizer()


        tfidf = vectorizer.fit_transform(
            combined_documents
        )


        words = vectorizer.get_feature_names_out()


        for index, topic_id in enumerate(topic_order):

            scores = tfidf[index].toarray()[0]


            top_indices = scores.argsort()[::-1][
                :self.top_n_words
            ]


            self.topic_representations_[topic_id] = [
                (
                    words[i],
                    float(scores[i])
                )
                for i in top_indices
            ]


        return self



    def get_topic(
        self,
        topic_id: int,
    ):

        return self.topic_representations_.get(
            topic_id,
            []
        )


    def get_topics(
        self,
    ):

        return self.topic_representations_