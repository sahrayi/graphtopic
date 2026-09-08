"""Offline example of a custom lexical representation without inheritance."""

import numpy as np

from graphtopic import ExactNeighborSearch, GraphTopic


class FirstWords:
    """Use the first available word in each topic as its display label."""

    def fit(self, documents, topics):
        self.words = {}
        for text, topic in zip(documents, topics, strict=True):
            self.words.setdefault(topic, [])
            if not self.words[topic] and text.split():
                self.words[topic] = [(text.split()[0], 1.0)]
        return self

    def get_topics(self):
        return self.words


def main():
    model = GraphTopic(
        embedding_model=None,
        neighbor_search=ExactNeighborSearch(1),
        n_neighbors=1,
        representation_model=FirstWords(),
    )
    model.fit(["apple fruit", "pear fruit"], embeddings=np.array([[1.0, 0.1], [1.0, 0.2]]))
    print(model.get_topics())


if __name__ == "__main__":
    main()
