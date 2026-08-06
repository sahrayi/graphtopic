from abc import ABC, abstractmethod


class BaseRepresentationModel(ABC):

    @abstractmethod
    def fit(
        self,
        documents: list[str],
        topic_ids: list[int],
    ):
        """
        Learn representations for discovered topics.
        """
        pass


    @abstractmethod
    def get_topic(
        self,
        topic_id: int,
    ):
        """
        Return representation of a topic.
        """
        pass


    @abstractmethod
    def get_topics(
        self,
    ):
        """
        Return all topic representations.
        """
        pass