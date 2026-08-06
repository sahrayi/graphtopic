from __future__ import annotations

import numpy as np


from graphtopic.core.models.topic_info import TopicInfo

from graphtopic.core.interfaces.embedding_model import (
    BaseEmbeddingModel,
)

from graphtopic.core.interfaces.representation_model import (
    BaseRepresentationModel,
)

from graphtopic.representation_models.ctfidf import (
    CTFIDFRepresentation,
)


from graphion.core.models import FeatureSet


from graphion.builders.relation.cosine_similarity import (
    CosineSimilarity,
)

from graphion.builders.graph.knn_threshold import (
    KNNThreshold,
)

from graphion.detectors.partition.leiden import (
    Leiden,
)

from graphion.reducers.identity import (
    IdentityReducer,
)

from graphion.refiners.graph.identity import (
    IdentityGraphRefiner,
)

from graphion.refiners.partition.identity import (
    IdentityPartitionRefiner,
)



class GraphTopic:
    """
    Graph based topic modeling.

    High level text topic modeling pipeline
    built on top of Graphion.

    Pipeline:

        Documents
            |
        Embedding
            |
        FeatureSet
            |
        Relation Graph
            |
        Community Detection
            |
        Topic Representation
    """

    def __init__(
        self,

        embedding_model: BaseEmbeddingModel | None = None,

        representation_model: BaseRepresentationModel | None = None,

        reducer=None,

        relation_builder=None,

        graph_builder=None,

        graph_refiner=None,

        partition_detector=None,

        partition_refiner=None,

    ) -> None:


        self.embedding_model = embedding_model


        self.representation_model = (
            representation_model
            if representation_model is not None
            else CTFIDFRepresentation()
        )


        self.reducer = (
            reducer
            if reducer is not None
            else IdentityReducer()
        )


        self.relation_builder = (
            relation_builder
            if relation_builder is not None
            else CosineSimilarity()
        )


        self.graph_builder = (
            graph_builder
            if graph_builder is not None
            else self._create_default_graph_builder()
        )


        self.graph_refiner = (
            graph_refiner
            if graph_refiner is not None
            else IdentityGraphRefiner()
        )


        self.partition_detector = (
            partition_detector
            if partition_detector is not None
            else Leiden()
        )


        self.partition_refiner = (
            partition_refiner
            if partition_refiner is not None
            else IdentityPartitionRefiner()
        )


        self._reset_state()



    # ==================================================
    # Public API
    # ==================================================


    def fit(
        self,
        documents: list[str],
        embeddings=None,
    ) -> "GraphTopic":


        self._reset_state()


        self._validate_documents(
            documents
        )


        self.documents_ = documents


        self.document_ids_ = tuple(
            range(
                len(documents)
            )
        )


        embeddings = self._get_embeddings(
            documents,
            embeddings,
        )


        embeddings = self._validate_embeddings(
            documents,
            embeddings,
        )


        self.embeddings_ = embeddings



        feature_set = self._create_feature_set(
            embeddings
        )


        self.feature_set_ = feature_set



        feature_set = self.reducer.reduce(
            feature_set
        )



        relations = self.relation_builder.build(
            feature_set
        )



        graph = self.graph_builder.build(
            relations,
            nodes=self.document_ids_,
        )

        if graph is None:
            raise ValueError(
                "Invalid graph returned by graph_builder"
            )


        graph = self.graph_refiner.refine(
            graph
        )


        self.graph_ = graph

        partition = self.partition_detector.detect(
            graph
        )

        self._validate_partition(
            partition
        )

        partition = self.partition_refiner.refine(
            partition
        )

        self._validate_partition(
            partition
        )


        self.partition_set_ = partition



        self.topics_ = self._extract_topic_ids(
            partition
        )



        self.representation_model.fit(
            documents,
            self.topics_,
        )


        self.topic_info_ = self._build_topic_info()


        return self



    def fit_transform(
        self,
        documents: list[str],
        embeddings=None,
    ) -> list[int]:

        self.fit(
            documents,
            embeddings,
        )

        return self.topics_

    def get_topic_info(
            self,
    ):
        if self.topic_info_ is None:
            raise ValueError(
                "Model is not fitted"
            )

        return self.topic_info_



    # ==================================================
    # State
    # ==================================================


    def _reset_state(
        self,
    ) -> None:

        self.documents_ = None

        self.document_ids_ = None

        self.embeddings_ = None

        self.feature_set_ = None

        self.graph_ = None

        self.partition_set_ = None

        self.topics_ = None

        self.topic_info_ = None



    # ==================================================
    # Defaults
    # ==================================================


    def _create_default_graph_builder(
        self,
    ):

        return KNNThreshold(
            relation_builder=self.relation_builder
        )



    # ==================================================
    # Validation
    # ==================================================


    def _validate_documents(
        self,
        documents: list[str],
    ) -> None:


        if len(documents) == 0:

            raise ValueError(
                "documents cannot be empty"
            )


        if any(
            not isinstance(doc, str)
            for doc in documents
        ):

            raise ValueError(
                "All documents must be strings"
            )



    def _validate_embeddings(
        self,
        documents,
        embeddings,
    ):


        try:

            matrix = np.asarray(
                embeddings,
                dtype=float,
            )


        except (
            ValueError,
            TypeError,
        ):

            raise ValueError(
                "Invalid embedding shape"
            )



        if matrix.ndim != 2:

            raise ValueError(
                "Invalid embedding shape"
            )



        if matrix.shape[0] != len(documents):

            raise ValueError(
                "Number of embeddings must match documents"
            )



        if matrix.shape[1] == 0:

            raise ValueError(
                "Embedding dimension cannot be zero"
            )


        return matrix

    def _validate_partition(
            self,
            partition,
    ):
        if partition is None:
            raise ValueError(
                "Invalid partition output"
            )

        if not hasattr(
                partition,
                "to_dict",
        ):
            raise ValueError(
                "Invalid partition output"
            )



    # ==================================================
    # Helpers
    # ==================================================


    def _get_embeddings(
        self,
        documents,
        embeddings,
    ):


        if embeddings is not None:

            return embeddings



        if self.embedding_model is None:

            raise ValueError(
                "Either embeddings or embedding_model "
                "must be provided."
            )



        return self.embedding_model.encode(
            documents
        )



    def _create_feature_set(
        self,
        embeddings,
    ):

        return FeatureSet.from_numpy(
            ids=self.document_ids_,
            matrix=embeddings,
        )



    def _extract_topic_ids(
        self,
        partition_set,
    ):

        _, labels = partition_set.to_labels(
            self.document_ids_
        )

        return list(labels)



    def _build_topic_info(
        self,
    ) -> list[TopicInfo]:

        topics = []


        mapping = (
            self.partition_set_.to_dict()
        )


        grouped: dict[int, list[int]] = {}


        for document_id, topic_id in mapping.items():

            grouped.setdefault(
                topic_id,
                []
            ).append(
                document_id
            )



        for topic_id, document_ids in grouped.items():

            topics.append(

                TopicInfo(

                    topic_id=topic_id,

                    size=len(document_ids),

                    representation=(
                        self.representation_model.get_topic(
                            topic_id
                        )
                    ),

                    document_ids=document_ids,

                )

            )


        return sorted(
            topics,
            key=lambda item: item.topic_id
        )