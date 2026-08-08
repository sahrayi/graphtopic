"""
Graph based topic modeling.
"""

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

    High-level text topic modeling pipeline built on top
    of Graphion.

    Pipeline
    --------
    Documents
        |
        v
    Embedding
        |
        v
    FeatureSet
        |
        v
    Feature Reduction
        |
        v
    Reduced FeatureSet
        |
        v
    Relation Building
        |
        v
    RelationSet
        |
        v
    Graph Construction
        |
        v
    Graph Refinement
        |
        v
    Community Detection
        |
        v
    Partition Refinement
        |
        v
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
        """
        Initialize GraphTopic.

        Parameters
        ----------
        embedding_model:
            Model used to generate document embeddings when
            embeddings are not supplied directly to ``fit``.

        representation_model:
            Model used to generate topic representations.

        reducer:
            Feature reduction component.

        relation_builder:
            Component responsible for computing relations
            between document feature vectors.

        graph_builder:
            Component responsible for constructing a graph
            from a RelationSet.

        graph_refiner:
            Optional graph refinement component.

        partition_detector:
            Community detection component.

        partition_refiner:
            Optional partition refinement component.
        """

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
            else self._create_default_relation_builder()
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
    ) -> GraphTopic:
        """
        Fit the topic model.

        Parameters
        ----------
        documents:
            Input documents.

        embeddings:
            Optional precomputed document embeddings.

            If omitted, ``embedding_model`` must be configured.

        Returns
        -------
        GraphTopic
            Fitted model.
        """

        print()
        print("=" * 70)
        print("GraphTopic fitting started")
        print("=" * 70)

        # --------------------------------------------------
        # Reset fitted state
        # --------------------------------------------------

        self._reset_state()

        # --------------------------------------------------
        # 1. Validate documents
        # --------------------------------------------------

        print()
        print("[1/10] Validating documents...")

        self._validate_documents(documents)

        self.documents_ = list(documents)

        self.document_ids_ = tuple(
            range(len(documents))
        )

        print(
            f"✓ Documents: {len(documents):,}"
        )

        # --------------------------------------------------
        # 2. Prepare embeddings
        # --------------------------------------------------

        print()
        print("[2/10] Preparing embeddings...")

        embeddings = self._get_embeddings(
            documents,
            embeddings,
        )

        embeddings = self._validate_embeddings(
            documents,
            embeddings,
        )

        self.embeddings_ = embeddings

        print(
            f"✓ Embeddings shape: {embeddings.shape}"
        )

        # --------------------------------------------------
        # 3. Create original FeatureSet
        # --------------------------------------------------

        print()
        print("[3/10] Creating feature set...")

        self.feature_set_ = self._create_feature_set(
            embeddings
        )

        print(
            "✓ Original FeatureSet created"
        )

        # --------------------------------------------------
        # 4. Reduce features
        # --------------------------------------------------

        print()
        print("[4/10] Reducing features...")

        self.reduced_feature_set_ = (
            self.reducer.reduce(
                self.feature_set_
            )
        )

        self._validate_feature_set(
            self.reduced_feature_set_,
            "reducer",
        )

        print(
            "✓ Feature reduction completed"
        )

        # --------------------------------------------------
        # 5. Build relations
        # --------------------------------------------------

        print()
        print("[5/10] Building relations...")

        relations = self.relation_builder.build(
            self.reduced_feature_set_
        )

        if relations is None:
            raise ValueError(
                "Invalid relation output from "
                "relation_builder"
            )

        self.relations_ = relations

        print(
            f"✓ Relations created: "
            f"{len(relations):,}"
        )

        # --------------------------------------------------
        # 6. Build graph
        # --------------------------------------------------

        print()
        print("[6/10] Building graph...")

        graph = self.graph_builder.build(
            relations,
            nodes=self.document_ids_,
        )

        if graph is None:
            raise ValueError(
                "Invalid graph returned by "
                "graph_builder"
            )

        self.graph_ = graph

        print(
            f"✓ Graph created "
            f"(nodes={len(graph.nodes)}, "
            f"edges={len(graph.edges)})"
        )

        # --------------------------------------------------
        # 7. Refine graph
        # --------------------------------------------------

        print()
        print("[7/10] Refining graph...")

        refined_graph = self.graph_refiner.refine(
            self.graph_
        )

        if refined_graph is None:
            raise ValueError(
                "Invalid graph returned by "
                "graph_refiner"
            )

        self.refined_graph_ = refined_graph

        print(
            "✓ Graph refinement completed"
        )

        # --------------------------------------------------
        # 8. Detect communities
        # --------------------------------------------------

        print()
        print("[8/10] Detecting communities...")

        partition = self.partition_detector.detect(
            self.refined_graph_
        )

        self._validate_partition(
            partition,
            "partition_detector",
        )

        self.partition_set_ = partition

        print(
            f"✓ Communities detected: "
            f"{self._count_topics(partition)}"
        )

        # --------------------------------------------------
        # 9. Refine partition
        # --------------------------------------------------

        print()
        print("[9/10] Refining partition...")

        refined_partition = (
            self.partition_refiner.refine(
                self.partition_set_
            )
        )

        self._validate_partition(
            refined_partition,
            "partition_refiner",
        )

        self.refined_partition_set_ = (
            refined_partition
        )

        self.topics_ = self._extract_topic_ids(
            self.refined_partition_set_
        )

        print(
            "✓ Partition refinement completed"
        )

        print(
            f"✓ Final topics: "
            f"{len(set(self.topics_))}"
        )

        # --------------------------------------------------
        # 10. Build topic representation
        # --------------------------------------------------

        print()
        print("[10/10] Building topic representation...")

        self.representation_model.fit(
            documents,
            self.topics_,
        )

        self.topic_info_ = self._build_topic_info()

        print(
            f"✓ Topics generated: "
            f"{len(self.topic_info_)}"
        )

        print()
        print("=" * 70)
        print("GraphTopic fitting completed")
        print("=" * 70)

        return self

    def fit_transform(
        self,
        documents: list[str],
        embeddings=None,
    ) -> list[int]:
        """
        Fit the model and return topic assignments.
        """

        self.fit(
            documents,
            embeddings,
        )

        return self.topics_

    def get_topic_info(
        self,
    ) -> list[TopicInfo]:
        """
        Return topic information.

        Raises
        ------
        ValueError
            If the model has not been fitted.
        """

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
        """
        Reset fitted state.

        The original and transformed objects are kept
        separately so that every stage of the pipeline
        remains inspectable.
        """

        self.documents_ = None

        self.document_ids_ = None

        self.embeddings_ = None

        # Original feature representation.
        self.feature_set_ = None

        # Feature representation after reduction.
        self.reduced_feature_set_ = None

        # Relations computed from reduced features.
        self.relations_ = None

        # Original graph built from relations.
        self.graph_ = None

        # Graph after graph refinement.
        self.refined_graph_ = None

        # Original partition returned by detector.
        self.partition_set_ = None

        # Partition after partition refinement.
        self.refined_partition_set_ = None

        # Final topic assignments.
        self.topics_ = None

        self.topic_info_ = None

    # ==================================================
    # Defaults
    # ==================================================

    @staticmethod
    def _create_default_relation_builder():
        """
        Create the default relation builder.
        """

        return CosineSimilarity()

    @staticmethod
    def _create_default_graph_builder():
        """
        Create the default graph builder.
        """

        return KNNThreshold(
            k=10,
            threshold=0.5,
        )

    # ==================================================
    # Validation
    # ==================================================

    @staticmethod
    def _validate_documents(
        documents: list[str],
    ) -> None:
        """
        Validate input documents.
        """

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

    @staticmethod
    def _validate_embeddings(
        documents,
        embeddings,
    ) -> np.ndarray:
        """
        Validate and convert embeddings.
        """

        matrix = np.asarray(
            embeddings,
            dtype=float,
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

    @staticmethod
    def _validate_feature_set(
        feature_set,
        source: str,
    ) -> None:
        """
        Validate a FeatureSet returned by a pipeline component.
        """

        if feature_set is None:
            raise ValueError(
                f"Invalid feature set returned by "
                f"{source}"
            )

        if not isinstance(
            feature_set,
            FeatureSet,
        ):
            raise ValueError(
                f"{source} must return a FeatureSet"
            )

    @staticmethod
    def _validate_partition(
        partition,
        source: str,
    ) -> None:
        """
        Validate a partition output.
        """

        if partition is None:
            raise ValueError(
                f"Invalid partition output from "
                f"{source}"
            )

        if not hasattr(
            partition,
            "to_dict",
        ):
            raise ValueError(
                f"Invalid partition output from "
                f"{source}"
            )

    # ==================================================
    # Embeddings
    # ==================================================

    def _get_embeddings(
        self,
        documents,
        embeddings,
    ):
        """
        Obtain document embeddings.

        Explicit embeddings take precedence over the
        configured embedding model.
        """

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

    # ==================================================
    # FeatureSet
    # ==================================================

    def _create_feature_set(
        self,
        embeddings,
    ) -> FeatureSet:
        """
        Create the original Graphion FeatureSet.

        Each document receives a stable integer identifier
        corresponding to its position in the input collection.
        """

        return FeatureSet.from_numpy(
            ids=self.document_ids_,
            matrix=embeddings,
        )

    # ==================================================
    # Topic extraction
    # ==================================================

    def _extract_topic_ids(
        self,
        partition_set,
    ) -> list[int]:
        """
        Extract topic labels in document order.
        """

        _, labels = partition_set.to_labels(
            self.document_ids_
        )

        return list(labels)

    # ==================================================
    # Topic information
    # ==================================================

    def _build_topic_info(
        self,
    ) -> list[TopicInfo]:
        """
        Build TopicInfo objects from the final partition.
        """

        topics = []

        mapping = (
            self.refined_partition_set_.to_dict()
        )

        grouped: dict[int, list[int]] = {}

        for document_id, topic_id in mapping.items():
            grouped.setdefault(
                topic_id,
                [],
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
            key=lambda item: item.topic_id,
        )

    # ==================================================
    # Utilities
    # ==================================================

    def _count_topics(
        self,
        partition_set,
    ) -> int:
        """
        Count communities in a partition.
        """

        mapping = partition_set.to_dict()

        return len(
            set(mapping.values())
        )