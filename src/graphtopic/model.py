"""Transactional GraphTopic facade and graph-reuse operations."""

import logging
import warnings
from collections.abc import Mapping
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
from numbers import Integral, Real
from time import perf_counter

import numpy as np
from sklearn.exceptions import NotFittedError

from ._validation import (
    canonical_labels,
    documents_and_ids,
    normalize_embeddings,
    ordered,
    positive_int,
    positive_real,
    seed_value,
)
from .community import LeidenDetector
from .embedding import SentenceTransformerEmbedding
from .graph import UnionMaxGraph
from .neighbors import NNDescentSearch
from .representation import CTFIDFRepresentation
from .results import DocumentGraph, GraphTopicResult, ResolutionPath

logger = logging.getLogger(__name__)


def _clone(component, name):
    try:
        cloned = deepcopy(component)
    except Exception as exc:
        raise TypeError(f"{name} must support deepcopy for isolated fitting") from exc
    if cloned is component:
        raise TypeError(f"{name} deepcopy must return an independent component")
    return cloned


def _run(stage, function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except Exception as exc:
        exc.add_note(f"GraphTopic pipeline stage: {stage}")
        raise


def _config(component):
    config = getattr(component, "effective_config_", None)
    return {
        "class": f"{type(component).__module__}.{type(component).__qualname__}",
        "effective": deepcopy(config) if config is not None else "unknown",
    }


def _words(representation, topics):
    words = representation.get_topics()
    if not isinstance(words, Mapping):
        raise ValueError("representation_model.get_topics() must return a mapping")
    if any(isinstance(t, bool) or not isinstance(t, Integral) for t in words):
        raise ValueError("representation_model topic keys must be integers")
    if set(words) != set(topics):
        raise ValueError("representation_model must return exactly the fitted topics")
    output = {}
    for topic, entries in words.items():
        if not isinstance(entries, list):
            raise ValueError("representation_model must return lists of (word, score) pairs")
        clean = []
        for entry in entries:
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                raise ValueError("representation_model entries must be (word, score) pairs")
            word, score = entry
            if (
                not isinstance(word, str)
                or isinstance(score, (bool, np.bool_))
                or not isinstance(score, Real)
                or not np.isfinite(score)
            ):
                raise ValueError("representation_model requires string words and finite scores")
            clean.append((word, float(score)))
        output[int(topic)] = clean
    return output


class GraphTopic:
    """Discover document topics with a replaceable sparse graph pipeline.

    Parameters
    ----------
    embedding_model : str, encoder or None
        Lazy Sentence Transformers model name, object with encode(documents), or
        None to require supplied embeddings.
    n_candidates : int, default=50
        Non-self candidate budget for the default NNDescent search.
    n_neighbors : int, default=20
        Maximum outgoing positive neighbors before union-max symmetrization.
    resolution : float, default=1.0
        RB-configuration resolution for the default Leiden detector.
    random_state : int or None, default=42
        Seed for default stochastic components.
    neighbor_search, graph_model, community_model, representation_model : object or None
        Structural component implementations. None constructs the default.
        Supplied components own their settings; related facade settings are ignored.
    verbose : bool, default=False
        Emit stage timings through the graphtopic logger (no global logging setup).
    """

    def __init__(
        self,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        n_candidates=50,
        n_neighbors=20,
        resolution=1.0,
        random_state=42,
        neighbor_search=None,
        graph_model=None,
        community_model=None,
        representation_model=None,
        verbose=False,
    ):
        self.embedding_model = embedding_model
        self.n_candidates = positive_int(n_candidates, "n_candidates")
        self.n_neighbors = positive_int(n_neighbors, "n_neighbors")
        if neighbor_search is None and graph_model is None and n_candidates < n_neighbors:
            raise ValueError("n_candidates must be >= n_neighbors")
        self.resolution = positive_real(resolution, "resolution")
        self.random_state = seed_value(random_state)
        if not isinstance(verbose, bool):
            raise ValueError("verbose must be bool")
        self.verbose = verbose
        self.neighbor_search = neighbor_search
        self.graph_model = graph_model
        self.community_model = community_model
        self.representation_model = representation_model
        overrides = [
            (neighbor_search, "neighbor_search", "n_candidates", n_candidates, 50),
            (graph_model, "graph_model", "n_neighbors", n_neighbors, 20),
            (community_model, "community_model", "resolution", resolution, 1.0),
            (neighbor_search, "neighbor_search", "random_state", random_state, 42),
            (community_model, "community_model", "random_state", random_state, 42),
        ]
        for component, name, parameter, value, default in overrides:
            if component is not None and value != default:
                warnings.warn(
                    f"{parameter}={value!r} does not configure supplied {name}",
                    UserWarning,
                    stacklevel=2,
                )

    def __repr__(self):
        def name(component, default):
            return type(component).__name__ if component is not None else default

        return (
            f"GraphTopic(neighbor_search={name(self.neighbor_search, 'NNDescentSearch')}, "
            f"graph_model={name(self.graph_model, 'UnionMaxGraph')}, "
            f"community_model={name(self.community_model, 'LeidenDetector')})"
        )

    def _templates(self):
        return (
            NNDescentSearch(self.n_candidates, random_state=self.random_state)
            if self.neighbor_search is None
            else self.neighbor_search,
            UnionMaxGraph(self.n_neighbors) if self.graph_model is None else self.graph_model,
            LeidenDetector(self.resolution, random_state=self.random_state)
            if self.community_model is None
            else self.community_model,
            CTFIDFRepresentation()
            if self.representation_model is None
            else self.representation_model,
        )

    def fit(self, documents, embeddings=None, document_ids=None):
        """Fit atomically, preserving the previous successful result on failure.

        Precomputed embeddings always take precedence over the encoder. Documents
        and embedding rows must already be aligned by the caller.
        """
        docs, ids = documents_and_ids(documents, document_ids)
        names = ("neighbor_search", "graph_model", "community_model", "representation_model")
        components = tuple(
            _clone(c, name) for c, name in zip(self._templates(), names, strict=True)
        )
        search, graph_model, detector, representation = components
        for component, name, methods in (
            (search, "neighbor_search", ("search",)),
            (graph_model, "graph_model", ("build",)),
            (detector, "community_model", ("fit_predict",)),
            (representation, "representation_model", ("fit", "get_topics")),
        ):
            for method in methods:
                if not callable(getattr(component, method, None)):
                    raise TypeError(f"{name} must implement {method}()")
        timings = {}

        def timed(stage, function, *args, **kwargs):
            start = perf_counter()
            result = _run(stage, function, *args, **kwargs)
            timings[stage] = perf_counter() - start
            if self.verbose:
                logger.info("%s completed in %.3fs", stage, timings[stage])
            return result

        source = "provided"
        if embeddings is None:
            source = "encoded"
            encoder = self.embedding_model
            if encoder is None:
                raise ValueError("Supply embeddings or configure embedding_model")
            if isinstance(encoder, str):
                if getattr(self, "_encoder_name", None) != encoder:
                    self._encoder_adapter = SentenceTransformerEmbedding(encoder)
                    self._encoder_name = encoder
                encoder = self._encoder_adapter
            if not callable(getattr(encoder, "encode", None)):
                raise TypeError("embedding_model must implement encode(documents)")
            embeddings = timed("embedding", encoder.encode, docs)
        matrix = timed("normalization", normalize_embeddings, embeddings, len(docs))
        candidates = timed("neighbor_search", search.search, matrix)
        graph = timed("graph_model", graph_model.build, matrix, candidates, document_ids=ids)
        if not isinstance(graph, DocumentGraph) or graph.document_ids != ids:
            raise ValueError("graph_model must return DocumentGraph with unchanged document_ids")
        raw = timed("community_model", detector.fit_predict, graph)
        topics, mapping = canonical_labels(raw, len(docs))
        timed("representation_model", representation.fit, docs, topics)
        words = _words(representation, topics)
        versions = {}
        for package in (
            "graphtopic",
            "numpy",
            "scipy",
            "scikit-learn",
            "pandas",
            "pynndescent",
            "igraph",
            "leidenalg",
        ):
            try:
                versions[package] = version(package)
            except PackageNotFoundError:
                versions[package] = "unknown"
        encoder_name = (
            (
                self.embedding_model
                if isinstance(self.embedding_model, str)
                else type(self.embedding_model).__qualname__
            )
            if source == "encoded"
            else "unknown"
        )
        component_metadata = {
            name: _config(component) for name, component in zip(names, components, strict=True)
        }
        metadata = {
            "versions": versions,
            "base_fit": {
                "embedding_source": source,
                "encoder": encoder_name,
                "dtype": str(matrix.dtype),
                "n_documents": len(docs),
                "n_dimensions": matrix.shape[1],
                "requested": {
                    "n_candidates": self.n_candidates,
                    "n_neighbors": self.n_neighbors,
                    "resolution": self.resolution,
                    "random_state": self.random_state,
                },
                "components": {
                    name: component_metadata[name] for name in ("neighbor_search", "graph_model")
                },
                "graph": graph.diagnostics,
            },
            "run": {
                "resolution": self.resolution,
                "graph_reused": False,
                "raw_to_topic": mapping,
                "components": {
                    name: component_metadata[name]
                    for name in ("community_model", "representation_model")
                },
                "timings": timings,
            },
        }
        result = GraphTopicResult(docs, ids, topics, graph, words, metadata)
        # Everything above can fail. Publish fitted state only after all validation succeeds.
        detector_template = _clone(self._templates()[2], "community_model")
        representation_template = _clone(self._templates()[3], "representation_model")
        self._result = result
        self._embeddings = matrix
        self._detector_template = detector_template
        self._representation_template = representation_template
        return self

    def fit_transform(self, documents, embeddings=None, document_ids=None):
        """Fit and return topic IDs, not a (topics, probabilities) tuple."""
        return self.fit(documents, embeddings, document_ids).topics_

    def _fitted(self):
        if not hasattr(self, "_result"):
            raise NotFittedError("GraphTopic has not been fitted")
        return self._result

    @property
    def result_(self):
        return self._fitted()

    @property
    def documents_(self):
        return self._fitted().documents

    @property
    def document_ids_(self):
        return self._fitted().document_ids

    @property
    def embeddings_(self):
        self._fitted()
        view = self._embeddings.view()
        view.flags.writeable = False
        return view

    @property
    def graph_(self):
        return self._fitted().graph

    @property
    def topics_(self):
        return list(self._fitted().topics)

    @property
    def topic_sizes_(self):
        return self._fitted().topic_sizes

    @property
    def topic_representations_(self):
        return self._fitted().topic_words

    def get_topics(self):
        return self.topic_representations_

    def get_topic(self, topic_id):
        return self.get_topics()[topic_id]

    def get_topic_info(self):
        return self._fitted().get_topic_info()

    def get_document_info(self):
        return self._fitted().get_document_info()

    def get_graph(self):
        return self.graph_.copy()

    def update_topics(self, representation_model=None):
        """Replace only the lexical explanation; preserve graph and assignments."""
        previous = self._fitted()
        template = (
            self._representation_template if representation_model is None else representation_model
        )
        new_template = _clone(template, "representation_model")
        representation = _clone(template, "representation_model")
        start = perf_counter()
        _run("representation_model", representation.fit, previous.documents, previous.topics)
        words = _words(representation, previous.topics)
        metadata = previous.metadata
        metadata["run"]["components"]["representation_model"] = _config(representation)
        metadata["run"]["timings"]["representation_model"] = perf_counter() - start
        result = GraphTopicResult(
            previous.documents,
            previous.document_ids,
            previous.topics,
            previous.graph,
            words,
            metadata,
        )
        self._result = result
        self._representation_template = new_template
        return self

    def resolution_path(self, resolutions):
        """Return independent partitions without recomputing or changing the graph."""
        previous = self._fitted()
        values = tuple(positive_real(v, "resolution") for v in ordered(resolutions, "resolutions"))
        if not values or len(set(values)) != len(values):
            raise ValueError("resolutions must be nonempty and unique")
        if not callable(getattr(self._detector_template, "with_resolution", None)):
            raise NotImplementedError("community_model must implement with_resolution(value)")
        results = []
        for value in values:
            template = _clone(self._detector_template, "community_model")
            detector = template.with_resolution(value)
            representation = _clone(self._representation_template, "representation_model")
            start = perf_counter()
            raw = _run("community_model", detector.fit_predict, previous.graph)
            topics, mapping = canonical_labels(raw, len(previous.documents))
            detection_time = perf_counter() - start
            start = perf_counter()
            _run("representation_model", representation.fit, previous.documents, topics)
            words = _words(representation, topics)
            metadata = previous.metadata
            metadata["run"]["resolution"] = value
            metadata["run"]["raw_to_topic"] = mapping
            metadata["run"]["graph_reused"] = True
            metadata["run"]["components"]["community_model"] = _config(detector)
            metadata["run"]["components"]["representation_model"] = _config(representation)
            metadata["run"]["timings"] = {
                "community_model": detection_time,
                "representation_model": perf_counter() - start,
            }
            results.append(
                GraphTopicResult(
                    previous.documents,
                    previous.document_ids,
                    topics,
                    previous.graph,
                    words,
                    metadata,
                )
            )
        return ResolutionPath(values, tuple(results))
