"""GraphTopic: sparse semantic graph topic modeling."""

import logging
from importlib.metadata import PackageNotFoundError, version

from .community import LeidenDetector
from .embedding import SentenceTransformerEmbedding
from .graph import UnionMaxGraph
from .model import GraphTopic
from .neighbors import ExactNeighborSearch, NNDescentSearch
from .representation import ClassTfidfTransformer, CTFIDFRepresentation
from .results import DocumentGraph, GraphTopicResult, ResolutionPath

try:
    __version__ = version("graphtopic")
except PackageNotFoundError:
    __version__ = "0.1.2"

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "GraphTopic",
    "DocumentGraph",
    "GraphTopicResult",
    "ResolutionPath",
    "SentenceTransformerEmbedding",
    "NNDescentSearch",
    "ExactNeighborSearch",
    "UnionMaxGraph",
    "LeidenDetector",
    "CTFIDFRepresentation",
    "ClassTfidfTransformer",
]
