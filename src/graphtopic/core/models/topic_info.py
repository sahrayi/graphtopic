from dataclasses import dataclass, field
from typing import Any


@dataclass
class TopicInfo:
    """
    Information about a discovered topic.

    Stores topic metadata and its human-readable
    representation.
    """

    topic_id: int

    size: int = 0

    representation: list[tuple[str, float]] = field(
        default_factory=list
    )

    document_ids: list[Any] = field(
        default_factory=list
    )