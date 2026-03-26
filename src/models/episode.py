"""Episode data model for AI-TV-Studio."""

from dataclasses import dataclass, field
from typing import List

from .scene import Scene


@dataclass
class Episode:
    """Represents a complete episode of the TV series."""

    id: str
    series_title: str
    episode_number: int
    episode_title: str
    logline: str  # one-sentence summary

    characters: List[str] = field(default_factory=list)  # Character IDs appearing
    scenes: List[Scene] = field(default_factory=list)

    # Metadata
    genre: str = "drama"
    runtime_estimate: int = 0  # seconds
