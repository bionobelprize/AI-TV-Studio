"""Scene data model for AI-TV-Studio."""

from dataclasses import dataclass, field
from typing import List

from .shot import Shot


@dataclass
class Scene:
    """Represents a single scene within an episode."""

    id: str
    episode_id: str
    scene_number: int
    location: str
    time_of_day: str  # dawn, day, dusk, night
    weather: str  # clear, rain, snow, fog
    mood: str  # tense, romantic, action, peaceful

    shots: List[Shot] = field(default_factory=list)

    # Background music
    bgm_mood: str = "neutral"
    bgm_tempo: str = "moderate"  # slow, moderate, fast

    # Environmental audio
    ambient_sounds: List[str] = field(default_factory=list)
