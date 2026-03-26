"""Character data models for AI-TV-Studio."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class CharacterEmotion(str, Enum):
    """Emotions that a character can express."""

    NEUTRAL = "neutral"
    HAPPY = "happy"
    ANGRY = "angry"
    SAD = "sad"
    SURPRISED = "surprised"
    FEARFUL = "fearful"


@dataclass
class CharacterVisualCore:
    """Visual identity that ensures character consistency across generations."""

    base_image_path: str
    reference_prompt: str
    key_features: str
    lora_trigger: Optional[str] = None

    # Multi-angle reference library
    front_view: Optional[str] = None
    side_view: Optional[str] = None
    three_quarter_view: Optional[str] = None

    # Expression reference library
    expressions: Dict[CharacterEmotion, str] = field(default_factory=dict)


@dataclass
class Character:
    """Represents a character in the TV series."""

    id: str
    name: str
    age: int
    gender: str
    occupation: str
    aliases: List[str] = field(default_factory=list)

    # Narrative attributes
    personality_traits: List[str] = field(default_factory=list)
    character_arc: str = ""

    # Visual consistency
    visual_core: Optional[CharacterVisualCore] = None

    # Voice attributes (for TTS)
    voice_id: Optional[str] = None
    voice_emotion_profile: Dict[str, float] = field(default_factory=dict)

    def get_expression_frame(self, emotion: CharacterEmotion) -> Optional[str]:
        """Retrieve the reference frame for a specific emotion.

        Returns the path to the expression image for the given emotion,
        falling back to the base image if no specific expression image exists.
        """
        if self.visual_core and emotion in self.visual_core.expressions:
            return self.visual_core.expressions[emotion]
        return self.visual_core.base_image_path if self.visual_core else None
