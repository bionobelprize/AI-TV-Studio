"""Shot data models for AI-TV-Studio."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from .character import CharacterEmotion


class GenerationMode(str, Enum):
    """Video generation mode for a shot."""

    TEXT_TO_VIDEO = "txt2video"
    FIRST_FRAME = "first_frame"
    FIRSTLAST_FRAME = "firstlast_frame"
    REFERENCE_TO_VIDEO = "ref2video"


@dataclass
class CameraMotion:
    """Describes camera movement for a shot."""

    type: str  # static, push_in, pull_out, pan_left, pan_right, tilt_up, tilt_down, orbit
    speed: float = 1.0  # normalized speed factor
    start_position: Optional[tuple] = None
    end_position: Optional[tuple] = None


@dataclass
class Shot:
    """Represents a single camera shot within a scene."""

    id: str
    scene_id: str
    sequence_number: int

    # Narrative content
    action_description: str
    dialogue: Optional[str] = None
    internal_thought: Optional[str] = None

    # Character management
    characters_in_shot: List[str] = field(default_factory=list)  # Character IDs
    character_emotions: Dict[str, CharacterEmotion] = field(default_factory=dict)

    # Visual parameters
    generation_mode: GenerationMode = GenerationMode.REFERENCE_TO_VIDEO
    duration: int = 8  # seconds, between 5-12
    camera_motion: Optional[CameraMotion] = None
    lighting_description: str = "cinematic lighting"

    # Input assets
    text_prompt: str = ""
    start_frame_path: Optional[str] = None
    end_frame_path: Optional[str] = None
    reference_images: List[str] = field(default_factory=list)

    # Output
    generated_video_path: Optional[str] = None
    generation_error: Optional[str] = None

    # Continuity tracking
    previous_shot_tail: Optional[str] = None  # path to last frame of previous shot
    is_transition_shot: bool = False
    transition_type: Optional[str] = None  # "character_entry", "scene_change", etc.
