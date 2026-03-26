"""Character Entry Stitching Algorithm for AI-TV-Studio.

This module implements the core innovation of AI-TV-Studio: introducing new
characters into scenes while maintaining visual continuity using a stitching
approach based on first-last frame interpolation.
"""

import uuid
from dataclasses import dataclass, field
from typing import List, Optional

from src.models.character import Character
from src.models.shot import GenerationMode, Shot


@dataclass
class CharacterEntryParameters:
    """Parameters for smoothly introducing a new character into a scene."""

    new_character_id: str
    entry_style: str  # "walk_in", "appear_from_behind", "turn_around", "emerges_from_door"
    entry_direction: str  # "left", "right", "top", "bottom", "center_back"
    entry_duration: int = 8  # seconds
    interaction_with_existing: bool = True
    existing_character_reaction: str = "notice"  # "notice", "ignore", "surprised"

    # Positioning in final frame
    final_position: tuple = field(default_factory=lambda: (0.5, 0.5))  # normalized coordinates (x, y)
    final_scale: float = 0.3  # relative to frame height

    # Intermediate steps (for complex entries)
    intermediate_steps: int = 0  # number of intermediate frames to generate


class CharacterEntryStitcher:
    """Implements the stitching approach for introducing new characters.

    Algorithm Overview:
    1. Extract tail frame from previous shot
    2. Generate entry frame with new character positioned appropriately
    3. If needed, generate intermediate frames for complex entries
    4. Generate transition video using first-last frame interpolation
    5. Concatenate with following shots
    """

    def __init__(self, image_generator, video_generator):
        """Initialize the stitcher with image and video generators.

        Args:
            image_generator: Text-to-image generator with reference capability.
            video_generator: First-last frame video generator.
        """
        self.image_gen = image_generator
        self.video_gen = video_generator

    def stitch_character_entry(
        self,
        previous_shot: Shot,
        new_character: Character,
        entry_params: CharacterEntryParameters,
        existing_characters: List[Character],
    ) -> List[Shot]:
        """Main algorithm for character entry stitching.

        Args:
            previous_shot: The shot preceding the character entry.
            new_character: The character being introduced.
            entry_params: Parameters controlling how the character enters.
            existing_characters: Characters already present in the scene.

        Returns:
            A list of transition shots replacing the original transition point.

        Raises:
            ValueError: If the previous shot has no generated video.
        """
        if not previous_shot.generated_video_path:
            raise ValueError("Previous shot must have generated video")

        # Step 1: Extract the tail frame (last frame) of previous shot
        tail_frame = self._extract_tail_frame(previous_shot.generated_video_path)

        # Step 2: Analyze existing scene composition
        scene_composition = self._analyze_composition(tail_frame, existing_characters)

        # Step 3: Generate entry frame with new character
        entry_frame = self._generate_entry_frame(
            tail_frame=tail_frame,
            new_character=new_character,
            existing_characters=existing_characters,
            entry_params=entry_params,
            scene_composition=scene_composition,
        )

        # Step 4: Determine if intermediate frames are needed
        if entry_params.intermediate_steps > 0:
            return self._generate_multi_stage_entry(
                tail_frame, entry_frame, entry_params, new_character
            )

        # Step 5: Generate single transition shot
        transition_shot = self._create_transition_shot(
            start_frame=tail_frame,
            end_frame=entry_frame,
            entry_params=entry_params,
            new_character=new_character,
        )

        return [transition_shot]

    def _extract_tail_frame(self, video_path: str) -> str:
        """Extract the last frame of a video file.

        Uses FFmpeg to extract the final frame and saves it as an image.

        Args:
            video_path: Path to the source video file.

        Returns:
            Path to the extracted tail frame image.
        """
        import os
        import subprocess

        output_path = video_path.rsplit(".", 1)[0] + "_tail_frame.png"
        subprocess.run(
            [
                "ffmpeg",
                "-sseof",
                "-1",
                "-i",
                video_path,
                "-update",
                "1",
                "-q:v",
                "1",
                output_path,
                "-y",
            ],
            check=True,
            capture_output=True,
        )
        return output_path

    def _analyze_composition(
        self, frame: str, characters: List[Character]
    ) -> dict:
        """Analyze existing frame to determine layout and available space.

        Determines:
        - Positions of existing characters
        - Lighting conditions
        - Available negative space for new character entry

        Args:
            frame: Path to the frame image to analyze.
            characters: Characters present in the frame.

        Returns:
            Dictionary with composition analysis results.
        """
        return {
            "character_positions": {},
            "lighting": "cinematic",
            "available_space": {"left": True, "right": True, "center": False},
        }

    def _generate_entry_frame(
        self,
        tail_frame: str,
        new_character: Character,
        existing_characters: List[Character],
        entry_params: CharacterEntryParameters,
        scene_composition: dict,
    ) -> str:
        """Generate the end frame containing both existing and new characters.

        Algorithm:
        1. Use the previous tail frame as background context
        2. Generate new character using reference image for consistency
        3. Use inpainting/control to blend character into scene
        4. Ensure lighting and perspective match

        Args:
            tail_frame: Path to the last frame of the preceding shot.
            new_character: The character being introduced.
            existing_characters: Characters already visible in the scene.
            entry_params: Parameters for the character entry.
            scene_composition: Analysis of the current scene layout.

        Returns:
            Path to the generated entry frame image.
        """
        prompt = self._build_entry_prompt(
            new_character=new_character,
            existing_characters=existing_characters,
            entry_params=entry_params,
            scene_composition=scene_composition,
        )

        entry_frame = self.image_gen.generate_with_reference(
            prompt=prompt,
            reference_images=[new_character.visual_core.base_image_path],
            background_image=tail_frame,
            pose_control=self._get_pose_from_params(entry_params),
            lighting_match=True,
        )

        return entry_frame

    def _build_entry_prompt(
        self,
        new_character: Character,
        existing_characters: List[Character],
        entry_params: CharacterEntryParameters,
        scene_composition: dict,
    ) -> str:
        """Construct the prompt for entry frame generation.

        Args:
            new_character: The character being introduced.
            existing_characters: Characters already present.
            entry_params: Entry parameters.
            scene_composition: Scene composition analysis.

        Returns:
            A descriptive prompt string for image generation.
        """
        existing_desc = ", ".join(c.name for c in existing_characters)
        return (
            f"{new_character.name} entering the scene from the {entry_params.entry_direction}, "
            f"{entry_params.entry_style}. "
            f"Existing characters: {existing_desc}. "
            f"{new_character.visual_core.key_features if new_character.visual_core else ''}. "
            f"Photorealistic, cinematic lighting, 4K quality."
        )

    def _get_pose_from_params(self, entry_params: CharacterEntryParameters) -> dict:
        """Convert entry parameters to pose control specification.

        Args:
            entry_params: Character entry parameters.

        Returns:
            Dictionary describing the pose control configuration.
        """
        return {
            "position": entry_params.final_position,
            "scale": entry_params.final_scale,
            "facing": entry_params.entry_direction,
        }

    def _generate_intermediate_frame(
        self,
        start_frame: str,
        end_frame: str,
        new_character: Character,
        progress: float,
        entry_params: CharacterEntryParameters,
    ) -> str:
        """Generate an intermediate frame showing a partially visible new character.

        Args:
            start_frame: Path to the initial frame (no new character).
            end_frame: Path to the final frame (new character fully present).
            new_character: The character being introduced.
            progress: Float from 0.0 to 1.0 representing entry progress.
            entry_params: Entry parameters.

        Returns:
            Path to the generated intermediate frame image.
        """
        prompt = (
            f"{new_character.name} {entry_params.entry_style} from the "
            f"{entry_params.entry_direction}, {int(progress * 100)}% visible. "
            f"{new_character.visual_core.key_features if new_character.visual_core else ''}. "
            f"Photorealistic, cinematic, 4K."
        )
        return self.image_gen.generate_with_reference(
            prompt=prompt,
            reference_images=[new_character.visual_core.base_image_path],
            background_image=start_frame,
            pose_control=self._get_pose_from_params(entry_params),
            lighting_match=True,
        )

    def _generate_multi_stage_entry(
        self,
        start_frame: str,
        end_frame: str,
        entry_params: CharacterEntryParameters,
        new_character: Character,
    ) -> List[Shot]:
        """Generate multiple transition shots for complex character entries.

        Algorithm:
        1. Generate N intermediate frames showing progressive entry
        2. Create N transition videos between consecutive frames
        3. Return as sequence of shots

        Args:
            start_frame: Path to the starting frame.
            end_frame: Path to the final frame (character fully entered).
            entry_params: Entry parameters.
            new_character: The character being introduced.

        Returns:
            List of transition shots covering the full entry sequence.
        """
        intermediate_shots: List[Shot] = []
        current_frame = start_frame
        steps = entry_params.intermediate_steps

        for i in range(steps):
            progress = (i + 1) / (steps + 1)

            intermediate_frame = self._generate_intermediate_frame(
                start_frame=start_frame,
                end_frame=end_frame,
                new_character=new_character,
                progress=progress,
                entry_params=entry_params,
            )

            transition_shot = self._create_transition_shot(
                start_frame=current_frame,
                end_frame=intermediate_frame,
                entry_params=entry_params,
                new_character=new_character,
                shot_id=f"transition_{i + 1}_of_{steps + 1}",
            )

            intermediate_shots.append(transition_shot)
            current_frame = intermediate_frame

        # Final transition to full entry frame
        final_transition = self._create_transition_shot(
            start_frame=current_frame,
            end_frame=end_frame,
            entry_params=entry_params,
            new_character=new_character,
            shot_id="transition_final",
        )
        intermediate_shots.append(final_transition)

        return intermediate_shots

    def _describe_entry(self, entry_params: CharacterEntryParameters) -> str:
        """Produce a human-readable description of the character entry.

        Args:
            entry_params: Entry parameters to describe.

        Returns:
            Descriptive string for use in video generation prompts.
        """
        return (
            f"character {entry_params.entry_style} from the {entry_params.entry_direction}"
        )

    def _create_transition_shot(
        self,
        start_frame: str,
        end_frame: str,
        entry_params: CharacterEntryParameters,
        new_character: Character,
        shot_id: Optional[str] = None,
    ) -> Shot:
        """Create a transition shot using first-last frame video generation.

        This is the core stitching operation: generating video that smoothly
        interpolates between the existing scene and the scene with the new
        character.

        Args:
            start_frame: Path to the first frame of the transition video.
            end_frame: Path to the last frame of the transition video.
            entry_params: Parameters controlling the entry style.
            new_character: The character being introduced.
            shot_id: Optional explicit shot identifier.

        Returns:
            A Shot configured for first-last frame video generation.
        """
        key_features = (
            new_character.visual_core.key_features if new_character.visual_core else ""
        )
        prompt = (
            f"Smooth transition: {self._describe_entry(entry_params)}. "
            f"Motion: {entry_params.entry_style} from the {entry_params.entry_direction}. "
            f"Character consistency: {key_features}. "
            f"The existing character(s) maintain their positions and expressions."
        )

        return Shot(
            id=shot_id or f"transition_{uuid.uuid4().hex[:8]}",
            scene_id="transition",
            sequence_number=0,
            action_description=f"{new_character.name} enters the scene",
            generation_mode=GenerationMode.FIRSTLAST_FRAME,
            duration=entry_params.entry_duration,
            start_frame_path=start_frame,
            end_frame_path=end_frame,
            text_prompt=prompt,
            is_transition_shot=True,
            transition_type="character_entry",
        )
