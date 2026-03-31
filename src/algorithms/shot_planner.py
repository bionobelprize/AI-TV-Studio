"""Shot Planning Algorithm for AI-TV-Studio.

Converts a narrative script into an optimized sequence of shots, inserting
transition shots where character entries or scene changes are needed.
"""

import uuid
from typing import List, Optional, Any

from src.models.character import Character
from src.models.episode import Episode
from src.models.scene import Scene
from src.models.shot import CameraMotion, GenerationMode, Shot


class ShotPlanner:
    """Converts narrative script into optimized shot sequence.

    Algorithm:
    1. Parse script into scenes and actions
    2. Identify character appearances and exits
    3. Create shot boundaries based on duration constraints
    4. Insert transition shots where needed
    5. Optimize generation mode selection
    6. Populate reference images for character-based shots
    """

    DEFAULT_SHOT_DURATION = 8  # seconds
    MIN_SHOT_DURATION = 5
    MAX_SHOT_DURATION = 12

    def __init__(self, character_registry: Optional[dict] = None, asset_manager: Optional[Any] = None):
        """Initialize the shot planner.

        Args:
            character_registry: Optional mapping of character ID to Character
                objects for character look-up during planning.
            asset_manager: Optional AssetManager for retrieving character
                reference images. If provided, reference images will be
                automatically populated for non-transition shots with characters.
        """
        self.character_registry: dict = character_registry or {}
        self.asset_manager = asset_manager

    def plan_episode(self, episode: Episode) -> Episode:
        """Plan shots for every scene in the episode.

        Iterates over all scenes and calls :meth:`plan_scene` on each.

        Args:
            episode: The episode whose scenes will be shot-planned.

        Returns:
            The same episode with shots populated on each scene.
        """
        for scene in episode.scenes:
            scene.shots = self.plan_scene(scene)
        return episode

    def plan_scene(self, scene: Scene) -> List[Shot]:
        """Plan the shot sequence for a single scene.

        Identifies character entries, determines shot boundaries, and inserts
        transition shots as needed. Automatically populates reference images
        for non-transition shots with characters.

        Args:
            scene: The scene to plan.

        Returns:
            An ordered list of shots covering the entire scene.
        """
        planned_shots: List[Shot] = []
        characters_present: List[str] = []

        for i, shot in enumerate(scene.shots):
            new_characters = [
                cid
                for cid in shot.characters_in_shot
                if cid not in characters_present
            ]

            if new_characters and characters_present:
                for char_id in new_characters:
                    transition = self._create_character_entry_transition(
                        scene_id=scene.id,
                        sequence_number=len(planned_shots),
                        new_character_id=char_id,
                        preceding_shot=planned_shots[-1] if planned_shots else None,
                    )
                    planned_shots.append(transition)

            shot.sequence_number = len(planned_shots)
            shot.generation_mode = self._select_generation_mode(
                shot, bool(planned_shots)
            )
            shot.duration = self._clamp_duration(shot.duration)

            if planned_shots:
                shot.previous_shot_tail = None  # populated during generation

            # Populate reference images for non-transition shots with characters
            if not shot.is_transition_shot and shot.characters_in_shot:
                self._populate_reference_images(shot)

            planned_shots.append(shot)
            characters_present.extend(new_characters)

        return planned_shots

    def _create_character_entry_transition(
        self,
        scene_id: str,
        sequence_number: int,
        new_character_id: str,
        preceding_shot: Optional[Shot],
    ) -> Shot:
        """Create a placeholder transition shot for a character entry.

        Args:
            scene_id: The scene this transition belongs to.
            sequence_number: Position of the shot in the scene sequence.
            new_character_id: ID of the character entering the scene.
            preceding_shot: The shot that immediately precedes this transition.

        Returns:
            A Shot configured as a character-entry transition.
        """
        character = self.character_registry.get(new_character_id)
        char_name = character.name if character else new_character_id

        return Shot(
            id=f"transition_{uuid.uuid4().hex[:8]}",
            scene_id=scene_id,
            sequence_number=sequence_number,
            action_description=f"{char_name} enters the scene",
            generation_mode=GenerationMode.FIRSTLAST_FRAME,
            duration=self.DEFAULT_SHOT_DURATION,
            is_transition_shot=True,
            transition_type="character_entry",
            characters_in_shot=[new_character_id],
        )

    def _select_generation_mode(
        self, shot: Shot, has_preceding_shot: bool
    ) -> GenerationMode:
        """Determine the optimal generation mode for a shot.

        Rules:
        - If already set to FIRSTLAST_FRAME, keep it.
        - If a transition shot, keep its assigned mode.
        - If NOT a transition shot AND has characters, use REFERENCE_TO_VIDEO
          for consistency and character coherence.
        - If a preceding shot exists and reference images are provided, use
          REFERENCE_TO_VIDEO for character consistency.
        - Otherwise fall back to TEXT_TO_VIDEO.

        Args:
            shot: The shot whose generation mode is being selected.
            has_preceding_shot: Whether a preceding shot exists in the scene.

        Returns:
            The selected GenerationMode.
        """
        if shot.generation_mode == GenerationMode.FIRSTLAST_FRAME:
            return shot.generation_mode

        if shot.start_frame_path and not shot.end_frame_path:
            return GenerationMode.FIRST_FRAME

        # Non-transition shots with characters should use reference images
        if not shot.is_transition_shot and shot.characters_in_shot:
            return GenerationMode.REFERENCE_TO_VIDEO

        if shot.reference_images or (
            shot.characters_in_shot and has_preceding_shot
        ):
            return GenerationMode.REFERENCE_TO_VIDEO

        return GenerationMode.TEXT_TO_VIDEO

    def _clamp_duration(self, duration: int) -> int:
        """Clamp shot duration to the valid range [5, 12] seconds.

        Args:
            duration: Requested shot duration in seconds.

        Returns:
            Duration clamped to [MIN_SHOT_DURATION, MAX_SHOT_DURATION].
        """
        return max(self.MIN_SHOT_DURATION, min(duration, self.MAX_SHOT_DURATION))

    def _populate_reference_images(self, shot: Shot) -> None:
        """Populate reference images for a shot based on its characters.

        For each character in the shot, retrieves their reference image from
        the asset manager and adds it to the shot's reference_images list.
        If the asset manager is not available, this is a no-op.

        Args:
            shot: The shot to populate with reference images.
        """
        if not self.asset_manager or not shot.characters_in_shot:
            return

        reference_images = []
        for char_id in shot.characters_in_shot:
            ref_image = self.asset_manager.get_character_reference_image(char_id)
            if ref_image:
                reference_images.append(ref_image)

        if reference_images:
            shot.reference_images = reference_images
