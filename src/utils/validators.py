"""Validation utilities for AI-TV-Studio data models.

Provides validation functions for Character, Shot, Scene, and Episode objects
to ensure they meet the constraints required by the generation pipeline before
any expensive API calls are made.
"""

from typing import List

from src.models.character import Character
from src.models.episode import Episode
from src.models.shot import Shot


class ValidationError(Exception):
    """Raised when a model object fails validation."""


def validate_character(character: Character) -> List[str]:
    """Validate a Character object and return a list of error messages.

    Args:
        character: The character to validate.

    Returns:
        List of validation error strings. Empty list means the character is valid.
    """
    errors: List[str] = []

    if not character.id or not character.id.strip():
        errors.append("Character must have a non-empty 'id'.")

    if not character.name or not character.name.strip():
        errors.append("Character must have a non-empty 'name'.")

    if character.age < 0:
        errors.append(f"Character age must be non-negative, got {character.age}.")

    if not character.gender or not character.gender.strip():
        errors.append("Character must have a non-empty 'gender'.")

    if not character.occupation or not character.occupation.strip():
        errors.append("Character must have a non-empty 'occupation'.")

    if character.visual_core is not None:
        if not character.visual_core.base_image_path:
            errors.append(
                "CharacterVisualCore must have a non-empty 'base_image_path'."
            )
        if not character.visual_core.key_features:
            errors.append(
                "CharacterVisualCore must have a non-empty 'key_features'."
            )

    return errors


def validate_shot(shot: Shot) -> List[str]:
    """Validate a Shot object and return a list of error messages.

    Args:
        shot: The shot to validate.

    Returns:
        List of validation error strings. Empty list means the shot is valid.
    """
    errors: List[str] = []

    if not shot.id or not shot.id.strip():
        errors.append("Shot must have a non-empty 'id'.")

    if not shot.scene_id or not shot.scene_id.strip():
        errors.append("Shot must have a non-empty 'scene_id'.")

    if not shot.action_description or not shot.action_description.strip():
        errors.append("Shot must have a non-empty 'action_description'.")

    if not (5 <= shot.duration <= 12):
        errors.append(
            f"Shot duration must be between 5 and 12 seconds, got {shot.duration}."
        )

    from src.models.shot import GenerationMode

    if shot.generation_mode == GenerationMode.FIRSTLAST_FRAME:
        if not shot.start_frame_path:
            errors.append(
                "FIRSTLAST_FRAME shots require 'start_frame_path'."
            )
        if not shot.end_frame_path:
            errors.append(
                "FIRSTLAST_FRAME shots require 'end_frame_path'."
            )

    return errors


def validate_episode(episode: Episode) -> List[str]:
    """Validate an Episode object and return a list of error messages.

    Validates the episode itself and recursively validates all contained
    scenes and shots.

    Args:
        episode: The episode to validate.

    Returns:
        List of validation error strings. Empty list means the episode is valid.
    """
    errors: List[str] = []

    if not episode.id or not episode.id.strip():
        errors.append("Episode must have a non-empty 'id'.")

    if not episode.series_title or not episode.series_title.strip():
        errors.append("Episode must have a non-empty 'series_title'.")

    if episode.episode_number < 1:
        errors.append(
            f"Episode number must be >= 1, got {episode.episode_number}."
        )

    if not episode.episode_title or not episode.episode_title.strip():
        errors.append("Episode must have a non-empty 'episode_title'.")

    if not episode.scenes:
        errors.append("Episode must contain at least one scene.")

    for scene in episode.scenes:
        if not scene.id or not scene.id.strip():
            errors.append(f"Scene {scene.scene_number} must have a non-empty 'id'.")

        if not scene.location or not scene.location.strip():
            errors.append(
                f"Scene {scene.scene_number} must have a non-empty 'location'."
            )

        for shot in scene.shots:
            shot_errors = validate_shot(shot)
            for err in shot_errors:
                errors.append(
                    f"Scene {scene.scene_number}, Shot {shot.sequence_number}: {err}"
                )

    return errors


def assert_valid_character(character: Character) -> None:
    """Validate a character, raising ValidationError on failure.

    Args:
        character: The character to validate.

    Raises:
        ValidationError: If the character fails validation.
    """
    errors = validate_character(character)
    if errors:
        raise ValidationError(
            f"Character '{character.id}' failed validation:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )


def assert_valid_shot(shot: Shot) -> None:
    """Validate a shot, raising ValidationError on failure.

    Args:
        shot: The shot to validate.

    Raises:
        ValidationError: If the shot fails validation.
    """
    errors = validate_shot(shot)
    if errors:
        raise ValidationError(
            f"Shot '{shot.id}' failed validation:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )


def assert_valid_episode(episode: Episode) -> None:
    """Validate an episode, raising ValidationError on failure.

    Args:
        episode: The episode to validate.

    Raises:
        ValidationError: If the episode fails validation.
    """
    errors = validate_episode(episode)
    if errors:
        raise ValidationError(
            f"Episode '{episode.id}' failed validation:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )
