"""Tests for validation utilities."""

import pytest

from src.models.character import Character, CharacterEmotion, CharacterVisualCore
from src.models.episode import Episode
from src.models.scene import Scene
from src.models.shot import GenerationMode, Shot
from src.utils.validators import (
    ValidationError,
    assert_valid_character,
    assert_valid_episode,
    assert_valid_shot,
    validate_character,
    validate_episode,
    validate_shot,
)


def _make_valid_character() -> Character:
    return Character(
        id="c1",
        name="Lin Xiao",
        age=28,
        gender="female",
        occupation="Detective",
    )


def _make_valid_shot() -> Shot:
    return Shot(
        id="shot_001",
        scene_id="scene_001",
        sequence_number=1,
        action_description="Character enters",
        duration=8,
    )


def _make_valid_scene(shot=None) -> Scene:
    if shot is None:
        shot = _make_valid_shot()
    return Scene(
        id="sc_001",
        episode_id="ep_001",
        scene_number=1,
        location="Office",
        time_of_day="day",
        weather="clear",
        mood="neutral",
        shots=[shot],
    )


def _make_valid_episode(scene=None) -> Episode:
    if scene is None:
        scene = _make_valid_scene()
    return Episode(
        id="ep_001",
        series_title="Test Show",
        episode_number=1,
        episode_title="Pilot",
        logline="A detective investigates.",
        scenes=[scene],
    )


class TestValidateCharacter:
    def test_valid_character_no_errors(self):
        char = _make_valid_character()
        assert validate_character(char) == []

    def test_missing_id(self):
        char = _make_valid_character()
        char.id = ""
        errors = validate_character(char)
        assert any("id" in e for e in errors)

    def test_missing_name(self):
        char = _make_valid_character()
        char.name = ""
        errors = validate_character(char)
        assert any("name" in e for e in errors)

    def test_negative_age(self):
        char = _make_valid_character()
        char.age = -1
        errors = validate_character(char)
        assert any("age" in e for e in errors)

    def test_missing_gender(self):
        char = _make_valid_character()
        char.gender = ""
        errors = validate_character(char)
        assert any("gender" in e for e in errors)

    def test_missing_occupation(self):
        char = _make_valid_character()
        char.occupation = ""
        errors = validate_character(char)
        assert any("occupation" in e for e in errors)

    def test_visual_core_missing_base_image(self):
        char = _make_valid_character()
        char.visual_core = CharacterVisualCore(
            base_image_path="",
            reference_prompt="A detective",
            key_features="dark hair",
        )
        errors = validate_character(char)
        assert any("base_image_path" in e for e in errors)

    def test_visual_core_missing_key_features(self):
        char = _make_valid_character()
        char.visual_core = CharacterVisualCore(
            base_image_path="/img/base.png",
            reference_prompt="A detective",
            key_features="",
        )
        errors = validate_character(char)
        assert any("key_features" in e for e in errors)

    def test_assert_valid_raises(self):
        char = _make_valid_character()
        char.name = ""
        with pytest.raises(ValidationError):
            assert_valid_character(char)

    def test_assert_valid_passes(self):
        char = _make_valid_character()
        assert_valid_character(char)  # should not raise


class TestValidateShot:
    def test_valid_shot_no_errors(self):
        shot = _make_valid_shot()
        assert validate_shot(shot) == []

    def test_missing_id(self):
        shot = _make_valid_shot()
        shot.id = ""
        errors = validate_shot(shot)
        assert any("id" in e for e in errors)

    def test_missing_scene_id(self):
        shot = _make_valid_shot()
        shot.scene_id = ""
        errors = validate_shot(shot)
        assert any("scene_id" in e for e in errors)

    def test_missing_action_description(self):
        shot = _make_valid_shot()
        shot.action_description = ""
        errors = validate_shot(shot)
        assert any("action_description" in e for e in errors)

    def test_duration_too_short(self):
        shot = _make_valid_shot()
        shot.duration = 3
        errors = validate_shot(shot)
        assert any("duration" in e for e in errors)

    def test_duration_too_long(self):
        shot = _make_valid_shot()
        shot.duration = 15
        errors = validate_shot(shot)
        assert any("duration" in e for e in errors)

    def test_firstlast_requires_frames(self):
        shot = _make_valid_shot()
        shot.generation_mode = GenerationMode.FIRSTLAST_FRAME
        shot.start_frame_path = None
        shot.end_frame_path = None
        errors = validate_shot(shot)
        assert any("start_frame_path" in e for e in errors)
        assert any("end_frame_path" in e for e in errors)

    def test_firstlast_valid_with_frames(self):
        shot = _make_valid_shot()
        shot.generation_mode = GenerationMode.FIRSTLAST_FRAME
        shot.start_frame_path = "/frames/start.png"
        shot.end_frame_path = "/frames/end.png"
        errors = validate_shot(shot)
        assert errors == []


class TestValidateEpisode:
    def test_valid_episode_no_errors(self):
        episode = _make_valid_episode()
        assert validate_episode(episode) == []

    def test_missing_id(self):
        episode = _make_valid_episode()
        episode.id = ""
        errors = validate_episode(episode)
        assert any("id" in e for e in errors)

    def test_missing_series_title(self):
        episode = _make_valid_episode()
        episode.series_title = ""
        errors = validate_episode(episode)
        assert any("series_title" in e for e in errors)

    def test_invalid_episode_number(self):
        episode = _make_valid_episode()
        episode.episode_number = 0
        errors = validate_episode(episode)
        assert any("episode_number" in e or "number" in e for e in errors)

    def test_missing_episode_title(self):
        episode = _make_valid_episode()
        episode.episode_title = ""
        errors = validate_episode(episode)
        assert any("episode_title" in e for e in errors)

    def test_no_scenes(self):
        episode = _make_valid_episode()
        episode.scenes = []
        errors = validate_episode(episode)
        assert any("scene" in e.lower() for e in errors)

    def test_propagates_shot_errors(self):
        bad_shot = _make_valid_shot()
        bad_shot.duration = 2  # invalid
        episode = _make_valid_episode(scene=_make_valid_scene(shot=bad_shot))
        errors = validate_episode(episode)
        assert any("duration" in e for e in errors)

    def test_assert_valid_raises(self):
        episode = _make_valid_episode()
        episode.id = ""
        with pytest.raises(ValidationError):
            assert_valid_episode(episode)
