"""Tests for AI-TV-Studio data models."""

import pytest
from src.models.character import Character, CharacterEmotion, CharacterVisualCore
from src.models.episode import Episode
from src.models.scene import Scene
from src.models.shot import CameraMotion, GenerationMode, Shot


class TestCharacterEmotion:
    def test_emotion_values(self):
        assert CharacterEmotion.NEUTRAL == "neutral"
        assert CharacterEmotion.HAPPY == "happy"
        assert CharacterEmotion.ANGRY == "angry"
        assert CharacterEmotion.SAD == "sad"
        assert CharacterEmotion.SURPRISED == "surprised"
        assert CharacterEmotion.FEARFUL == "fearful"


class TestCharacterVisualCore:
    def test_defaults(self):
        vc = CharacterVisualCore(
            base_image_path="/img/char.png",
            reference_prompt="A detective",
            key_features="short black hair, blue eyes",
        )
        assert vc.lora_trigger is None
        assert vc.front_view is None
        assert vc.expressions == {}


class TestCharacter:
    def _make_character(self, **kwargs):
        defaults = dict(
            id="char_001",
            name="Lin Xiao",
            age=28,
            gender="female",
            occupation="Detective",
        )
        defaults.update(kwargs)
        return Character(**defaults)

    def test_basic_creation(self):
        char = self._make_character()
        assert char.id == "char_001"
        assert char.name == "Lin Xiao"
        assert char.age == 28

    def test_defaults(self):
        char = self._make_character()
        assert char.aliases == []
        assert char.personality_traits == []
        assert char.character_arc == ""
        assert char.visual_core is None
        assert char.voice_id is None

    def test_get_expression_frame_no_visual_core(self):
        char = self._make_character()
        result = char.get_expression_frame(CharacterEmotion.HAPPY)
        assert result is None

    def test_get_expression_frame_falls_back_to_base(self):
        vc = CharacterVisualCore(
            base_image_path="/img/base.png",
            reference_prompt="A detective",
            key_features="short black hair",
        )
        char = self._make_character(visual_core=vc)
        result = char.get_expression_frame(CharacterEmotion.HAPPY)
        assert result == "/img/base.png"

    def test_get_expression_frame_specific_emotion(self):
        vc = CharacterVisualCore(
            base_image_path="/img/base.png",
            reference_prompt="A detective",
            key_features="short black hair",
            expressions={CharacterEmotion.HAPPY: "/img/happy.png"},
        )
        char = self._make_character(visual_core=vc)
        result = char.get_expression_frame(CharacterEmotion.HAPPY)
        assert result == "/img/happy.png"

    def test_get_expression_frame_missing_emotion_falls_back(self):
        vc = CharacterVisualCore(
            base_image_path="/img/base.png",
            reference_prompt="A detective",
            key_features="short black hair",
            expressions={CharacterEmotion.HAPPY: "/img/happy.png"},
        )
        char = self._make_character(visual_core=vc)
        # SAD not in expressions → returns base image
        result = char.get_expression_frame(CharacterEmotion.SAD)
        assert result == "/img/base.png"


class TestShot:
    def test_defaults(self):
        shot = Shot(
            id="shot_001",
            scene_id="scene_001",
            sequence_number=1,
            action_description="Character walks in",
        )
        assert shot.generation_mode == GenerationMode.REFERENCE_TO_VIDEO
        assert shot.duration == 8
        assert shot.is_transition_shot is False
        assert shot.characters_in_shot == []
        assert shot.generated_video_path is None

    def test_transition_shot(self):
        shot = Shot(
            id="trans_001",
            scene_id="scene_001",
            sequence_number=2,
            action_description="Character enters",
            generation_mode=GenerationMode.FIRSTLAST_FRAME,
            start_frame_path="/frames/start.png",
            end_frame_path="/frames/end.png",
            is_transition_shot=True,
            transition_type="character_entry",
        )
        assert shot.is_transition_shot is True
        assert shot.transition_type == "character_entry"
        assert shot.generation_mode == GenerationMode.FIRSTLAST_FRAME


class TestGenerationMode:
    def test_values(self):
        assert GenerationMode.TEXT_TO_VIDEO == "txt2video"
        assert GenerationMode.FIRSTLAST_FRAME == "firstlast_frame"
        assert GenerationMode.REFERENCE_TO_VIDEO == "ref2video"


class TestScene:
    def test_creation(self):
        scene = Scene(
            id="scene_001",
            episode_id="ep_001",
            scene_number=1,
            location="Police station",
            time_of_day="day",
            weather="clear",
            mood="tense",
        )
        assert scene.shots == []
        assert scene.ambient_sounds == []
        assert scene.bgm_tempo == "moderate"

    def test_with_shots(self):
        shot = Shot(
            id="shot_001",
            scene_id="scene_001",
            sequence_number=1,
            action_description="Detective examines evidence",
        )
        scene = Scene(
            id="scene_001",
            episode_id="ep_001",
            scene_number=1,
            location="Crime lab",
            time_of_day="night",
            weather="clear",
            mood="tense",
            shots=[shot],
        )
        assert len(scene.shots) == 1


class TestEpisode:
    def test_creation(self):
        episode = Episode(
            id="ep_001",
            series_title="Midnight Detective",
            episode_number=1,
            episode_title="The Missing Person",
            logline="Detective Lin investigates a disappearance.",
        )
        assert episode.genre == "drama"
        assert episode.runtime_estimate == 0
        assert episode.scenes == []
        assert episode.characters == []
