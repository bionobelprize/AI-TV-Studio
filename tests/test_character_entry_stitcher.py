"""Tests for the character entry stitcher algorithm."""

import pytest
from unittest.mock import MagicMock, patch

from src.algorithms.character_entry_stitcher import (
    CharacterEntryParameters,
    CharacterEntryStitcher,
)
from src.models.character import Character, CharacterEmotion, CharacterVisualCore
from src.models.shot import GenerationMode, Shot


def _make_character(char_id: str = "char_001", name: str = "Lin Xiao") -> Character:
    vc = CharacterVisualCore(
        base_image_path="/img/base.png",
        reference_prompt="A detective",
        key_features="short black hair, blue eyes",
    )
    return Character(
        id=char_id,
        name=name,
        age=28,
        gender="female",
        occupation="Detective",
        visual_core=vc,
    )


def _make_previous_shot(video_path: str = "/videos/shot_001.mp4") -> Shot:
    shot = Shot(
        id="shot_001",
        scene_id="scene_001",
        sequence_number=1,
        action_description="Detective walks to desk",
        generated_video_path=video_path,
    )
    return shot


class TestCharacterEntryParameters:
    def test_defaults(self):
        params = CharacterEntryParameters(
            new_character_id="char_002",
            entry_style="walk_in",
            entry_direction="left",
        )
        assert params.entry_duration == 8
        assert params.intermediate_steps == 0
        assert params.final_position == (0.5, 0.5)
        assert params.final_scale == 0.3
        assert params.interaction_with_existing is True

    def test_custom_values(self):
        params = CharacterEntryParameters(
            new_character_id="char_003",
            entry_style="emerges_from_door",
            entry_direction="right",
            entry_duration=10,
            intermediate_steps=2,
            final_position=(0.7, 0.6),
            final_scale=0.4,
        )
        assert params.entry_duration == 10
        assert params.intermediate_steps == 2


class TestCharacterEntryStitcher:
    def setup_method(self):
        self.image_gen = MagicMock()
        self.video_gen = MagicMock()
        self.stitcher = CharacterEntryStitcher(
            image_generator=self.image_gen,
            video_generator=self.video_gen,
        )

    def test_raises_if_no_video_path(self):
        shot = Shot(
            id="shot_001",
            scene_id="scene_001",
            sequence_number=1,
            action_description="Walk",
            generated_video_path=None,
        )
        params = CharacterEntryParameters(
            new_character_id="char_002",
            entry_style="walk_in",
            entry_direction="left",
        )
        new_char = _make_character("char_002", "Wang")

        with pytest.raises(ValueError, match="must have generated video"):
            self.stitcher.stitch_character_entry(shot, new_char, params, [])

    def test_single_transition_no_intermediate(self):
        shot = _make_previous_shot()
        params = CharacterEntryParameters(
            new_character_id="char_002",
            entry_style="walk_in",
            entry_direction="left",
            intermediate_steps=0,
        )
        new_char = _make_character("char_002", "Wang")
        existing = [_make_character()]

        with (
            patch.object(
                self.stitcher,
                "_extract_tail_frame",
                return_value="/frames/tail.png",
            ),
            patch.object(
                self.stitcher,
                "_generate_entry_frame",
                return_value="/frames/entry.png",
            ),
        ):
            result = self.stitcher.stitch_character_entry(
                shot, new_char, params, existing
            )

        assert len(result) == 1
        transition = result[0]
        assert transition.is_transition_shot is True
        assert transition.transition_type == "character_entry"
        assert transition.generation_mode == GenerationMode.FIRSTLAST_FRAME
        assert transition.start_frame_path == "/frames/tail.png"
        assert transition.end_frame_path == "/frames/entry.png"

    def test_multi_stage_entry(self):
        shot = _make_previous_shot()
        params = CharacterEntryParameters(
            new_character_id="char_002",
            entry_style="walk_in",
            entry_direction="right",
            intermediate_steps=2,
        )
        new_char = _make_character("char_002", "Wang")

        with (
            patch.object(
                self.stitcher,
                "_extract_tail_frame",
                return_value="/frames/tail.png",
            ),
            patch.object(
                self.stitcher,
                "_generate_entry_frame",
                return_value="/frames/entry.png",
            ),
            patch.object(
                self.stitcher,
                "_generate_intermediate_frame",
                side_effect=["/frames/int_1.png", "/frames/int_2.png"],
            ),
        ):
            result = self.stitcher.stitch_character_entry(
                shot, new_char, params, []
            )

        # 2 intermediate steps → 2 intermediate transitions + 1 final = 3 shots
        assert len(result) == 3

    def test_create_transition_shot_prompt_contains_character_name(self):
        params = CharacterEntryParameters(
            new_character_id="char_002",
            entry_style="walk_in",
            entry_direction="left",
        )
        new_char = _make_character("char_002", "Captain Wang")
        shot = self.stitcher._create_transition_shot(
            start_frame="/frames/start.png",
            end_frame="/frames/end.png",
            entry_params=params,
            new_character=new_char,
        )
        assert "Captain Wang" in shot.action_description
        assert shot.generation_mode == GenerationMode.FIRSTLAST_FRAME
        assert shot.duration == params.entry_duration

    def test_describe_entry(self):
        params = CharacterEntryParameters(
            new_character_id="c1",
            entry_style="emerges_from_door",
            entry_direction="right",
        )
        desc = self.stitcher._describe_entry(params)
        assert "emerges_from_door" in desc
        assert "right" in desc

    def test_build_entry_prompt_mentions_character_name(self):
        params = CharacterEntryParameters(
            new_character_id="c2",
            entry_style="walk_in",
            entry_direction="left",
        )
        new_char = _make_character("c2", "Detective Lin")
        prompt = self.stitcher._build_entry_prompt(
            new_character=new_char,
            existing_characters=[],
            entry_params=params,
            scene_composition={},
        )
        assert "Detective Lin" in prompt
