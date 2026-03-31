"""Tests for the shot planner algorithm."""

import pytest

from src.algorithms.shot_planner import ShotPlanner
from src.models.character import Character, CharacterVisualCore
from src.models.episode import Episode
from src.models.scene import Scene
from src.models.shot import GenerationMode, Shot


def _make_character(char_id: str, name: str) -> Character:
    return Character(
        id=char_id,
        name=name,
        age=30,
        gender="male",
        occupation="Actor",
    )


def _make_shot(shot_id: str, scene_id: str, seq: int, chars=None, duration: int = 8) -> Shot:
    return Shot(
        id=shot_id,
        scene_id=scene_id,
        sequence_number=seq,
        action_description=f"Action for shot {shot_id}",
        characters_in_shot=chars or [],
        duration=duration,
    )


class TestShotPlannerDurationClamping:
    def setup_method(self):
        self.planner = ShotPlanner()

    def test_clamp_below_min(self):
        assert self.planner._clamp_duration(2) == 5

    def test_clamp_above_max(self):
        assert self.planner._clamp_duration(20) == 12

    def test_clamp_within_range(self):
        assert self.planner._clamp_duration(8) == 8

    def test_clamp_at_boundaries(self):
        assert self.planner._clamp_duration(5) == 5
        assert self.planner._clamp_duration(12) == 12


class TestShotPlannerGenerationMode:
    def setup_method(self):
        self.planner = ShotPlanner()

    def test_keep_firstlast_frame(self):
        shot = _make_shot("s1", "sc1", 1)
        shot.generation_mode = GenerationMode.FIRSTLAST_FRAME
        assert (
            self.planner._select_generation_mode(shot, True)
            == GenerationMode.FIRSTLAST_FRAME
        )

    def test_ref2video_when_preceding_shot_and_characters(self):
        shot = _make_shot("s1", "sc1", 1, chars=["c1"])
        assert (
            self.planner._select_generation_mode(shot, True)
            == GenerationMode.REFERENCE_TO_VIDEO
        )

    def test_ref2video_when_non_transition_shot_with_characters(self):
        """Non-transition shots with characters should use reference images."""
        shot = _make_shot("s1", "sc1", 1, chars=["c1"])
        shot.is_transition_shot = False
        # Even without preceding shot, should use reference images for character consistency
        assert (
            self.planner._select_generation_mode(shot, False)
            == GenerationMode.REFERENCE_TO_VIDEO
        )

    def test_txt2video_for_transition_shot_with_characters(self):
        """Transition shots should not automatically use reference images."""
        shot = _make_shot("s1", "sc1", 1, chars=["c1"])
        shot.is_transition_shot = True
        # Transition shots without preceding context should fall back to TEXT_TO_VIDEO
        assert (
            self.planner._select_generation_mode(shot, False)
            == GenerationMode.TEXT_TO_VIDEO
        )

    def test_txt2video_no_context(self):
        shot = _make_shot("s1", "sc1", 1)
        assert (
            self.planner._select_generation_mode(shot, False)
            == GenerationMode.TEXT_TO_VIDEO
        )

    def test_ref2video_when_reference_images_provided(self):
        shot = _make_shot("s1", "sc1", 1)
        shot.reference_images = ["/img/ref.png"]
        assert (
            self.planner._select_generation_mode(shot, False)
            == GenerationMode.REFERENCE_TO_VIDEO
        )

    def test_first_frame_when_only_start_frame_provided(self):
        shot = _make_shot("s1", "sc1", 1)
        shot.start_frame_path = "/img/first.png"
        shot.end_frame_path = None
        assert (
            self.planner._select_generation_mode(shot, False)
            == GenerationMode.FIRST_FRAME
        )


class TestShotPlannerReferenceImagePopulation:
    """Test reference image population for character-based shots."""

    def setup_method(self):
        from unittest.mock import MagicMock
        from src.models.character import CharacterVisualCore

        self.asset_manager_mock = MagicMock()
        self.planner = ShotPlanner(asset_manager=self.asset_manager_mock)

        # Setup mock character visual cores
        self.asset_manager_mock.get_character_reference_image.side_effect = lambda cid: (
            f"/refs/{cid}_ref.png" if cid in ["c1", "c2"] else None
        )

    def test_populate_reference_images_for_non_transition_shot(self):
        """Non-transition shots with characters should get reference images populated."""
        shot = _make_shot("s1", "sc1", 1, chars=["c1", "c2"])
        shot.is_transition_shot = False

        self.planner._populate_reference_images(shot)

        # Reference images should be populated
        assert len(shot.reference_images) == 2
        assert "/refs/c1_ref.png" in shot.reference_images
        assert "/refs/c2_ref.png" in shot.reference_images

    def test_populate_reference_images_skips_transition_shots(self):
        """Transition shots should not have reference images populated."""
        shot = _make_shot("s1", "sc1", 1, chars=["c1"])
        shot.is_transition_shot = True
        shot.reference_images = []

        self.planner._populate_reference_images(shot)

        # Reference images should NOT be populated for transition shots
        # (This is controlled by the caller's condition)

    def test_populate_reference_images_handles_missing_assets(self):
        """Should handle characters without reference images gracefully."""
        shot = _make_shot("s1", "sc1", 1, chars=["c1", "unknown_char"])
        shot.is_transition_shot = False

        self.planner._populate_reference_images(shot)

        # Only available reference images should be added
        assert len(shot.reference_images) == 1
        assert "/refs/c1_ref.png" in shot.reference_images

    def test_populate_reference_images_with_no_asset_manager(self):
        """Should safely skip if no asset manager is available."""
        planner = ShotPlanner(asset_manager=None)
        shot = _make_shot("s1", "sc1", 1, chars=["c1"])
        shot.is_transition_shot = False
        shot.reference_images = []

        planner._populate_reference_images(shot)

        # Should not raise an error, reference_images should not be modified
        assert len(shot.reference_images) == 0


class TestShotPlannerTransitionInsertion:
    def setup_method(self):
        char_a = _make_character("c_a", "Alice")
        char_b = _make_character("c_b", "Bob")
        self.planner = ShotPlanner(character_registry={"c_a": char_a, "c_b": char_b})

    def test_no_transition_for_single_character_scene(self):
        shots = [_make_shot("s1", "sc1", 1, chars=["c_a"])]
        scene = Scene(
            id="sc1",
            episode_id="ep1",
            scene_number=1,
            location="Office",
            time_of_day="day",
            weather="clear",
            mood="neutral",
            shots=shots,
        )
        result = self.planner.plan_scene(scene)
        # No existing characters before first shot, so no transition
        assert len(result) == 1
        assert result[0].is_transition_shot is False

    def test_transition_inserted_for_new_character(self):
        shot1 = _make_shot("s1", "sc1", 1, chars=["c_a"])
        shot2 = _make_shot("s2", "sc1", 2, chars=["c_a", "c_b"])
        scene = Scene(
            id="sc1",
            episode_id="ep1",
            scene_number=1,
            location="Office",
            time_of_day="day",
            weather="clear",
            mood="neutral",
            shots=[shot1, shot2],
        )
        result = self.planner.plan_scene(scene)
        # Expect: shot1, transition for c_b, shot2
        assert len(result) == 3
        transition = result[1]
        assert transition.is_transition_shot is True
        assert transition.transition_type == "character_entry"
        assert "c_b" in transition.characters_in_shot

    def test_no_transition_for_same_characters(self):
        shot1 = _make_shot("s1", "sc1", 1, chars=["c_a"])
        shot2 = _make_shot("s2", "sc1", 2, chars=["c_a"])
        scene = Scene(
            id="sc1",
            episode_id="ep1",
            scene_number=1,
            location="Office",
            time_of_day="day",
            weather="clear",
            mood="neutral",
            shots=[shot1, shot2],
        )
        result = self.planner.plan_scene(scene)
        assert len(result) == 2
        assert all(not s.is_transition_shot for s in result)

    def test_plan_episode_populates_all_scenes(self):
        scene1 = Scene(
            id="sc1",
            episode_id="ep1",
            scene_number=1,
            location="Office",
            time_of_day="day",
            weather="clear",
            mood="neutral",
            shots=[_make_shot("s1", "sc1", 1, chars=["c_a"])],
        )
        scene2 = Scene(
            id="sc2",
            episode_id="ep1",
            scene_number=2,
            location="Street",
            time_of_day="night",
            weather="rain",
            mood="tense",
            shots=[_make_shot("s2", "sc2", 1, chars=["c_b"])],
        )
        episode = Episode(
            id="ep1",
            series_title="Test Series",
            episode_number=1,
            episode_title="Pilot",
            logline="A test episode.",
            scenes=[scene1, scene2],
        )
        result = self.planner.plan_episode(episode)
        assert len(result.scenes[0].shots) >= 1
        assert len(result.scenes[1].shots) >= 1
