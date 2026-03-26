"""Tests for the Asset Manager pipeline module."""

import os
import shutil
import tempfile

import pytest

from src.models.character import Character, CharacterEmotion
from src.pipeline.asset_manager import AssetManager


def _make_character(char_id: str = "c1", name: str = "Lin") -> Character:
    return Character(
        id=char_id,
        name=name,
        age=28,
        gender="female",
        occupation="Detective",
    )


class TestAssetManager:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.manager = AssetManager(base_dir=self.tmpdir)

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_directories_created(self):
        assert os.path.isdir(os.path.join(self.tmpdir, "characters"))
        assert os.path.isdir(os.path.join(self.tmpdir, "scenes"))
        assert os.path.isdir(os.path.join(self.tmpdir, "generated"))
        assert os.path.isdir(os.path.join(self.tmpdir, "output"))

    def test_register_and_get_character(self):
        char = _make_character()
        self.manager.register_character(char)
        assert self.manager.get_character("c1") is char

    def test_get_nonexistent_character_returns_none(self):
        assert self.manager.get_character("does_not_exist") is None

    def test_get_all_characters(self):
        char1 = _make_character("c1", "Alice")
        char2 = _make_character("c2", "Bob")
        self.manager.register_character(char1)
        self.manager.register_character(char2)
        all_chars = self.manager.get_all_characters()
        assert "c1" in all_chars
        assert "c2" in all_chars

    def test_get_character_reference_image_no_visual_core(self):
        char = _make_character()
        self.manager.register_character(char)
        result = self.manager.get_character_reference_image("c1")
        assert result is None

    def test_register_character_assets(self):
        char = _make_character()
        updated = self.manager.register_character_assets(
            character=char,
            base_image_path="/img/base.png",
            reference_prompt="A young detective",
            key_features="short black hair, blue eyes",
        )
        assert updated.visual_core is not None
        assert updated.visual_core.base_image_path == "/img/base.png"
        assert self.manager.get_character("c1") is updated

    def test_get_character_reference_image_with_visual_core(self):
        char = _make_character()
        self.manager.register_character_assets(
            character=char,
            base_image_path="/img/base.png",
            reference_prompt="Detective",
            key_features="dark hair",
        )
        result = self.manager.get_character_reference_image("c1")
        assert result == "/img/base.png"

    def test_get_scene_background_nonexistent(self):
        result = self.manager.get_scene_background("scene_999")
        assert result is None

    def test_save_generated_asset(self):
        # Create a temporary source file
        src = tempfile.NamedTemporaryFile(
            suffix=".png", dir=self.tmpdir, delete=False
        )
        src.write(b"fake image data")
        src.close()

        dest = self.manager.save_generated_asset("frame", "test_001", src.name)
        assert os.path.exists(dest)
        assert "frame_test_001" in dest

    def test_list_generated_videos_empty(self):
        videos = self.manager.list_generated_videos()
        assert videos == []
