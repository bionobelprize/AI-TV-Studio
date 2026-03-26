"""Asset Manager for AI-TV-Studio.

Manages character reference images, scene backgrounds, and generated assets,
providing a central registry for all visual resources used in production.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

from src.models.character import Character, CharacterEmotion, CharacterVisualCore


class AssetManager:
    """Central registry for all production assets.

    Manages:
    - Character reference image libraries
    - Scene background assets
    - Generated video/image output tracking
    """

    def __init__(self, base_dir: str = "data"):
        """Initialize the asset manager.

        Args:
            base_dir: Root directory for all asset storage.
        """
        self.base_dir = Path(base_dir)
        self.characters_dir = self.base_dir / "characters"
        self.scenes_dir = self.base_dir / "scenes"
        self.generated_dir = self.base_dir / "generated"
        self.output_dir = self.base_dir / "output"

        self._character_registry: Dict[str, Character] = {}
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create asset storage directories if they don't exist."""
        for directory in [
            self.characters_dir,
            self.scenes_dir,
            self.generated_dir,
            self.output_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    def register_character(self, character: Character) -> None:
        """Register a character in the asset registry.

        Args:
            character: The character to register.
        """
        self._character_registry[character.id] = character

    def get_character(self, character_id: str) -> Optional[Character]:
        """Retrieve a registered character by ID.

        Args:
            character_id: Unique identifier of the character.

        Returns:
            The Character object, or None if not found.
        """
        return self._character_registry.get(character_id)

    def get_all_characters(self) -> Dict[str, Character]:
        """Return all registered characters.

        Returns:
            Dictionary mapping character ID to Character object.
        """
        return dict(self._character_registry)

    def get_character_reference_image(
        self, character_id: str, emotion: Optional[CharacterEmotion] = None
    ) -> Optional[str]:
        """Get the reference image path for a character.

        Args:
            character_id: ID of the character.
            emotion: Optional emotion to retrieve a specific expression image.

        Returns:
            Path to the reference image, or None if the character is not found
            or has no visual core.
        """
        character = self.get_character(character_id)
        if not character:
            return None
        if emotion:
            return character.get_expression_frame(emotion)
        return (
            character.visual_core.base_image_path if character.visual_core else None
        )

    def register_character_assets(
        self,
        character: Character,
        base_image_path: str,
        reference_prompt: str,
        key_features: str,
        expression_images: Optional[Dict[CharacterEmotion, str]] = None,
        front_view: Optional[str] = None,
        side_view: Optional[str] = None,
        three_quarter_view: Optional[str] = None,
        lora_trigger: Optional[str] = None,
    ) -> Character:
        """Attach a visual core to a character and register it.

        Args:
            character: The character to update with visual assets.
            base_image_path: Path to the character's primary reference image.
            reference_prompt: Text prompt describing the character's appearance.
            key_features: Comma-separated list of distinctive visual features.
            expression_images: Optional mapping of emotion to expression image path.
            front_view: Optional path to a front-view reference image.
            side_view: Optional path to a side-view reference image.
            three_quarter_view: Optional path to a 3/4-view reference image.
            lora_trigger: Optional LoRA trigger word for the character.

        Returns:
            The updated Character with the visual core attached.
        """
        character.visual_core = CharacterVisualCore(
            base_image_path=base_image_path,
            reference_prompt=reference_prompt,
            key_features=key_features,
            lora_trigger=lora_trigger,
            front_view=front_view,
            side_view=side_view,
            three_quarter_view=three_quarter_view,
            expressions=expression_images or {},
        )
        self.register_character(character)
        return character

    def get_scene_background(self, scene_id: str) -> Optional[str]:
        """Retrieve the background image path for a scene.

        Args:
            scene_id: Unique identifier of the scene.

        Returns:
            Path to the background image, or None if none has been generated.
        """
        candidate = self.scenes_dir / f"{scene_id}_background.png"
        return str(candidate) if candidate.exists() else None

    def save_generated_asset(
        self, asset_type: str, asset_id: str, source_path: str
    ) -> str:
        """Copy a generated asset into the managed output directory.

        Args:
            asset_type: Category of asset (e.g., "frame", "video").
            asset_id: Unique identifier for this asset.
            source_path: Current path of the generated file.

        Returns:
            Final managed path of the asset.
        """
        import shutil

        ext = Path(source_path).suffix
        dest = self.generated_dir / f"{asset_type}_{asset_id}{ext}"
        shutil.copy2(source_path, dest)
        return str(dest)

    def list_generated_videos(self) -> List[str]:
        """Return paths to all generated video files.

        Returns:
            Sorted list of video file paths in the generated directory.
        """
        return sorted(
            str(p)
            for p in self.generated_dir.iterdir()
            if p.suffix in {".mp4", ".mov", ".avi"}
        )
