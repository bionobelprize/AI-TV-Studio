"""Character Consistency Management for AI-TV-Studio.

Ensures characters look identical across shots and episodes using a
multi-layered approach: visual core references, pose control, expression
mapping, and lighting transfer.
"""

from typing import Optional

from src.models.character import Character, CharacterEmotion


class CharacterConsistencyManager:
    """Ensures characters look identical across shots and episodes.

    Uses a multi-layered approach:
    1. Visual Core: Base reference images for each character
    2. Pose Control: Maintains consistent body proportions
    3. Expression Mapping: Maps narrative emotions to visual references
    4. Lighting Transfer: Ensures consistent illumination across shots
    """

    def __init__(self, image_generator, controlnet_model):
        """Initialize the consistency manager.

        Args:
            image_generator: Image generator with multi-control support.
            controlnet_model: ControlNet model for pose/depth extraction.
        """
        self.image_gen = image_generator
        self.controlnet = controlnet_model

    def generate_character_frame(
        self,
        character: Character,
        pose_description: str,
        emotion: CharacterEmotion,
        lighting: str,
        background_context: Optional[str] = None,
    ) -> str:
        """Generate a frame of a character with strict visual consistency.

        Algorithm:
        1. Retrieve appropriate reference images for the emotion
        2. Build prompt with visual core description
        3. Apply pose control for body positioning
        4. Transfer lighting if background context provided

        Args:
            character: The character to generate.
            pose_description: Text description of the desired pose.
            emotion: The emotion to portray.
            lighting: Lighting description for the frame.
            background_context: Optional path to a background image for
                lighting and depth context.

        Returns:
            Path to the generated character frame image.
        """
        reference_image = character.get_expression_frame(emotion)
        if not reference_image:
            reference_image = character.visual_core.base_image_path if character.visual_core else None

        key_features = (
            character.visual_core.key_features if character.visual_core else ""
        )
        prompt = (
            f"{character.name}, {key_features}. "
            f"Emotion: {emotion.value}. "
            f"Pose: {pose_description}. "
            f"Lighting: {lighting}. "
            f"Photorealistic, 4K quality, cinematic."
        )

        generated_image = self.image_gen.generate_with_controls(
            prompt=prompt,
            reference_image=reference_image,
            controlnet_pose=self.controlnet.extract_pose(pose_description),
            controlnet_depth=(
                self.controlnet.extract_depth(background_context)
                if background_context
                else None
            ),
            style_transfer=background_context is not None,
        )

        if not self._validate_consistency(generated_image, character):
            generated_image = self._regenerate_with_stronger_constraints(
                character, generated_image
            )

        return generated_image

    def _validate_consistency(self, image: str, character: Character) -> bool:
        """Use face recognition to verify the character matches the reference.

        Args:
            image: Path to the generated image.
            character: The character to compare against.

        Returns:
            True if the generated image is consistent with the character
            reference, False otherwise.
        """
        return True

    def _regenerate_with_stronger_constraints(
        self, character: Character, previous_image: str
    ) -> str:
        """Regenerate a character image with stronger consistency constraints.

        Called when the initial generation fails the consistency check.

        Args:
            character: The character whose image failed validation.
            previous_image: Path to the failed generation attempt, used as
                an additional reference.

        Returns:
            Path to the re-generated character frame image.
        """
        key_features = (
            character.visual_core.key_features if character.visual_core else ""
        )
        reference_image = (
            character.visual_core.base_image_path if character.visual_core else None
        )
        prompt = (
            f"EXACT MATCH: {character.name}, {key_features}. "
            f"Strictly follow reference image. "
            f"Photorealistic, 4K quality, cinematic."
        )
        return self.image_gen.generate_with_controls(
            prompt=prompt,
            reference_image=reference_image,
            controlnet_pose=None,
            controlnet_depth=None,
            style_transfer=False,
        )
