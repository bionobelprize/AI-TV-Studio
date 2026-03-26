"""Image processing utilities for AI-TV-Studio.

Provides helpers for frame extraction, image resizing, aspect-ratio
normalization, and basic compositing operations used throughout the
generation pipeline.
"""

from pathlib import Path
from typing import Optional, Tuple


class ImageProcessor:
    """Utility class for image manipulation operations.

    Wraps Pillow (PIL) for common image operations needed by the pipeline:
    resizing, cropping, compositing, and format conversion.
    """

    SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

    def __init__(self, default_resolution: Tuple[int, int] = (1280, 720)):
        """Initialize the image processor.

        Args:
            default_resolution: Default (width, height) used when no explicit
                resolution is provided.
        """
        self.default_resolution = default_resolution

    def resize(
        self,
        image_path: str,
        output_path: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        maintain_aspect: bool = True,
    ) -> str:
        """Resize an image to the specified dimensions.

        Args:
            image_path: Path to the source image.
            output_path: Path where the resized image will be saved.
            width: Target width in pixels. If None, uses the default width.
            height: Target height in pixels. If None, uses the default height.
            maintain_aspect: If True, preserve the aspect ratio by fitting
                the image within the target dimensions.

        Returns:
            Path to the resized image.
        """
        from PIL import Image

        target_w = width or self.default_resolution[0]
        target_h = height or self.default_resolution[1]

        with Image.open(image_path) as img:
            if maintain_aspect:
                img.thumbnail((target_w, target_h), Image.LANCZOS)
            else:
                img = img.resize((target_w, target_h), Image.LANCZOS)
            img.save(output_path)

        return output_path

    def crop_to_aspect(
        self,
        image_path: str,
        output_path: str,
        aspect_ratio: Tuple[int, int] = (16, 9),
    ) -> str:
        """Crop an image to a specific aspect ratio by center-cropping.

        Args:
            image_path: Path to the source image.
            output_path: Path where the cropped image will be saved.
            aspect_ratio: Target aspect ratio as (width, height) tuple.

        Returns:
            Path to the cropped image.
        """
        from PIL import Image

        ar_w, ar_h = aspect_ratio

        with Image.open(image_path) as img:
            img_w, img_h = img.size
            target_ratio = ar_w / ar_h
            current_ratio = img_w / img_h

            if current_ratio > target_ratio:
                # Image is wider than target: crop sides
                new_w = int(img_h * target_ratio)
                left = (img_w - new_w) // 2
                img = img.crop((left, 0, left + new_w, img_h))
            elif current_ratio < target_ratio:
                # Image is taller than target: crop top/bottom
                new_h = int(img_w / target_ratio)
                top = (img_h - new_h) // 2
                img = img.crop((0, top, img_w, top + new_h))

            img.save(output_path)

        return output_path

    def composite_character(
        self,
        background_path: str,
        character_path: str,
        output_path: str,
        position: Tuple[float, float] = (0.5, 0.5),
        scale: float = 0.3,
    ) -> str:
        """Composite a character image onto a background.

        The character image is scaled relative to the background height and
        placed at the given normalised position.

        Args:
            background_path: Path to the background image.
            character_path: Path to the character image (should have alpha).
            output_path: Path where the composited image will be saved.
            position: Normalised (x, y) position for the character centre,
                where (0, 0) is top-left and (1, 1) is bottom-right.
            scale: Character height as a fraction of the background height.

        Returns:
            Path to the composited image.
        """
        from PIL import Image

        with Image.open(background_path).convert("RGBA") as bg:
            bg_w, bg_h = bg.size
            char_h = int(bg_h * scale)

            with Image.open(character_path).convert("RGBA") as char:
                char_w, char_img_h = char.size
                char_scale = char_h / char_img_h
                char_w_scaled = int(char_w * char_scale)
                char_resized = char.resize(
                    (char_w_scaled, char_h), Image.LANCZOS
                )

            # Centre position in pixels
            cx = int(bg_w * position[0])
            cy = int(bg_h * position[1])
            paste_x = cx - char_w_scaled // 2
            paste_y = cy - char_h // 2

            bg.paste(char_resized, (paste_x, paste_y), char_resized)
            bg.convert("RGB").save(output_path)

        return output_path

    def normalize_format(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        target_format: str = "PNG",
    ) -> str:
        """Convert an image to a normalised format and resolution.

        Args:
            image_path: Path to the source image.
            output_path: Optional output path. If None, the source file is
                overwritten (with the new extension if the format changes).
            target_format: Pillow format string (e.g., ``"PNG"``, ``"JPEG"``).

        Returns:
            Path to the normalised image.
        """
        from PIL import Image

        src = Path(image_path)
        ext = "." + target_format.lower()
        dest_path = output_path or str(src.with_suffix(ext))

        with Image.open(image_path) as img:
            if target_format == "JPEG":
                img = img.convert("RGB")
            img.save(dest_path, format=target_format)

        return dest_path

    def is_valid_image(self, image_path: str) -> bool:
        """Check whether a file is a readable, valid image.

        Args:
            image_path: Path to the image file.

        Returns:
            True if the file exists and can be opened as an image.
        """
        path = Path(image_path)
        if not path.exists() or path.suffix.lower() not in self.SUPPORTED_FORMATS:
            return False
        try:
            from PIL import Image

            with Image.open(image_path) as img:
                img.verify()
            return True
        except Exception:
            return False
