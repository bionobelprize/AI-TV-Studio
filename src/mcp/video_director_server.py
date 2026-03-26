"""MCP Video Director Server for AI-TV-Studio.

Provides a Model Context Protocol (MCP) server that bridges the orchestration
layer with external video and image generation APIs, exposing tools for:
- Text-to-video generation
- First-last frame video generation
- Reference-to-video generation
- Image generation with reference controls
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class VideoDirectorServer:
    """MCP server that exposes video and image generation as callable tools.

    Acts as the bridge between the AI-TV-Studio orchestration layer and
    external AI generation APIs (e.g., Runway, Pika, Stable Video Diffusion).

    Supported generation modes:
    - ``txt2video``: Generate video from a text prompt alone.
    - ``firstlast_frame``: Interpolate video between a start and end frame.
    - ``ref2video``: Generate video guided by reference images.
    - ``txt2image``: Generate an image from a text prompt.
    - ``ref2image``: Generate an image guided by reference images.
    """

    def __init__(self, api_client, config: Optional[Dict[str, Any]] = None):
        """Initialize the video director server.

        Args:
            api_client: Client for the external generation API.
            config: Optional configuration overrides (e.g., default resolution,
                default duration).
        """
        self.api = api_client
        self.config = config or {}
        self._tools = self._register_tools()

    def _register_tools(self) -> Dict[str, callable]:
        """Register all available MCP tools.

        Returns:
            Dictionary mapping tool name to handler method.
        """
        return {
            "generate_text_to_video": self.generate_text_to_video,
            "generate_firstlast_frame_video": self.generate_firstlast_frame_video,
            "generate_reference_video": self.generate_reference_video,
            "generate_image": self.generate_image,
            "generate_reference_image": self.generate_reference_image,
        }

    def call_tool(self, tool_name: str, **kwargs: Any) -> Dict[str, Any]:
        """Dispatch a tool call by name.

        Args:
            tool_name: Name of the tool to invoke.
            **kwargs: Keyword arguments forwarded to the tool handler.

        Returns:
            Dictionary containing at minimum a ``"path"`` key with the output
            file path, plus any additional metadata returned by the API.

        Raises:
            ValueError: If the requested tool is not registered.
        """
        if tool_name not in self._tools:
            raise ValueError(
                f"Unknown tool '{tool_name}'. "
                f"Available tools: {list(self._tools.keys())}"
            )
        handler = self._tools[tool_name]
        logger.info("Calling MCP tool: %s", tool_name)
        return handler(**kwargs)

    def generate_text_to_video(
        self,
        prompt: str,
        duration: int = 8,
        resolution: str = "1280x720",
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a video from a text prompt.

        Args:
            prompt: Text description of the video to generate.
            duration: Desired video duration in seconds (5-12).
            resolution: Output resolution as ``"WxH"`` string.
            output_path: Optional path where the video will be saved.

        Returns:
            Dictionary with ``"path"`` and ``"duration"`` keys.
        """
        logger.debug("txt2video | prompt=%s | duration=%ds", prompt[:80], duration)
        result = self.api.generate_video(
            mode="txt2video",
            prompt=prompt,
            duration=duration,
            resolution=resolution,
        )
        path = output_path or result.get("path", "")
        return {"path": path, "duration": duration, "mode": "txt2video"}

    def generate_firstlast_frame_video(
        self,
        start_frame_path: str,
        end_frame_path: str,
        prompt: str = "",
        duration: int = 8,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a video that interpolates between two keyframes.

        This is the core of the character entry stitching mechanism.

        Args:
            start_frame_path: Path to the first frame image.
            end_frame_path: Path to the last frame image.
            prompt: Optional text prompt to guide the motion.
            duration: Desired video duration in seconds.
            output_path: Optional path where the video will be saved.

        Returns:
            Dictionary with ``"path"``, ``"duration"``, and ``"mode"`` keys.
        """
        logger.debug(
            "firstlast_frame | start=%s | end=%s | duration=%ds",
            start_frame_path,
            end_frame_path,
            duration,
        )
        result = self.api.generate_video(
            mode="firstlast_frame",
            start_frame=start_frame_path,
            end_frame=end_frame_path,
            prompt=prompt,
            duration=duration,
        )
        path = output_path or result.get("path", "")
        return {"path": path, "duration": duration, "mode": "firstlast_frame"}

    def generate_reference_video(
        self,
        prompt: str,
        reference_images: List[str],
        duration: int = 8,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a video guided by reference images for character consistency.

        Args:
            prompt: Text description of the video content.
            reference_images: List of paths to reference images.
            duration: Desired video duration in seconds.
            output_path: Optional path where the video will be saved.

        Returns:
            Dictionary with ``"path"``, ``"duration"``, and ``"mode"`` keys.
        """
        logger.debug(
            "ref2video | prompt=%s | refs=%d | duration=%ds",
            prompt[:80],
            len(reference_images),
            duration,
        )
        result = self.api.generate_video(
            mode="ref2video",
            prompt=prompt,
            reference_images=reference_images,
            duration=duration,
        )
        path = output_path or result.get("path", "")
        return {"path": path, "duration": duration, "mode": "ref2video"}

    def generate_image(
        self,
        prompt: str,
        resolution: str = "1280x720",
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate an image from a text prompt.

        Args:
            prompt: Text description of the image to generate.
            resolution: Output resolution as ``"WxH"`` string.
            output_path: Optional path where the image will be saved.

        Returns:
            Dictionary with ``"path"`` key.
        """
        logger.debug("txt2image | prompt=%s", prompt[:80])
        result = self.api.generate_image(
            mode="txt2image",
            prompt=prompt,
            resolution=resolution,
        )
        path = output_path or result.get("path", "")
        return {"path": path, "mode": "txt2image"}

    def generate_reference_image(
        self,
        prompt: str,
        reference_images: List[str],
        background_image: Optional[str] = None,
        pose_control: Optional[Dict[str, Any]] = None,
        lighting_match: bool = False,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate an image guided by reference images and optional controls.

        Used by the character entry stitcher to generate consistent character
        frames.

        Args:
            prompt: Text description of the image.
            reference_images: List of reference image paths.
            background_image: Optional background image for scene context.
            pose_control: Optional pose specification dictionary.
            lighting_match: If True, transfer lighting from the background.
            output_path: Optional path where the image will be saved.

        Returns:
            Dictionary with ``"path"`` key.
        """
        logger.debug(
            "ref2image | prompt=%s | refs=%d", prompt[:80], len(reference_images)
        )
        result = self.api.generate_image(
            mode="ref2image",
            prompt=prompt,
            reference_images=reference_images,
            background_image=background_image,
            pose_control=pose_control,
            lighting_match=lighting_match,
        )
        path = output_path or result.get("path", "")
        return {"path": path, "mode": "ref2image"}

    def list_tools(self) -> List[str]:
        """Return the names of all registered tools.

        Returns:
            Sorted list of tool name strings.
        """
        return sorted(self._tools.keys())
