"""Ark provider client for video generation tasks.

This adapter wraps Ark content generation APIs and exposes the
``generate_video`` / ``generate_image`` methods expected by
``VideoDirectorServer``.
"""

import base64
import mimetypes
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.api_key_config import load_video_api_keys

try:
    from volcenginesdkarkruntime import Ark
except Exception:  # pragma: no cover - optional dependency at runtime
    Ark = None


class ArkVideoAPIClient:
    """Video/image generation adapter for Volcengine Ark."""

    DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
    DEFAULT_MODELS = {
        "txt2video": "doubao-seedance-1-5-pro-251215",
        "first_frame": "doubao-seedance-1-5-pro-251215",
        "firstlast_frame": "doubao-seedance-1-5-pro-251215",
        "ref2video": "doubao-seedance-1-0-lite-i2v-250428",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        config_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        poll_interval_seconds: int = 10,
        timeout_seconds: int = 600,
    ):
        """Initialize Ark client from explicit params or config file."""
        if Ark is None:
            raise ImportError(
                "volcengine-python-sdk[ark] is required for ArkVideoAPIClient"
            )

        cfg = load_video_api_keys(config_path=config_path)
        ark_cfg = cfg.get("ark", {}) if isinstance(cfg, dict) else {}
        self.api_key = (
            api_key
            or ark_cfg.get("api_key")
            or os.environ.get("ARK_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "ARK API key not configured. Set config/video_api_keys.yaml"
                " -> ark.api_key or environment variable ARK_API_KEY."
            )

        self.base_url = base_url or ark_cfg.get("base_url") or self.DEFAULT_BASE_URL
        self.output_dir = Path(
            output_dir or ark_cfg.get("output_dir") or "./outputs"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model_map = dict(self.DEFAULT_MODELS)
        self.model_map.update(ark_cfg.get("models", {}))

        self.poll_interval_seconds = max(1, poll_interval_seconds)
        self.timeout_seconds = max(10, timeout_seconds)

        self.client = Ark(base_url=self.base_url, api_key=self.api_key)

    def generate_video(
        self,
        mode: str,
        prompt: str,
        duration: int = 8,
        resolution: str = "1280x720",
        start_frame: Optional[str] = None,
        end_frame: Optional[str] = None,
        first_frame: Optional[str] = None,
        reference_images: Optional[List[str]] = None,
        ratio: Optional[str] = None,
        watermark: bool = False,
        generate_audio: bool = False,
    ) -> Dict[str, Any]:
        """Generate video with Ark and return local download path metadata."""
        mode = self._normalize_mode(mode)
        model = self.model_map.get(mode) or self.model_map["txt2video"]

        content = self._build_video_content(
            mode=mode,
            prompt=prompt,
            start_frame=start_frame,
            end_frame=end_frame,
            first_frame=first_frame,
            reference_images=reference_images or [],
        )

        task = self.client.content_generation.tasks.create(
            model=model,
            content=content,
            ratio=ratio or self._resolution_to_ratio(resolution),
            duration=duration,
            watermark=watermark,
            generate_audio=generate_audio,
        )
        task_result = self._wait_for_task(task_id=task.id)

        media_url = self._extract_media_url(task_result, suffixes=(".mp4",))
        if not media_url:
            raise RuntimeError("Task succeeded but no downloadable .mp4 URL found.")

        local_path = self._download_to_output(media_url, fallback_name=f"{task.id}.mp4")
        return {
            "path": local_path,
            "task_id": task.id,
            "remote_url": media_url,
            "status": "succeeded",
            "mode": mode,
        }

    def generate_image(
        self,
        mode: str,
        prompt: str,
        resolution: str = "1280x720",
        reference_images: Optional[List[str]] = None,
        background_image: Optional[str] = None,
        watermark: bool = False,
    ) -> Dict[str, Any]:
        """Generate image-like output through Ark content generation.

        This method supports a text+reference submission pattern and extracts
        the first downloadable image URL from task results.
        """
        model = self.model_map.get("ref2image") or self.model_map.get("txt2image")
        if not model:
            raise ValueError(
                "No Ark image model configured. Add ark.models.ref2image or txt2image"
                " in config/video_api_keys.yaml"
            )

        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in reference_images or []:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._to_image_url(img)},
                    "role": "reference_image",
                }
            )
        if background_image:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._to_image_url(background_image)},
                }
            )

        task = self.client.content_generation.tasks.create(
            model=model,
            content=content,
            ratio=self._resolution_to_ratio(resolution),
            watermark=watermark,
        )
        task_result = self._wait_for_task(task_id=task.id)
        media_url = self._extract_media_url(
            task_result,
            suffixes=(".png", ".jpg", ".jpeg", ".webp"),
        )
        if not media_url:
            raise RuntimeError(
                "Task succeeded but no downloadable image URL found."
            )

        ext = Path(urllib.parse.urlparse(media_url).path).suffix or ".png"
        local_path = self._download_to_output(media_url, fallback_name=f"{task.id}{ext}")
        return {
            "path": local_path,
            "task_id": task.id,
            "remote_url": media_url,
            "status": "succeeded",
            "mode": mode,
        }

    def _build_video_content(
        self,
        mode: str,
        prompt: str,
        start_frame: Optional[str],
        end_frame: Optional[str],
        first_frame: Optional[str],
        reference_images: List[str],
    ) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]

        if mode == "firstlast_frame":
            if not start_frame or not end_frame:
                raise ValueError("firstlast_frame mode requires start_frame and end_frame")
            content.extend(
                [
                    {
                        "type": "image_url",
                        "image_url": {"url": self._to_image_url(start_frame)},
                        "role": "first_frame",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": self._to_image_url(end_frame)},
                        "role": "last_frame",
                    },
                ]
            )
            return content

        if mode == "first_frame":
            frame = first_frame or start_frame
            if not frame:
                raise ValueError("first_frame mode requires first_frame or start_frame")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._to_image_url(frame)},
                    "role": "first_frame",
                }
            )
            return content

        if mode == "ref2video":
            if not reference_images:
                raise ValueError("ref2video mode requires at least one reference image")
            for ref in reference_images:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": self._to_image_url(ref)},
                        "role": "reference_image",
                    }
                )

        return content

    def _wait_for_task(self, task_id: str) -> Any:
        start_time = time.time()
        while True:
            result = self.client.content_generation.tasks.get(task_id=task_id)
            status = getattr(result, "status", "")
            if status == "succeeded":
                return result
            if status == "failed":
                error = getattr(result, "error", "unknown error")
                raise RuntimeError(f"Ark task {task_id} failed: {error}")

            if time.time() - start_time > self.timeout_seconds:
                raise TimeoutError(
                    f"Ark task {task_id} timed out after {self.timeout_seconds}s"
                )
            time.sleep(self.poll_interval_seconds)

    def _extract_media_url(self, obj: Any, suffixes: tuple) -> Optional[str]:
        if isinstance(obj, str):
            lower = obj.lower()
            if lower.startswith(("http://", "https://")) and any(
                s in lower for s in suffixes
            ):
                return obj
            return None

        if isinstance(obj, dict):
            for value in obj.values():
                url = self._extract_media_url(value, suffixes=suffixes)
                if url:
                    return url
            return None

        if isinstance(obj, (list, tuple)):
            for item in obj:
                url = self._extract_media_url(item, suffixes=suffixes)
                if url:
                    return url
            return None

        if hasattr(obj, "model_dump"):
            try:
                return self._extract_media_url(obj.model_dump(), suffixes=suffixes)
            except Exception:
                return None

        if hasattr(obj, "dict"):
            try:
                return self._extract_media_url(obj.dict(), suffixes=suffixes)
            except Exception:
                return None

        if hasattr(obj, "__dict__"):
            return self._extract_media_url(vars(obj), suffixes=suffixes)

        return None

    def _download_to_output(self, media_url: str, fallback_name: str) -> str:
        parsed = urllib.parse.urlparse(media_url)
        filename = os.path.basename(parsed.path) or fallback_name
        output_path = self.output_dir / filename
        urllib.request.urlretrieve(media_url, output_path)
        return str(output_path)

    def _to_image_url(self, image_path_or_url: str) -> str:
        if image_path_or_url.startswith("http://") or image_path_or_url.startswith(
            "https://"
        ):
            return image_path_or_url

        path = Path(image_path_or_url)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path_or_url}")

        mime_type, _ = mimetypes.guess_type(str(path))
        mime_type = mime_type or "image/png"
        with path.open("rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("ascii")
        return f"data:{mime_type};base64,{b64}"

    def _normalize_mode(self, mode: str) -> str:
        aliases = {
            "first_last_frame": "firstlast_frame",
            "firstlast": "firstlast_frame",
            "first_frame_video": "first_frame",
        }
        return aliases.get(mode, mode)

    def _resolution_to_ratio(self, resolution: str) -> str:
        if not resolution or "x" not in resolution.lower():
            return "adaptive"
        try:
            width_str, height_str = resolution.lower().split("x", maxsplit=1)
            width = int(width_str)
            height = int(height_str)
            if width == 0 or height == 0:
                return "adaptive"

            gcd = self._gcd(width, height)
            return f"{width // gcd}:{height // gcd}"
        except Exception:
            return "adaptive"

    def _gcd(self, a: int, b: int) -> int:
        while b:
            a, b = b, a % b
        return a
