"""AI-TV-Studio: Main orchestration module.

Provides the AITVStudio class that coordinates the complete production
pipeline from series configuration through final episode assembly.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from src.algorithms.character_entry_stitcher import (
    CharacterEntryParameters,
    CharacterEntryStitcher,
)
from src.algorithms.shot_planner import ShotPlanner
from src.models.character import Character
from src.models.episode import Episode
from src.pipeline.asset_manager import AssetManager
from src.pipeline.script_generator import ScriptGenerator
from src.pipeline.video_assembler import VideoAssembler
from src.utils.ffmpeg_helper import FFmpegHelper
import src.model_load as model_load

logger = logging.getLogger(__name__)


class AITVStudio:
    """Main orchestration class for the AI-TV-Studio production pipeline.

    Pipeline Stages:
    1. Script Generation (LLM) → Structured Episode
    2. Character Asset registration
    3. Shot Planning → Optimised shot sequence with transitions
    4. Video Generation → Through MCP video director
    5. Post-Processing → Stitching, audio, assembly

    Example::

        studio = AITVStudio(config={
            "llm_model": "deepseek-chat",
            "video_api": "runway",
            "output_dir": "./output",
        })
        episode = studio.produce_episode(
            series_config=series_config,
            episode_outline="Detective Lin investigates a mysterious case.",
        )
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the studio with the provided configuration.

        The LLM client is loaded automatically from :mod:`src.model_load`
        (``deepseek-chat`` via ``DEEPSEEK_API_KEY``) following the same
        pattern as :class:`~src.pipeline.script_generator.ScriptGenerator`.
        Call :meth:`configure_llm` to override with a custom client.

        Args:
            config: Configuration dictionary. Supported keys:
                - ``llm_model``: Model identifier (default: ``"deepseek-chat"``).
                  The LLM provider is always DeepSeek (via :mod:`src.model_load`).
                - ``video_api``: Video API provider (default: ``"runway"``).
                - ``output_dir``: Output directory (default: ``"./output"``).
                - ``data_dir``: Asset data directory (default: ``"./data"``).
        """
        self.config = config or {}
        output_dir = self.config.get("output_dir", "./output")
        data_dir = self.config.get("data_dir", "./data")

        self.asset_manager = AssetManager(base_dir=data_dir)
        self.video_assembler = VideoAssembler(output_dir=output_dir)
        self.shot_planner: Optional[ShotPlanner] = None
        self.script_generator: Optional[ScriptGenerator] = None
        self._mcp_server = None

        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure basic logging for the studio."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    def configure_llm(self, llm_client, model: Optional[str] = None) -> None:
        """Attach a custom LLM client for script generation.

        This overrides the default client loaded from :mod:`src.model_load`.
        Both the client and model follow the same interface as used in
        :class:`~src.pipeline.script_generator.ScriptGenerator`.

        Args:
            llm_client: LangChain-compatible LLM client (e.g. ``ChatDeepSeek``).
            model: Optional model identifier override (default: ``"deepseek-chat"``).
        """
        model = model or self.config.get("llm_model", "deepseek-chat")
        self.script_generator = ScriptGenerator(llm_client=llm_client, model=model)

    def configure_mcp(self, mcp_server) -> None:
        """Attach an MCP video director server for generation calls.

        Args:
            mcp_server: Configured VideoDirectorServer instance.
        """
        self._mcp_server = mcp_server
        self.shot_planner = ShotPlanner(
            character_registry=self.asset_manager.get_all_characters(),
            asset_manager=self.asset_manager
        )

    def register_character(self, character: Character) -> None:
        """Register a character in the asset manager.

        Args:
            character: Character to register.
        """
        self.asset_manager.register_character(character)
        logger.info("Registered character: %s (%s)", character.name, character.id)

    def produce_episode(
        self,
        series_config: Dict[str, Any],
        episode_outline: str,
        episode_number: int = 1,
    ) -> Episode:
        """Run the full production pipeline for a single episode.

        Steps:
        1. Generate a structured script via the LLM.
        2. Plan shot sequences and insert transition shots.
        3. Generate each shot video via the MCP server.
        4. Stitch character entry transitions.
        5. Assemble all shots into the final episode video.

        Args:
            series_config: Dictionary with series metadata and characters.
            episode_outline: One-paragraph description of the episode.
            episode_number: Episode number within the series.

        Returns:
            The produced Episode with all shots generated and assembled.

        Raises:
            RuntimeError: If the MCP server has not been configured.
        """
        if not self._mcp_server:
            raise RuntimeError(
                "MCP server not configured. Call configure_mcp() first."
            )

        # Auto-initialize the ScriptGenerator from model_load if not already
        # configured via configure_llm(), following the same pattern as
        # ScriptGenerator itself (see src/pipeline/script_generator.py).
        if not self.script_generator:
            self.script_generator = ScriptGenerator(
                llm_client=model_load.load(),
                model=self.config.get("llm_model", "deepseek-chat"),
            )

        logger.info(
            "Starting production for episode %d of '%s'",
            episode_number,
            series_config.get("title", "Untitled"),
        )

        # Stage 1: Script generation
        episode = self.script_generator.generate_episode(
            series_config=series_config,
            episode_outline=episode_outline,
            episode_number=episode_number,
        )
        logger.info(
            "Script generated: '%s' (%d scenes)",
            episode.episode_title,
            len(episode.scenes),
        )

        # Stage 2: Shot planning
        planner = ShotPlanner(
            character_registry=self.asset_manager.get_all_characters(),
            asset_manager=self.asset_manager
        )
        episode = planner.plan_episode(episode)
        total_shots = sum(len(s.shots) for s in episode.scenes)
        logger.info("Shot planning complete: %d shots planned", total_shots)

        # Stage 3: Video generation
        episode = self._generate_shots(episode)

        # Stage 4: Assembly
        output_path = self.video_assembler.assemble_episode(episode)
        logger.info("Episode assembled: %s", output_path)

        return episode

    def _generate_shots(self, episode: Episode) -> Episode:
        """Generate video for all shots in the episode via the MCP server.

        Uses a two-phase approach to ensure correct continuity for transition
        shots:

        **Phase 1** — Generate all non-transition shots first (modes
        REFERENCE_TO_VIDEO, FIRST_FRAME, TEXT_TO_VIDEO).

        **Phase 2** — For each transition shot (FIRSTLAST_FRAME), derive the
        start/end frames from the already-generated adjacent videos:

        - ``start_frame_path`` ← tail frame (last frame) of the nearest
          preceding non-transition shot.
        - ``end_frame_path`` ← head frame (first frame) of the nearest
          following non-transition shot.

        Then generate each transition video.

        Args:
            episode: Episode with planned shots.

        Returns:
            Episode with ``generated_video_path`` populated on each shot.
        """
        failed_shots = []
        successful_count = 0

        # Flat, ordered list of all shots across all scenes.
        all_shots = []
        for scene in episode.scenes:
            all_shots.extend(scene.shots)

        # ------------------------------------------------------------------
        # Phase 1: Generate all non-transition shots first.
        # ------------------------------------------------------------------
        for shot in all_shots:
            if shot.is_transition_shot:
                continue
            if self._generate_single_shot(shot):
                successful_count += 1
            else:
                failed_shots.append(
                    (shot.id, shot.generation_error or "unknown error")
                )

        # ------------------------------------------------------------------
        # Phase 2: Populate each transition shot's frames from the adjacent
        # generated videos, then generate the transition shot.
        # ------------------------------------------------------------------
        ffmpeg = FFmpegHelper()
        for i, shot in enumerate(all_shots):
            if not shot.is_transition_shot:
                continue

            if not self._resolve_transition_frames(all_shots, i, ffmpeg):
                failed_shots.append((shot.id, shot.generation_error))
                continue

            if self._generate_single_shot(shot):
                successful_count += 1
            else:
                failed_shots.append(
                    (shot.id, shot.generation_error or "unknown error")
                )

        if successful_count == 0 and failed_shots:
            first_shot_id, first_error = failed_shots[0]
            raise RuntimeError(
                "Shot generation failed for all shots. "
                f"first_failed_shot={first_shot_id}; error={first_error}"
            )

        episode.runtime_estimate = self._compute_runtime(episode)
        return episode

    def resume_episode_generation(self, episode: Episode) -> Episode:
        """Resume generation from a persisted episode snapshot.

        Successful shots whose output files still exist are reused. Only shots
        with missing or failed outputs are regenerated, allowing iterative runs
        from a saved ``episode_*_video_generated.json`` snapshot.
        """
        failed_shots = []
        all_shots = []
        for scene in episode.scenes:
            all_shots.extend(scene.shots)

        for shot in all_shots:
            if shot.is_transition_shot:
                continue
            if self._shot_has_usable_video(shot):
                shot.generation_error = None
                continue
            if self._generate_single_shot(shot):
                continue
            failed_shots.append((shot.id, shot.generation_error or "unknown error"))

        ffmpeg = FFmpegHelper()
        for i, shot in enumerate(all_shots):
            if not shot.is_transition_shot:
                continue
            if self._shot_has_usable_video(shot):
                shot.generation_error = None
                continue
            if not self._resolve_transition_frames(all_shots, i, ffmpeg):
                failed_shots.append((shot.id, shot.generation_error or "unknown error"))
                continue
            if self._generate_single_shot(shot):
                continue
            failed_shots.append((shot.id, shot.generation_error or "unknown error"))

        if failed_shots:
            logger.warning(
                "Resume completed with %d failed shots", len(failed_shots)
            )

        episode.runtime_estimate = self._compute_runtime(episode)
        return episode

    def _shot_has_usable_video(self, shot) -> bool:
        """Return True when a shot already has a video file on disk."""
        if not shot.generated_video_path:
            return False
        return Path(shot.generated_video_path).exists()

    def _resolve_transition_frames(self, all_shots, index: int, ffmpeg: FFmpegHelper) -> bool:
        """Resolve start/end frames for a transition shot from neighbouring videos."""
        shot = all_shots[index]
        transition_trace = shot.generation_trace.setdefault(
            "transition_frame_resolution", {}
        )
        extraction_errors = []

        prev_shot = next(
            (
                all_shots[j]
                for j in range(index - 1, -1, -1)
                if not all_shots[j].is_transition_shot
                and self._shot_has_usable_video(all_shots[j])
            ),
            None,
        )
        transition_trace["prev_shot_id"] = prev_shot.id if prev_shot else None
        transition_trace["prev_generated_video_path"] = (
            prev_shot.generated_video_path if prev_shot else None
        )

        next_shot = next(
            (
                all_shots[j]
                for j in range(index + 1, len(all_shots))
                if not all_shots[j].is_transition_shot
                and self._shot_has_usable_video(all_shots[j])
            ),
            None,
        )
        transition_trace["next_shot_id"] = next_shot.id if next_shot else None
        transition_trace["next_generated_video_path"] = (
            next_shot.generated_video_path if next_shot else None
        )

        if prev_shot:
            tail_path = prev_shot.generated_video_path.rsplit(".", 1)[0] + "_tail.png"
            try:
                ffmpeg.extract_frame(prev_shot.generated_video_path, tail_path)
                shot.start_frame_path = tail_path
                transition_trace["resolved_start_frame_path"] = tail_path
            except Exception as exc:
                extraction_errors.append(
                    {
                        "stage": "extract_tail_frame",
                        "shot_id": prev_shot.id,
                        "video_path": prev_shot.generated_video_path,
                        "error": str(exc),
                    }
                )
                logger.warning(
                    "Cannot extract tail frame from shot %s: %s",
                    prev_shot.id,
                    exc,
                )

        if next_shot:
            head_path = next_shot.generated_video_path.rsplit(".", 1)[0] + "_head.png"
            try:
                ffmpeg.extract_frame(
                    next_shot.generated_video_path, head_path, timestamp=0.0
                )
                shot.end_frame_path = head_path
                transition_trace["resolved_end_frame_path"] = head_path
            except Exception as exc:
                extraction_errors.append(
                    {
                        "stage": "extract_head_frame",
                        "shot_id": next_shot.id,
                        "video_path": next_shot.generated_video_path,
                        "error": str(exc),
                    }
                )
                logger.warning(
                    "Cannot extract head frame from shot %s: %s",
                    next_shot.id,
                    exc,
                )

        if extraction_errors:
            transition_trace["extraction_errors"] = extraction_errors

        transition_trace["final_start_frame_path"] = shot.start_frame_path
        transition_trace["final_end_frame_path"] = shot.end_frame_path

        if shot.start_frame_path and shot.end_frame_path:
            return True

        missing = []
        if not shot.start_frame_path:
            missing.append("start_frame_path")
        if not shot.end_frame_path:
            missing.append("end_frame_path")

        shot.generation_error = (
            "Transition frame extraction failed before MCP call: missing "
            + ", ".join(missing)
        )
        shot.generation_trace["mcp_response"] = {
            "status": "skipped",
            "error": shot.generation_error,
        }
        return False

    def _generate_single_shot(self, shot) -> bool:
        """Generate video for a single shot via the MCP server.

        Dispatches to the appropriate MCP tool based on ``shot.generation_mode``
        and stores the result on the shot in-place.

        Args:
            shot: The :class:`~src.models.shot.Shot` to generate.

        Returns:
            ``True`` if generation succeeded and a video path was obtained,
            ``False`` otherwise.
        """
        from src.models.shot import GenerationMode

        prompt = self._build_generation_prompt(shot)
        shot.effective_prompt = prompt

        logger.info(
            "Generating shot %s (mode=%s, duration=%ds)",
            shot.id,
            shot.generation_mode,
            shot.duration,
        )
        try:
            if shot.generation_mode == GenerationMode.FIRSTLAST_FRAME:
                tool_name = "generate_firstlast_frame_video"
                tool_kwargs = {
                    "start_frame_path": shot.start_frame_path,
                    "end_frame_path": shot.end_frame_path,
                    "prompt": prompt,
                    "duration": shot.duration,
                }
            elif shot.generation_mode == GenerationMode.FIRST_FRAME:
                tool_name = "generate_first_frame_video"
                tool_kwargs = {
                    "first_frame_path": shot.start_frame_path,
                    "prompt": prompt,
                    "duration": shot.duration,
                }
            elif shot.generation_mode == GenerationMode.REFERENCE_TO_VIDEO:
                tool_name = "generate_reference_video"
                tool_kwargs = {
                    "prompt": prompt,
                    "reference_images": shot.reference_images,
                    "duration": shot.duration,
                }
            else:
                tool_name = "generate_text_to_video"
                tool_kwargs = {
                    "prompt": prompt,
                    "duration": shot.duration,
                }

            shot.generation_trace = {
                "source_fields": self._collect_generation_source_fields(shot),
                "effective_prompt": prompt,
                "mcp_request": {
                    "tool_name": tool_name,
                    **tool_kwargs,
                },
            }

            result = self._mcp_server.call_tool(tool_name, **tool_kwargs)
            shot.generated_video_path = result.get("path")
            shot.generation_error = None
            shot.generation_trace["mcp_response"] = self._sanitize_generation_result(result)
            return bool(shot.generated_video_path)
        except Exception as exc:
            shot.generation_error = str(exc)
            shot.generation_trace.setdefault(
                "mcp_response",
                {
                    "status": "failed",
                    "error": str(exc),
                },
            )
            logger.warning("Failed to generate shot %s: %s", shot.id, exc)
            return False

    def _collect_generation_source_fields(self, shot) -> Dict[str, Any]:
        """Capture the structured shot data that feeds prompt construction."""
        return {
            "generation_mode": getattr(shot.generation_mode, "value", shot.generation_mode),
            "text_prompt": shot.text_prompt,
            "action_description": shot.action_description,
            "dialogue": shot.dialogue,
            "internal_thought": shot.internal_thought,
            "characters_in_shot": list(shot.characters_in_shot or []),
            "character_emotions": self._normalize_character_emotions(shot),
            "camera_motion": self._serialize_camera_motion(shot),
            "lighting_description": shot.lighting_description,
            "duration": shot.duration,
            "start_frame_path": shot.start_frame_path,
            "end_frame_path": shot.end_frame_path,
            "reference_images": list(shot.reference_images or []),
            "is_transition_shot": shot.is_transition_shot,
            "transition_type": shot.transition_type,
        }

    def _normalize_character_emotions(self, shot) -> Dict[str, Any]:
        """Convert shot character emotions into JSON-safe values."""
        emotions = getattr(shot, "character_emotions", None) or {}
        return {
            char_id: getattr(emotion, "value", emotion)
            for char_id, emotion in emotions.items()
        }

    def _serialize_camera_motion(self, shot) -> Dict[str, Any]:
        """Convert camera motion into a JSON-safe structure."""
        camera_motion = getattr(shot, "camera_motion", None)
        if not camera_motion:
            return {}

        return {
            "type": camera_motion.type,
            "speed": camera_motion.speed,
            "start_position": camera_motion.start_position,
            "end_position": camera_motion.end_position,
        }

    def _sanitize_generation_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Keep generation metadata JSON-safe and focused on debugging value."""
        sanitized = dict(result or {})
        return sanitized

    def _build_generation_prompt(self, shot) -> str:
        """Compose the prompt actually sent to the video model.

        The script stores structured shot metadata across several fields, but the
        downstream video API accepts a single prompt string. This method merges
        the important fields into one prompt so the model receives the action,
        motion, lighting, and emotional context instead of only ``text_prompt``.
        """
        parts = []

        if shot.text_prompt and shot.text_prompt.strip():
            parts.append(shot.text_prompt.strip())

        if shot.action_description and shot.action_description.strip():
            parts.append(f"Action and staging: {shot.action_description.strip()}")

        camera_motion = self._describe_camera_motion(shot)
        if camera_motion:
            parts.append(f"Camera motion: {camera_motion}")

        if shot.lighting_description and shot.lighting_description.strip():
            parts.append(f"Lighting: {shot.lighting_description.strip()}")

        if shot.dialogue and shot.dialogue.strip():
            parts.append(f"Spoken dialogue: {shot.dialogue.strip()}")

        if shot.internal_thought and shot.internal_thought.strip():
            parts.append(f"Internal thought or subtext: {shot.internal_thought.strip()}")

        emotion_summary = self._describe_character_emotions(shot)
        if emotion_summary:
            parts.append(f"Character emotions: {emotion_summary}")

        return "\n".join(parts).strip()

    def _describe_camera_motion(self, shot) -> str:
        """Convert a shot camera_motion object into prompt text."""
        camera_motion = getattr(shot, "camera_motion", None)
        if not camera_motion or not getattr(camera_motion, "type", None):
            return ""

        details = [str(camera_motion.type)]
        if getattr(camera_motion, "speed", None) not in (None, "", 1.0):
            details.append(f"speed={camera_motion.speed}")
        if getattr(camera_motion, "start_position", None):
            details.append(f"start={camera_motion.start_position}")
        if getattr(camera_motion, "end_position", None):
            details.append(f"end={camera_motion.end_position}")
        return ", ".join(details)

    def _describe_character_emotions(self, shot) -> str:
        """Convert structured character emotions into prompt text."""
        emotions = getattr(shot, "character_emotions", None) or {}
        if not emotions:
            return ""

        return ", ".join(
            f"{char_id}={getattr(emotion, 'value', emotion)}"
            for char_id, emotion in emotions.items()
        )

    def _compute_runtime(self, episode: Episode) -> int:
        """Sum all shot durations to estimate total episode runtime.

        Args:
            episode: Episode whose shots are summed.

        Returns:
            Total duration in seconds.
        """
        return sum(
            shot.duration
            for scene in episode.scenes
            for shot in scene.shots
        )
