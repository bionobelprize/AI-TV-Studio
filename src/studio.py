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
            "llm_provider": "openai",
            "llm_model": "gpt-4",
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

        Args:
            config: Configuration dictionary. Supported keys:
                - ``llm_provider``: LLM provider name (default: ``"openai"``).
                - ``llm_model``: Model identifier (default: ``"gpt-4"``).
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
        """Attach an LLM client for script generation.

        Args:
            llm_client: LLM client instance with a ``chat`` method.
            model: Optional model identifier override.
        """
        model = model or self.config.get("llm_model", "gpt-4")
        self.script_generator = ScriptGenerator(llm_client=llm_client, model=model)

    def configure_mcp(self, mcp_server) -> None:
        """Attach an MCP video director server for generation calls.

        Args:
            mcp_server: Configured VideoDirectorServer instance.
        """
        self._mcp_server = mcp_server
        self.shot_planner = ShotPlanner(
            character_registry=self.asset_manager.get_all_characters()
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
            RuntimeError: If the LLM or MCP clients have not been configured.
        """
        if not self.script_generator:
            raise RuntimeError(
                "LLM client not configured. Call configure_llm() first."
            )
        if not self._mcp_server:
            raise RuntimeError(
                "MCP server not configured. Call configure_mcp() first."
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
            character_registry=self.asset_manager.get_all_characters()
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

        Args:
            episode: Episode with planned shots.

        Returns:
            Episode with ``generated_video_path`` populated on each shot.
        """
        from src.models.shot import GenerationMode

        for scene in episode.scenes:
            for shot in scene.shots:
                logger.info(
                    "Generating shot %s (mode=%s, duration=%ds)",
                    shot.id,
                    shot.generation_mode,
                    shot.duration,
                )
                try:
                    if shot.generation_mode == GenerationMode.FIRSTLAST_FRAME:
                        result = self._mcp_server.call_tool(
                            "generate_firstlast_frame_video",
                            start_frame_path=shot.start_frame_path,
                            end_frame_path=shot.end_frame_path,
                            prompt=shot.text_prompt,
                            duration=shot.duration,
                        )
                    elif shot.generation_mode == GenerationMode.REFERENCE_TO_VIDEO:
                        result = self._mcp_server.call_tool(
                            "generate_reference_video",
                            prompt=shot.text_prompt,
                            reference_images=shot.reference_images,
                            duration=shot.duration,
                        )
                    else:
                        result = self._mcp_server.call_tool(
                            "generate_text_to_video",
                            prompt=shot.text_prompt,
                            duration=shot.duration,
                        )
                    shot.generated_video_path = result.get("path")
                except Exception as exc:
                    logger.warning(
                        "Failed to generate shot %s: %s", shot.id, exc
                    )

        episode.runtime_estimate = self._compute_runtime(episode)
        return episode

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
