"""Script Generator for AI-TV-Studio.

Uses an LLM to generate structured episode scripts from series configuration
and episode outlines, producing structured Episode objects ready for
shot planning and video generation.
"""

import json
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from src.models.character import Character, CharacterEmotion, CharacterVisualCore
from src.models.episode import Episode
from src.models.scene import Scene
from src.models.shot import CameraMotion, GenerationMode, Shot
import src.model_load as model_load


class ShotOutput(BaseModel):
    sequence_number: int = Field(default=1)
    action_description: str = Field(default="")
    dialogue: Optional[str] = Field(default=None)
    characters_in_shot: List[str] = Field(default_factory=list)
    character_emotions: Dict[str, str] = Field(default_factory=dict)
    duration: int = Field(default=8)
    camera_motion: Dict[str, Any] = Field(default_factory=lambda: {"type": "static"})
    lighting_description: str = Field(default="cinematic lighting")
    text_prompt: str = Field(default="")


class SceneOutput(BaseModel):
    scene_number: int = Field(default=1)
    location: str = Field(default="Unknown")
    time_of_day: str = Field(default="day")
    weather: str = Field(default="clear")
    mood: str = Field(default="neutral")
    bgm_mood: str = Field(default="neutral")
    bgm_tempo: str = Field(default="moderate")
    ambient_sounds: List[str] = Field(default_factory=list)
    shots: List[ShotOutput] = Field(default_factory=list)


class EpisodeScriptOutput(BaseModel):
    episode_title: str = Field(default="")
    logline: str = Field(default="")
    scenes: List[SceneOutput] = Field(default_factory=list)


class ScriptGenerator:
    """Generates structured TV episode scripts using an LLM.

    The generator prompts the LLM to produce a JSON-structured script that
    maps directly onto the Episode/Scene/Shot data model, then parses the
    response into typed Python objects.
    """

    SYSTEM_PROMPT = (
        "You are an expert television screenwriter and cinematographer. "
        "Generate structured episode scripts as valid JSON following the "
        "provided schema exactly. Focus on cinematic storytelling with "
        "clear shot descriptions, character emotions, and camera directions. "
        "Every shot must be visual and production-ready for text-to-video generation."
    )

    SCRIPT_PROMPT_TEMPLATE = """You are writing Episode {episode_number} of a TV series.

Series Title: {series_title}
Genre: {genre}
Characters:
{characters_desc}

Episode Outline:
{episode_outline}

Please produce a complete cinematic episode plan with:
1. A compelling episode title and concise logline
2. Multiple coherent scenes with clear progression
3. Shot-by-shot visual storytelling in each scene
4. Character emotions aligned with each shot
5. Practical camera motion and strong text prompts for generation

Output must be in JSON format as specified below.
{format_instructions}"""

    def __init__(self, llm_client=None, model: str = "deepseek-chat"):
        """Initialize the script generator.

        Args:
            llm_client: Optional custom LLM client. If omitted, loads default
                reference model from ``model_load``.
            model: Model identifier to use for generation.
        """
        if llm_client is not None:
            self.llm = llm_client
        else:
            self.llm = model_load.load()
        self.model = model
        self.parser = JsonOutputParser(pydantic_object=EpisodeScriptOutput)
        self.prompt = PromptTemplate(
            template=self.SCRIPT_PROMPT_TEMPLATE,
            input_variables=[
                "series_title",
                "genre",
                "characters_desc",
                "episode_outline",
                "episode_number",
            ],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )
        self.chain = self.prompt | self.llm | self.parser

    def generate_episode(
        self,
        series_config: Dict[str, Any],
        episode_outline: str,
        episode_number: int = 1,
    ) -> Episode:
        """Generate a full episode from a series configuration and outline.

        Args:
            series_config: Dictionary containing series title, genre, and
                character definitions.
            episode_outline: One-paragraph description of the episode story.
            episode_number: Episode number within the series.

        Returns:
            A fully structured Episode ready for shot planning.
        """
        script_data = self._call_llm(series_config, episode_outline, episode_number)
        return self._build_episode(script_data, series_config, episode_number)

    def _call_llm(
        self,
        series_config: Dict[str, Any],
        episode_outline: str,
        episode_number: int,
    ) -> Dict[str, Any]:
        """Send prompt through the LangChain structured-output pipeline.

        Args:
            series_config: Series metadata and character definitions.
            episode_outline: High-level story outline.
            episode_number: Episode number.

        Returns:
            Parsed JSON dictionary representing the generated episode script.
        """
        characters_desc = json.dumps(series_config.get("characters", []), indent=2)
        input_dict = {
            "series_title": series_config.get("title", "Untitled"),
            "genre": series_config.get("genre", "drama"),
            "characters_desc": characters_desc,
            "episode_outline": episode_outline,
            "episode_number": episode_number,
        }
        # Prepend system instruction to keep output quality consistent.
        input_dict["episode_outline"] = (
            f"{self.SYSTEM_PROMPT}\n\n{input_dict['episode_outline']}"
        )

        try:
            return self.chain.invoke(input_dict)
        except Exception:
            fallback_prompt = self.prompt.format(**input_dict)
            response = self.llm.invoke(fallback_prompt)
            raw_text = self._extract_text_from_response(response)
            return self._parse_response(raw_text)

    def _extract_text_from_response(self, response: Any) -> str:
        """Normalize different LLM response object types to plain text."""
        if isinstance(response, str):
            return response

        content = getattr(response, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        parts.append(str(text))
                elif hasattr(item, "text") and getattr(item, "text"):
                    parts.append(str(getattr(item, "text")))
            if parts:
                return "\n".join(parts)

        return str(response)

    def _parse_response(self, raw_response: str) -> Dict[str, Any]:
        """Parse the LLM response into a structured dictionary.

        Attempts to extract a JSON object from the response, stripping any
        surrounding markdown code fences if present.

        Args:
            raw_response: Raw text response from the LLM.

        Returns:
            Parsed JSON data as a Python dictionary.

        Raises:
            ValueError: If valid JSON cannot be extracted.
        """
        text = raw_response.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse LLM response as JSON: {exc}") from exc

    def _build_episode(
        self,
        script_data: Dict[str, Any],
        series_config: Dict[str, Any],
        episode_number: int,
    ) -> Episode:
        """Convert parsed script data into an Episode object.

        Args:
            script_data: Parsed JSON dictionary from the LLM.
            series_config: Series metadata for episode context.
            episode_number: Episode number.

        Returns:
            A populated Episode object.
        """
        character_ids = [
            c["id"] for c in series_config.get("characters", [])
        ]

        scenes = [
            self._build_scene(scene_data, f"ep{episode_number}")
            for scene_data in script_data.get("scenes", [])
        ]

        episode = Episode(
            id=f"ep_{uuid.uuid4().hex[:8]}",
            series_title=series_config.get("title", "Untitled"),
            episode_number=episode_number,
            episode_title=script_data.get("episode_title", f"Episode {episode_number}"),
            logline=script_data.get("logline", ""),
            characters=character_ids,
            scenes=scenes,
            genre=series_config.get("genre", "drama"),
        )
        episode.runtime_estimate = self._compute_runtime(episode)
        return episode

    def _compute_runtime(self, episode: Episode) -> int:
        """Sum shot durations to estimate the episode runtime in seconds.

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

    def _build_scene(
        self, scene_data: Dict[str, Any], episode_id: str
    ) -> Scene:
        """Build a Scene object from raw scene data.

        Args:
            scene_data: Dictionary of scene attributes from the parsed script.
            episode_id: Parent episode identifier.

        Returns:
            A populated Scene object.
        """
        scene = Scene(
            id=f"scene_{uuid.uuid4().hex[:8]}",
            episode_id=episode_id,
            scene_number=scene_data.get("scene_number", 1),
            location=scene_data.get("location", "Unknown"),
            time_of_day=scene_data.get("time_of_day", "day"),
            weather=scene_data.get("weather", "clear"),
            mood=scene_data.get("mood", "neutral"),
            bgm_mood=scene_data.get("bgm_mood", "neutral"),
            bgm_tempo=scene_data.get("bgm_tempo", "moderate"),
            ambient_sounds=scene_data.get("ambient_sounds", []),
        )
        scene.shots = [
            self._build_shot(shot_data, scene.id)
            for shot_data in scene_data.get("shots", [])
        ]
        return scene

    def _build_shot(
        self, shot_data: Dict[str, Any], scene_id: str
    ) -> Shot:
        """Build a Shot object from raw shot data.

        Args:
            shot_data: Dictionary of shot attributes from the parsed script.
            scene_id: Parent scene identifier.

        Returns:
            A populated Shot object.
        """
        camera_data = shot_data.get("camera_motion")
        camera_motion: Optional[CameraMotion] = None
        if camera_data:
            camera_motion = CameraMotion(
                type=camera_data.get("type", "static"),
                speed=camera_data.get("speed", 1.0),
                start_position=camera_data.get("start_position"),
                end_position=camera_data.get("end_position"),
            )

        raw_emotions = shot_data.get("character_emotions", {})
        valid_emotion_values = {e.value for e in CharacterEmotion}
        character_emotions = {
            char_id: CharacterEmotion(emotion)
            for char_id, emotion in raw_emotions.items()
            if emotion in valid_emotion_values
        }

        return Shot(
            id=f"shot_{uuid.uuid4().hex[:8]}",
            scene_id=scene_id,
            sequence_number=shot_data.get("sequence_number", 0),
            action_description=shot_data.get("action_description", ""),
            dialogue=shot_data.get("dialogue"),
            internal_thought=shot_data.get("internal_thought"),
            characters_in_shot=shot_data.get("characters_in_shot", []),
            character_emotions=character_emotions,
            generation_mode=GenerationMode.REFERENCE_TO_VIDEO,
            duration=shot_data.get("duration", 8),
            camera_motion=camera_motion,
            lighting_description=shot_data.get(
                "lighting_description", "cinematic lighting"
            ),
            text_prompt=shot_data.get("text_prompt", ""),
        )
