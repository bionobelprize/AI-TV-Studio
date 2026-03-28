"""Script Generator for AI-TV-Studio.

Uses an LLM to generate structured episode scripts from series configuration
and episode outlines, producing structured Episode objects ready for
shot planning and video generation.

Script generation is performed in two phases to improve JSON stability:
  Phase 1 – Blueprint: generate the episode title, logline, and per-scene
    metadata (location, mood, a brief scene summary, etc.) in a single,
    compact LLM call.
  Phase 2 – Shots: for each scene in the blueprint, call the LLM once to
    generate that scene's shots.  Every shot-generation prompt includes the
    full episode blueprint so that the model can maintain narrative coherence
    and artistic continuity across scenes.
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


# ---------------------------------------------------------------------------
# Phase-1 models: episode blueprint (no shots)
# ---------------------------------------------------------------------------

class SceneBlueprintOutput(BaseModel):
    """Compact scene description used in the episode blueprint (Phase 1)."""

    scene_number: int = Field(default=1)
    location: str = Field(default="Unknown")
    time_of_day: str = Field(default="day")
    weather: str = Field(default="clear")
    mood: str = Field(default="neutral")
    bgm_mood: str = Field(default="neutral")
    bgm_tempo: str = Field(default="moderate")
    ambient_sounds: List[str] = Field(default_factory=list)
    scene_summary: str = Field(
        default="",
        description="One-to-two sentence description of what happens in this scene.",
    )


class EpisodeBlueprintOutput(BaseModel):
    """Top-level episode blueprint returned by Phase 1."""

    episode_title: str = Field(default="")
    logline: str = Field(default="")
    scenes: List[SceneBlueprintOutput] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Phase-2 model: shots for a single scene
# ---------------------------------------------------------------------------

class SceneShotsOutput(BaseModel):
    """Shots list returned for one scene by Phase 2."""

    shots: List[ShotOutput] = Field(default_factory=list)


class ScriptGenerator:
    """Generates structured TV episode scripts using an LLM.

    Generation is split into two phases to keep each LLM response small and
    structurally stable:

    **Phase 1 – Blueprint**
        One compact LLM call produces the episode title, logline, and a list
        of scene blueprints (location, mood, brief scene summary, etc.) but
        *no* individual shots.  The small output size makes JSON parsing
        reliable.

    **Phase 2 – Shots (one call per scene)**
        For every scene in the blueprint the generator makes a separate LLM
        call that yields only that scene's shots.  Every shot-generation
        prompt embeds the *complete* episode blueprint so the model maintains
        narrative coherence and artistic continuity across all scenes.
    """

    SYSTEM_PROMPT = (
        "You are an expert television screenwriter and cinematographer. "
        "Generate structured episode scripts as valid JSON following the "
        "provided schema exactly. Focus on cinematic storytelling with "
        "clear shot descriptions, character emotions, and camera directions. "
        "Every shot must be visual and production-ready for text-to-video generation."
    )

    # ------------------------------------------------------------------
    # Phase 1: blueprint prompt
    # ------------------------------------------------------------------

    BLUEPRINT_PROMPT_TEMPLATE = """You are writing the structural blueprint for Episode {episode_number} of a TV series.

Series Title: {series_title}
Genre: {genre}
Characters:
{characters_desc}

Episode Outline:
{episode_outline}

STRICT OUTPUT FORMAT – Phase 1 Blueprint:
Plan the high-level structure of this episode. For each scene provide:
- scene_number, location, time_of_day, weather, mood, bgm_mood, bgm_tempo, ambient_sounds
- scene_summary: a 1-2 sentence description of the dramatic action in this scene

Do NOT generate individual shots here – only the scene-level structure.

{format_instructions}"""

    # ------------------------------------------------------------------
    # Phase 2: per-scene shots prompt
    # ------------------------------------------------------------------

    SCENE_SHOTS_PROMPT_TEMPLATE = """You are writing the shot-by-shot breakdown for ONE scene of Episode {episode_number}.

Series Title: {series_title}
Genre: {genre}
Characters:
{characters_desc}

=== EPISODE BLUEPRINT (for narrative coherence) ===
Episode Title: {episode_title}
Logline: {logline}

All scenes in this episode:
{all_scenes_summary}
===================================================

Now write detailed shots for Scene {scene_number}:
Location: {location}
Time of day: {time_of_day}
Weather: {weather}
Mood: {mood}
BGM mood: {bgm_mood}
BGM tempo: {bgm_tempo}
Ambient sounds: {ambient_sounds}
Scene summary: {scene_summary}

Generate a sequence of cinematic shots for this scene that:
1. Advance the story described in the scene summary
2. Are visually consistent with the overall episode mood and arc
3. Include strong text_prompt values suitable for text-to-video generation
4. Include character emotions and camera motions

{format_instructions}"""

    # ------------------------------------------------------------------
    # Legacy single-call prompt (kept for _call_llm backward compatibility)
    # ------------------------------------------------------------------

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
            llm_client: Optional custom LLM client. If provided, it is used
                directly for all LLM calls and ``model_load`` is not invoked.
                Useful for testing with stub clients.
            model: Model identifier used when loading the default client.
        """
        self.llm = llm_client if llm_client is not None else model_load.load()
        self.model = model

        # Phase 1 – blueprint chain
        self._blueprint_parser = JsonOutputParser(pydantic_object=EpisodeBlueprintOutput)
        self._blueprint_prompt = PromptTemplate(
            template=self.BLUEPRINT_PROMPT_TEMPLATE,
            input_variables=[
                "series_title",
                "genre",
                "characters_desc",
                "episode_outline",
                "episode_number",
            ],
            partial_variables={
                "format_instructions": self._blueprint_parser.get_format_instructions()
            },
        )
        self._blueprint_chain = self._blueprint_prompt | self.llm | self._blueprint_parser

        # Phase 2 – per-scene shots chain
        self._shots_parser = JsonOutputParser(pydantic_object=SceneShotsOutput)
        self._shots_prompt = PromptTemplate(
            template=self.SCENE_SHOTS_PROMPT_TEMPLATE,
            input_variables=[
                "series_title",
                "genre",
                "characters_desc",
                "episode_number",
                "episode_title",
                "logline",
                "all_scenes_summary",
                "scene_number",
                "location",
                "time_of_day",
                "weather",
                "mood",
                "bgm_mood",
                "bgm_tempo",
                "ambient_sounds",
                "scene_summary",
            ],
            partial_variables={
                "format_instructions": self._shots_parser.get_format_instructions()
            },
        )
        self._shots_chain = self._shots_prompt | self.llm | self._shots_parser

        # Legacy single-call chain kept for _call_llm backward compatibility
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
        """Generate a full episode using a two-phase multi-call approach.

        Phase 1: generate the episode blueprint (title, logline, scene
        metadata) in a single LLM call.
        Phase 2: for each scene in the blueprint, call the LLM once to
        expand the scene into individual shots, passing the full blueprint
        as context to ensure narrative coherence.

        Args:
            series_config: Dictionary containing series title, genre, and
                character definitions.
            episode_outline: One-paragraph description of the episode story.
            episode_number: Episode number within the series.

        Returns:
            A fully structured Episode ready for shot planning.
        """
        blueprint = self._generate_blueprint(
            series_config, episode_outline, episode_number
        )

        all_scenes_summary = self._format_all_scenes_summary(blueprint.get("scenes", []))
        scenes_data: List[Dict[str, Any]] = []
        for scene_bp in blueprint.get("scenes", []):
            shots_data = self._generate_scene_shots(
                series_config=series_config,
                episode_number=episode_number,
                blueprint=blueprint,
                all_scenes_summary=all_scenes_summary,
                scene_blueprint=scene_bp,
            )
            scene_dict = dict(scene_bp)
            scene_dict["shots"] = shots_data.get("shots", [])
            scenes_data.append(scene_dict)

        final_script: Dict[str, Any] = {
            "episode_title": blueprint.get("episode_title", ""),
            "logline": blueprint.get("logline", ""),
            "scenes": scenes_data,
        }
        return self._build_episode(final_script, series_config, episode_number)

    def _generate_blueprint(
        self,
        series_config: Dict[str, Any],
        episode_outline: str,
        episode_number: int,
    ) -> Dict[str, Any]:
        """Phase 1: Generate the episode blueprint (no shots).

        Args:
            series_config: Series metadata and character definitions.
            episode_outline: High-level story outline.
            episode_number: Episode number.

        Returns:
            Parsed dictionary with ``episode_title``, ``logline``, and
            ``scenes`` (each scene has metadata + ``scene_summary``).
        """
        characters_desc = json.dumps(series_config.get("characters", []), indent=2)
        input_dict = {
            "series_title": series_config.get("title", "Untitled"),
            "genre": series_config.get("genre", "drama"),
            "characters_desc": characters_desc,
            "episode_outline": episode_outline,
            "episode_number": episode_number,
        }
        return self._blueprint_chain.invoke(input_dict)

    def _generate_scene_shots(
        self,
        series_config: Dict[str, Any],
        episode_number: int,
        blueprint: Dict[str, Any],
        all_scenes_summary: str,
        scene_blueprint: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Phase 2: Generate shots for one scene, given the full blueprint.

        Args:
            series_config: Series metadata and character definitions.
            episode_number: Episode number.
            blueprint: Full episode blueprint returned by Phase 1.
            all_scenes_summary: Pre-formatted summary of all scenes for the
                prompt (avoids reformatting inside each call).
            scene_blueprint: The specific scene's blueprint dict.

        Returns:
            Parsed dictionary with a ``shots`` key containing the shot list.
        """
        characters_desc = json.dumps(series_config.get("characters", []), indent=2)
        ambient = scene_blueprint.get("ambient_sounds", [])
        input_dict = {
            "series_title": series_config.get("title", "Untitled"),
            "genre": series_config.get("genre", "drama"),
            "characters_desc": characters_desc,
            "episode_number": episode_number,
            "episode_title": blueprint.get("episode_title", ""),
            "logline": blueprint.get("logline", ""),
            "all_scenes_summary": all_scenes_summary,
            "scene_number": scene_blueprint.get("scene_number", 1),
            "location": scene_blueprint.get("location", "Unknown"),
            "time_of_day": scene_blueprint.get("time_of_day", "day"),
            "weather": scene_blueprint.get("weather", "clear"),
            "mood": scene_blueprint.get("mood", "neutral"),
            "bgm_mood": scene_blueprint.get("bgm_mood", "neutral"),
            "bgm_tempo": scene_blueprint.get("bgm_tempo", "moderate"),
            "ambient_sounds": ", ".join(ambient) if ambient else "none",
            "scene_summary": scene_blueprint.get("scene_summary", ""),
        }
        return self._shots_chain.invoke(input_dict)

    @staticmethod
    def _format_all_scenes_summary(scenes: List[Dict[str, Any]]) -> str:
        """Format the scene list from the blueprint into a readable summary.

        Args:
            scenes: List of scene blueprint dicts.

        Returns:
            A multi-line string summarising every scene.
        """
        lines = []
        for s in scenes:
            lines.append(
                f"Scene {s.get('scene_number', '?')} – "
                f"{s.get('location', 'Unknown')} "
                f"({s.get('time_of_day', 'day')}, {s.get('mood', 'neutral')}): "
                f"{s.get('scene_summary', '')}"
            )
        return "\n".join(lines)

    def _call_llm(
        self,
        series_config: Dict[str, Any],
        episode_outline: str,
        episode_number: int,
    ) -> Dict[str, Any]:
        """Legacy single-call LLM path (kept for backward compatibility).

        Prefer ``generate_episode`` which uses the two-phase approach.

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

        return self.chain.invoke(input_dict)

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

