"""Script Generator for AI-TV-Studio.

Uses an LLM to generate structured episode scripts from series configuration
and episode outlines, producing structured Episode objects ready for
shot planning and video generation.
"""

import json
import uuid
from typing import Any, Dict, List, Optional

from src.models.character import Character, CharacterEmotion, CharacterVisualCore
from src.models.episode import Episode
from src.models.scene import Scene
from src.models.shot import CameraMotion, GenerationMode, Shot
import src.model_load as model_load


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
        "clear shot descriptions, character emotions, and camera directions."
    )

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
        prompt = self._build_prompt(series_config, episode_outline, episode_number)
        raw_response = self._call_llm(prompt)
        script_data = self._parse_response(raw_response)
        return self._build_episode(script_data, series_config, episode_number)

    def _build_prompt(
        self,
        series_config: Dict[str, Any],
        episode_outline: str,
        episode_number: int,
    ) -> str:
        """Construct the prompt sent to the LLM.

        Args:
            series_config: Series metadata and character list.
            episode_outline: High-level story outline for the episode.
            episode_number: Episode number.

        Returns:
            The full prompt string.
        """
        characters_desc = json.dumps(
            series_config.get("characters", []), indent=2
        )
        return (
            f"Series: {series_config.get('title', 'Untitled')}\n"
            f"Genre: {series_config.get('genre', 'drama')}\n"
            f"Characters:\n{characters_desc}\n\n"
            f"Episode {episode_number} Outline:\n{episode_outline}\n\n"
            "Generate a structured JSON episode script with the following format:\n"
            "{\n"
            '  "episode_title": "...",\n'
            '  "logline": "...",\n'
            '  "scenes": [\n'
            "    {\n"
            '      "scene_number": 1,\n'
            '      "location": "...",\n'
            '      "time_of_day": "day",\n'
            '      "weather": "clear",\n'
            '      "mood": "tense",\n'
            '      "bgm_mood": "suspenseful",\n'
            '      "bgm_tempo": "moderate",\n'
            '      "ambient_sounds": ["city noise"],\n'
            '      "shots": [\n'
            "        {\n"
            '          "sequence_number": 1,\n'
            '          "action_description": "...",\n'
            '          "dialogue": "...",\n'
            '          "characters_in_shot": ["char_id"],\n'
            '          "character_emotions": {"char_id": "neutral"},\n'
            '          "duration": 8,\n'
            '          "camera_motion": {"type": "static"},\n'
            '          "lighting_description": "cinematic lighting",\n'
            '          "text_prompt": "..."\n'
            "        }\n"
            "      ]\n"
            "    }\n"
            "  ]\n"
            "}"
        )

    def _call_llm(self, prompt: str) -> str:
        """Send prompt to the LLM and return the raw text response.

        Args:
            prompt: The user prompt to send.

        Returns:
            Raw response string from the LLM.
        """
        full_prompt = (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"{prompt}\n\n"
            "Return only valid JSON without markdown fences or extra text."
        )
        # Keep invocation style aligned with reference_code tooling.
        response = self.llm.invoke(full_prompt)
        return self._extract_text_from_response(response)

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
