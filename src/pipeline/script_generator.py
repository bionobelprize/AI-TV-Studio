"""Script Generator for AI-TV-Studio.

Uses an LLM to generate structured episode scripts from series configuration
and episode outlines, producing structured Episode objects ready for
shot planning and video generation.

Script generation is performed in three phases to improve JSON stability and
narrative coherence:
  Phase 1 – Blueprint: generate the episode title, logline, and per-scene
    metadata (location, mood, a brief scene summary, etc.) in a single,
    compact LLM call.
  Phase 2 – Shot Plan: for each scene in the blueprint, call the LLM once to
    generate a minimal shot plan (sequence numbers, brief descriptions, and
    character lists) for that scene.  The full episode blueprint is embedded
    for narrative coherence.
  Phase 3 – Individual Shots: for each shot in the plan, call the LLM once
    to expand it into full shot details.  Every shot-generation prompt
    embeds the complete episode blueprint, the scene context, the registered
    character registry (with exact IDs), and a summary of all shots already
    generated in the scene so that the model maintains continuity.
"""

import json
import logging
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


logger = logging.getLogger(__name__)


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
# Phase-2 model: shots for a single scene (legacy – kept for backward compat)
# ---------------------------------------------------------------------------

class SceneShotsOutput(BaseModel):
    """Shots list returned for one scene by Phase 2."""

    shots: List[ShotOutput] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Phase-2 (new) model: minimal shot plan for a single scene
# ---------------------------------------------------------------------------

class ShotPlanItem(BaseModel):
    """Minimal description of one shot in a scene's plan (Phase 2)."""

    sequence_number: int = Field(default=1)
    brief_description: str = Field(
        default="",
        description="One-sentence visual action summary (≤ 80 characters).",
    )
    characters_in_shot: List[str] = Field(
        default_factory=list,
        description="Exact character IDs from the registered character list.",
    )


class SceneShotPlanOutput(BaseModel):
    """Shot plan returned for one scene by Phase 2 (new)."""

    shots: List[ShotPlanItem] = Field(default_factory=list)


class ScriptGenerator:
    """Generates structured TV episode scripts using an LLM.

    Generation is split into three phases to keep each LLM response small and
    structurally stable:

    **Phase 1 – Blueprint**
        One compact LLM call produces the episode title, logline, and a list
        of scene blueprints (location, mood, brief scene summary, etc.) but
        *no* individual shots.  The small output size makes JSON parsing
        reliable.

    **Phase 2 – Shot Plan (one call per scene)**
        For every scene in the blueprint the generator makes a separate LLM
        call that yields a minimal shot plan for that scene: each planned shot
        has a sequence number, a brief description, and the list of character
        IDs that appear in it.  The full episode blueprint is embedded so the
        model can plan the right number and sequence of shots.

    **Phase 3 – Individual Shots (one call per shot)**
        For each planned shot the generator makes a separate LLM call that
        expands the plan item into full shot details (action description,
        dialogue, camera motion, text prompt, etc.).  Every single-shot prompt
        embeds the complete episode blueprint, the scene context, the
        registered character registry (with exact IDs so the model uses them
        verbatim), and a running summary of shots already generated in the
        scene to ensure visual and narrative continuity.
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
5. Contain 6-10 shots to avoid overlong outputs

JSON SAFETY RULES:
- Return ONE JSON object only (no markdown, no commentary)
- Every string value must use double quotes
- Keep each action_description concise (<= 120 Chinese characters)
- Keep each text_prompt concise (<= 220 English characters)

{format_instructions}"""

    SHOTS_JSON_REPAIR_PROMPT_TEMPLATE = """The following model output is intended to be JSON but may be malformed.

Repair it into valid JSON that strictly follows this schema:
{format_instructions}

Rules:
- Return ONE JSON object only
- Do not wrap in markdown code fences
- Do not add explanations
- Preserve the original semantic intent as much as possible

Broken output:
{broken_output}
"""

    # ------------------------------------------------------------------
    # Phase 2 (new): per-scene shot plan prompt
    # ------------------------------------------------------------------

    SCENE_SHOT_PLAN_PROMPT_TEMPLATE = """You are planning the shot sequence for ONE scene of Episode {episode_number}.

Series Title: {series_title}
Genre: {genre}

=== REGISTERED CHARACTERS (use EXACTLY these IDs in characters_in_shot) ===
{characters_registry}
===========================================================================

=== EPISODE BLUEPRINT (for narrative coherence) ===
Episode Title: {episode_title}
Logline: {logline}

All scenes in this episode:
{all_scenes_summary}
===================================================

Plan shots for Scene {scene_number}:
Location: {location}
Time of day: {time_of_day}
Mood: {mood}
Scene summary: {scene_summary}

Create a shot plan with 6-10 shots that advance the scene's story.
For each shot specify:
- sequence_number: integer starting from 1
- brief_description: one sentence describing the visual action (≤ 80 characters)
- characters_in_shot: list of character IDs that appear in this shot.
  IMPORTANT: use ONLY IDs exactly as listed in the REGISTERED CHARACTERS section above.
  Use an empty list [] if no character from the registry appears in this shot.

JSON SAFETY RULES:
- Return ONE JSON object only (no markdown, no commentary)
- Every string value must use double quotes

{format_instructions}"""

    # ------------------------------------------------------------------
    # Phase 3: per-shot full detail prompt
    # ------------------------------------------------------------------

    SINGLE_SHOT_PROMPT_TEMPLATE = """You are writing the full cinematic details for a SINGLE shot in Episode {episode_number}.

Series Title: {series_title}
Genre: {genre}

=== REGISTERED CHARACTERS (use EXACTLY these IDs in characters_in_shot and character_emotions) ===
{characters_registry}
================================================================================================

=== EPISODE BLUEPRINT (for narrative coherence) ===
Episode Title: {episode_title}
Logline: {logline}

All scenes in this episode:
{all_scenes_summary}
===================================================

Scene {scene_number} context:
Location: {location}
Time of day: {time_of_day}
Weather: {weather}
Mood: {mood}
BGM mood: {bgm_mood}
BGM tempo: {bgm_tempo}
Ambient sounds: {ambient_sounds}
Scene summary: {scene_summary}

=== SHOTS ALREADY GENERATED IN THIS SCENE (for continuity) ===
{previous_shots_summary}
==============================================================

Now generate the FULL details for Shot {shot_sequence_number}:
Brief description: {shot_brief_description}
Characters in this shot: {shot_characters}

Fill in all fields:
- sequence_number: must be {shot_sequence_number}
- action_description: vivid cinematic description (≤ 120 Chinese characters)
- dialogue: character line if any, or null
- characters_in_shot: MUST use ONLY the exact IDs from the REGISTERED CHARACTERS section above
- character_emotions: dict mapping each character_id to an emotion string
- duration: 5–12 seconds (integer)
- camera_motion: object with "type" and optional "speed", "start_position", "end_position"
- lighting_description: specific lighting for this shot
- text_prompt: English description for text-to-video generation (≤ 220 characters)

JSON SAFETY RULES:
- Return ONE JSON object only (no markdown, no commentary)
- Every string value must use double quotes
- Keep text_prompt in English

{format_instructions}"""

    SINGLE_SHOT_JSON_REPAIR_PROMPT_TEMPLATE = """The following model output is intended to be JSON for a single shot but may be malformed.

Repair it into valid JSON that strictly follows this schema:
{format_instructions}

Rules:
- Return ONE JSON object only
- Do not wrap in markdown code fences
- Do not add explanations
- Preserve the original semantic intent as much as possible

Broken output:
{broken_output}
"""

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

    def __init__(self, llm_client=None, model: str = "deepseek-chat", asset_manager=None):
        """Initialize the script generator.

        Args:
            llm_client: Optional custom LLM client. If provided, it is used
                directly for all LLM calls and ``model_load`` is not invoked.
                Useful for testing with stub clients.
            model: Model identifier used when loading the default client.
            asset_manager: Optional ``AssetManager`` instance.  When provided,
                the registered character registry is used to build the
                character list embedded in shot-generation prompts, ensuring
                that LLM-generated character IDs match the registry exactly.
                Falls back to the ``series_config`` characters list when not
                provided or when the registry is empty.
        """
        self.llm = llm_client if llm_client is not None else model_load.load()
        self.model = model
        self._asset_manager = asset_manager

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

        # Phase 2 (legacy) – per-scene shots chain (kept for backward compatibility)
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
        self._shots_chain = self._shots_prompt | self.llm

        # Phase 2 (new) – per-scene shot plan chain
        self._shot_plan_parser = JsonOutputParser(pydantic_object=SceneShotPlanOutput)
        self._shot_plan_prompt = PromptTemplate(
            template=self.SCENE_SHOT_PLAN_PROMPT_TEMPLATE,
            input_variables=[
                "series_title",
                "genre",
                "characters_registry",
                "episode_number",
                "episode_title",
                "logline",
                "all_scenes_summary",
                "scene_number",
                "location",
                "time_of_day",
                "mood",
                "scene_summary",
            ],
            partial_variables={
                "format_instructions": self._shot_plan_parser.get_format_instructions()
            },
        )
        self._shot_plan_chain = self._shot_plan_prompt | self.llm

        # Phase 3 – per-shot full-detail chain
        self._single_shot_parser = JsonOutputParser(pydantic_object=ShotOutput)
        self._single_shot_prompt = PromptTemplate(
            template=self.SINGLE_SHOT_PROMPT_TEMPLATE,
            input_variables=[
                "series_title",
                "genre",
                "characters_registry",
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
                "previous_shots_summary",
                "shot_sequence_number",
                "shot_brief_description",
                "shot_characters",
            ],
            partial_variables={
                "format_instructions": self._single_shot_parser.get_format_instructions()
            },
        )
        self._single_shot_chain = self._single_shot_prompt | self.llm

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
        self._scene_shots_max_attempts = 3

    def generate_episode(
        self,
        series_config: Dict[str, Any],
        episode_outline: str,
        episode_number: int = 1,
    ) -> Episode:
        """Generate a full episode using a three-phase multi-call approach.

        Phase 1: generate the episode blueprint (title, logline, scene
        metadata) in a single LLM call.
        Phase 2: for each scene in the blueprint, call the LLM once to
        produce a minimal shot plan (sequence numbers, brief descriptions,
        and character lists), passing the full blueprint for coherence.
        Phase 3: for each planned shot, call the LLM once to expand it into
        full shot details.  The registered character registry and a running
        summary of previously generated shots in the scene are embedded in
        every prompt so that character IDs are consistent and visual
        continuity is maintained.

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
        characters = self._get_characters_for_registry(series_config)
        characters_registry = self._format_characters_registry(characters)

        scenes_data: List[Dict[str, Any]] = []
        for scene_bp in blueprint.get("scenes", []):
            # Phase 2: plan the shots for this scene (minimal data)
            shot_plan = self._generate_scene_shot_plan(
                series_config=series_config,
                episode_number=episode_number,
                blueprint=blueprint,
                all_scenes_summary=all_scenes_summary,
                scene_blueprint=scene_bp,
                characters_registry=characters_registry,
            )

            # Phase 3: expand each planned shot into full details
            shots_data: List[Dict[str, Any]] = []
            for plan_item in shot_plan.get("shots", []):
                shot_data = self._generate_single_shot(
                    series_config=series_config,
                    episode_number=episode_number,
                    blueprint=blueprint,
                    all_scenes_summary=all_scenes_summary,
                    scene_blueprint=scene_bp,
                    characters_registry=characters_registry,
                    previous_shots=shots_data,
                    plan_item=plan_item,
                )
                shots_data.append(shot_data)

            scene_dict = dict(scene_bp)
            scene_dict["shots"] = shots_data
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
        last_error: Optional[Exception] = None
        for attempt in range(1, self._scene_shots_max_attempts + 1):
            raw_response = self._shots_chain.invoke(input_dict)
            raw_text = self._extract_text_from_response(raw_response)

            try:
                return self._parse_scene_shots_output(raw_text)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Scene shots parse failed (attempt %d/%d) for scene %s: %s",
                    attempt,
                    self._scene_shots_max_attempts,
                    scene_blueprint.get("scene_number", "?"),
                    exc,
                )

                try:
                    repaired_text = self._repair_scene_shots_output(raw_text)
                    return self._parse_scene_shots_output(repaired_text)
                except Exception as repair_exc:
                    last_error = repair_exc
                    logger.warning(
                        "Scene shots JSON repair failed (attempt %d/%d) for scene %s: %s",
                        attempt,
                        self._scene_shots_max_attempts,
                        scene_blueprint.get("scene_number", "?"),
                        repair_exc,
                    )

        raise ValueError(
            f"Failed to generate valid scene shots JSON after {self._scene_shots_max_attempts} attempts"
        ) from last_error

    def _parse_scene_shots_output(self, raw_text: str) -> Dict[str, Any]:
        """Parse and validate Phase-2 scene-shots output."""
        parsed = self._shots_parser.parse(raw_text)
        validated = SceneShotsOutput.model_validate(parsed)
        return validated.model_dump()

    def _repair_scene_shots_output(self, broken_output: str) -> str:
        """Ask the LLM to repair malformed scene-shots JSON."""
        repair_prompt = self.SHOTS_JSON_REPAIR_PROMPT_TEMPLATE.format(
            format_instructions=self._shots_parser.get_format_instructions(),
            broken_output=broken_output,
        )

        if hasattr(self.llm, "invoke"):
            repaired = self.llm.invoke(repair_prompt)
        else:
            repaired = self.llm(repair_prompt)

        return self._extract_text_from_response(repaired)

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

    def _get_characters_for_registry(
        self, series_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Return the character list to embed in LLM prompts.

        Prefers the ``asset_manager`` registry (if provided and non-empty)
        so that character IDs in generated shots match the production registry
        exactly.  Falls back to the ``series_config`` ``characters`` list.

        Args:
            series_config: Series configuration dictionary.

        Returns:
            List of character dicts, each containing at least ``id`` and
            ``name`` keys.
        """
        if self._asset_manager is not None:
            all_chars = self._asset_manager.get_all_characters()
            if all_chars:
                return [
                    {
                        "id": c.id,
                        "name": c.name,
                        "occupation": c.occupation,
                        "age": c.age,
                        "gender": c.gender,
                    }
                    for c in all_chars.values()
                ]
        return series_config.get("characters", [])

    @staticmethod
    def _format_characters_registry(characters: List[Dict[str, Any]]) -> str:
        """Format the character list for embedding in prompts.

        The output clearly highlights the exact ``id`` values so the LLM uses
        them verbatim in ``characters_in_shot`` fields.

        Args:
            characters: List of character dicts.

        Returns:
            A multi-line string describing each character with its exact ID.
        """
        if not characters:
            return "(No characters defined)"
        lines = []
        for c in characters:
            char_id = c.get("id", "")
            name = c.get("name", "")
            occupation = c.get("occupation", "")
            line = f'- ID: "{char_id}" | Name: {name}'
            if occupation:
                line += f" | Occupation: {occupation}"
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _format_previous_shots_summary(shots: List[Dict[str, Any]]) -> str:
        """Format shots already generated in a scene as a continuity context.

        Args:
            shots: List of shot dicts generated so far in the current scene.

        Returns:
            A multi-line string summarising each previous shot.
        """
        if not shots:
            return "(This is the first shot in the scene – no prior shots.)"
        lines = []
        for s in shots:
            chars = ", ".join(s.get("characters_in_shot", [])) or "no characters"
            dialogue = ""
            if s.get("dialogue"):
                dialogue = f' | Dialogue: "{s["dialogue"]}"'
            lines.append(
                f"Shot {s.get('sequence_number', '?')}: "
                f"{s.get('action_description', s.get('brief_description', ''))}"
                f" [Characters: {chars}]{dialogue}"
            )
        return "\n".join(lines)

    def _generate_scene_shot_plan(
        self,
        series_config: Dict[str, Any],
        episode_number: int,
        blueprint: Dict[str, Any],
        all_scenes_summary: str,
        scene_blueprint: Dict[str, Any],
        characters_registry: str,
    ) -> Dict[str, Any]:
        """Phase 2 (new): Generate a minimal shot plan for one scene.

        Produces only the sequence numbers, brief descriptions, and character
        lists for each planned shot.  Full details are expanded in Phase 3.

        Args:
            series_config: Series metadata and character definitions.
            episode_number: Episode number.
            blueprint: Full episode blueprint returned by Phase 1.
            all_scenes_summary: Pre-formatted summary of all scenes.
            scene_blueprint: The specific scene's blueprint dict.
            characters_registry: Pre-formatted character registry string.

        Returns:
            Parsed dictionary with a ``shots`` key containing
            :class:`ShotPlanItem` dicts.
        """
        input_dict = {
            "series_title": series_config.get("title", "Untitled"),
            "genre": series_config.get("genre", "drama"),
            "characters_registry": characters_registry,
            "episode_number": episode_number,
            "episode_title": blueprint.get("episode_title", ""),
            "logline": blueprint.get("logline", ""),
            "all_scenes_summary": all_scenes_summary,
            "scene_number": scene_blueprint.get("scene_number", 1),
            "location": scene_blueprint.get("location", "Unknown"),
            "time_of_day": scene_blueprint.get("time_of_day", "day"),
            "mood": scene_blueprint.get("mood", "neutral"),
            "scene_summary": scene_blueprint.get("scene_summary", ""),
        }
        last_error: Optional[Exception] = None
        for attempt in range(1, self._scene_shots_max_attempts + 1):
            raw_response = self._shot_plan_chain.invoke(input_dict)
            raw_text = self._extract_text_from_response(raw_response)

            try:
                return self._parse_shot_plan_output(raw_text)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Shot plan parse failed (attempt %d/%d) for scene %s: %s",
                    attempt,
                    self._scene_shots_max_attempts,
                    scene_blueprint.get("scene_number", "?"),
                    exc,
                )

                try:
                    repaired_text = self._repair_generic_output(
                        raw_text,
                        self._shot_plan_parser.get_format_instructions(),
                    )
                    return self._parse_shot_plan_output(repaired_text)
                except Exception as repair_exc:
                    last_error = repair_exc
                    logger.warning(
                        "Shot plan JSON repair failed (attempt %d/%d) for scene %s: %s",
                        attempt,
                        self._scene_shots_max_attempts,
                        scene_blueprint.get("scene_number", "?"),
                        repair_exc,
                    )

        raise ValueError(
            f"Failed to generate valid shot plan JSON after "
            f"{self._scene_shots_max_attempts} attempts"
        ) from last_error

    def _parse_shot_plan_output(self, raw_text: str) -> Dict[str, Any]:
        """Parse and validate Phase-2 shot-plan output."""
        parsed = self._shot_plan_parser.parse(raw_text)
        validated = SceneShotPlanOutput.model_validate(parsed)
        return validated.model_dump()

    def _generate_single_shot(
        self,
        series_config: Dict[str, Any],
        episode_number: int,
        blueprint: Dict[str, Any],
        all_scenes_summary: str,
        scene_blueprint: Dict[str, Any],
        characters_registry: str,
        previous_shots: List[Dict[str, Any]],
        plan_item: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Phase 3: Generate full details for a single planned shot.

        Each call embeds the complete episode blueprint, the scene context,
        the registered character registry, and a summary of shots already
        generated in the scene, ensuring that character IDs are consistent
        with the production registry and that visual continuity is maintained.

        Args:
            series_config: Series metadata and character definitions.
            episode_number: Episode number.
            blueprint: Full episode blueprint returned by Phase 1.
            all_scenes_summary: Pre-formatted summary of all scenes.
            scene_blueprint: The specific scene's blueprint dict.
            characters_registry: Pre-formatted character registry string with
                exact character IDs.
            previous_shots: List of shot dicts already generated for this
                scene (used to build the continuity context).
            plan_item: The :class:`ShotPlanItem` dict for the shot to expand.

        Returns:
            A :class:`ShotOutput`-compatible dict with full shot details.
        """
        ambient = scene_blueprint.get("ambient_sounds", [])
        shot_chars = plan_item.get("characters_in_shot", [])
        input_dict = {
            "series_title": series_config.get("title", "Untitled"),
            "genre": series_config.get("genre", "drama"),
            "characters_registry": characters_registry,
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
            "previous_shots_summary": self._format_previous_shots_summary(previous_shots),
            "shot_sequence_number": plan_item.get("sequence_number", 1),
            "shot_brief_description": plan_item.get("brief_description", ""),
            "shot_characters": ", ".join(shot_chars) if shot_chars else "none",
        }
        last_error: Optional[Exception] = None
        for attempt in range(1, self._scene_shots_max_attempts + 1):
            raw_response = self._single_shot_chain.invoke(input_dict)
            raw_text = self._extract_text_from_response(raw_response)

            try:
                return self._parse_single_shot_output(raw_text)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Single shot parse failed (attempt %d/%d) for scene %s shot %s: %s",
                    attempt,
                    self._scene_shots_max_attempts,
                    scene_blueprint.get("scene_number", "?"),
                    plan_item.get("sequence_number", "?"),
                    exc,
                )

                try:
                    repaired_text = self._repair_single_shot_output(raw_text)
                    return self._parse_single_shot_output(repaired_text)
                except Exception as repair_exc:
                    last_error = repair_exc
                    logger.warning(
                        "Single shot JSON repair failed (attempt %d/%d) for scene %s shot %s: %s",
                        attempt,
                        self._scene_shots_max_attempts,
                        scene_blueprint.get("scene_number", "?"),
                        plan_item.get("sequence_number", "?"),
                        repair_exc,
                    )

        raise ValueError(
            f"Failed to generate valid single-shot JSON after "
            f"{self._scene_shots_max_attempts} attempts"
        ) from last_error

    def _parse_single_shot_output(self, raw_text: str) -> Dict[str, Any]:
        """Parse and validate Phase-3 single-shot output."""
        parsed = self._single_shot_parser.parse(raw_text)
        validated = ShotOutput.model_validate(parsed)
        return validated.model_dump()

    def _repair_single_shot_output(self, broken_output: str) -> str:
        """Ask the LLM to repair malformed single-shot JSON."""
        return self._repair_generic_output(
            broken_output,
            self._single_shot_parser.get_format_instructions(),
        )

    def _repair_generic_output(
        self, broken_output: str, format_instructions: str
    ) -> str:
        """Ask the LLM to repair malformed JSON given a format schema."""
        repair_prompt = self.SINGLE_SHOT_JSON_REPAIR_PROMPT_TEMPLATE.format(
            format_instructions=format_instructions,
            broken_output=broken_output,
        )
        if hasattr(self.llm, "invoke"):
            repaired = self.llm.invoke(repair_prompt)
        else:
            repaired = self.llm(repair_prompt)
        return self._extract_text_from_response(repaired)

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
            generation_mode=(
                GenerationMode.REFERENCE_TO_VIDEO
                if shot_data.get("characters_in_shot")
                else GenerationMode.TEXT_TO_VIDEO
            ),
            duration=shot_data.get("duration", 8),
            camera_motion=camera_motion,
            lighting_description=shot_data.get(
                "lighting_description", "cinematic lighting"
            ),
            text_prompt=shot_data.get("text_prompt", ""),
        )

