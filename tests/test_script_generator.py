"""Tests for the ScriptGenerator pipeline module."""

import json
import os

import pytest

from src.models.character import CharacterEmotion
from src.pipeline.script_generator import ScriptGenerator


class DummyLLM:
    """Simple LLM stub that records prompts and returns a fixed response.

    Compatible with LangChain chain composition (``prompt | llm | parser``)
    because it implements ``__call__``, which LangChain wraps in a
    ``RunnableLambda`` automatically.
    """

    def __init__(self, response):
        self.response = response
        self.calls = []

    def __call__(self, prompt):
        to_str = getattr(prompt, "to_string", None)
        prompt_str = to_str() if callable(to_str) else str(prompt)
        self.calls.append(prompt_str)
        return self.response

    def invoke(self, prompt):
        return self(prompt)


class ResponseObject:
    """Mimics an object response with a content field."""

    def __init__(self, content):
        self.content = content


class TestScriptGenerator:
    def _series_config(self):
        return {
            "title": "Midnight Detective",
            "genre": "crime",
            "characters": [
                {"id": "char_lin", "name": "Lin Xiao"},
                {"id": "char_chen", "name": "Chen Wei"},
            ],
        }

    def _valid_script_payload(self):
        return {
            "episode_title": "The Silent Alley",
            "logline": "Lin follows a clue into an alley and finds a witness.",
            "scenes": [
                {
                    "scene_number": 1,
                    "location": "Old Alley",
                    "time_of_day": "night",
                    "weather": "fog",
                    "mood": "tense",
                    "bgm_mood": "suspenseful",
                    "bgm_tempo": "moderate",
                    "ambient_sounds": ["distant siren"],
                    "shots": [
                        {
                            "sequence_number": 1,
                            "action_description": "Lin enters the alley.",
                            "dialogue": "Stay where you are!",
                            "characters_in_shot": ["char_lin"],
                            "character_emotions": {
                                "char_lin": "neutral",
                                "char_chen": "not_an_emotion",
                            },
                            "duration": 7,
                            "camera_motion": {"type": "push_in", "speed": 1.2},
                            "lighting_description": "cold streetlight",
                            "text_prompt": "cinematic noir alley at night",
                        },
                        {
                            "sequence_number": 2,
                            "action_description": "A shadow disappears.",
                            "duration": 9,
                            "text_prompt": "shadow running through fog",
                        },
                    ],
                }
            ],
        }

    def test_generate_episode_success(self):
        payload = self._valid_script_payload()
        llm = DummyLLM(json.dumps(payload))
        generator = ScriptGenerator(llm_client=llm)

        episode = generator.generate_episode(
            series_config=self._series_config(),
            episode_outline="A clue leads Lin into a dangerous alley.",
            episode_number=3,
        )

        assert episode.series_title == "Midnight Detective"
        assert episode.episode_number == 3
        assert episode.episode_title == "The Silent Alley"
        assert episode.characters == ["char_lin", "char_chen"]
        assert len(episode.scenes) == 1
        assert len(episode.scenes[0].shots) == 2
        assert episode.runtime_estimate == 16

        shot = episode.scenes[0].shots[0]
        assert shot.camera_motion is not None
        assert shot.camera_motion.type == "push_in"
        assert shot.character_emotions == {"char_lin": CharacterEmotion.NEUTRAL}

        assert len(llm.calls) == 1
        assert "STRICT OUTPUT FORMAT" in llm.calls[0]

    def test_parse_response_accepts_markdown_wrapped_json(self):
        payload = self._valid_script_payload()
        raw = """```json\n""" + json.dumps(payload) + """\n```"""
        generator = ScriptGenerator(llm_client=DummyLLM("{}"))

        parsed = generator._parse_response(raw)
        assert parsed["episode_title"] == "The Silent Alley"

    def test_parse_response_raises_on_invalid_json(self):
        generator = ScriptGenerator(llm_client=DummyLLM("{}"))

        with pytest.raises(ValueError, match="Failed to parse LLM response as JSON"):
            generator._parse_response("this is not json")

    def test_extract_text_from_response_object_content_list(self):
        response = ResponseObject(
            [
                {"text": '{"episode_title": "E1", "logline": "L", "scenes": []}'},
                {"ignored": "x"},
            ]
        )
        generator = ScriptGenerator(llm_client=DummyLLM("{}"))

        text = generator._extract_text_from_response(response)
        assert '"episode_title": "E1"' in text

    def test_compute_runtime_sums_all_shot_durations(self):
        payload = self._valid_script_payload()
        llm = DummyLLM(json.dumps(payload))
        generator = ScriptGenerator(llm_client=llm)

        episode = generator.generate_episode(
            series_config=self._series_config(),
            episode_outline="Outline",
            episode_number=1,
        )
        assert generator._compute_runtime(episode) == 16


@pytest.mark.integration
def test_script_generator_real_run_prints_runtime_status():
    """Run ScriptGenerator with the real configured LLM and print key outputs.

    Enable with:
    RUN_REAL_SCRIPT_GENERATOR=1 python -m pytest -s -q tests/test_script_generator.py -k real_run
    """

    series_config = {
        "title": "Midnight Detective",
        "genre": "crime",
        "characters": [
            {"id": "char_lin", "name": "Lin Xiao"},
            {"id": "char_chen", "name": "Chen Wei"},
        ],
    }
    episode_outline = (
        "Lin Xiao receives a late-night tip about a witness hiding in an old alley, "
        "confronts danger, and secures a critical clue before dawn."
    )

    generator = ScriptGenerator()
    episode = generator.generate_episode(
        series_config=series_config,
        episode_outline=episode_outline,
        episode_number=1,
    )

    print("\n=== ScriptGenerator Real Run ===")
    print(f"episode_id={episode.id}")
    print(f"title={episode.episode_title}")
    print(f"logline={episode.logline}")
    print(f"scenes={len(episode.scenes)}")
    print(f"runtime_estimate={episode.runtime_estimate}")
    for scene in episode.scenes:
        print(
            f"scene#{scene.scene_number} location={scene.location} shots={len(scene.shots)}"
        )
        for shot in scene.shots:
            motion = shot.camera_motion.type if shot.camera_motion else "none"
            print(
                f"  shot#{shot.sequence_number} duration={shot.duration} motion={motion}"
            )

    assert episode.episode_title
    assert len(episode.scenes) > 0
    assert episode.runtime_estimate > 0

if __name__ == "__main__":
    test_script_generator_real_run_prints_runtime_status()