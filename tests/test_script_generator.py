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


class MultiResponseDummyLLM:
    """LLM stub that returns a different response for each successive call.

    Responses are consumed in order; the last response is repeated if more
    calls are made than responses provided.  All prompts are recorded in
    ``calls`` for assertion in tests.
    """

    def __init__(self, responses):
        self.responses = responses if isinstance(responses, list) else [responses]
        self.calls = []
        self._call_index = 0

    def __call__(self, prompt):
        to_str = getattr(prompt, "to_string", None)
        prompt_str = to_str() if callable(to_str) else str(prompt)
        self.calls.append(prompt_str)
        idx = min(self._call_index, len(self.responses) - 1)
        self._call_index += 1
        return self.responses[idx]

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

    def _valid_blueprint_payload(self):
        """Phase 1 response: episode structure without individual shots."""
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
                    "scene_summary": "Lin enters the alley and spots a shadowy figure.",
                }
            ],
        }

    def _valid_shots_payload(self):
        """Phase 2 response: shots for a single scene."""
        return {
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
            ]
        }

    def _valid_script_payload(self):
        """Legacy combined payload (kept for _parse_response tests)."""
        payload = self._valid_blueprint_payload()
        payload["scenes"][0]["shots"] = self._valid_shots_payload()["shots"]
        return payload

    def test_generate_episode_success(self):
        """Two-phase generation: blueprint call then per-scene shots call."""
        llm = MultiResponseDummyLLM([
            json.dumps(self._valid_blueprint_payload()),
            json.dumps(self._valid_shots_payload()),
        ])
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

        # Two-phase: 1 blueprint call + 1 shots call (one per scene)
        assert len(llm.calls) == 2
        # Phase 1 blueprint prompt must include the strict format marker
        assert "STRICT OUTPUT FORMAT" in llm.calls[0]
        # Phase 2 shots prompt must embed the full episode blueprint for coherence
        assert "EPISODE BLUEPRINT" in llm.calls[1]

    def test_generate_episode_multiple_scenes(self):
        """Each scene in the blueprint triggers its own shots LLM call."""
        blueprint = {
            "episode_title": "Two Scenes",
            "logline": "A two-scene episode.",
            "scenes": [
                {
                    "scene_number": 1,
                    "location": "Rooftop",
                    "time_of_day": "dusk",
                    "weather": "clear",
                    "mood": "tense",
                    "bgm_mood": "dramatic",
                    "bgm_tempo": "fast",
                    "ambient_sounds": [],
                    "scene_summary": "Chase begins.",
                },
                {
                    "scene_number": 2,
                    "location": "Alley",
                    "time_of_day": "night",
                    "weather": "rain",
                    "mood": "dark",
                    "bgm_mood": "ominous",
                    "bgm_tempo": "slow",
                    "ambient_sounds": ["rain"],
                    "scene_summary": "Chase ends.",
                },
            ],
        }
        shots = {"shots": [{"sequence_number": 1, "action_description": "Shot.", "duration": 5, "text_prompt": "p"}]}
        llm = MultiResponseDummyLLM([
            json.dumps(blueprint),
            json.dumps(shots),
            json.dumps(shots),
        ])
        generator = ScriptGenerator(llm_client=llm)

        episode = generator.generate_episode(
            series_config=self._series_config(),
            episode_outline="A chase across the city.",
            episode_number=2,
        )

        assert len(episode.scenes) == 2
        # 1 blueprint call + 2 shots calls (one per scene)
        assert len(llm.calls) == 3
        # Both shots prompts embed the full blueprint
        assert "EPISODE BLUEPRINT" in llm.calls[1]
        assert "EPISODE BLUEPRINT" in llm.calls[2]
        # Each shots prompt references all scene summaries for coherence
        assert "Chase begins" in llm.calls[1]
        assert "Chase begins" in llm.calls[2]

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
        llm = MultiResponseDummyLLM([
            json.dumps(self._valid_blueprint_payload()),
            json.dumps(self._valid_shots_payload()),
        ])
        generator = ScriptGenerator(llm_client=llm)

        episode = generator.generate_episode(
            series_config=self._series_config(),
            episode_outline="Outline",
            episode_number=1,
        )
        assert generator._compute_runtime(episode) == 16

    def test_format_all_scenes_summary(self):
        """Utility method formats scene list into a readable narrative overview."""
        scenes = [
            {"scene_number": 1, "location": "Park", "time_of_day": "morning", "mood": "calm", "scene_summary": "Hero wakes up."},
            {"scene_number": 2, "location": "Office", "time_of_day": "afternoon", "mood": "tense", "scene_summary": "Conflict arises."},
        ]
        summary = ScriptGenerator._format_all_scenes_summary(scenes)

        # Check that both scenes appear and in the right order
        idx1 = summary.index("Scene 1")
        idx2 = summary.index("Scene 2")
        assert idx1 < idx2, "Scene 1 must appear before Scene 2"

        # Verify key fields are present for each scene
        assert "Park" in summary
        assert "morning" in summary
        assert "calm" in summary
        assert "Hero wakes up" in summary

        assert "Office" in summary
        assert "afternoon" in summary
        assert "tense" in summary
        assert "Conflict arises" in summary


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