"""Integration test for production pipeline with Ark text-to-video generation.

This test runs a real pipeline path:
AITVStudio -> ScriptGenerator -> ShotPlanner -> VideoDirectorServer -> ArkVideoAPIClient

It is intentionally opt-in and requires:
1. Valid API key in config/video_api_keys.yaml (or ARK_API_KEY)
2. Network access
3. ENABLE_REAL_PIPELINE_TEST=1
"""

import json
import os
from pathlib import Path
import tempfile

import pytest

from src.mcp import ArkVideoAPIClient, VideoDirectorServer
from src.studio import AITVStudio


class _DeterministicLLMClient:
    """LLM stub that returns deterministic JSON script payload."""

    def chat(self, model, messages):
        _ = model
        _ = messages
        return json.dumps(
            {
                "episode_title": "Pipeline Integration Pilot",
                "logline": "A single-shot integration run for pipeline validation.",
                "scenes": [
                    {
                        "scene_number": 1,
                        "location": "Cyberpunk street",
                        "time_of_day": "night",
                        "weather": "clear",
                        "mood": "mysterious",
                        "shots": [
                            {
                                "sequence_number": 1,
                                "action_description": "Camera slowly pushes through a neon-lit alley.",
                                "duration": 5,
                                "characters_in_shot": [],
                                "text_prompt": "cinematic, neon alley at night, slow push-in, realistic"
                            }
                        ],
                    }
                ],
            }
        )


def _run_production_pipeline_txt2video_and_download(tmp_path: Path) -> None:

    cfg_path = Path("config/video_api_keys.yaml")
    if not cfg_path.exists():
        raise RuntimeError("Missing config/video_api_keys.yaml")

    studio_output = tmp_path / "final_output"
    generated_output = tmp_path / "generated_videos"

    studio = AITVStudio(
        config={
            "output_dir": str(studio_output),
            "data_dir": str(tmp_path / "data"),
        }
    )
    studio.configure_llm(_DeterministicLLMClient(), model="gpt-4")

    ark_client = ArkVideoAPIClient(
        config_path=str(cfg_path),
        output_dir=str(generated_output),
        poll_interval_seconds=10,
        timeout_seconds=900,
    )
    studio.configure_mcp(VideoDirectorServer(api_client=ark_client))

    episode = studio.produce_episode(
        series_config={"title": "Integration Series", "genre": "sci-fi", "characters": []},
        episode_outline="Validate end-to-end generation pipeline with one text-to-video shot.",
        episode_number=1,
    )

    all_shots = [shot for scene in episode.scenes for shot in scene.shots]
    assert all_shots, "No shots produced by pipeline"

    generated_paths = [Path(s.generated_video_path) for s in all_shots if s.generated_video_path]
    assert generated_paths, "No generated video path found on any shot"
    assert generated_paths[0].exists(), "Downloaded video file does not exist"
    assert generated_paths[0].parent == generated_output

    assembled_episode_path = studio_output / "episode_01.mp4"
    assert assembled_episode_path.exists(), "Final assembled episode was not produced"


def test_production_pipeline_txt2video_and_download(tmp_path):
    if os.environ.get("ENABLE_REAL_PIPELINE_TEST") != "1":
        pytest.skip("Set ENABLE_REAL_PIPELINE_TEST=1 to run real Ark pipeline test")
    if not Path("config/video_api_keys.yaml").exists():
        pytest.skip("Missing config/video_api_keys.yaml")

    _run_production_pipeline_txt2video_and_download(tmp_path=tmp_path)


if __name__ == "__main__":
    run_dir = Path(tempfile.mkdtemp(prefix="ai_tv_pipeline_smoke_"))
    print(f"Smoke test workspace: {run_dir}")
    try:
        _run_production_pipeline_txt2video_and_download(tmp_path=run_dir)
        print("Pipeline smoke test succeeded.")
        print(f"Generated videos dir: {run_dir / 'generated_videos'}")
        print(f"Final episode dir: {run_dir / 'final_output'}")
    except Exception as exc:
        print(f"Generated videos dir: {run_dir / 'generated_videos'}")
        print(f"Final episode dir: {run_dir / 'final_output'}")
        print(f"Workspace preserved for debugging: {run_dir}")
        print(f"Pipeline smoke test failed: {exc}")
        raise
