"""Resume video generation from a persisted episode snapshot.

This utility reloads an ``episode_*_video_generated.json`` snapshot, retries
only missing or failed shots, and optionally assembles the final episode once
all shots are available.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from src.mcp.ark_video_client import ArkVideoAPIClient
from src.mcp.video_director_server import VideoDirectorServer
from src.models.character import CharacterEmotion
from src.models.episode import Episode
from src.models.scene import Scene
from src.models.shot import CameraMotion, GenerationMode, Shot
from src.studio import AITVStudio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume failed or missing shot generation from an episode snapshot."
    )
    parser.add_argument(
        "--episode-json",
        required=True,
        help="Path to an episode_*_video_generated.json snapshot.",
    )
    parser.add_argument(
        "--video-api-config",
        default="config/video_api_keys.yaml",
        help="Path to video API credentials/config.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Where to write the updated episode JSON. Defaults to in-place overwrite.",
    )
    parser.add_argument(
        "--output-dir",
        default="./outputs",
        help="Directory for generated videos and assembled output.",
    )
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Data directory for studio initialization.",
    )
    parser.add_argument(
        "--skip-assembly",
        action="store_true",
        help="Do not assemble the final episode video after resuming generation.",
    )
    parser.add_argument(
        "--allow-incomplete-assembly",
        action="store_true",
        help="Assemble even if some shots are still missing. Disabled by default.",
    )
    parser.add_argument(
        "--assemble-only",
        action="store_true",
        help=(
            "Skip all generation and immediately assemble the episode from "
            "whatever video files are already on disk. "
            "Implies --allow-incomplete-assembly."
        ),
    )
    return parser.parse_args()


def load_episode_snapshot(snapshot_path: str) -> Episode:
    path = Path(snapshot_path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    scenes = [load_scene(scene_data) for scene_data in payload.get("scenes", [])]
    return Episode(
        id=payload["id"],
        series_title=payload.get("series_title", "Untitled"),
        episode_number=int(payload.get("episode_number", 1)),
        episode_title=payload.get("episode_title", "Untitled Episode"),
        logline=payload.get("logline", ""),
        characters=payload.get("characters", []),
        scenes=scenes,
        genre=payload.get("genre", "drama"),
        runtime_estimate=payload.get("runtime_estimate", 0),
    )


def load_scene(scene_data: Dict[str, Any]) -> Scene:
    shots = [load_shot(shot_data) for shot_data in scene_data.get("shots", [])]
    return Scene(
        id=scene_data["id"],
        episode_id=scene_data.get("episode_id", ""),
        scene_number=int(scene_data.get("scene_number", 0)),
        location=scene_data.get("location", ""),
        time_of_day=scene_data.get("time_of_day", "day"),
        weather=scene_data.get("weather", "clear"),
        mood=scene_data.get("mood", "neutral"),
        shots=shots,
        bgm_mood=scene_data.get("bgm_mood", "neutral"),
        bgm_tempo=scene_data.get("bgm_tempo", "moderate"),
        ambient_sounds=scene_data.get("ambient_sounds", []),
    )


def load_shot(shot_data: Dict[str, Any]) -> Shot:
    camera_motion = None
    camera_data = shot_data.get("camera_motion") or None
    if camera_data:
        camera_motion = CameraMotion(
            type=camera_data.get("type", "static"),
            speed=camera_data.get("speed", 1.0),
            start_position=camera_data.get("start_position"),
            end_position=camera_data.get("end_position"),
        )

    character_emotions = {}
    for char_id, emotion in (shot_data.get("character_emotions") or {}).items():
        if emotion in {item.value for item in CharacterEmotion}:
            character_emotions[char_id] = CharacterEmotion(emotion)
        else:
            character_emotions[char_id] = emotion

    generation_mode = shot_data.get("generation_mode", GenerationMode.TEXT_TO_VIDEO.value)
    if generation_mode in {item.value for item in GenerationMode}:
        generation_mode = GenerationMode(generation_mode)

    return Shot(
        id=shot_data["id"],
        scene_id=shot_data.get("scene_id", ""),
        sequence_number=int(shot_data.get("sequence_number", 0)),
        action_description=shot_data.get("action_description", ""),
        dialogue=shot_data.get("dialogue"),
        internal_thought=shot_data.get("internal_thought"),
        characters_in_shot=shot_data.get("characters_in_shot", []),
        character_emotions=character_emotions,
        generation_mode=generation_mode,
        duration=int(shot_data.get("duration", 8)),
        camera_motion=camera_motion,
        lighting_description=shot_data.get("lighting_description", "cinematic lighting"),
        text_prompt=shot_data.get("text_prompt", ""),
        start_frame_path=shot_data.get("start_frame_path"),
        end_frame_path=shot_data.get("end_frame_path"),
        reference_images=shot_data.get("reference_images", []),
        generated_video_path=shot_data.get("generated_video_path"),
        generation_error=shot_data.get("generation_error"),
        effective_prompt=shot_data.get("effective_prompt", ""),
        generation_trace=shot_data.get("generation_trace", {}),
        previous_shot_tail=shot_data.get("previous_shot_tail"),
        is_transition_shot=bool(shot_data.get("is_transition_shot", False)),
        transition_type=shot_data.get("transition_type"),
    )


def save_episode_snapshot(episode: Episode, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(episode), f, ensure_ascii=False, indent=2)


def count_missing_or_failed_shots(episode: Episode) -> int:
    count = 0
    for scene in episode.scenes:
        for shot in scene.shots:
            if not shot.generated_video_path or not Path(shot.generated_video_path).exists():
                count += 1
    return count


def main() -> None:
    args = parse_args()
    snapshot_path = Path(args.episode_json)
    output_json = Path(args.output_json) if args.output_json else snapshot_path

    episode = load_episode_snapshot(str(snapshot_path))

    studio = AITVStudio(
        config={
            "output_dir": args.output_dir,
            "data_dir": args.data_dir,
        }
    )

    if args.assemble_only:
        missing = count_missing_or_failed_shots(episode)
        print(f"Assemble-only mode. Shots missing on disk: {missing}")
        assembled_path = studio.video_assembler.assemble_episode(episode)
        print(f"Assembled episode: {assembled_path}")
        return

    ark_client = ArkVideoAPIClient(
        config_path=args.video_api_config,
        output_dir=args.output_dir,
    )
    studio.configure_mcp(VideoDirectorServer(api_client=ark_client))

    before_missing = count_missing_or_failed_shots(episode)
    episode = studio.resume_episode_generation(episode)
    after_missing = count_missing_or_failed_shots(episode)
    save_episode_snapshot(episode, str(output_json))

    print(f"Snapshot updated: {output_json}")
    print(f"Shots needing work before resume: {before_missing}")
    print(f"Shots still missing after resume: {after_missing}")

    if args.skip_assembly:
        return

    if after_missing and not args.allow_incomplete_assembly:
        print(
            "Assembly skipped because some shots are still missing. "
            "Use --allow-incomplete-assembly to override."
        )
        return

    assembled_path = studio.video_assembler.assemble_episode(episode)
    print(f"Assembled episode: {assembled_path}")


if __name__ == "__main__":
    main()